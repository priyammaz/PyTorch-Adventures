import os    
import math
import numpy as np
import random
import json
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator
import argparse
import shutil
import itertools

### Import our Diffusion Model and DDPM Sampler ###
from model import Diffusion
from scheduler import Sampler

### DONT USE PARALLEL TOKENIZER ###
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#### DEFINE ARGUMENT PARSER ###
parser = argparse.ArgumentParser(description="Arguments for Denoising Diffusion")
parser.add_argument("--experiment_name", 
                    help="Name of Training Run", 
                    required=True, 
                    type=str)
parser.add_argument("--path_to_data",
                    help="Path to COCO Dataset",
                    required=True,
                    type=str)
parser.add_argument("--working_directory",
                    help="Working Directory where Checkpoints and Logs are stored",
                    required=True, 
                    type=str)
parser.add_argument("--num_keep_checkpoints",   
                    help="Number of the most recent checkpoints to keep",
                    default=1,
                    type=int)
parser.add_argument("--generated_directory",
                    help="Path to folder to store all generated images during training", 
                    required=True, 
                    type=str)
parser.add_argument("--num_diffusion_timesteps",
                    help="Number of timesteps for forward/reverse diffusion process", 
                    default=1000, 
                    type=int)
parser.add_argument("--plot_freq_interval",
                    help="Time pacing between generated images for reverse diffusion visuals", 
                    default=100, 
                    type=int)
parser.add_argument("--num_generations",
                    help="Number of generated images in each visual",
                    default=5, 
                    type=int)
parser.add_argument("--num_training_steps", 
                    help="Number of training steps to take",
                    default=200000,
                    type=int)
parser.add_argument("--evaluation_interval", 
                    help="Number of iterations for every evaluation and plotting",
                    default=5000, 
                    type=int)
parser.add_argument("--batch_size",
                    help="Effective batch size per GPU, multiplied by number of GPUs used",
                    default=256, 
                    type=int)
parser.add_argument("--gradient_accumulation_steps", 
                    help="Number of gradient accumulation steps, splitting set batchsize",
                    default=1, 
                    type=int)
parser.add_argument("--learning_rate", 
                    help="Max learning rate for Cosine LR Scheduler", 
                    default=1e-4, 
                    type=float)
parser.add_argument("--warmup_steps",
                    help="Number of learning rate warmup steps of training",
                    default=5000, 
                    type=int)
parser.add_argument("--bias_weight_decay", 
                    help="Apply weight decay to bias",
                    default=False, 
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--norm_weight_decay",
                    help="Apply weight decay to normalization weight and bias",
                    default=False,
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--max_grad_norm",
                    help="Maximum gradient norm for clipping",
                    default=1.0, 
                    type=float)
parser.add_argument("--weight_decay",
                    help="Weight decay constant for AdamW optimizer", 
                    default=1e-4, 
                    type=float)
parser.add_argument("--loss_fn", 
                    choices=("mae", "mse", "huber"),
                    default="mse", 
                    type=str)
parser.add_argument("--img_size", 
                    help="Width and Height of Images passed to model", 
                    default=192, 
                    type=int)
parser.add_argument("--starting_channels",
                    help="Number of channels in first convolutional projection", 
                    default=224, 
                    type=int)
parser.add_argument("--cfg_weight",
                    help="Weight for Classifier Free Guidance",
                    default=0.1,
                    type=float)
parser.add_argument("--num_workers", 
                    help="Number of workers for DataLoader", 
                    default=32, 
                    type=int)
parser.add_argument("--resume_from_checkpoint", 
                    help="Checkpoint folder for model to resume training from, inside the experiment folder", 
                    default=None, 
                    type=str)
args = parser.parse_args()

##################################################
############### STUFF WE NEED ####################
##################################################

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


#############################################################
####################### BUILD DATASET #######################
#############################################################

class COCODataset(Dataset):
    def __init__(self, path_to_root, train=True, img_size=128):

        self.path_to_root = path_to_root
        
        if train:
            path_to_annotations = os.path.join(self.path_to_root, "annotations", "captions_train2017.json")
            self.path_to_images = os.path.join(self.path_to_root, "train2017")

        else:
            path_to_annotations = os.path.join(self.path_to_root, "annotations", "captions_val2017.json")
            self.path_to_images = os.path.join(self.path_to_root, "val2017")


        self._prepare_annotations(path_to_annotations)


        self.image2tensor = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Lambda(lambda t: (t*2) - 1)
            ]
        )
        
    def _prepare_annotations(self, path_to_annotations):

        ### Load Annotation Json ###
        with open(path_to_annotations, "r") as f:
            annotation_json = json.load(f)
            
        ### For Each Image ID Get the Corresponding Annotations ###
        id_annotations = {}
    
        for annot in annotation_json["annotations"]:
            image_id = annot["image_id"]
            caption = annot["caption"]
    
            if image_id not in id_annotations:
                id_annotations[image_id] = [caption]
            else:
                id_annotations[image_id].append(caption)
    
        ### Coorespond Image Id to Filename ###
        path_id_coorespondance = {}
    
        for image in annotation_json["images"]:
            file_name = image["file_name"]
            image_id = image["id"]
        
            path_id_coorespondance[file_name] = id_annotations[image_id]

        self.filenames = list(path_id_coorespondance.keys())
        self.annotations = path_id_coorespondance


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        ### Grab Filename and Cooresponding Annotations ###
        filename = self.filenames[idx]
        annotation = self.annotations[filename]

        ### If more than 1 annotation, randomly select ###
        annotation = random.choice(annotation)

        ### Remove Any Whitespace from Text ###
        annotation = annotation.strip()

        ### Error Handling on any Broken Images ###
        try:
            ### Load Image ###
            path_to_img = os.path.join(self.path_to_images, filename)
            img = Image.open(path_to_img).convert("RGB")

            ### Apply Image Transforms ###
            img = self.image2tensor(img)
            
            return img, annotation

        except Exception as e:
            print("Exception:", e)
            return None, None

def collate_fn(examples):
    
    ### Remove any potential Nones ###
    examples = [i for i in examples if i[0] is not None]

    if len(examples) > 0:

        ### Grab Images and add batch dimension ###
        images = [i[0].unsqueeze(0) for i in examples]

        ### Grab Text Annotations ###
        annot = [i[1] for i in examples]

        ### Stick All Images Together along Batch ###
        images = torch.concatenate(images)

        ### Tokenize Annotations with Padding ###
        annotation = tokenizer(annot, padding=True, return_tensors="pt")

        ### Store Batch as Dictionary ###
        batch = {"images": images, 
                "context": annotation["input_ids"], 
                "mask": ~annotation["attention_mask"].bool()} # Flipped mask, so True on pad tokens rather than actual tokens

        return batch
    
    else:
        print("Broken Batch!")
        return None


########################################################
############## Sample Generations ######################
########################################################

@torch.no_grad()
def sample_plot_images(step_idx, 
                       total_timesteps, 
                       sampler, 
                       image_size, 
                       num_channels, 
                       plot_freq, 
                       model, 
                       conditional, 
                       path_to_save, 
                       device):

    prompts = ['A red kite flying through a cloudy blue sky.', 
               'A sailboat floating on the ocean', 
               'A man with an umbrella walking through the rain']

    num_gens = len(prompts)
    
    if not conditional:
        prompts = None
        mask = None
        
    else:
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        prompts = tokenized["input_ids"].to(device)
        mask = tokenized["attention_mask"].to(device)
        mask = ~mask.bool()
        
    tensor2image_transform = transforms.Compose([
        transforms.Lambda(lambda t: t.squeeze(0)),
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: torch.clamp(t, min=0, max=255)),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    images = torch.randn((num_gens, num_channels, image_size, image_size))
    num_images_per_gen = (total_timesteps // plot_freq)

    images_to_vis = [[] for _ in range(num_gens)]

    for t in np.arange(total_timesteps)[::-1]:
        ts = torch.full((num_gens, ), t)
        noise_pred = model(images.to(device), 
                           ts.to(device),
                           context=prompts,
                           mask=mask).detach().cpu()
        
        images = sampler.remove_noise(images, ts, noise_pred)

        if t % plot_freq == 0:
            for idx, image in enumerate(images):
                images_to_vis[idx].append(tensor2image_transform(image))

    images_to_vis = list(itertools.chain(*images_to_vis))

    fig, axes = plt.subplots(nrows=num_gens, ncols=num_images_per_gen, figsize=(num_images_per_gen, num_gens))
    plt.tight_layout()
    for ax, image in zip(axes.ravel(), images_to_vis):
        ax.imshow(image)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(path_to_save, f"{step_idx}.png"), dpi=300)
    return fig


########################################################
############## TRAINING SCRIPT #########################
########################################################

torch.backends.cudnn.benchmark = True

### Instantiate MultiGPU ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment, 
                          gradient_accumulation_steps=args.gradient_accumulation_steps, 
                          log_with="wandb")
accelerator.init_trackers(args.experiment_name)

### Prep  Dataset ###
mini_batch_size = args.batch_size // args.gradient_accumulation_steps
dataset = COCODataset(args.path_to_data, train=True, img_size=args.img_size)
trainloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True, 
                         num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

### Prep Model ###
model = Diffusion(in_channels=3, 
                  context_embed_dim=512, # Embed dim for CLIP Text Encoder
                  start_dim=args.starting_channels, 
                  dim_mults=(1,2,3,4), 
                  residual_blocks_per_group=2, 
                  groupnorm_num_groups=16, 
                  time_embed_dim=128, 
                  time_embed_dim_ratio=4)


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of Parameters:", params)

### PREPARE OPTIMIZER ###
if (not args.bias_weight_decay) or (not args.norm_weight_decay):
    accelerator.print("Disabling Weight Decay on Some Parameters")
    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():

        if param.requires_grad:
            
            ### Dont have Weight decay on any bias parameter (including norm) ###
            if "bias" in name and not args.bias_weight_decay:
                no_weight_decay_params.append(param)

            ### Dont have Weight Decay on any Norm scales params (weights) ###
            elif "groupnorm" in name and not args.norm_weight_decay:
                no_weight_decay_params.append(param)

            else:
                weight_decay_params.append(param)

    optimizer_group = [
        {"params": weight_decay_params, "weight_decay": args.weight_decay},
        {"params": no_weight_decay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_group, lr=args.learning_rate)

else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

### DEFINE SCHEDULER ###
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                            num_warmup_steps=args.warmup_steps*accelerator.num_processes, 
                                            num_training_steps=args.num_training_steps*accelerator.num_processes)


### Prepare Everything ###
model, optimizer, trainloader, scheduler = accelerator.prepare(
    model, optimizer, trainloader, scheduler
)
accelerator.register_for_checkpointing(scheduler)

### RESUME FROM CHECKPOINT ###
if args.resume_from_checkpoint is not None:
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    accelerator.load_state(path_to_checkpoint)
    completed_steps = int(args.resume_from_checkpoint.split("_")[-1])
    accelerator.print(f"Resuming from Iteration: {completed_steps}")
else:
    completed_steps = 0

### DEFINE DDPM SAMPLER ###
ddpm_sampler = Sampler(total_timesteps=args.num_diffusion_timesteps)


### DEFINE LOSS FN ###
loss_functions = {"mse": nn.MSELoss(), 
                  "mae": nn.L1Loss(), 
                  "huber": nn.HuberLoss()}
loss_fn = loss_functions[args.loss_fn]
 
### DEFINE TRAINING LOOP ###
progress_bar = tqdm(range(completed_steps, args.num_training_steps), disable=not accelerator.is_main_process)
accumulated_loss = 0
train = True


while train:
    for batch in trainloader:
        if batch is not None:
            images, context, mask = batch["images"], batch["context"], batch["mask"]
            images, context, mask = images.to(accelerator.device), context.to(accelerator.device), mask.to(accelerator.device)
            
            with accelerator.accumulate(model):

                ### Grab Number of Samples in Batch ###
                batch_size = images.shape[0]
                
                ### Random Sample Timesteps ###
                timesteps = torch.randint(0, args.num_diffusion_timesteps, (batch_size,))

                ### Get Noisy Images ###
                noisy_images, noise = ddpm_sampler.add_noise(images, timesteps)

                ### Predict Noise with Diffusion Model ###
                noise_pred = model(noisy_images.to(accelerator.device), 
                                timesteps.to(accelerator.device),
                                context,
                                mask,
                                cfg_weight=args.cfg_weight)

                ### Compute Error ###
                loss = loss_fn(noise_pred, noise.to(accelerator.device))
                accumulated_loss += loss / args.gradient_accumulation_steps

                ### Compute Gradients ###
                accelerator.backward(loss)

                ### Clip Gradients ###
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
                mean_loss_gathered = torch.mean(loss_gathered).item()
                accelerator.log({"loss": mean_loss_gathered,
                                "learning_rate": scheduler.get_last_lr()[0],
                                "iteration": completed_steps},
                                step=completed_steps)

                ### Reset and Iterate ###
                accumulated_loss = 0
                completed_steps += 1
                progress_bar.update(1)

            
                if completed_steps % args.evaluation_interval == 0:

                    ### Save Checkpoint ### 
                    path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")
                    accelerator.save_state(output_dir=path_to_checkpoint)

                    ### Delete Old Checkpoints ###
                    if accelerator.is_main_process:
                        all_checkpoints = os.listdir(path_to_experiment)
                        all_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split(".")[0].split("_")[-1]))
                        
                        if len(all_checkpoints) > args.num_keep_checkpoints:
                            checkpoints_to_delete = all_checkpoints[:-args.num_keep_checkpoints]

                            for checkpoint_to_delete in checkpoints_to_delete:
                                path_to_checkpoint_to_delete = os.path.join(path_to_experiment, checkpoint_to_delete)
                                if os.path.isdir(path_to_checkpoint_to_delete):
                                    shutil.rmtree(path_to_checkpoint_to_delete)




                    ### Inference Model and Save Results ###
                    if accelerator.is_main_process:
                        accelerator.print("Generating Conditional Images")
                        cond_fig =  sample_plot_images(step_idx=completed_steps, 
                                                    total_timesteps=args.num_diffusion_timesteps, 
                                                    sampler=ddpm_sampler, 
                                                    image_size=args.img_size, 
                                                    num_channels=3, 
                                                    plot_freq=args.plot_freq_interval, 
                                                    model=model, 
                                                    conditional=True, 
                                                    path_to_save=os.path.join(args.generated_directory, "conditional"), 
                                                    device=accelerator.device)

                        accelerator.print("Generating Unconditional Images")
                        uncond_fig = sample_plot_images(step_idx=completed_steps, 
                                                        total_timesteps=args.num_diffusion_timesteps, 
                                                        sampler=ddpm_sampler, 
                                                        image_size=args.img_size, 
                                                        num_channels=3, 
                                                        plot_freq=args.plot_freq_interval, 
                                                        model=model, 
                                                        conditional=False, 
                                                        path_to_save=os.path.join(args.generated_directory, "unconditional"), 
                                                        device=accelerator.device)
                        
                        accelerator.log({"conditional_plot": cond_fig,
                                        "unconditional_plot": uncond_fig},
                                        step=completed_steps)


        if completed_steps >= args.num_training_steps:
            train = False
            accelerator.print("Completed Training")
            break

accelerator.end_training()

            
            





















