import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import utils
from model import VisionTransformer

import warnings 
warnings.filterwarnings("ignore")

def parse_args():
    ### Parse Training Arguments ###
    parser = argparse.ArgumentParser(description="Arguments for Image Classification Training")

    ### EXPERIMENT CONFIG ###
    parser.add_argument("--experiment_name", 
                        help="Name of Experiment being Launched", 
                        required=True, 
                        type=str)
    parser.add_argument("--wandb_run_name",
                        required=True, 
                        type=str)
    parser.add_argument("--path_to_data", 
                        help="Path to ImageNet root folder which should contain \train and \validation folders", 
                        required=True, 
                        type=str)
    parser.add_argument("--working_directory", 
                        help="Working Directory where checkpoints and logs are stored, inside a \
                        folder labeled by the experiment name", 
                        required=True, 
                        type=str)


    ### TRAINING CONFIG ###
    parser.add_argument("--num_classes",
                        help="Number of output classes for Image Classification",
                        default=1000,
                        type=int)
    parser.add_argument("--epochs",
                        help="Number of Epochs to Train",
                        default=300, 
                        type=int)
    parser.add_argument("--warmup_epochs",
                        help="Number of Epochs to Train",
                        default=30, 
                        type=int)
    parser.add_argument("--save_checkpoint_interval", 
                        help="After how many epochs to save model checkpoints",
                        default=10,
                        type=int)
    parser.add_argument("--per_gpu_batch_size", 
                        help="Effective batch size. If split_batches is false, batch size is \
                            multiplied by number of GPUs utilized ", 
                        default=256, 
                        type=int)
    parser.add_argument("--gradient_accumulation_steps", 
                        help="Number of Gradient Accumulation Steps for Training", 
                        default=1, 
                        type=int)
    parser.add_argument("--learning_rate", 
                        help="Starting Learning Rate for StepLR", 
                        default=0.003,
                        type=float)
    parser.add_argument("--weight_decay", 
                        help="Weight decay for optimizer", 
                        default=0.1, 
                        type=float)
    parser.add_argument("--rand_augment_strength", 
                        help="Magnitude of random augments, if 0 the no random augment will be applied",
                        default=9, 
                        type=int)
    parser.add_argument("--mixup_alpha",
                        help="Alpha parameter for Beta distribution from which mixup lambda is sampled",
                        default=1.0,
                        type=float)
    parser.add_argument("--cutmix_alpha",
                        help="Alpha parameter for Beta distribution from which cutmix lambda is samples",
                        default=1.0,
                        type=float)
    parser.add_argument("--label_smoothing",
                        help="smooths labels when computing loss, mix between ground truth and uniform",
                        default=0, 
                        type=float)
    parser.add_argument("--custom_weight_init", 
                        help="Do you want to initialize the model with truncated normal layers?", 
                        default=False, 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--bias_weight_decay",
                        help="Apply weight decay to bias",
                        default=False, 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--norm_weight_decay",
                        help="Apply weight decay to normalization weight and bias",
                        default=False, 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_grad_norm", 
                        help="Maximum norm for gradient clipping", 
                        default=1.0, 
                        type=float)

    ### DATALOADER CONFIG ###
    parser.add_argument("--img_size", 
                        help="Width and Height of Images passed to model", 
                        default=224, 
                        type=int)
    parser.add_argument("--num_workers", 
                        help="Number of workers for DataLoader", 
                        default=32, 
                        type=int)
    

    ### EXTRA CONFIGS ###
    parser.add_argument("--log_wandb",
                        action=argparse.BooleanOptionalAction, 
                        default=False)

    parser.add_argument("--resume_from_checkpoint", 
                        help="Checkpoint folder for model to resume training from, inside the experiment folder", 
                        default=None, 
                        type=str)

    args = parser.parse_args()

    return args

### Grab Arguments ###
args = parse_args()

### Init Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          log_with="wandb" if args.log_wandb else None)

### Weights and Biases Logger ###
if args.log_wandb:
    experiment_config = {"epochs": args.epochs,
                        "effective_batch_size": args.per_gpu_batch_size*accelerator.num_processes, 
                        "learning_rate": args.learning_rate,
                        "warmup_epochs": args.warmup_epochs,
                        "rand_augment": args.rand_augment_strength,
                        "cutmix_alpha": args.cutmix_alpha,
                        "mixup_alpha": args.mixup_alpha,
                        "custom_weight_init": args.custom_weight_init}
    
    accelerator.init_trackers(args.experiment_name, config=experiment_config, init_kwargs={"wandb": {"name": args.wandb_run_name}})

### Load Model ###
model = VisionTransformer(num_classes=args.num_classes, custom_weight_init=args.custom_weight_init)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of Trainable Parameters:", params)

### Load Datasets w/ Transforms ###
train_transforms = utils.train_transformations(image_size=(args.img_size, args.img_size),
                                               random_aug_magnitude=args.rand_augment_strength)
eval_transforms = utils.eval_transformations(image_size=(args.img_size, args.img_size))

train_set = ImageFolder(os.path.join(args.path_to_data, "train"), transform=train_transforms)
test_set = ImageFolder(os.path.join(args.path_to_data, "validation"), transform=eval_transforms)

### Load DataLoader ###
minibach_size = args.per_gpu_batch_size // args.gradient_accumulation_steps
collate_fn = utils.mixup_cutmix_collate_fn(mixup_alpha=args.mixup_alpha, 
                                           cutmix_alpha=args.cutmix_alpha,
                                           num_classes=args.num_classes)
trainloader = DataLoader(train_set, 
                         batch_size=minibach_size, 
                         collate_fn=collate_fn, 
                         num_workers=args.num_workers, 
                         pin_memory=True)

testloader = DataLoader(test_set, 
                        batch_size=minibach_size,
                        num_workers=args.num_workers, 
                        pin_memory=True)


### Define Loss Function ###
loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

### Define Optimizer (And seperate out weight decay and no weight decay parameters) ###
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
            elif "bn" in name and not args.norm_weight_decay:
                no_weight_decay_params.append(param)

            else:
                weight_decay_params.append(param)

    optimizer_group = [
        {"params": weight_decay_params, "weight_decay": args.weight_decay},
        {"params": no_weight_decay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_group, lr=args.learning_rate, weight_decay=args.weight_decay)

else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

### Define Scheduler (Compute number of training steps from epochs and adjust for num_gpus) ###
num_training_steps = len(trainloader) * args.epochs // args.gradient_accumulation_steps
num_warmup_steps = len(trainloader) * args.warmup_epochs // args.gradient_accumulation_steps
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                            num_warmup_steps=num_warmup_steps * accelerator.num_processes, 
                                            num_training_steps=num_training_steps * accelerator.num_processes)
### Prepare Everything ###
model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
    model, optimizer, trainloader, testloader, scheduler
)
accelerator.register_for_checkpointing(scheduler)

### Check if we are resuming from checkpoint ###
if args.resume_from_checkpoint is not None:
    accelerator.print(f"Resuming from Checkpoint: {args.resume_from_checkpoint}")
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    accelerator.load_state(path_to_checkpoint)
    starting_checkpoint = int(args.resume_from_checkpoint.split("_")[-1])
else:
    starting_checkpoint = 0

for epoch in range(starting_checkpoint, args.epochs):
    
    accelerator.print(f"Training Epoch {epoch}")

    ### Storage for Everything ###
    train_loss = []
    test_loss = []
    train_top1_acc = []
    train_top5_acc = []
    test_top1_acc = []
    test_top5_acc = []

    accumulated_loss = 0 
    accumulated_top1_accuracy = 0
    accumulated_top5_accuracy = 0

    ### Training Progress Bar ###
    progress_bar = tqdm(range(len(trainloader)//args.gradient_accumulation_steps), 
                        disable=not accelerator.is_local_main_process)

    model.train()
    for images, targets in trainloader:

        ### Move Data to Correct GPU ###
        images, targets = images.to(accelerator.device), targets.to(accelerator.device)

        with accelerator.accumulate(model):
            
            ### Pass Through Model ###
            pred = model(images)

            ### Compute and Store Loss ##
            loss = loss_fn(pred, targets)
            accumulated_loss += loss / args.gradient_accumulation_steps

            ### Compute and Store Accuracy ###
            top1_accuracy, top5_accuracy = utils.accuracy(pred, targets)
            accumulated_top1_accuracy += top1_accuracy / args.gradient_accumulation_steps
            accumulated_top5_accuracy += top5_accuracy / args.gradient_accumulation_steps

            ### Compute Gradients ###
            accelerator.backward(loss)

            ### Clip Gradients ###
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            ### Update Model ###
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()


        ### Only when GPUs are being synchronized (When all the grad accumulation is done) store metrics ###
        if accelerator.sync_gradients:
            
            ### Gather Metrics Across GPUs ###
            loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
            accuracy_top1_gathered = accelerator.gather_for_metrics(accumulated_top1_accuracy)
            accuracy_top5_gathered = accelerator.gather_for_metrics(accumulated_top5_accuracy)

            ### Store Current Iteration Error ###
            train_loss.append(torch.mean(loss_gathered).item())
            train_top1_acc.append(torch.mean(accuracy_top1_gathered).item())
            train_top5_acc.append(torch.mean(accuracy_top5_gathered).item())

            ### Reset Accumulated for next Accumulation ###
            accumulated_loss, accumulated_top1_accuracy, accumulated_top5_accuracy = 0, 0, 0

            ### Iterate Progress Bar ###
            progress_bar.update(1)


    model.eval()
    for images, targets in tqdm(testloader, disable=not accelerator.is_local_main_process):
        images, targets = images.to(accelerator.device), targets.to(accelerator.device)
        with torch.no_grad():
            pred = model(images)

        ### Compute Loss ###
        loss = loss_fn(pred, targets)

        ### Computed Accuracy ###
        top1_accuracy, top5_accuracy = utils.accuracy(pred, targets)

        ### Gather across GPUs ###
        loss_gathered = accelerator.gather_for_metrics(loss)
        top1_accuracy_gathered = accelerator.gather_for_metrics(top1_accuracy)
        top5_accuracy_gathered = accelerator.gather_for_metrics(top5_accuracy)

        ### Store Current Iteration Error ###
        test_loss.append(torch.mean(loss_gathered).item())
        test_top1_acc.append(torch.mean(top1_accuracy_gathered).item())
        test_top5_acc.append(torch.mean(top5_accuracy_gathered).item())
    
    epoch_train_loss = np.mean(train_loss)
    epoch_test_loss = np.mean(test_loss)
    epoch_train_top1_acc = np.mean(train_top1_acc)
    epoch_train_top5_acc = np.mean(train_top5_acc)
    epoch_test_top1_acc = np.mean(test_top1_acc)
    epoch_test_top5_acc = np.mean(test_top5_acc)

    accelerator.print(f"Training Top 1 Accuracy: ", epoch_train_top1_acc, "Training Top 5 Accuracy", epoch_train_top5_acc, "Training Loss:", epoch_train_loss)
    accelerator.print(f"Testing Top 1 Accuracy: ", epoch_test_top1_acc, "Testing Top 5 Accuracy", epoch_test_top5_acc, "Training Loss:", epoch_train_loss)
        
    ### Log with Weights and Biases ###
    accelerator.log({"learning_rate": scheduler.get_last_lr()[0],
                     "training_loss": epoch_train_loss,
                     "testing_loss": epoch_test_loss, 
                     "training_top1_acc": epoch_train_top1_acc, 
                     "training_top5_acc": epoch_train_top5_acc,
                     "testing_top1_acc": epoch_test_top1_acc, 
                     "testing_top5_acc": epoch_test_top5_acc}, step=epoch)
    
    ### Checkpoint Model ###
    if epoch % args.save_checkpoint_interval == 0:
        path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{epoch}")
        accelerator.save_state(output_dir=path_to_checkpoint)

### End Training for Trackers to Exit ###
accelerator.end_training()