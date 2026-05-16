import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from PIL import ImageFilter

from model import IJEPA
 
import warnings
warnings.filterwarnings("ignore")


class IJEPAMaskSampler:
    """
    Samples context and target patch index sets for a batch of images,
    following the multi-block masking strategy from the I-JEPA paper.
 
    Sample `num_target_blocks` non-overlapping target blocks.
      Each block is a random rectangle whose area is drawn uniformly
      from [min_target_scale, max_target_scale] × num_patches, with
      a random aspect ratio.
    Context = all patches EXCEPT those in any target block.
      This is the "full context" variant no extra context masking.
 

    """
    def __init__(
        self,
        num_patches, # total patches in the grid (e.g. 196 for 224/16)
        grid_size,
        num_target_blocks = 4,
        min_target_scale = 0.15,
        max_target_scale = 0.2,
        min_aspect_ratio = 0.75,
        max_aspect_ratio = 1.5,
    ):
        self.num_patches = num_patches
        self.grid_size = grid_size
        self.num_target_blocks = num_target_blocks
        self.min_target_scale = min_target_scale
        self.max_target_scale = max_target_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
 
    def _sample_block_ids(self) -> set:
        """Sample one rectangular block, return its patch flat-indices as a set."""
        G = self.grid_size
        for _ in range(20):    # retry if proposed block is out of bounds
            scale = np.random.uniform(self.min_target_scale, self.max_target_scale) # random sample a scale
            area = scale * self.num_patches # get the number of patches this scale covers
            aspect = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio) # random sample an aspet ratio

            # convert area to h x w (basically create a rectangle covering area with wanted aspect)
            h = max(1, int(round(math.sqrt(area / aspect))))
            w = max(1, int(round(math.sqrt(area * aspect))))

            # if out of bounds try again!
            if h > G or w > G:
                continue
            
            # random placement in image
            top = np.random.randint(0, G - h + 1)
            left = np.random.randint(0, G - w + 1)

            # flatten 2d to 1d indexing
            # (0,0) → 0
            # (0,1) → 1
            # ...
            # (1,0) → G
            ids = set()
            for r in range(top, top + h):
                for c in range(left, left + w):
                    ids.add(r * G + c)

            return ids
        
        # Fallback: single random patch
        return {np.random.randint(0, self.num_patches)}
 
    def sample(self, batch_size: int):

        all_ctx, all_tgt = [], []
        Nc_min = self.num_patches   # we'll pad to the smallest context size

        # for every sample
        for _ in range(batch_size):
            target_set = set()

            # grab however many target blocks we want
            # and keep adding new locations to our target set (union)
            for _ in range(self.num_target_blocks):
                target_set = target_set | self._sample_block_ids()

            # everything not in the targets is context
            context_set = set(range(self.num_patches)) - target_set

            # store for the batch 
            all_ctx.append(sorted(context_set))
            all_tgt.append(sorted(target_set))
            Nc_min = min(Nc_min, len(context_set))

        # pad and truncate for batching. We may have different number of
        # selected context per sample, just keep as many as the smallest set
        all_ctx = [ids[:Nc_min] for ids in all_ctx]
        context_ids = torch.tensor(all_ctx, dtype=torch.long)   # (B, Nc)
 
        # Similarly, pad/truncate targets to the same length
        Nt = min(len(t) for t in all_tgt)
        all_tgt = [ids[:Nt] for ids in all_tgt]
        target_ids = torch.tensor(all_tgt, dtype=torch.long)    # (B, Nt)
 
        return context_ids, target_ids

def linear_ema_momentum(step, total_steps, start=0.996, end=1.0):
    return start + (end - start) * step / total_steps


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
def jepa_train_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):

    
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)

    return transform

def parse_args():
    parser = argparse.ArgumentParser(description="I-JEPA Self-Supervised Pre-Training")
 
    parser.add_argument("--experiment_name", required=True, type=str)
    parser.add_argument("--wandb_run_name", required=True, type=str)
    parser.add_argument("--path_to_data", required=True, type=str,
                        help="Path to ImageNet root (contains /train subfolder)")
    parser.add_argument("--working_directory",required=True,  type=str)
 
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--warmup_epochs", default=40, type=int)
    parser.add_argument("--per_gpu_batch_size", default=256, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", default=1.5e-4, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--save_checkpoint_interval", default=10, type=int)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
 
    parser.add_argument("--ema_momentum_start", default=0.996, type=float,
                        help="Starting EMA momentum (annealed up toward ema_momentum_end)")
    parser.add_argument("--ema_momentum_end", default=1.000, type=float)
 
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--encoder_embed_dim", default=768, type=int)
    parser.add_argument("--encoder_depth", default=12, type=int)
    parser.add_argument("--encoder_num_heads", default=12, type=int)
 
    parser.add_argument("--predictor_embed_dim", default=384, type=int)
    parser.add_argument("--predictor_depth", default=6, type=int)
    parser.add_argument("--predictor_num_heads", default=12, type=int)
 
    parser.add_argument("--num_target_blocks", default=4, type=int)
    parser.add_argument("--min_target_scale", default=0.15, type=float)
    parser.add_argument("--max_target_scale", default=0.2, type=float)
    parser.add_argument("--min_aspect_ratio", default=0.75, type=float)
    parser.add_argument("--max_aspect_ratio", default=1.5, type=float)
 
    parser.add_argument("--num_workers", default=16, type=int)
 
    parser.add_argument("--log_wandb",  action=argparse.BooleanOptionalAction, default=False)
 
    return parser.parse_args()

args = parse_args()
 
### Init Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(
    project_dir=path_to_experiment,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    log_with="wandb" if args.log_wandb else None,
)
 
### Weights & Biases ###
if args.log_wandb:
    accelerator.init_trackers(
        args.experiment_name,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.wandb_run_name}},
    )
 
### Model ###
model = IJEPA(
    img_size=args.img_size,
    patch_size=args.patch_size,
    encoder_embed_dim=args.encoder_embed_dim,
    encoder_depth=args.encoder_depth,
    encoder_num_heads=args.encoder_num_heads,
    predictor_embed_dim=args.predictor_embed_dim,
    predictor_depth=args.predictor_depth,
    predictor_num_heads=args.predictor_num_heads,
)
 
### Data ###
train_transforms = jepa_train_transforms(args.img_size)
path_to_train = os.path.join(args.path_to_data, "train")
trainset = datasets.ImageFolder(path_to_train, transform=train_transforms)
 
mini_batchsize = args.per_gpu_batch_size // args.gradient_accumulation_steps
trainloader = DataLoader(
    trainset,
    batch_size=mini_batchsize,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,    # keeps batch size constant — important for mask sampler
)
 
### Masking Sampler ###
num_patches = (args.img_size // args.patch_size) ** 2
grid_size = args.img_size  // args.patch_size
mask_sampler = IJEPAMaskSampler(
    num_patches=num_patches,
    grid_size=grid_size,
    num_target_blocks=args.num_target_blocks,
    min_target_scale=args.min_target_scale,
    max_target_scale=args.max_target_scale,
    min_aspect_ratio=args.min_aspect_ratio,
    max_aspect_ratio=args.max_aspect_ratio,
)
 
### Optimizer  (only context_encoder + predictor, NOT target_encoder) ###
trainable_params = (
    list(model.context_encoder.parameters())
    + list(model.predictor.parameters())
)
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.95),   # slightly higher β2 is common for ViT pre-training
)
 
### LR schedule ###
num_training_steps = len(trainloader) * args.epochs // args.gradient_accumulation_steps
num_warmup_steps   = len(trainloader) * args.warmup_epochs // args.gradient_accumulation_steps
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)
 
### Prepare with Accelerate ###
model, optimizer, trainloader, scheduler = accelerator.prepare(
    model, optimizer, trainloader, scheduler
)
accelerator.register_for_checkpointing(scheduler)
 
### Resume ###
starting_epoch = 0
if args.resume_from_checkpoint is not None:
    accelerator.print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    path_to_ckpt = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    accelerator.load_state(path_to_ckpt)
    starting_epoch = int(args.resume_from_checkpoint.split("_")[-1])
 
### Global step counter for EMA schedule ###
global_step = starting_epoch * (len(trainloader) // args.gradient_accumulation_steps)
  
for epoch in range(starting_epoch, args.epochs):
 
    accelerator.print(f"\n── Epoch {epoch} ──")
    model.train()
 
    train_losses  = []
    accumulated_loss = 0.0
    progress_bar = tqdm(
        range(len(trainloader) // args.gradient_accumulation_steps),
        disable=not accelerator.is_local_main_process,
    )
 
    for images, _ in trainloader:   # labels not needed for JEPA
        images = images.to(accelerator.device)
        B = images.shape[0]
 
        # Sample fresh masks for this batch (on CPU, then move indices to GPU)
        context_ids, target_ids = mask_sampler.sample(B)
        context_ids = context_ids.to(accelerator.device)
        target_ids  = target_ids.to(accelerator.device)
 
        with accelerator.accumulate(model):
 
            loss = model(images, context_ids, target_ids)
            accumulated_loss += loss.detach() / args.gradient_accumulation_steps
 
            accelerator.backward(loss)
 
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
 
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
 
        if accelerator.sync_gradients:
            # EMA update of target encoder
            momentum = linear_ema_momentum(
                global_step, num_training_steps,
                args.ema_momentum_start, args.ema_momentum_end,
            )
            # unwrap in case of DDP wrapping
            raw_model = accelerator.unwrap_model(model)
            raw_model.update_target_encoder(momentum)
 
            # Gather loss across GPUs for logging
            loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
            train_losses.append(torch.mean(loss_gathered).item())
            accumulated_loss = 0.0
 
            global_step  += 1
            progress_bar.update(1)
 
    epoch_loss = float(np.mean(train_losses))
    accelerator.print(f"  Loss: {epoch_loss:.4f}  |  LR: {scheduler.get_last_lr()[0]:.2e}")
 
    if args.log_wandb:
        accelerator.log({
            "train_loss":    epoch_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "ema_momentum":  momentum,
        }, step=epoch)
 
    if epoch % args.save_checkpoint_interval == 0:
        ckpt_dir = os.path.join(path_to_experiment, f"checkpoint_{epoch}")
        accelerator.save_state(output_dir=ckpt_dir)
 
accelerator.end_training()

if __name__ == "__main__":

    def plot_mask(context_ids, target_ids, grid_size):
        grid = np.zeros((grid_size, grid_size))

        # mark context = 1
        for idx in context_ids:
            r, c = divmod(idx, grid_size)
            grid[r, c] = 1

        # mark target = 2 (overrides context visually)
        for idx in target_ids:
            r, c = divmod(idx, grid_size)
            grid[r, c] = 2

        plt.figure(figsize=(5, 5))
        plt.imshow(grid, cmap="viridis", interpolation="nearest")

        plt.title("I-JEPA Masking Pattern")
        plt.colorbar(label="0=unused, 1=context, 2=target")
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
    sampler = IJEPAMaskSampler(num_patches=196)

    ctx, tgt = sampler.sample(batch_size=1)

    plot_mask(ctx[0].numpy(), tgt[0].numpy(), grid_size=14)

