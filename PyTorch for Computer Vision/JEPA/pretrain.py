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
import matplotlib.pyplot as plt
from PIL import ImageFilter

from model import IJEPA
from masking import IJEPAMaskSampler
 
import warnings
warnings.filterwarnings("ignore")

### SCHEDULERS ###
def linear_ema_momentum(step, total_steps, start=0.996, end=1.0):
    """linear ema schedule"""
    return start + (end - start) * step / total_steps

def get_lr(step, warmup_steps, total_steps, start_lr, peak_lr, final_lr):
    """linear increase for warmup steps + cos decay for remaining"""
    if step < warmup_steps:
        return start_lr + (peak_lr - start_lr) * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr + (peak_lr - final_lr) * cosine

def get_wd(step, total_steps, start_wd, final_wd):
    """linear wd schedule"""
    return start_wd + (final_wd - start_wd) * step / total_steps

### TRANSFORMS ###
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
    parser.add_argument("--working_directory", required=True, type=str)
 
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--warmup_epochs", default=15, type=int)
    parser.add_argument("--per_gpu_batch_size", default=512, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--start_lr", default=1e-4, type=float,
                        help="LR at start of linear warmup")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="Peak LR reached at end of warmup")
    parser.add_argument("--final_lr", default=1e-6, type=float,
                        help="LR at end of cosine decay")
    parser.add_argument("--start_weight_decay", default=0.04, type=float,
                        help="Weight decay at start of training")
    parser.add_argument("--final_weight_decay", default=0.4, type=float,
                        help="Weight decay at end of training (linearly increased)")
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

def main():
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
        drop_last=True,
    )
    
    ### Masking Sampler ###
    mask_sampler = IJEPAMaskSampler(
        img_size=args.img_size,
        patch_size=args.patch_size,
        pred_mask_scale=(args.min_target_scale, args.max_target_scale),
        aspect_ratio=(args.min_aspect_ratio, args.max_aspect_ratio),
        npred=args.num_target_blocks,
    )
        
    ### Optimizer  (only context_encoder + predictor, NOT target_encoder) ###
    trainable_params = (
        list(model.context_encoder.parameters())
        + list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.start_lr, # will be overridden each step manually
        weight_decay=args.start_weight_decay,
        betas=(0.9, 0.95),
    )
 
    
    ### Schedule constants (computed in steps, not epochs) ###
    num_training_steps = len(trainloader) * args.epochs // args.gradient_accumulation_steps
    num_warmup_steps   = len(trainloader) * args.warmup_epochs // args.gradient_accumulation_steps

    ### Prepare with Accelerate ###
    model, optimizer, trainloader = accelerator.prepare(
        model, optimizer, trainloader
    )
    
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
    
        accelerator.print(f"\n --- Epoch {epoch} --- ")
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
            target_ids = target_ids.to(accelerator.device)

            context_ids = context_ids[:, 0, :] # (B, Nc)
            target_ids  = target_ids.flatten(1) # (B, npred*Nt)

            with accelerator.accumulate(model):
    
                loss = model(images, context_ids, target_ids)
                accumulated_loss += loss.detach() / args.gradient_accumulation_steps
    
                accelerator.backward(loss)
    
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                current_lr = get_lr(global_step, num_warmup_steps, num_training_steps,
                                 args.start_lr, args.learning_rate, args.final_lr)
                current_wd = get_wd(global_step, num_training_steps,
                                    args.start_weight_decay, args.final_weight_decay)
                
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                    param_group["weight_decay"] = current_wd

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    
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
                loss_gathered = torch.mean(loss_gathered).item()
                train_losses.append(loss_gathered)
                accumulated_loss = 0.0
    
                global_step  += 1
                progress_bar.update(1)

                if args.log_wandb:
                    accelerator.log({
                        "train_loss":    loss_gathered,
                        "learning_rate": current_lr,
                        "weight_decay": current_wd,
                        "ema_momentum":  momentum,
                    }, step=global_step)
    
        epoch_loss = float(np.mean(train_losses))
        accelerator.print(f"Loss: {epoch_loss:.4f}")
    
        if epoch % args.save_checkpoint_interval == 0:
            ckpt_dir = os.path.join(path_to_experiment, f"checkpoint_{epoch}")
            accelerator.save_state(output_dir=ckpt_dir)
    
    accelerator.end_training()

if __name__ == "__main__":
    main()