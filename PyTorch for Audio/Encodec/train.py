import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import accelerate
from accelerate import Accelerator
import argparse 
import random
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from modules.encodec import EncodecModel
from modules.discriminator import MultiScaleSTFTDiscriminator
from dataset import AudioDataset
from loss import generator_loss, discriminator_loss
from utils import load_audios, save_audios
from balancer import Balancer

torch.backends.cudnn.benchmark = True

def parse_args():

    parser = argparse.ArgumentParser()

    ### SETUP CONFIG ###
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--working_directory", type=str, required=True)
    parser.add_argument("--path_to_train_manifest", type=str, required=True)
    parser.add_argument("--path_to_val_manifest", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    ### TRAINING CONFIG ###
    parser.add_argument("--sampling_rate", type=int, default=24000)
    parser.add_argument("--segment_length", type=int, default=72000)
    parser.add_argument("--training_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=float, default=5)
    parser.add_argument("--console_out_iters", type=int, default=5)
    parser.add_argument("--wandb_log_iters", type=int, default=5)
    parser.add_argument("--checkpoint_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--disc_learning_rate", type=float, default=0.0003)
    parser.add_argument("--disc_update_prob", type=float, default=0.666) # 2/3 from paper
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)

    ### LOSS BALANCE ###
    parser.add_argument("--time_loss", type=float, default=0.1)
    parser.add_argument("--frequency_loss", type=float, default=1)
    parser.add_argument("--feature_loss", type=float, default=3)
    parser.add_argument("--generator_loss", type=float, default=3)
    parser.add_argument("--disable_balancer", action="store_true")


    ### INFERENCE CONFIG ###
    parser.add_argument("--num_samples_for_reconstruction", type=int, default=3)

    ### DATASET CONFIG ###
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--log_wandb", action="store_true")
    
    args = parser.parse_args()

    return args

args = parse_args()

### Init Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb" if args.log_wandb else None)
if args.log_wandb:
    accelerator.init_trackers(args.experiment_name)

### Load Datasets ###
trainset = AudioDataset(args.path_to_train_manifest, 
                        args.segment_length, 
                        args.sampling_rate)

testset = AudioDataset(args.path_to_val_manifest, 
                       args.segment_length, 
                       args.sampling_rate)


trainloader = DataLoader(trainset, batch_size=args.batch_size, 
                         num_workers=args.num_workers, 
                         shuffle=True)

testloader = DataLoader(testset, batch_size=args.batch_size, 
                        num_workers=args.num_workers, 
                        shuffle=False)

### Sample some files for inference ###
paths = random.sample(testset.audio_paths, k=args.num_samples_for_reconstruction)
cached_audios = load_audios(paths, args.sampling_rate)

### Save the cached samples for comparison later ###
path_to_save_dir = os.path.join(path_to_experiment, f"gens_original")
os.makedirs(path_to_save_dir, exist_ok=True)
path_to_saves = [os.path.join(path_to_save_dir, file) for file in [f"gen_{i}.wav" for i in range(len(cached_audios))]]
save_audios(cached_audios, path_to_saves, args.sampling_rate)

### Load Model ###
model = EncodecModel(accelerator=accelerator)

### Load Discriminator ###
disc_model = MultiScaleSTFTDiscriminator()

### Print Training Run Config to Console ###
def count_params(model):
    total = 0
    for param in model.parameters():
        total += param.numel()
    for param in model.buffers():
        total += param.numel()
    return total

accelerator.print("=" * 60)
accelerator.print(f"{'TRAINING CONFIGURATION':^60}")
accelerator.print("=" * 60)
accelerator.print(f"{'Training Samples':<25}: {len(trainset):>15,}")
accelerator.print(f"{'Testing Samples':<25}: {len(testset):>15,}")
accelerator.print(f"{'Sampling Rate':<25}: {args.sampling_rate:>15,}")
accelerator.print(f"{'Segment Size':<25}: {args.segment_length:>15,}")
accelerator.print("-" * 60)
accelerator.print(f"{'Model Parameters':<25}: {count_params(model):>15,}")
accelerator.print(f"{'Discriminator Parameters':<25}: {count_params(disc_model):>15,}")
accelerator.print("-" * 60)
resume_status = "Yes" if args.resume_from_checkpoint else "No"
checkpoint_path = args.resume_from_checkpoint if args.resume_from_checkpoint else "—"
accelerator.print(f"{'Resume From Checkpoint':<25}: {resume_status:>15}")
accelerator.print(f"{'Checkpoint Path':<25}: {checkpoint_path:>15}")
accelerator.print("=" * 60)

### Load Optimizers ###
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], 
                             lr=args.learning_rate)
disc_optimizer = torch.optim.Adam(disc_model.parameters(),
                                  lr=args.disc_learning_rate)

### Load Scheduler ###
steps_per_epoch = len(trainloader) // accelerator.num_processes
warmup_steps = steps_per_epoch * args.warmup_epochs
total_training_steps = steps_per_epoch * args.training_epochs

scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=warmup_steps * accelerator.num_processes, 
                                            num_training_steps=total_training_steps * accelerator.num_processes)
disc_scheduler = get_cosine_schedule_with_warmup(disc_optimizer, 
                                                 num_warmup_steps=warmup_steps * accelerator.num_processes, 
                                                 num_training_steps=total_training_steps * accelerator.num_processes)

### Prepare Everything ###
(
    model, 
    disc_model, 
    optimizer, 
    disc_optimizer, 
    scheduler, 
    disc_scheduler, 
    trainloader, 
    testloader
) = accelerator.prepare(
    model, 
    disc_model, 
    optimizer, 
    disc_optimizer, 
    scheduler, 
    disc_scheduler, 
    trainloader, 
    testloader
)

### Initialize Balancer ###
weights = {"time_loss": args.time_loss, 
           "frequency_loss": args.frequency_loss,
           "generator_loss": args.generator_loss,
           "feature_loss": args.feature_loss}

if not args.disable_balancer:
    balancer = Balancer(weights=weights, accelerator=accelerator)

### Resume From Checkpoint ###
if args.resume_from_checkpoint is not None:
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    accelerator.load_state(path_to_checkpoint)

    if not args.disable_balancer:
        path_to_balancer_state = os.path.join(path_to_checkpoint, "balancer_state.bin")
        balancer_state = torch.load(path_to_balancer_state)
        balancer.load_state(balancer_state)

    starting_epoch = int(args.resume_from_checkpoint.split("_")[-1])
    completed_steps = steps_per_epoch * starting_epoch
    accelerator.print(f"RESUMING FROM EPOCH: {starting_epoch}")
else:
    starting_epoch = 0
    completed_steps = 0

### Train Model ###
for epoch in range(starting_epoch, args.training_epochs):

    model.train()

    log = {"accum_test_time_loss": [], "accum_test_freq_loss": []}

    for waveforms in trainloader:

        ### Move to GPU ###
        waveforms = waveforms.to("cuda")
        
        ######################
        ### Generator Step ###
        ######################
        optimizer.zero_grad()

        ### Get output ###
        output = model(waveforms)

        ### pass real and fake into disc ###
        logits_real, fmap_real = disc_model(waveforms)
        logits_fake, fmap_fake = disc_model(output["decoded"])

        ### Compute Generator Loss ###
        losses = generator_loss(
            fmap_real, fmap_fake, logits_fake, 
            waveforms, output["decoded"], 
            sample_rate=args.sampling_rate   
        )

        ### Standard backward pass without the balancer ###
        if args.disable_balancer:
            total_gen_loss = weights["time_loss"] * losses["time_loss"] + \
                             weights["frequency_loss"] * losses["frequency_loss"] + \
                             weights["generator_loss"] * losses["generator_loss"] + \
                             weights["feature_loss"] * losses["feature_loss"] + \
                             output["quantizer_loss"]
            
            ### Update model ###
            accelerator.backward(total_gen_loss)
            
        ### Otherwise use the autobalancer ###
        else:

            ### Retain graph as we need to do a second backward ###
            balancer.backward(losses, output["decoded"], retain_graph=True)

            ### Compute separately grads w.r.t commitment loss ###
            ### as this only effects the encoder portion of the model ###
            ### https://github.com/facebookresearch/encodec/issues/20
            accelerator.backward(output["quantizer_loss"])

        ### Grad Clipping ###
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        
        ### Update the model with the compute grads ###
        optimizer.step()

        ### Update Generator Scheduler ###
        scheduler.step()

        ### Accumulate Loss ###
        if accelerator.num_processes > 1:
            train_time_loss = accelerator.gather_for_metrics(losses["time_loss"]).mean()
            train_freq_loss = accelerator.gather_for_metrics(losses["frequency_loss"]).mean()
            train_gen_loss = accelerator.gather_for_metrics(losses["generator_loss"]).mean()
            train_feat_loss = accelerator.gather_for_metrics(losses["feature_loss"]).mean()
            train_commit_loss = accelerator.gather_for_metrics(output["quantizer_loss"]).mean()
        else:
            train_time_loss = losses["time_loss"]
            train_freq_loss = losses["frequency_loss"]
            train_gen_loss = losses["generator_loss"]
            train_feat_loss = losses["feature_loss"]
            train_commit_loss = output["quantizer_loss"]
            
        ###########################
        ### Dimscriminator Step ###
        ###########################

        disc_optimizer.zero_grad()

        ### Random sample value between 0 and 1 ###
        ### Ensure all GPUs have the same value ###
        rand = torch.rand(size=(), device=accelerator.device)
        rand = accelerate.utils.broadcast(rand, from_process=0)

        if rand < args.disc_update_prob:
        
            ### Pass Through Discriminator ###
            logits_real, _ = disc_model(waveforms)
            logits_fake, _ = disc_model(output["decoded"].detach())

            ### Compute Discriminator Loss ###
            disc_loss = discriminator_loss(logits_real, logits_fake)

            ### Update Disc ###
            accelerator.backward(disc_loss)

            ### Grad Clipping ###
            accelerator.clip_grad_norm_(disc_model.parameters(), 1.0)

            ### Update Discriminator ###
            disc_optimizer.step()
        
        else:
            disc_loss = "skipped"

        ### Update Disc Scheduler anyway even if we skipped ###
        disc_scheduler.step()

        ### Logging ###
        progress = completed_steps / (len(trainloader) * args.training_epochs) * 100
        if completed_steps % args.console_out_iters == 0:
            disc_loss_str = f"{disc_loss:.3f}" if isinstance(disc_loss, (int, float, torch.Tensor)) else str(disc_loss)
            accelerator.print(
                f"Step {completed_steps:>6} ({progress:5.1f}%) | "
                f"Time: {train_time_loss:.3f} | "
                f"Freq: {train_freq_loss:.3f} | "
                f"Gen: {train_gen_loss:.3f} | "
                f"Feat: {train_feat_loss:.3f} | "
                f"Disc: {disc_loss_str} | "
                f"Quant: {train_commit_loss:.2e} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"D_LR: {disc_scheduler.get_last_lr()[0]:.2e}"
            )

        if (completed_steps % args.wandb_log_iters == 0) and args.log_wandb:

            log_dict = {
                "time_loss": train_time_loss,
                "freq_loss": train_freq_loss, 
                "gen_loss": train_gen_loss, 
                "feat_loss": train_feat_loss, 
                "quant": train_commit_loss,
                "lr": scheduler.get_last_lr()[0], 
                "disc_lr": disc_scheduler.get_last_lr()[0]
            }

            if not isinstance(disc_loss, str):
                log_dict["disc_loss"] = disc_loss

            accelerator.log(log_dict, step=completed_steps)

        ### Completed 1 Step of Training ###
        completed_steps += 1

    ### Validation ###
    model.eval()

    for waveforms in tqdm(testloader, disable=not accelerator.is_main_process):

        waveforms = waveforms.to("cuda")

        with torch.no_grad():
            output = model(waveforms)

        ### pass real and fake into disc ###
        logits_real, fmap_real = disc_model(waveforms)
        logits_fake, fmap_fake = disc_model(output["decoded"])

        ### Compute Generator Loss ###
        losses = generator_loss(
            fmap_real, fmap_fake, logits_fake, 
            waveforms, output["decoded"], 
            sample_rate=args.sampling_rate   
        )

        ### Accumulate Loss ###
        if accelerator.num_processes > 1:
            test_time_loss = accelerator.gather_for_metrics(losses["time_loss"]).mean()
            test_freq_loss = accelerator.gather_for_metrics(losses["frequency_loss"]).mean()
        else:
            test_time_loss = losses["time_loss"]
            test_freq_loss = losses["frequency_loss"]

        log["accum_test_time_loss"].append(test_time_loss.item())
        log["accum_test_freq_loss"].append(test_freq_loss.item())
    
    test_time_loss = np.mean(log["accum_test_time_loss"])
    test_freq_loss = np.mean(log["accum_test_freq_loss"])
    
    accelerator.print(
        f"VALIDATION LOSS: "
        f"Time: {test_time_loss:.3f} | "
        f"Freq: {test_freq_loss:.3f}"
    )

    if args.log_wandb:
        accelerator.log(
                {"test_time_loss": train_time_loss,
                 "test_freq_loss": train_freq_loss},
                 step=completed_steps
            )

    ### Inference our cached audio every epoch ###
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)

        gens = [unwrapped_model.passthrough(c.to(accelerator.device)).detach().cpu() for c in cached_audios]

        ### Dir to save in ###
        path_to_save_dir = os.path.join(path_to_experiment, f"gens_epoch_{epoch}")
        os.makedirs(path_to_save_dir, exist_ok=True)
        path_to_saves = [os.path.join(path_to_save_dir, file) for file in [f"gen_{i}.wav" for i in range(len(gens))]]
        
        save_audios(gens, path_to_saves, args.sampling_rate)

    accelerator.wait_for_everyone()

    ### Checkpoint Model ###
    if epoch % args.checkpoint_epochs == 0:
        output_dir = os.path.join(path_to_experiment, f"checkpoint_{epoch}")
        accelerator.save_state(output_dir, safe_serialization=False) # disable to use .bin due to weird lstm weights and safetensors

        if not args.disable_balancer:
            output_balancer = os.path.join(path_to_experiment, f"checkpoint_{epoch}", "balancer_state.bin")
            balancer_state = balancer.state_dict()
            torch.save(balancer_state, output_balancer)

### Store Final Checkpoint ###
output_dir = os.path.join(path_to_experiment, f"final_checkpoint")
accelerator.save_state(output_dir, safe_serialization=False)
