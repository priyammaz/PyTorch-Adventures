import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import argparse 
from transformers import get_cosine_schedule_with_warmup

from modules.encodec import EncodecModel
from modules.discriminator import MultiScaleSTFTDiscriminator
from dataset import AudioDataset
from loss import generator_loss, discriminator_loss

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
    parser.add_argument("--segment_length", type=int, default=24000)
    parser.add_argument("--training_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=float, default=5)
    parser.add_argument("--console_out_iters", type=int, default=5)
    parser.add_argument("--wandb_log_iters", type=int, default=5)
    parser.add_argument("--checkpoint_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--disc_learning_rate", type=float, default=0.0003)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)

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

testloader = DataLoader(trainset, batch_size=args.batch_size, 
                        num_workers=args.num_workers, 
                        shuffle=False)

### Load Model ###
model = EncodecModel(accelerator=accelerator)

### Load Discriminator ###
disc_model = MultiScaleSTFTDiscriminator()

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
                                            num_warmup_steps=warmup_steps, 
                                            num_training_steps=total_training_steps)
disc_scheduler = get_cosine_schedule_with_warmup(disc_optimizer, 
                                                 num_warmup_steps=warmup_steps, 
                                                 num_training_steps=total_training_steps)


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

### Train Model ###
for epoch in range(args.training_epochs):

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

        total_gen_loss = 0.1 * losses["time_loss"] + 1 * losses["frequency_loss"] + \
                            3 * losses["generator_loss"] + 3 * losses["feature_loss"]
        
        
        ### Update model ###
        accelerator.backward(total_gen_loss)
        optimizer.step()

        ### Update Generator Scheduler ###
        scheduler.step()

        ###########################
        ### Dimscriminator Step ###
        ###########################
        disc_optimizer.zero_grad()

        logits_real, _ = disc_model(waveforms)
        logits_fake, _ = disc_model(output["decoded"].detach())

        disc_loss = discriminator_loss(logits_real, logits_fake)

        ### Update Disc ###
        accelerator.backward(disc_loss)
        disc_optimizer.step()

        ### Update Disc Scheduler ###
        disc_scheduler.step()

        accelerator.print(scheduler.get_last_lr()[0], losses["frequency_loss"])


        