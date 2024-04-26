import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from accelerate import Accelerator

### TRAINING ARGS ###
num_iterations = 150000
eval_iterations = 2500
learning_rate = 0.0005
warmup_steps = 2500
model_start_dim = 128
model_dim_mults = (1,2,2,4)
model_latent_channels = 8
mini_batch_size = 128
gradient_accumulation_steps = 1
kl_loss_weight = 1
reconstruction_loss_weight = 1
train_perc = 0.85
project_dir = "work_dir/vae"


#### DEFINE MODEL ARCHITECTURE ###
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"), 
            nn.ReLU(), 
            nn.GroupNorm(8, out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )

        if in_channels != out_channels:
            self.conv_identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        else:
            self.conv_identity = nn.Identity()

    def forward(self, x):
        
        return self.conv_identity(x) + self.conv_block(x)
        
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels):

        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.down(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels):
        
        super().__init__()
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )

    def forward(self, x):
        return self.upsample(x)


class ConvolutionalVAE(nn.Module):
    def __init__(self, in_channels=3, start_dim=128, dim_mults=(1,2,4), latent_channels=8):

        super().__init__()

        ### Channel Compression for Latents ###
        self.latent_channels = latent_channels

        ### Starting Convolution to start_dim ###
        self.enc_start = nn.Conv2d(in_channels=in_channels, out_channels=start_dim, kernel_size=3, padding="same")

        ### Define Convolutional Encoder: Downsample + 2x Residual ###
        self.encoder = nn.ModuleList()
        for idx, i in enumerate(dim_mults):

            block_input_dim = i * start_dim
            block_output_dim = dim_mults[idx+1] * start_dim if idx != (len(dim_mults) - 1) else i * start_dim
            
            ### Downsample Block ###
            self.encoder.append(
                DownsampleBlock(in_channels=block_input_dim)
            )

            ### Residual Blocks ###
            self.encoder.append(
                ResidualBlock(in_channels=block_input_dim, out_channels=block_input_dim)
            )

            self.encoder.append(
                ResidualBlock(in_channels=block_input_dim, 
                                out_channels=block_output_dim)
            )
        
        self.encoder = nn.Sequential(*self.encoder)

        ### Map to Mu and LogVar for VAE ###
        self.conv_mu = nn.Conv2d(in_channels=start_dim*dim_mults[-1],
                                 out_channels=self.latent_channels, 
                                 kernel_size=3, 
                                 padding="same")

        self.conv_logvar = nn.Conv2d(in_channels=start_dim*dim_mults[-1],
                                     out_channels=self.latent_channels, 
                                     kernel_size=3, 
                                     padding="same")

        ### Define Convolutional Decoder ###

        ### Starting the Decoder Back to High Dimensional Channels
        self.dec_start = nn.Conv2d(in_channels=self.latent_channels, 
                                   out_channels=start_dim*dim_mults[-1],
                                   kernel_size=3,
                                   padding="same")


        ### Create the Remaining Decoder Blocks going 2xResidual + Upsample ###
        self.decoder = nn.ModuleList()
        dim_mults = dim_mults[::-1]
        for idx, i in enumerate(dim_mults):
            block_input_dim = i*start_dim
            block_output_dim = dim_mults[idx+1] * start_dim if idx != (len(dim_mults) - 1) else i * start_dim

            ### Residual Blocks ###
            self.decoder.append(
                ResidualBlock(in_channels=block_input_dim, out_channels=block_input_dim)
            )

            self.decoder.append(
                ResidualBlock(in_channels=block_input_dim, 
                                out_channels=block_output_dim)
            )
            
            ### Upsample Block ###
            self.decoder.append(
                UpsampleBlock(in_channels=block_output_dim)
            )

        ### Add a Sigmoid to Map values btwn 0 and 1 ###
        self.decoder = nn.Sequential(*self.decoder)

        ### Map Back to Image Space and Map to Range [0,1] ###
        self.final_mapping = nn.Sequential(
            nn.Conv2d(in_channels=start_dim*dim_mults[-1], 
                      out_channels=in_channels, 
                      kernel_size=3, 
                      padding="same"), 

            nn.Sigmoid()
        )

            
    def encode(self, x):

        ### Starting Encoder Convolution ###
        x = self.enc_start(x)
        
        ### Pass through Encoder ###
        conv_enc = self.encoder(x)

        ### Get Mu and Sigma ###
        mu = self.conv_mu(conv_enc)
        logvar = self.conv_logvar(conv_enc)

        ### Sample with Reparamaterization Trick ###
        sigma = torch.exp(0.5*logvar)
        noise = torch.randn_like(sigma, device=sigma.device)
        z = mu + sigma*noise

        return z, mu, logvar

    def decode(self, x):

        ### Starting Decoder Convolution ###
        x = self.dec_start(x)

        ### Pass through Decoder ###
        x = self.decoder(x)

        ### Map Back to Pixel Space ###
        x = self.final_mapping(x)
        
        return x
        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        decoded = self.decode(z)
        return z, decoded, mu, logvar

### Define Loss funciton ###

def VAELoss(x, x_hat, mean, log_var, kl_weight=kl_loss_weight, reconstruction_weight=reconstruction_loss_weight):

    ### Compute the MSE For Every Pixel [B, C, H, W] ###
    pixel_mse = ((x-x_hat)**2)

    ### Flatten Each Image in Batch to Vector [B, C*H*W] ###
    pixel_mse = pixel_mse.flatten(1)

    ### Sum  Up Pixel Loss Per Image and Average Across Batch ###
    reconstruction_loss = pixel_mse.sum(axis=-1).mean()

    ### Compute KL Per Image and Sum Across Flattened Latent###
    kl = (1 + log_var - mean**2 - torch.exp(log_var)).flatten(1)
    kl_per_image = - 0.5 * torch.sum(kl, dim=-1)

    ### Average KL Across the Batch ###
    kl_loss = torch.mean(kl_per_image)
    
    return reconstruction_weight*reconstruction_loss + kl_weight*kl_loss
    

### Define Training Script ###

accelerator = Accelerator(project_dir=project_dir,
                          gradient_accumulation_steps=gradient_accumulation_steps)

img_transforms = transforms.Compose(
    [
        transforms.Resize((128,128)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

dataset = ImageFolder("../../../data/CelebA/", transform=img_transforms)
train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*train_perc), len(dataset)-int(len(dataset)*train_perc)])


trainloader = DataLoader(train_set, batch_size=mini_batch_size, shuffle=True, num_workers=16, pin_memory=True)
testloader = DataLoader(test_set, batch_size=mini_batch_size, shuffle=False, num_workers=16, pin_memory=True)

model = ConvolutionalVAE(start_dim=model_start_dim, dim_mults=model_dim_mults, latent_channels=model_latent_channels)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                            num_warmup_steps=2500, 
                                            num_training_steps=num_iterations)


model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
    model, optimizer, trainloader, testloader, scheduler
)

if accelerator.is_local_main_process:
    print("Number of Parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

### Logging and Visuals ###
progress_bar = tqdm(range(num_iterations), disable=not accelerator.is_local_main_process)
completed_steps = 0
grad_iters = 0
eval_flag = False
best_eval_loss = np.inf
all_training_losses, all_eval_losses = [], []

train = True
while train:
    training_losses = []
    eval_losses = []
    for images, _ in trainloader:

        with accelerator.accumulate(model):

            ### Prep Images and Move to GPU ###
            batch_size = images.shape[0]

            ### Pass Through Model and Compute/Store Loss###
            latent, decoded, mu, logvar = model(images)
            loss = VAELoss(images, decoded, mu, logvar)
            training_losses.append(loss.item())

            ### Update Model ###
            loss.backward()
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            grad_iters += 1

        if grad_iters % gradient_accumulation_steps == 0:
            progress_bar.update(1)
            completed_steps += 1
            grad_iters = 0
            eval_flag = True

        if (completed_steps % eval_iterations) == 0 and accelerator.is_local_main_process and eval_flag:
            
            with torch.no_grad():
                for images, _ in testloader:
                    latent, decoded, mu, logvar = model(images)
                    loss = VAELoss(images, decoded, mu, logvar)
                    eval_losses.append(loss.item())

            train_loss_mean = np.mean(training_losses)
            eval_loss_mean = np.mean(eval_losses)
        
            print("Training Losses:", train_loss_mean)
            print("Eval Losses:", eval_loss_mean)

            all_training_losses.append(train_loss_mean)
            all_eval_losses.append(eval_loss_mean)
            
            if eval_loss_mean < best_eval_loss:
                best_eval_loss = eval_loss_mean
                print("---SAVING MODEL---")
                accelerator.save_state(os.path.join(project_dir, "best_vae"), safe_serialization=False)

            ### Reset Training Stuff ###
            training_losses = []
            eval_losses = []
            eval_flag = False

        if completed_steps >= num_iterations:
            print("Done Training")
            accelerator.save_state(os.path.join(project_dir, "final_vae"), safe_serialization=False)
            train = False
            break
