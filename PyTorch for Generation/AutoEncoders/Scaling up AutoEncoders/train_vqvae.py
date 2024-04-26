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
codebook_size = 512
model_latent_channels = 8
mini_batch_size = 128
gradient_accumulation_steps = 1
kl_loss_weight = 1
reconstruction_loss_weight = 1
train_perc = 0.85
project_dir = "work_dir/vqvae"

### Extra that I computed Ahead of Time for Normalized MSE Following: https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb
precomputed_data_variance = 0.0892


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

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=1024, latent_dim=2):
        super().__init__()
        
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

    def forward(self, x, efficient=True):

        batch_size = x.shape[0]
        
        ### Bad Implementation That Requires Matrix Expansion ###
        if not efficient:

            # C: Codebook Size, L: Latent Dim
            
            ### Embedding: [C, L] -> [B, C, L]
            emb = self.embedding.weight.unsqueeze(0).repeat(batch_size,1,1)

            ### X: [B, L] -> [B, 1, L]
            x = x.unsqueeze(1)

            ### [B, C]
            distances = torch.sum(((x - emb)**2), dim=-1)

        ### Alternative more Efficient Implementation ###
        else:
            ### Distance btwn every Latent and Code: (L-C)**2 = (L**2 - 2LC + C**2 ) ###

            ### L2: [B, L] -> [B, 1]
            L2 = torch.sum(x**2, dim=1, keepdim=True)

            ### C2: [C, L] -> [C]
            C2 = torch.sum(self.embedding.weight**2, dim=1).unsqueeze(0)

            ### CL: [B,L]@[L,C] -> [B, C]
            CL = x@self.embedding.weight.t()

            ### [B, 1] - 2 * [B, C] + [C] -> [B, C]
            distances = L2 - 2*CL + C2
        
        ### Grab Closest Indexes, create matrix of corresponding vectors ###
        ### Closest: [B, 1]
        closest = torch.argmin(distances, dim=-1)

        ### Create Empty Quantized Latents Embedding ###
        # latents_idx: [B, C]
        quantized_latents_idx = torch.zeros(batch_size, self.codebook_size, device=x.device)

        ### Place a 1 at the Indexes for each sample for the codebook we want ###
        batch_idx = torch.arange(batch_size)
        quantized_latents_idx[batch_idx,closest] = 1

        ### Matrix Multiplication to Grab Indexed Latents from Embeddings ###

        # quantized_latents: [B, C] @ [C, L] -> [B, L]
        quantized_latents = quantized_latents_idx @ self.embedding.weight

        return quantized_latents 
        

class ConvolutionalVQVAE(nn.Module):
    def __init__(self, in_channels=3, start_dim=128, dim_mults=(1,2,4), 
                 latent_channels=8, codebook_size=512):

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


        ### Bottleneck + Vector Quantization Module ###

        self.pre_vq_conv = nn.Conv2d(in_channels=start_dim*dim_mults[-1], 
                                     out_channels=latent_channels,
                                     kernel_size=3, 
                                     padding="same")
        
        self.vq = VectorQuantizer(codebook_size=codebook_size,
                                  latent_dim=latent_channels)

        
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
        
        ### Pass Through Pre Vector Quantize Conv BottleNeck ###
        conv_enc = self.pre_vq_conv(conv_enc)

        return conv_enc

    def quantize(self, z):

        ### Quantize Z ###
        codes = self.vq(z)

        ### Compute VQ Loss ###
        codebook_loss = torch.mean((codes - z.detach())**2)
        commitment_loss = torch.mean((codes.detach() - z)**2)

        ### Straight Through (Copy Gradients) ###
        codes = z + (codes - z).detach()

        return codes, codebook_loss, commitment_loss

    def decode(self, x):

        batch_size, channels, height, width = x.shape

        ### Reshape x for Quantization ###
        x = x.permute(0,2,3,1)
        x = torch.flatten(x, start_dim=0, end_dim=-2)

        ### Quantize Data ###
        codes, codebook_loss, commitment_loss = self.quantize(x)

        ### Reshape Back to B x C x H x W ###
        codes = codes.reshape(batch_size, height, width, channels)
        codes = codes.permute(0,3,1,2)

        ### Starting Decoder Convolution ###
        conv_dec = self.dec_start(codes)

        ### Pass through Decoder ###
        conv_dec = self.decoder(conv_dec)

        ### Map Back to Pixel Space ###
        conv_dec = self.final_mapping(conv_dec)
        
        return codes, conv_dec, codebook_loss, commitment_loss
        
    def forward(self, x):
        latents = self.encode(x)
        quantized_latents, decoded, codebook_loss, commitment_loss = self.decode(latents)
        return latents, quantized_latents, decoded, codebook_loss, commitment_loss

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

### Compute the Data Variance if not Provided (Memory intensive, probably better way to do this)
if precomputed_data_variance is None:
    data = []
    for image, _ in tqdm(train_set):
        data.append(image.unsqueeze(0))

    precomputed_data_variance = torch.concatenate(data)

model = ConvolutionalVQVAE(start_dim=model_start_dim, dim_mults=model_dim_mults, 
                           latent_channels=model_latent_channels, codebook_size=codebook_size)

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
            latents, quantized_latents, decoded, codebook_loss, commitment_loss = model(images)
            reconstruction_loss = torch.mean((images-decoded)**2) / precomputed_data_variance

            ### Loss from VQVAE Paper ###
            loss = reconstruction_loss + codebook_loss + 0.25*commitment_loss
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
                    latents, quantized_latents, decoded, codebook_loss, commitment_loss = model(images)
                    reconstruction_loss = torch.mean((images-decoded)**2) / precomputed_data_variance
                    loss = reconstruction_loss + codebook_loss + 0.25*commitment_loss
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
