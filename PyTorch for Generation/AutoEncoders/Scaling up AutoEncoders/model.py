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
        noise = torch.rand_like(sigma, device=sigma.device)
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
        

