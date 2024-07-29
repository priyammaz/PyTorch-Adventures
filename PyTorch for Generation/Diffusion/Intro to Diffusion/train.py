import os    
import shutil
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from torchvision import transforms 
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import argparse
import itertools

#### DEFINE ARGUMENT PARSER ###
parser = argparse.ArgumentParser(description="Arguments for Denoising Diffusion")
parser.add_argument("--experiment_name", 
                    help="Name of Training Run", 
                    required=True, 
                    type=str)
parser.add_argument("--path_to_data",
                    help="Path to CelebA Dataset",
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
                    default=150000,
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
parser.add_argument("--num_workers", 
                    help="Number of workers for DataLoader", 
                    default=32, 
                    type=int)
parser.add_argument("--resume_from_checkpoint", 
                    help="Checkpoint folder for model to resume training from, inside the experiment folder", 
                    default=None, 
                    type=str)
args = parser.parse_args()

### DEFINE DDPM SAMPLER ###
class Sampler:
    def __init__(self, total_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.total_timesteps = total_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        ### Define Basic Beta Scheduler ###
        self.beta_schedule = self.linear_beta_schedule()

        ### Compute Alphas for Direction 0 > t Noise Calculation ###
        self.alpha = 1 - self.beta_schedule
        self.alpha_cumulative_prod = torch.cumprod(self.alpha, dim=-1)
    
    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.total_timesteps)

    def _repeated_unsqueeze(self, target_shape, input):
        while target_shape.dim() > input.dim():
            input = input.unsqueeze(-1)
        return input
    
    def add_noise(self, inputs, timesteps):

        batch_size, c, h, w = inputs.shape

        ### Grab the Device we want to place tensors on ###
        device = inputs.device
        
        alpha_cumulative_prod_timesteps = self.alpha_cumulative_prod[timesteps].to(device)
        
        ### Compute Mean Coefficient ###
        mean_coeff = alpha_cumulative_prod_timesteps ** 0.5

        ### Compute Variance Coefficient ###
        var_coeff = (1 - alpha_cumulative_prod_timesteps) ** 0.5

        ### Reshape mean_coeff and var_coeff to have shape (batch x 1 x 1 x 1) so we can broadcast with input (batch x c x height x width) ###
        mean_coeff = self._repeated_unsqueeze(inputs, mean_coeff)
        var_coeff = self._repeated_unsqueeze(inputs, var_coeff)

        ### Generate some Noise X ~ N(0,1) (rand_like will automatically place on same device as the inputs) ###
        noise = torch.randn_like(inputs)
        
        ### Compute Mean (mean_coef * x_0) ###
        mean = mean_coeff * inputs

        ### Compute Variance ###
        var = var_coeff * noise

        ### Compute Noisy Data ###
        noisy_image = mean + var

        return noisy_image, noise
        
    def remove_noise(self, input, timestep, predicted_noise):

        assert (input.shape == predicted_noise.shape), "Shapes of noise pattern and input image must be identical!!"
        
        b, c, h, w = input.shape

        ### Grab Device to Place Tensors On ###
        device = input.device

        ### Create a mask (if timestep == 0 sigma_z will also be 0 so we need to save this for later ###
        greater_than_0_mask = (timestep >= 1).int()

        
        ### Compute Sigma (b_t * (1 - cumulative_a_(t-1)) / (1 - cumulative_a)) * noise ###
        alpha_cumulative_t = self.alpha_cumulative_prod[timestep].to(device)
        alpha_cumulative_prod_t_prev = self.alpha_cumulative_prod[timestep - 1].to(device) # (timestep - 1) if timestep is 0 is WRONG! we will multiply by 0 later
        beta_t = self.beta_schedule[timestep].to(device)
        noise = torch.randn_like(input)
        variance = beta_t * (1 - alpha_cumulative_prod_t_prev) / (1 - alpha_cumulative_t)

        ### 0 out the variance for if the timestep == 0 ###
        variance = variance * greater_than_0_mask
        variance = self._repeated_unsqueeze(input, variance)
        sigma_z = noise * variance**0.5

        ### Compute Noise Coefficient (1 - a_t / sqrt(1 - cumulative_a)) where 1 - a_t = b_t ###
        beta_t = self.beta_schedule[timestep].to(device)
        alpha_cumulative_t = self.alpha_cumulative_prod[timestep].to(device)
        root_one_minus_cumulative_alpha_t = (1 - alpha_cumulative_t) ** 0.5
        noise_coefficient = beta_t / root_one_minus_cumulative_alpha_t
        noise_coefficient = self._repeated_unsqueeze(input, noise_coefficient)
        

        ### Compute 1 / sqrt(a_t) ###
        reciprocal_root_a_t = (self.alpha[timestep]**-0.5).to(device)
        reciprocal_root_a_t = self._repeated_unsqueeze(input, reciprocal_root_a_t)
        
        ### Compute Denoised Image ###
        denoised = reciprocal_root_a_t * (input - (noise_coefficient * predicted_noise)) + sigma_z
 
        return denoised

### DEFINE DIFFUSION MODEL ###

class SelfAttention(nn.Module):

  def __init__(self,
               in_channels,
               num_heads=12, 
               attn_p=0,
               proj_p=0,
               fused_attn=True):

    super().__init__()
    assert in_channels % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = int(in_channels / num_heads)
    self.scale = self.head_dim ** -0.5
    self.fused_attn = fused_attn  

    self.qkv = nn.Linear(in_channels, in_channels*3)
    self.attn_p = attn_p
    self.attn_drop = nn.Dropout(attn_p)
    self.proj = nn.Linear(in_channels, in_channels)
    self.proj_drop = nn.Dropout(proj_p)

  def forward(self, x):
    batch_size, seq_len, embed_dim = x.shape
      
    qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2,0,3,1,4)
    q,k,v = qkv.unbind(0)

    if self.fused_attn:
      x = F.scaled_dot_product_attention(q,k,v, dropout_p=self.attn_p)
    else:
      attn = (q @ k.transpose(-2,-1)) * self.scale
      attn = attn.softmax(dim=-1)
      attn = self.attn_drop(attn)
      x = attn @ v
    
    x = x.transpose(1,2).reshape(batch_size, seq_len, embed_dim)
    x = self.proj(x)
    x = self.proj_drop(x)
    
    return x

class MLP(nn.Module):
    def __init__(self, 
                 in_channels,
                 mlp_ratio=4, 
                 act_layer=nn.GELU,
                 mlp_p=0):

        super().__init__()
        hidden_features = int(in_channels * mlp_ratio)
        self.fc1 = nn.Linear(in_channels, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2 = nn.Linear(hidden_features, in_channels)
        self.drop2 = nn.Dropout(mlp_p)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
        
class TransformerBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 fused_attention=True,
                 num_heads=4, 
                 mlp_ratio=2,
                 proj_p=0,
                 attn_p=0,
                 mlp_p=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.norm1 = norm_layer(in_channels, eps=1e-6)

        self.attn = SelfAttention(in_channels=in_channels,
                                  num_heads=num_heads, 
                                  attn_p=attn_p,
                                  proj_p=proj_p,
                                  fused_attn=fused_attention)
        
        self.norm2 = norm_layer(in_channels, eps=1e-6)
        self.mlp = MLP(in_channels=in_channels,
                       mlp_ratio=mlp_ratio,
                       act_layer=act_layer,
                       mlp_p=mlp_p)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
      
        ### Reshape to batch_size x (height*width) x channels
        x = x.reshape(batch_size, channels, height*width).permute(0,2,1)
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0,2,1).reshape(batch_size, channels, height, width)
        return x


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim, scaled_time_embed_dim):
        super().__init__()
        self.inv_freqs = nn.Parameter(1.0 / (10000 ** (torch.arange(0, time_embed_dim, 2).float() / (time_embed_dim/2))), requires_grad=False)
        
        self.time_mlp = nn.Sequential(nn.Linear(time_embed_dim, scaled_time_embed_dim), 
                                      nn.SiLU(), 
                                      nn.Linear(scaled_time_embed_dim, scaled_time_embed_dim), 
                                      nn.SiLU())
    def forward(self, timesteps):
        timestep_freqs = timesteps.unsqueeze(1) * self.inv_freqs.unsqueeze(0)
        embeddings = torch.cat([torch.sin(timestep_freqs), torch.cos(timestep_freqs)], axis=-1)
        embeddings = self.time_mlp(embeddings)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groupnorm_num_groups, time_embed_dim):
        super().__init__()
        
        ### Time Embedding Expansion to Out Channels ###
        self.time_expand = nn.Linear(time_embed_dim, out_channels)

        ### Input Convolutions + GroupNorm ###
        self.groupnorm_1 = nn.GroupNorm(groupnorm_num_groups, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

        ### Input + Time Embedding Convolutions + GroupNorm ###
        self.groupnorm_2 = nn.GroupNorm(groupnorm_num_groups, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")

        ### Residual Layer ###
        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embeddings):

        residual_connection = x

        ### Time Expansion to Out Channels ###
        time_embed = self.time_expand(time_embeddings)
        
        ### Input GroupNorm and Convolutions ###
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        ### Add Time Embeddings ###
        x = x + time_embed.reshape((*time_embed.shape, 1, 1))

        ### Group Norm and Conv Again! ###
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        ### Add Residual and Return ###
        x = x + self.residual_connection(residual_connection)
        
        return x

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        )

    def forward(self, inputs):
        batch, channels, height, width = inputs.shape
        upsampled = self.upsample(inputs)
        assert (upsampled.shape == (batch, channels, height*2, width*2))
        return upsampled


class UNET(nn.Module):
    def __init__(self, in_channels=3, start_dim=64, dim_mults=(1,2,4), residual_blocks_per_group=1, groupnorm_num_groups=16, time_embed_dim=128):
        super().__init__()

        #######################################
        ### COMPUTE ALL OF THE CONVOLUTIONS ###
        #######################################
        
        ### Store Number of Input channels from Original Image ###
        self.input_image_channels = in_channels
        
        ### Get Number of Channels at Each Block ###
        channel_sizes = [start_dim*i for i in dim_mults]
        starting_channel_size, ending_channel_size = channel_sizes[0], channel_sizes[-1]

        ### Compute the Input/Output Channel Sizes for Every Convolution of Encoder ###
        self.encoder_config = []
        
        for idx, d in enumerate(channel_sizes):
            ### For Every Channel Size add "residual_blocks_per_group" number of Residual Blocks that DONT Change the number of channels ###
            for _ in range(residual_blocks_per_group):
                self.encoder_config.append(((d, d), "residual")) # Shape: (Batch x Channels x Height x Width) -> (Batch x Channels x Height x Width)

            ### After Residual Blocks include Downsampling (by factor of 2) but dont change number of channels ###
            self.encoder_config.append(((d,d), "downsample")) # Shape: (Batch x Channels x Height x Width) -> (Batch x Channels x Height/2 x Width/2)

            ### Compute Attention ###
            self.encoder_config.append((d, "attention"))
            
            ### If we are not at the last channel size include a channel upsample (typically by factor of 2) ###
            if idx < len(channel_sizes) - 1:
                self.encoder_config.append(((d,channel_sizes[idx+1]), "residual")) # Shape: (Batch x Channels x Height x Width) -> (Batch x Channels*2 x Height x Width)
            
        ### The Bottleneck will have "residual_blocks_per_group" number of ResidualBlocks each with the input/output of our final channel size###
        self.bottleneck_config = []
        for _ in range(residual_blocks_per_group):
            self.bottleneck_config.append(((ending_channel_size, ending_channel_size), "residual"))

        ### Store a variable of the final Output Shape of our Encoder + Bottleneck so we can compute Decoder Shapes ###
        out_dim = ending_channel_size

        ### Reverse our Encoder config to compute the Decoder ###
        reversed_encoder_config = self.encoder_config[::-1]

        ### The output of our reversed encoder will be the number of channels added for residual connections ###
        self.decoder_config = []
        for idx, (metadata, type) in enumerate(reversed_encoder_config):
            ### Flip in_channels, out_channels with the previous out_dim added on ###
            if type != "attention":
                enc_in_channels, enc_out_channels = metadata
            
                self.decoder_config.append(((out_dim+enc_out_channels, enc_in_channels), "residual"))
                        
                if type == "downsample":
                    ### If we did a downsample in our encoder, we need to upsample in our decoder ###
                    self.decoder_config.append(((enc_in_channels, enc_in_channels), "upsample"))
    
                ### The new out_dim will be the number of output channels from our block (or the cooresponding encoder input channels) ###
                out_dim = enc_in_channels
            else:
                in_channels = metadata
                self.decoder_config.append((in_channels, "attention"))

        ### Add Extra Residual Block for residual from input convolution ###
        # hint: We know that the initial convolution will have starting_channel_size
        # and the output of our decoder will also have starting_channel_size, so the
        # final ResidualBlock we need will need to go from starting_channel_size*2 to starting_channel_size

        self.decoder_config.append(((starting_channel_size*2, starting_channel_size), "residual"))
        
        
        #######################################
        ### ACTUALLY BUILD THE CONVOLUTIONS ###
        #######################################

        ### Intial Convolution Block ###
        self.conv_in_proj = nn.Conv2d(self.input_image_channels, 
                                      starting_channel_size, 
                                      kernel_size=3, 
                                      padding="same")
        
        self.encoder = nn.ModuleList()
        for metadata, type in self.encoder_config:
            if type == "residual":
                in_channels, out_channels = metadata
                self.encoder.append(ResidualBlock(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  groupnorm_num_groups=groupnorm_num_groups,
                                                  time_embed_dim=time_embed_dim))
            elif type == "downsample":
                in_channels, out_channels = metadata
                self.encoder.append(
                    nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=3, 
                              stride=2, 
                              padding=1)
                    )
            elif type == "attention":
                in_channels = metadata
                self.encoder.append(TransformerBlock(in_channels))

        
        ### Build Encoder Blocks ###
        self.bottleneck = nn.ModuleList()
        
        for (in_channels, out_channels), _ in self.bottleneck_config:
            self.bottleneck.append(ResidualBlock(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 groupnorm_num_groups=groupnorm_num_groups,
                                                 time_embed_dim=time_embed_dim))

        ### Build Decoder Blocks ###
        self.decoder = nn.ModuleList()
        for metadata, type in self.decoder_config:
            if type == "residual":
                in_channels, out_channels = metadata
                self.decoder.append(ResidualBlock(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  groupnorm_num_groups=groupnorm_num_groups,
                                                  time_embed_dim=time_embed_dim))
            elif type == "upsample":
                in_channels, out_channels = metadata
                self.decoder.append(UpSampleBlock(in_channels=in_channels, 
                                                  out_channels=out_channels))

            elif type == "attention":
                in_channels = metadata
                self.decoder.append(TransformerBlock(in_channels))

        ### Output Convolution ###
        self.conv_out_proj = nn.Conv2d(in_channels=starting_channel_size, 
                                       out_channels=self.input_image_channels,
                                       kernel_size=3, 
                                       padding="same")

        
    def forward(self, x, time_embeddings):
        residuals = []

        ### Pass Through Projection and Store Residual ###
        x = self.conv_in_proj(x)
        residuals.append(x)

        ### Pass through encoder and store residuals ##
        for module in self.encoder:
            if isinstance(module, (ResidualBlock)):
                x = module(x, time_embeddings)
                residuals.append(x)
            elif isinstance(module, nn.Conv2d):
                x = module(x)
                residuals.append(x)
            else:
                x = module(x)

        ### Pass Through BottleNeck ###
        for module in self.bottleneck:
            x = module(x, time_embeddings)

        ### Pass through Decoder while Concatenating Residuals ###
        for module in self.decoder:
            if isinstance(module, ResidualBlock):
                residual_tensor = residuals.pop()
                x  = torch.cat([x, residual_tensor], axis=1)
                x = module(x, time_embeddings)
            else:
                x = module(x)

        ### Map back to num_channels for final output ###
        x = self.conv_out_proj(x)
        
        return x

class Diffusion(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 start_dim=128, 
                 dim_mults=(1,2,4), 
                 residual_blocks_per_group=1, 
                 groupnorm_num_groups=16, 
                 time_embed_dim=128, 
                 time_embed_dim_ratio=2):

        super().__init__()
        self.in_channels = in_channels
        self.start_dim = start_dim
        self.dim_mults = dim_mults
        self.residual_blocks_per_group = residual_blocks_per_group
        self.groupnorm_num_groups = groupnorm_num_groups

        self.time_embed_dim = time_embed_dim
        self.scaled_time_embed_dim = int(time_embed_dim * time_embed_dim_ratio)

        self.sinusoid_time_embeddings = SinusoidalTimeEmbedding(time_embed_dim=self.time_embed_dim,
                                                                scaled_time_embed_dim=self.scaled_time_embed_dim)

        self.unet = UNET(in_channels=in_channels, 
                         start_dim=start_dim, 
                         dim_mults=dim_mults, 
                         residual_blocks_per_group=residual_blocks_per_group, 
                         groupnorm_num_groups=groupnorm_num_groups,  
                         time_embed_dim=self.scaled_time_embed_dim)

    def forward(self, noisy_inputs, timesteps):

        ### Embed the Timesteps ###
        timestep_embeddings = self.sinusoid_time_embeddings(timesteps)
        
        ### Pass Images + Time Embeddings through UNET ###
        noise_pred = self.unet(noisy_inputs, timestep_embeddings)

        return noise_pred

### DEFINE PLOTTING FUNCTION ###
@torch.no_grad()
def sample_plot_image(step_idx, 
                      total_timesteps, 
                      sampler, 
                      image_size,
                      num_channels,
                      plot_freq, 
                      model,
                      num_gens,
                      path_to_generated_dir,
                      device):

    ### Conver Tensor back to Image (From Huggingface Annotated Diffusion) ###
    tensor2image_transform = transforms.Compose([
        transforms.Lambda(lambda t: t.squeeze(0)) if num_channels==3 else nn.Identity(),
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    images = torch.randn((num_gens, num_channels, image_size, image_size))
    num_images_per_gen = (total_timesteps // plot_freq)

    images_to_vis = [[] for _ in range(num_gens)]
    for t in np.arange(total_timesteps)[::-1]:
        ts = torch.full((num_gens, ), t)
        noise_pred = model(images.to(device), ts.to(device)).detach().cpu()
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
    plt.savefig(os.path.join(path_to_generated_dir, f"step_{step_idx}.png"))
    
    return fig

### DEFINE DATASET CLASS ###
class CelebADataset(Dataset):
    def __init__(self, path_to_data, transform=None):
        self.transforms = transform
        self.path_to_files = [os.path.join(path_to_data, file) for file in os.listdir(path_to_data)]
    
    def __len__(self):
        return len(self.path_to_files)
    
    def __getitem__(self, idx):
        img_path = self.path_to_files[idx]
        img = Image.open(img_path)
        return self.transforms(img)

### PREP ACCELERATOR AND TRACKERS ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment, 
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          log_with="wandb")
accelerator.init_trackers(args.experiment_name)
wandb_tracker = accelerator.get_tracker("wandb")

### PREP DATALOADER ###
image2tensor = transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(), 
                    transforms.Lambda(lambda t: (t*2) - 1)
                ])
mini_batch_size = args.batch_size // args.gradient_accumulation_steps
dataset = CelebADataset(args.path_to_data, transform=image2tensor)
trainloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


### DEFINE MODEL ###
model = Diffusion(in_channels=3, 
                  start_dim=args.starting_channels, 
                  dim_mults=(1,2,3,4), 
                  residual_blocks_per_group=2,
                  time_embed_dim=args.starting_channels*2)

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

### PREPARE EVERYTHING ###
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
    for images in trainloader:
        images = images.to(accelerator.device)

        with accelerator.accumulate(model):
            
            ### Grab Number of Samples in Batch ###
            batch_size = images.shape[0]
        
            ### Random Sample T ###
            timesteps = torch.randint(0,args.num_diffusion_timesteps,(batch_size,))
        
            ### Get Noisy Images ###
            noisy_images, noise = ddpm_sampler.add_noise(images, timesteps)
        
            ### Get Noise Prediction ###
            noise_pred = model(noisy_images.to(accelerator.device), timesteps.to(accelerator.device))

            ### Compute Error ###
            loss = loss_fn(noise_pred, noise.to(accelerator.device))
            accumulated_loss += loss / args.gradient_accumulation_steps

            ### Compute Gradients ###
            accelerator.backward(loss)

            ### Clip Gradients ###
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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

            ### EVALUATION CHECK ###
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
                accelerator.print("Generating Images")
                if accelerator.is_main_process:
                    fig = sample_plot_image(step_idx=completed_steps, 
                                            total_timesteps=args.num_diffusion_timesteps, 
                                            sampler=ddpm_sampler, 
                                            image_size=args.img_size,
                                            num_channels=3,
                                            plot_freq=args.plot_freq_interval, 
                                            model=model,
                                            num_gens=args.num_generations,
                                            path_to_generated_dir=args.generated_directory,
                                            device=accelerator.device)
                    wandb_tracker.log({"plot": fig},
                                      step=completed_steps)

            if completed_steps >= args.num_training_steps:
                train = False
                accelerator.print("Completed Training")
                break

accelerator.end_training()
