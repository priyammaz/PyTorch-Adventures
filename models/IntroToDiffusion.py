import torch
import torchvision
import matplotlib.pyplot as plt
import os    
import math
from torchvision import transforms 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
import itertools

class Sampler:
    def __init__(self, num_training_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_training_steps = num_training_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        ### Define Basic Beta Scheduler ###
        self.beta_schedule = self.linear_beta_schedule()

        ### Compute Alphas for Direction 0 > t Noise Calculation ###
        self.alpha = 1 - self.beta_schedule
        self.alpha_cumulative_prod = torch.cumprod(self.alpha, dim=-1)
    
    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_training_steps)

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
        
        ### Compute Previous X_t-1 ###
        prev_img_pred = reciprocal_root_a_t * (input - (noise_coefficient * predicted_noise)) + sigma_z

        # return prev_img_pred
        return prev_img_pred
    

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

    def forward(self, input, time_embeddings):

        residual_connection = input

        ### Time Expansion to Out Channels ###
        time_embed = self.time_expand(time_embeddings)
        
        ### Input GroupNorm and Convolutions ###
        input = self.groupnorm_1(input)
        input = F.silu(input)
        input = self.conv_1(input)

        ### Add Time Embeddings ###
        input = input + time_embed.reshape((*time_embed.shape, 1, 1))

        ### Group Norm and Conv Again! ###
        input = self.groupnorm_2(input)
        input = F.silu(input)
        input = self.conv_2(input)

        ### Add Residual and Return ###
        input = input + self.residual_connection(residual_connection)
        return input

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
    def __init__(self, in_channels=3, start_dim=64, dim_mults=(1,2,4), residual_blocks_per_group=2, groupnorm_num_groups=16, time_embed_dim=128):
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

        ### Pass Through Projection ###
        x = self.conv_in_proj(x)
        
        ### Pass through encoder and store residuals ##
        for module in self.encoder:
            if isinstance(module, ResidualBlock):
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
    
        x = self.conv_out_proj(x)

        return x
            


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim, scaled_time_embed_dim, device):
        super().__init__()
        self.device = device
        self.inv_freqs = nn.Parameter(1.0 / (10000 ** (torch.arange(0, time_embed_dim, 2).float() / (time_embed_dim/2))), requires_grad=False).to(device)
        
        self.time_mlp = nn.Sequential(nn.Linear(time_embed_dim, scaled_time_embed_dim), 
                                      nn.SiLU(), 
                                      nn.Linear(scaled_time_embed_dim, scaled_time_embed_dim), 
                                      nn.SiLU())
    def forward(self, timesteps):
        timestep_freqs = timesteps.unsqueeze(1).to(self.device) * self.inv_freqs.unsqueeze(0)
        embeddings = torch.cat([torch.cos(timestep_freqs), torch.sin(timestep_freqs)], axis=-1)

        embeddings = self.time_mlp(embeddings)
        return embeddings


class Diffusion(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 start_dim=128, 
                 dim_mults=(1,2,4,8), 
                 residual_blocks_per_group=2, 
                 groupnorm_num_groups=16, 
                 time_embed_dim=128, 
                 time_embed_dim_ratio=4,
                 device="cuda"):

        super().__init__()
        self.in_channels = in_channels
        self.start_dim = start_dim
        self.dim_mults = dim_mults
        self.residual_blocks_per_group = residual_blocks_per_group
        self.groupnorm_num_groups = groupnorm_num_groups

        self.time_embed_dim = time_embed_dim
        self.scaled_time_embed_dim = int(time_embed_dim * time_embed_dim_ratio)

        self.sinusoid_time_embeddings = SinusoidalTimeEmbedding(time_embed_dim=self.time_embed_dim,
                                                                scaled_time_embed_dim=self.scaled_time_embed_dim,
                                                                device=device)

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


@torch.no_grad()
def sample_plot_image(step_idx, 
                      total_timesteps, 
                      sampler, 
                      image_size,
                      num_channels,
                      plot_freq, 
                      model,
                      num_gens,
                      path_to_save,
                      device):

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
    plt.savefig(os.path.join(path_to_save, f"{step_idx}.png"), dpi=300)
    plt.close()


def train(image_size=128, 
          evaluation_interval=2000,
          total_timesteps=1000, 
          plot_freq_interval=100, 
          num_generations=5, 
          num_training_steps=250000, 
          num_input_channels=3, 
          batch_size=256, 
          gradient_accumulation_steps=4,
          path_to_save_gens="generated/repeatmodel",
          path_to_model_save="save_model/repeatmodel"):

    accelerator = Accelerator() 
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if accelerator.is_local_main_process:
        if not os.path.isdir(path_to_save_gens):
            os.mkdir(path_to_save_gens)

        
    ### Define Basic Image Transformations ###
    image2tensor = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    transforms.Lambda(lambda t: (t*2) - 1)
                ])
    dataset = ImageFolder("/mnt/datadrive/data/CelebA", transform=image2tensor)
    trainloader = DataLoader(dataset, batch_size=batch_size//gradient_accumulation_steps, shuffle=True, num_workers=24, pin_memory=True)

    model = Diffusion(in_channels=3, 
                      start_dim=128, 
                      dim_mults=(1,1,2,2,4,4), 
                      residual_blocks_per_group=2, 
                      groupnorm_num_groups=16, 
                      time_embed_dim=128, 
                      time_embed_dim_ratio=4,
                      device=accelerator.device)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of Parameters:", params)

    ### MODEL TRAINING INPUTS ###
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0008)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps=5000, 
                                                num_training_steps=num_training_steps)

    model, optimizer, trainloader, scheduler = accelerator.prepare(model, optimizer, trainloader, scheduler)

    ddpm_sampler = Sampler(num_training_steps=total_timesteps)

    loss_fn = nn.HuberLoss()

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_loss = torch.inf

    while True:
        training_losses = []

        for images, _ in trainloader:
            batch_size = images.shape[0]
        
            ### Random Sample T ###
            timesteps = torch.randint(0,total_timesteps,(batch_size,))
        
            ### Get Noisy Images ###
            noisy_images, noise = ddpm_sampler.add_noise(images, timesteps)
                
            with accelerator.accumulate(model):
                ### Get Noise Prediction ###
                noise_pred = model(noisy_images, timesteps)

                ### Compute Error ###
                loss = loss_fn(noise_pred, noise)

                training_losses.append(loss.cpu().item()*gradient_accumulation_steps)

                ### Compute Gradients ###
                accelerator.backward(loss)

                ### Clip Gradients just Incase ###
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                
            progress_bar.update(1)
            completed_steps += 1

            if (completed_steps % evaluation_interval == 0) and accelerator.is_local_main_process:
                loss_mean = np.mean(training_losses)
                print("Training Loss:", loss_mean)
                print("Learning Rate:", optimizer.param_groups[-1]["lr"])

                if loss_mean < best_loss:
                    print("Saving Model!")
                    best_loss = loss_mean
                    accelerator.save_model(model, path_to_model_save)

                training_losses = []
                print("Saving Image Generation")
                sample_plot_image(step_idx=completed_steps, 
                                total_timesteps=total_timesteps, 
                                sampler=ddpm_sampler, 
                                image_size=image_size,
                                num_channels=num_input_channels,
                                plot_freq=plot_freq_interval, 
                                model=model,
                                num_gens=num_generations,
                                path_to_save=path_to_save_gens,
                                device=accelerator.device)
                
                print("Completed Image Generation")

            if completed_steps >= num_training_steps:
                sample_plot_image(step_idx=completed_steps, 
                                total_timesteps=total_timesteps, 
                                sampler=ddpm_sampler, 
                                image_size=image_size,
                                num_channels=num_input_channels,
                                plot_freq=plot_freq_interval, 
                                model=model,
                                num_gens=num_generations,
                                path_to_save=path_to_save_gens,
                                device=accelerator.device)
                break


if __name__ == "__main__":
    train()
