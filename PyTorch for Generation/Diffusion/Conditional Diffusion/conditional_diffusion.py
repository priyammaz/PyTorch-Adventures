import os    
import math
import numpy as np
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer, CLIPTextModel
import itertools


###################################################
################ DEFINE SAMPLER ###################
###################################################

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
        
        ### Compute Denoised Image ###
        denoised = reciprocal_root_a_t * (input - (noise_coefficient * predicted_noise)) + sigma_z
 
        return denoised


###################################################
################  DEFINE MODEL  ###################
###################################################

class SelfAttention(nn.Module):

  def __init__(self,
               embed_dim,
               num_heads=8, 
               attn_p=0,
               proj_p=0,
               fused_attn=True):

    super().__init__()
    assert embed_dim % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = int(embed_dim / num_heads)
    self.scale = self.head_dim ** -0.5
    self.fused_attn = fused_attn  

    self.qkv = nn.Linear(embed_dim, embed_dim*3)
    self.attn_p = attn_p
    self.attn_drop = nn.Dropout(attn_p)
    self.proj = nn.Linear(embed_dim, embed_dim)
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
        
class CrossAttention(nn.Module):
  def __init__(self,
               embed_dim,
               num_heads=8, 
               attn_p=0,
               proj_p=0):

    super().__init__()
    assert embed_dim % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = int(embed_dim / num_heads)
    self.scale = self.head_dim ** -0.5
      
    self.query = nn.Linear(embed_dim, embed_dim)
    self.key = nn.Linear(embed_dim, embed_dim)
    self.value = nn.Linear(embed_dim, embed_dim)
      
    self.attn_p = attn_p
    self.attn_drop = nn.Dropout(attn_p)
    self.proj = nn.Linear(embed_dim, embed_dim)
    self.proj_drop = nn.Dropout(proj_p)

  def forward(self, x, context, mask=None):
    batch_size, x_seq_len, embed_dim = x.shape
    batch_size, c_seq_len, embed_dim = context.shape

    q = self.query(x)
    k = self.key(context)
    v = self.value(context)
    
    q = q.reshape(batch_size, x_seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
    k = k.reshape(batch_size, c_seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
    v = v.reshape(batch_size, c_seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)

    attn = (q @ k.transpose(-2,-1)) * self.scale

    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(1)
        attn = attn.masked_fill_(mask, float("-inf"))

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn @ v
    
    x = x.transpose(1,2).reshape(batch_size, x_seq_len, embed_dim)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x


class MLP(nn.Module):
    def __init__(self, 
                 embed_dim,
                 mlp_ratio=4, 
                 act_layer=nn.GELU,
                 mlp_p=0):

        super().__init__()
        hidden_features = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2 = nn.Linear(hidden_features, embed_dim)
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
                 embed_dim,
                 context_embed_dim,
                 fused_attention=True,
                 num_heads=4, 
                 mlp_ratio=2,
                 proj_p=0,
                 attn_p=0,
                 mlp_p=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        
        super().__init__()

        ### Linear Layer to Map Context Embed Dim to Image Embed Dim (Number of Channels) ###
        self.context_map = nn.Linear(context_embed_dim, embed_dim)
        
        ### Self Attention ###
        self.norm1 = norm_layer(embed_dim, eps=1e-6)
        self.attn = SelfAttention(embed_dim=embed_dim,
                                  num_heads=num_heads, 
                                  attn_p=attn_p,
                                  proj_p=proj_p,
                                  fused_attn=fused_attention)
        
        

        ### Cross Attention ###
        self.norm2 = norm_layer(embed_dim, eps=1e-6)
        self.cross_attn = CrossAttention(embed_dim=embed_dim,
                                         num_heads=num_heads, 
                                         attn_p=attn_p,
                                         proj_p=proj_p)

        
        ### MLP ###
        self.norm3 = norm_layer(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim=embed_dim,
                       mlp_ratio=mlp_ratio,
                       act_layer=act_layer,
                       mlp_p=mlp_p)


    def forward(self, x, context, mask=None):

        batch_size, channels, height, width = x.shape

        ### Store Copy for Residual ###
        input_residual = x

        ### Map Context Embed Dim to Num Channels (Image embed Dim
        context = self.context_map(context)
      
        ### Reshape to batch_size x (height*width) x channels
        x = x.reshape(batch_size, channels, height*width).permute(0,2,1)

        ### Pass Through Attentions ###
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context=context, mask=mask)

        ### Pass Though MLP Layers ###
        x = x + self.mlp(self.norm3(x))

        ### Return Back to (Batch x Channels x Height x Width ###
        x = x.permute(0,2,1).reshape(batch_size, channels, height, width)

        ### Add to Input Residual ###
        x = x + input_residual
        
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
    def __init__(self, 
                 in_channels=3, 
                 context_embed_dim=768,
                 start_dim=64, 
                 dim_mults=(1,2,4), 
                 residual_blocks_per_group=1, 
                 groupnorm_num_groups=16, 
                 time_embed_dim=128):
        
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
                self.encoder.append(TransformerBlock(in_channels, context_embed_dim))

        
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
                self.decoder.append(TransformerBlock(in_channels, context_embed_dim))

        ### Output Convolution ###
        self.conv_out_proj = nn.Conv2d(in_channels=starting_channel_size, 
                                       out_channels=self.input_image_channels,
                                       kernel_size=3, 
                                       padding="same")

        
    def forward(self, x, time_embeddings, context, mask=None):
        residuals = []

        ### Pass Through Projection and Store Residual ###
        x = self.conv_in_proj(x)
        residuals.append(x)

        ### Pass through encoder and store residuals ##
        for module in self.encoder:
            if isinstance(module, ResidualBlock):
                x = module(x, time_embeddings)
                residuals.append(x)
            elif isinstance(module, nn.Conv2d):
                x = module(x)
                residuals.append(x)
            elif isinstance(module, TransformerBlock):
                x = module(x, context, mask)
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
            elif isinstance(module, TransformerBlock):
                x = module(x, context, mask)
            else:
                x = module(x)

        ### Map back to num_channels for final output ###
        x = self.conv_out_proj(x)
        
        return x

class Diffusion(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 context_embed_dim=512,
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

        ### Text Encoding CLIP Model ###
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder.eval()

        ### Diffusion Model Parts ###
        self.sinusoid_time_embeddings = SinusoidalTimeEmbedding(time_embed_dim=self.time_embed_dim,
                                                                scaled_time_embed_dim=self.scaled_time_embed_dim)
        self.unet = UNET(in_channels=in_channels, 
                         context_embed_dim=context_embed_dim,
                         start_dim=start_dim, 
                         dim_mults=dim_mults, 
                         residual_blocks_per_group=residual_blocks_per_group, 
                         groupnorm_num_groups=groupnorm_num_groups,  
                         time_embed_dim=self.scaled_time_embed_dim)

    @torch.no_grad()
    def encode_text(self, context):
        encoded_text = self.text_encoder(context).last_hidden_state
        return encoded_text
        
    def forward(self, noisy_inputs, timesteps, context, mask=None):

        ### Encode Text ###
        encoded_text = self.encode_text(context)
        
        ### Embed the Timesteps ###
        timestep_embeddings = self.sinusoid_time_embeddings(timesteps)

        ### Pass Images + Time Embeddings through UNET ###
        noise_pred = self.unet(noisy_inputs, timestep_embeddings, encoded_text, mask)

        return noise_pred



#############################################################
####################### BUILD DATASET #######################
#############################################################

class COCODataset(Dataset):
    def __init__(self, path_to_root, train=True, img_size=128):

        self.path_to_root = path_to_root
        
        if train:
            path_to_annotations = os.path.join(self.path_to_root, "annotations", "captions_train2017.json")
            self.path_to_images = os.path.join(self.path_to_root, "train2017")

        else:
            path_to_annotations = os.path.join(self.path_to_root, "annotations", "captions_val2017.json")
            self.path_to_images = os.path.join(self.path_to_root, "val2017")


        self._prepare_annotations(path_to_annotations)


        self.image2tensor = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Lambda(lambda t: (t*2) - 1)
            ]
        )
        
    def _prepare_annotations(self, path_to_annotations):

        ### Load Annotation Json ###
        with open(path_to_annotations, "r") as f:
            annotation_json = json.load(f)
            
        ### For Each Image ID Get the Corresponding Annotations ###
        id_annotations = {}
    
        for annot in annotation_json["annotations"]:
            image_id = annot["image_id"]
            caption = annot["caption"]
    
            if image_id not in id_annotations:
                id_annotations[image_id] = [caption]
            else:
                id_annotations[image_id].append(caption)
    
        ### Coorespond Image Id to Filename ###
        path_id_coorespondance = {}
    
        for image in annotation_json["images"]:
            file_name = image["file_name"]
            image_id = image["id"]
        
            path_id_coorespondance[file_name] = id_annotations[image_id]

        self.filenames = list(path_id_coorespondance.keys())
        self.annotations = path_id_coorespondance


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        ### Grab Filename and Cooresponding Annotations ###
        filename = self.filenames[idx]
        annotation = self.annotations[filename]

        ### If more than 1 annotation, randomly select ###
        annotation = random.choice(annotation)

        ### Remove Any Whitespace from Text ###
        annotation = annotation.strip()

        ### Load Image ###
        path_to_img = os.path.join(self.path_to_images, filename)
        img = Image.open(path_to_img).convert("RGB")

        ### Apply Image Transforms ###
        img = self.image2tensor(img)
        
        return img, annotation



def collate_fn(examples):

    ### Grab Images and add batch dimension ###
    images = [i[0].unsqueeze(0) for i in examples]

    ### Grab Text Annotations ###
    annot = [i[1] for i in examples]

    ### Stick All Images Together along Batch ###
    images = torch.concatenate(images)

    ### Tokenize Annotations wiht Padding ###
    annotation = tokenizer(annot, padding=True, return_tensors="pt")

    ### Store Batch as Dictionary ###
    batch = {"images": images, 
             "context": annotation["input_ids"], 
             "mask": ~annotation["attention_mask"].bool()} # Flipped mask, so True on pad tokens rather than actual tokens

    return batch


d = COCODataset("../../../data/coco2017", train=True)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
loader = DataLoader(d, batch_size=32, collate_fn=collate_fn, num_workers=16, pin_memory=True)

for d in tqdm(loader):
    break

model = Diffusion().to("cuda")

images, context, mask = d["images"].to("cuda"), d["context"].to("cuda"), d["mask"].to("cuda")
timesteps = torch.randint(0,1000, (32,)).to("cuda")

out = model(images, timesteps, context, mask=mask)
print(out.shape)



















