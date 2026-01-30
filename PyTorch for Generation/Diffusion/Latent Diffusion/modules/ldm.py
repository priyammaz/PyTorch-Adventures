import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from utils import save_generated_images

from .vae import VAE, VQVAE
from .unet import UNet2DModel
from .embeddings import PositionalEncoding, ClassConditionalEmbeddings, \
    TextConditionalEmbeddings
from .scheduler import Sampler

loss_functions = {"mse": nn.MSELoss(), 
                  "mae": nn.L1Loss(), 
                  "huber": nn.HuberLoss()}

class LDM(nn.Module):
    
    def __init__(self, config):
    
        super(LDM, self).__init__()

        self.config = config

        ### Load VAE Model ###
        if self.config.quantize:
            self.vae = VQVAE(config=config)
        else:
            self.vae = VAE(config=config)  

        if config.vae_scale_factor is None:
            print("vae_scale_factor not set in LDMConfig! Defaulting to 1 unless passed into forward") 

        ### VAE is PreTrained, Disable Gradients ###
        for param in self.vae.parameters():
            param.requires_grad = False

        ### Load Diffusion Time Embeddings ###
        self.sinusoidal_time_embeddings = PositionalEncoding(
            max_len=config.num_diffusion_timesteps, 
            time_embed_start_dim=config.time_embed_start_dim,
            time_embed_proj_dim=config.time_embed_proj_dim
        )

        ### Load Text Conditioning Model (if we have text and its not preencoded) ###
        if config.text_conditioning:
            self.text_encoder = TextConditionalEmbeddings(
                pre_encoded_text=config.pre_encoded_text, 
                text_conditioning_hf_model=config.text_conditioning_hf_model,
                text_embed_dim=config.text_embed_dim
            )

        ### Load Class Conditioning Module ###
        if config.class_conditioning:
            self.class_encoder = ClassConditionalEmbeddings(
                num_classes=config.num_classes, 
                embed_dim=config.class_embed_dim
            )

        ### Load UNET Model ###
        self.unet = UNet2DModel(config)

        ### Get DDPM Sampler ###
        self.ddpm_sampler = Sampler(total_timesteps=config.num_diffusion_timesteps,
                                    beta_start=config.beta_start, 
                                    beta_end=config.beta_end)

    def _load_vae_state_dict(self, state_dict):
        sucess = self.vae.load_state_dict(state_dict)
        print(sucess)
        
    @torch.no_grad()
    def _vae_encode_images(self, x, scale_factor=None):
        return self.vae.encode(x, return_stats=False, scale_factor=scale_factor)["posterior"]
    
    @torch.no_grad()
    def _vae_decode_images(self, x, scale_factor=None):
        return self.vae.decode(x, scale_factor=scale_factor)
    
    def forward(self, 
                images, 
                text_conditioning=None, 
                text_attention_mask=None,
                class_conditioning=None,
                scale_factor=None,
                cfg_dropout_prob=0):
        
        ### If we have text conditioning, then encode ###
        if self.config.text_conditioning:
            
            text_conditioning = self.text_encoder(
                batch_size=images.shape[0],
                text_conditioning=text_conditioning, 
                text_attention_mask=text_attention_mask, 
                cfg_dropout_prob=cfg_dropout_prob

            )

        ### Get Class Conditioning ###
        if self.config.class_conditioning:
            
            class_conditioning = self.class_encoder(
                batch_size=images.shape[0],
                class_conditioning=class_conditioning, 
                cfg_dropout_prob=cfg_dropout_prob

            )

        ### Compress Images using AutoEncoder ###
        compressed_images = self._vae_encode_images(images, scale_factor=scale_factor)

        ### Sample Random Timesteps for Noise ###
        timesteps = torch.randint(0, self.config.num_diffusion_timesteps, (images.shape[0], ))

        ### Add Noise to Images ###
        noisy_images, noise = self.ddpm_sampler.add_noise(compressed_images, timesteps)
        
        ### Get Timestep Embeddings ###
        timestep_embeddings = self.sinusoidal_time_embeddings(timesteps.to(images.device))

        ### Predict Noise with UNet Model ###
        noise_pred = self.unet(noisy_images.to(images.device), 
                               timestep_embeddings, 
                               text_conditioning=text_conditioning, 
                               text_attention_mask=text_attention_mask,
                               class_conditioning=class_conditioning)

        ### Compute Loss ###
        loss = loss_functions[self.config.diffusion_loss_fn](noise_pred, noise.to(noise_pred.device))

        return loss

    
    @torch.no_grad()
    def inference(self,
                  text_conditioning=None, 
                  text_attention_mask=None,
                  class_conditioning=None, 
                  device="cuda",
                  path_to_save="gen.png"):
        """
        We assume single sample generation in inference!
        """

        if text_conditioning is not None:

            model = CLIPTextModel.from_pretrained(self.config.text_conditioning_hf_model).to(device)
            tokenizer = CLIPTokenizer.from_pretrained(self.config.text_conditioning_hf_model)

            text_conditioning = text_conditioning.strip()

            tokenized = tokenizer(text_conditioning,
                                  truncation=True, 
                                  max_length=77)
            
            input_ids = torch.tensor(tokenized.input_ids).to(device).unsqueeze(0)
            
            with torch.no_grad():
                text_conditioning = model(input_ids).last_hidden_state

        else:
            
            ### If conditioning is enabled but we dont pass a prompt then we do unconditional generation ###
            if self.config.text_conditioning:
                text_conditioning = self.text_encoder.null_token

        ### Compute Latent Dimension Shape ###
        latent_resolution = self.config.img_size // 2**(len(self.config.vae_channels_per_block)-1)

        ### Sample Noise ###
        image = torch.randn(1, self.config.latent_channels, latent_resolution, latent_resolution)
        
        ### Diffusion on the Latent Space ###
        for t in tqdm(torch.flip(torch.arange(self.config.num_diffusion_timesteps), dims=[0])):
            
            ### Create Timestep Index ###
            timesteps = torch.full((1, ), t).to(device)

            ### Get Timestep Embeddings ###
            timestep_embeddings = self.sinusoidal_time_embeddings(timesteps)
            
            ### Predict Noise ###
            noise_pred = self.unet(image.to(device), 
                                   timestep_embeddings, 
                                   text_conditioning=text_conditioning, 
                                   text_attention_mask=text_attention_mask,
                                   class_conditioning=class_conditioning)
            
            ### Denoise ###
            image = self.ddpm_sampler.remove_noise(image, t, noise_pred.detach().cpu())
        
        ### Decode Latent Back to Image Space ###
        images = self._vae_decode_images(image.to(device))

        if path_to_save is not None:
            save_generated_images(images,
                                path_to_save=path_to_save)    

        return images
