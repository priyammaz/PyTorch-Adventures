"""
Script to generate Images from VAE
"""

import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from modules import VAE, LDMConfig
import argparse
from PIL import Image
from safetensors.torch import load_file
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("VAE Inference Script")

parser.add_argument("--model_config",
                    help="Path to config file for all model information",
                    required=True, 
                    type=str)
parser.add_argument("--path_to_vae_backbone",
                    help="Path to model.safetensors",
                    required=True, 
                    type=str)
parser.add_argument("--path_to_image",
                    help="Path to an image file to pass through VAE",
                    required=True, 
                    type=str)

args = parser.parse_args()

with open(args.model_config, "r") as f:
    vae_config = yaml.safe_load(f)["vae"]
    config = LDMConfig(**vae_config)

device = "cuda" if torch.cuda.is_available() else "cpu"

### Load Model ###
model = VAE(config).to(device)
model.eval()

### Load Weights ###
state_dict = load_file(args.path_to_vae_backbone)
model.load_state_dict(state_dict)

### Load Image ###
image = Image.open(args.path_to_image).convert('RGB').resize((config.img_size, config.img_size))
image = transforms.ToTensor()(image) 
image = (image - 0.5) / 0.5 # Scale between -1 and 1 
image = image.to(device).unsqueeze(0)

### Inference ###
with torch.no_grad():
    reconstruction = model(image)["reconstruction"]

### Rescale ###
image = (image + 1) / 2
reconstruction = (reconstruction + 1) / 2

image = image.clip(0,1)
reconstruction = reconstruction.clip(0,1)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(image.squeeze().cpu().permute(1,2,0).numpy())
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(reconstruction.squeeze().cpu().permute(1,2,0).numpy())
axes[1].set_title("Reconstruction")
axes[1].axis("off")

plt.tight_layout()
plt.show()
