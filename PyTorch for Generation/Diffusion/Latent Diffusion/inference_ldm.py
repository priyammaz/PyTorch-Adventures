"""
Script to generate Images from VAE
"""

import yaml
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from modules import LDM, LDMConfig
import argparse
from PIL import Image
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import CLIPTokenizer

parser = argparse.ArgumentParser("VAE Inference Script")

parser.add_argument("--model_config",
                    help="Path to config file for all model information",
                    required=True, 
                    type=str)
parser.add_argument("--path_to_weights",
                    help="Path to model.safetensors",
                    required=True, 
                    type=str)
parser.add_argument("--text_conditional",
                    action="store_true", 
                    help="Is this a text conditional model?",
                    default=False)
parser.add_argument("--training_dataset",
                    help="What dataset was this trained on?",
                    required=True,
                    choices=("conceptual_captions", "celebahq"))
parser.add_argument("--prompt",
                    help="What prompt do you want to generate from?",
                    type=str, 
                    default=None)
parser.add_argument("--text_encoding_model", 
                    default="openai/clip-vit-large-patch14", 
                    type=str)

args = parser.parse_args()

with open(args.model_config, "r") as f:
    ldm_config = yaml.safe_load(f)

config = LDMConfig(**ldm_config["vae"], **ldm_config["unet"])
scaling_constants = ldm_config["scaling_constants"]
device = "cuda" if torch.cuda.is_available() else "cpu"
config.text_conditioning = args.text_conditional
config.vae_scale_factor = scaling_constants[args.training_dataset]

### Load Model ###
model = LDM(config).to(device)
model.eval()

### Load Weights ###
state_dict = load_file(args.path_to_weights)
model.load_state_dict(state_dict)

### Inference ###
gen = model.inference(args.prompt)                   

### Rescale ###
gen = (gen + 1) / 2
gen = gen.clip(0,1)

plt.imshow(gen.squeeze().cpu().permute(1,2,0).numpy())
plt.tight_layout()
plt.show()
