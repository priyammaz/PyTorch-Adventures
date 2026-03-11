import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple
from VisionTransformer import PatchEmbed, EncoderBlock
from transformers import ViTMAEModel
from UperNet import UperNetHead
from dataclasses import dataclass
from safetensors.torch import load_file
from utils import sincos_embeddings, random_masking, patchify

@dataclass
class MAEConfig:

    ### Input Image Config ###
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    ### MAE Encoder Config ###
    encoder_embed_dim: int = 768
    encoder_depth: int = 12
    encoder_num_heads: int = 12
    encoder_attn_p: float = 0.0
    encoder_mlp_p: float = 0.0
    encoder_proj_p: float = 0.0
    encoder_mlp_ratio: int = 4

    ### MAE Decoder Config ###
    decoder_embed_dim: int = 512
    decoder_depth: int =  8
    decoder_num_heads: int = 16
    decoder_attn_p: float = 0.0
    decoder_mlp_p: float = 0.0
    decoder_proj_p: float = 0.0
    decoder_mlp_ratio: int = 4

    ### Image Classification Config ###
    num_classes: int = 1000

    ### Segmentation Head Config ###
    psp_bin_size: Tuple = (1,2,4,6)
    feature_layer_idx: Tuple = (3,5,7,11)
    channels_per_layer: Tuple = (768, 768, 768, 768)
    rescales: Tuple = (4,1,2,0.5)
    num_segmentation_classes: Tuple = 2
    
    ### MAE Settings ###
    fused_attention: bool = True
    learnable_positional_encodings: bool = True
    custom_weight_init: bool = True

    ### Masking Config ###
    mask_ratio: float = 0.75

    ### PreTrained Weights Config ###
    hf_model_name: str = "facebook/vit-mae-base"
    pretrained_backbone: Literal["pretrained", "pretrained_huggingface", "random"] = "pretrained"
    path_to_pretrained_weights: str = None


class VITMAEEncoder(nn.Module):

    """
    MAE Encoder as described in "Masked AutoEncoders are Scalable Vision Learners" (https://arxiv.org/pdf/2111.06377)

    The MAE Encoder is nearly identical to the Vision Transformer (https://arxiv.org/abs/2010.11929) with the exception
    of the Random Masking Strategy. Unlike Masked Language Modeling seen in BERT/RoBERTa, where tokens are selected
    to be masked and replaced with a mask token, the MAE Encoder completely removes randomly selected masked tokens. When
    mask_ratio is set to 0, the VITMAEEncoder acts as a normal Vision Transformer. 
    """
    def __init__(self, config):
        
        super(VITMAEEncoder, self).__init__()

        self.config = config

        ### Define Encoder ###
        self.patch_embed = PatchEmbed(img_size=config.img_size, 
                                      patch_size=config.patch_size, 
                                      in_chans=config.in_channels, 
                                      embed_dim=config.encoder_embed_dim)
        
        ### CLS Token and SinCos Positional Embeddings ###
        self.enc_cls_token = nn.Parameter(torch.zeros(1,1,config.encoder_embed_dim))
        self.enc_pos_embed = sincos_embeddings(num_tokens=self.patch_embed.num_patches+1,
                                               embed_dim=config.encoder_embed_dim,  
                                               requires_grad=config.learnable_positional_encodings)
        
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embed_dim=config.encoder_embed_dim,
                             num_heads=config.encoder_num_heads, 
                             mlp_ratio=config.encoder_mlp_ratio, 
                             proj_p=config.encoder_proj_p, 
                             attn_p=config.encoder_attn_p, 
                             mlp_p=config.encoder_mlp_p, 
                             fused_attention=config.fused_attention)

                for _ in range(config.encoder_depth)
            ]
        )

        self.encoder_layer_norm = nn.LayerNorm(config.encoder_embed_dim, eps=1e-6)

    def forward(self, x, mask_ratio=0.0, output_hidden_states=False):
        
        batch_size, channels, height, width = x.shape 

        ### Patch Embedding ###
        x = self.patch_embed(x)
        
        ### Add Position Embedding without CLS token ###
        x = x + self.enc_pos_embed[:, 1:, :]
        
        ### Random Masking ###
        if mask_ratio > 0.0:

            x, mask, restore_idx = random_masking(x, mask_ratio=mask_ratio)

        else: 
            
            mask = None
            restore_idx = None

        ### Add Position Information and Concatenate on CLS Token ###
        cls_token  = self.enc_cls_token + self.enc_pos_embed[:, :1, :]

        ### Expand to Batch Dimension on CLS Token and Concat ###
        cls_token = cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        ### Pass through Transformer Blocks ###
        hidden_states = []
        for block in self.encoder_blocks:
            x = block(x)
            hidden_states.append(x)

        ### Normalize ###
        x = self.encoder_layer_norm(x)

        if output_hidden_states:
            return x, mask, restore_idx, hidden_states
        else:
            return x, mask, restore_idx

class VITMAEDecoder(nn.Module):

    """
    MAE Decoder used for Masked Image Pretraining. This is a lightweight decoder that does two things:
        
        1) Projecting the embeddings from the embed_dim of the encoder to the embed_dim of the decoder
        2) Place the masked output of the encoder back into its original positions, and place masked tokens
           in the positions that were selected for masking
    """
    def __init__(self, config, num_patches):
        super(VITMAEDecoder, self).__init__()

        self.config = config 
        self.num_patches = num_patches

        ### Linear Layer to Project from Encoder to Decoder Embed Dims ###
        self.encoder2decoder_embbedding_proj = nn.Linear(config.encoder_embed_dim, 
                                                         config.decoder_embed_dim)

        ### Mask Token PlaceHolder (Same as RoBERTa Implementation) ###
        self.mask_token = nn.Parameter(torch.zeros(1,1,config.decoder_embed_dim))
        
        ### Decoder SinCos Positional Embeddings ###
        self.dec_pos_embed = sincos_embeddings(num_tokens=num_patches,
                                               embed_dim=config.decoder_embed_dim,  
                                               requires_grad=config.learnable_positional_encodings)

        self.decoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embed_dim=config.decoder_embed_dim,
                             num_heads=config.decoder_num_heads, 
                             mlp_ratio=config.decoder_mlp_ratio, 
                             proj_p=config.decoder_proj_p, 
                             attn_p=config.decoder_attn_p, 
                             mlp_p=config.decoder_mlp_p, 
                             fused_attention=config.fused_attention)

                for _ in range(config.decoder_depth)
            ]
        )

        self.decoder_layer_norm = nn.LayerNorm(config.decoder_embed_dim, eps=1e-6)

    def forward(self, x, restore_idx, output_hidden_states=False):
        
        ### Project Encoder Embeddings to Decoder Embeddings ###
        x = self.encoder2decoder_embbedding_proj(x)

        ### Remove CLS Token from Encoder Output ###
        x = x[:, 1:, :]

        ### Track the number of selected idx (excluding the CLS Token now) ###
        batch_size, num_selected_idx, embed_dim = x.shape
        
        ### Expand Mask Token (repeating for the number of tokens that were masked) ###
        mask_token = self.mask_token.repeat(batch_size, self.num_patches - num_selected_idx, 1)

        ### Place Selected Tokens Back in Original Location and Fill Masked Locations with Mask Token ### 
        x = torch.cat([x, mask_token], dim=1)
        restore_idx_repeat = restore_idx.unsqueeze(-1).repeat(1,1,embed_dim)
        x = torch.gather(x, dim=1, index=restore_idx_repeat)

        ### Add Decoder Positional Embeddings ###
        x = x + self.dec_pos_embed

        ### Pass Through Transformer Blocks ###
        hidden_states = []
        for block in self.decoder_blocks:
            x = block(x)
            hidden_states.append(x)
        
        ### Normalize Output ###
        x = self.decoder_layer_norm(x)
        
        if output_hidden_states:
            return x, hidden_states
        else:
            return x

class ViTMAEForPreTraining(nn.Module):
    """
    The default Vision Transformer takes images of size 224 with patch size 16, providing us 
    196 images patches. With the default 75% masking strategy, 147 of the image patches are randomly selected and 
    removed, and only the remaining 49 image patches are passed to the encoder. 

    The encoded image patches are then passed to the MAE Decoder, where we place the selected 49 tokens back in their 
    original positions, a learnable mask token in the other 147 positions. These 196 tokens (again the 49 
    outputs from the Encoder and 147 mask tokens) are then passed to the decoder. The output of the decoder is 
    projected back to the orignal image space, and an MSE Loss is done between masked patches and the original image 
    patches. 

    KEY IDEA: The only reason this works is because, we add the positional embeddings to our image tokens BEFORE MASKING!!!
    Once the positional information is added, the order of the image tokens no longer matter (Transformers are Permutation Invariant)
    Therefore, when the encoder sees the 49 randomly selected image patches, it also sees the position of them (and therefore
    the relative position between them). 

    This means when we are done pretraining, we can pass in all 196 image tokens to our model (with their cooresponding 
    positional information), so we can finetune something like a classification model using the Encoder.
    """
    def __init__(self, config, mask_ratio=0.75):
        super(ViTMAEForPreTraining, self).__init__()

        self.config = config 
        self.mask_ratio = mask_ratio

        ### Define the Encoder/Decoder of the MAE ###
        self.encoder = VITMAEEncoder(config=config)
        self.decoder = VITMAEDecoder(config=config, 
                                     num_patches=self.encoder.patch_embed.num_patches)
        
        self.embed2image = nn.Linear(self.config.decoder_embed_dim, 
                                     self.config.in_channels * self.config.patch_size**2)
        
        if self.config.custom_weight_init:
            self.apply(_init_weights_)

    def forward(self, x, return_mask=False):
        
        ### Encoder and Decoder Images ###
        encoded, mask, restore_idx = self.encoder(x, mask_ratio=self.mask_ratio)
        decoded = self.decoder(encoded, restore_idx)

        ### Project Decoded Back to Pixel Space ###
        logits = self.embed2image(decoded)

        ### Patchify Original Images to Compute Loss on Masked Patches ###
        patched_target = patchify(x)

        ### Compute Loss ###
        loss = (logits - patched_target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        if not return_mask:
            return encoded, decoded, logits, loss
        else:
            return encoded, decoded, logits, loss, mask
            
class ViTMAEForDownstreamTasks(nn.Module):
    """
    Dummy class that holds load_backbone method to
    toggle between our own pretrained weights and 
    the huggingface pretrained weights
    """
    def __init__(self):
        super().__init__()

    def load_backbone(self, config):

        """
        Helper function to load the VITMAEEncoder, where we identify
        if we want to load our own pre-trained weights or if we want to use
        the huggingface weights! All of this is indicated in the config:

        config arguments:

            1) hf_model_name: Name of pretrained model from huggingface
            2) pretrained_backbone: Options include
                - pretrained: This will load our own pretrained weights
                - pretrained_huggingface: This will load the indicated huggingface backbone
                - random: Loads a randomly initialized backbone
            3) path_to_pretrained_weights: if pretrained_backbone is "pretrained"
                                        then we also provide path to weights
        """

        if config.pretrained_backbone == "pretrained_huggingface":
            print("Loading Huggingface MAE Backbone:", config.hf_model_name)
            backbone = ViTMAEModel.from_pretrained(config.hf_model_name)
            
            ### No Need for Random Masking when Finetuning, so set mask_ratio to 0 ###
            backbone.config.mask_ratio= 0.0

        else:
            backbone = VITMAEEncoder(config)

            if config.pretrained_backbone == "pretrained":
                if config.path_to_pretrained_weights is None:
                    raise Exception("Provide the argument `path_to_pretrained_weights` in the config, else we cant load them!")
                else:
                    if not os.path.isfile(config.path_to_pretrained_weights):
                        raise Exception(f"Provided path to safetensors weights {self.config.path_to_pretrained_weights} is invalid!")
                    
                    print(f"Loading Masked Autoencoder Backbone from:", config.path_to_pretrained_weights)

                    ### Load Weights with load_file from safetensors ###
                    state_dict = load_file(config.path_to_pretrained_weights)
                    
                    ### Cleanup Weights we dont need (Remove Backbone) ###
                    backbone_keys = {}
                    for key in state_dict.keys():
                
                        if ("decoder" in key) | ("embed2image" in key):
                            continue

                        else:

                            new_key = key.replace("encoder.", "") if key.startswith("encoder.") else key
                            backbone_keys[new_key] = state_dict[key]
                            
                    backbone.load_state_dict(backbone_keys)
                        
        return backbone


class ViTMAEForImageClassification(ViTMAEForDownstreamTasks):

    def __init__(self, config):

        super(ViTMAEForImageClassification, self).__init__()

        self.config = config

        self.encoder = self.load_backbone(config)

        self.hf_backbone = False
        if config.pretrained_backbone == "pretrained_huggingface":
            self.hf_backbone = True

        self.head = nn.Linear(self.encoder.config.hidden_size if self.hf_backbone else config.encoder_embed_dim,
                              config.num_classes)
        
    def forward(self, x):

        if self.hf_backbone:
            output = self.encoder(x)["last_hidden_state"]
        else:
            output, _, _ = self.encoder(x)

        ### Index out CLS Token ###
        cls_token = output[:, 0]

        ### Compute Final Output ###
        logits = self.head(cls_token)

        return logits

        
class ViTMAEForSegmentation(ViTMAEForDownstreamTasks):
    def __init__(self, config):

        super(ViTMAEForSegmentation, self).__init__()

        self.config = config

        self.encoder = self.load_backbone(config)

        self.hf_backbone = False
        if config.pretrained_backbone == "pretrained_huggingface":
            self.hf_backbone = True

        self.upernet = UperNetHead(config)

    def forward(self, x):

        input_size = (x.shape[2], x.shape[3])

        if self.hf_backbone:
            hidden_states = self.encoder(x, output_hidden_states=True)["hidden_states"]
        else:
            _, _, _, hidden_states = self.encoder(x, output_hidden_states=True)

        selected_hidden_states = []
        for idx, state in enumerate(hidden_states):
            if idx in self.config.feature_layer_idx:
                
                ### Remove CLS token ###
                state = state[:, 1:, :]

                ### Get Edge Length ###
                batch, num_tokens, embed_dim = state.shape
                edge_length = int(num_tokens**0.5)

                ### Convert State from (B x S x E) -> (B x E x H x W) ###
                ### where H=W=S**0.5 ###
                state = state.permute(0,2,1)
                state = state.permute(0,2,1).reshape(-1, embed_dim, edge_length, edge_length)

                ### Store the Tensor ###
                selected_hidden_states.append(state)

        ### Pass Through UperNet Head ###
        output = self.upernet(selected_hidden_states)
        
        ### Interpolate Back to Image Space ###
        output = F.interpolate(output, size=input_size, mode="bilinear")

        return output

        
def _init_weights_(module: nn.Module):

    if isinstance(module, VITMAEEncoder):
        module.enc_cls_token.data = nn.init.normal_(module.enc_cls_token.data, mean=0, std=0.02)
    if isinstance(module, VITMAEDecoder):
        module.mask_token.data = nn.init.normal_(module.mask_token.data, mean=0, std=0.02)
    elif isinstance(module, PatchEmbed):
        torch.nn.init.xavier_uniform_(module.proj.weight.data.flatten(1))
    elif isinstance(module, nn.Linear):
        module.weight.data = nn.init.normal_(module.weight.data, mean=0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

if __name__ == "__main__":

    rand = torch.randn(4,3,224,224)
    mae_config = MAEConfig(pretrained_backbone="pretrained_huggingface", 
                           path_to_pretrained_weights="work_dir/MAE Pretraining/checkpoint_50/model.safetensors")
    model = ViTMAEForSegmentation(mae_config)
    rand = torch.randn(4,3,224,224)
    model(rand)
    