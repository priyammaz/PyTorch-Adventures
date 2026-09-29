"""
Minimal copy of the MAE encoder, used to compare against I-JEPA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def sincos_embeddings(num_tokens, embed_dim, requires_grad=False):
    ### Create Tensors for Position and Embedding Idx ###
    encoding = torch.zeros(num_tokens, embed_dim, dtype=torch.float)
    position_idx = torch.arange(0, num_tokens, dtype=torch.float).reshape(-1, 1)
    embed_dim_skip_idx = torch.arange(0, embed_dim, step=2, dtype=torch.float)

    ### Attention is All You Need Pos Embed Formula ###
    encoding[:, 0::2] = torch.sin(position_idx / (10000 ** (embed_dim_skip_idx / embed_dim)))
    encoding[:, 1::2] = torch.cos(position_idx / (10000 ** (embed_dim_skip_idx / embed_dim)))

    return nn.Parameter(encoding.unsqueeze(0), requires_grad=requires_grad)


class PatchEmbed(nn.Module):
    """Image to Patch Embeddings via Convolution"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super(PatchEmbed, self).__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class SelfAttentionEncoder(nn.Module):
    """Self Attention Proposed in `Attention is All You Need` https://arxiv.org/abs/1706.03762"""
    def __init__(self, embed_dim=768, num_heads=12, attn_p=0., proj_p=0., fused_attn=True):
        super(SelfAttentionEncoder, self).__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_p = attn_p
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p if self.training else 0.)
        else:
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            x = self.attn_drop(attn) @ v

        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    """Multi Layer Perceptron used in the Vision Transformer Architecture"""
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, mlp_p=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(mlp_p)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class EncoderBlock(nn.Module):
    """Single Transformer Block consisting of Attention and MLP"""
    def __init__(self, fused_attention=True, embed_dim=768, num_heads=12, mlp_ratio=4,
                 proj_p=0., attn_p=0., mlp_p=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(EncoderBlock, self).__init__()
        self.norm1 = norm_layer(embed_dim, eps=1e-6)
        self.attn = SelfAttentionEncoder(embed_dim, num_heads, attn_p, proj_p, fused_attention)
        self.norm2 = norm_layer(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, act_layer, mlp_p)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VITMAEEncoder(nn.Module):

    """
    MAE Encoder from "Masked AutoEncoders are Scalable Vision Learners" (https://arxiv.org/pdf/2111.06377)

    During pretraining 75% of the patches are dropped and only the visible 25% are encoded. At
    evaluation time we set mask_ratio=0, at which point this is just a normal Vision Transformer
    that sees all 196 patches plus the CLS token.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4, fused_attention=True, learnable_positional_encodings=True):

        super(VITMAEEncoder, self).__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_channels, embed_dim=embed_dim)

        ### CLS Token and SinCos Positional Embeddings (197 = 196 patches + CLS) ###
        self.enc_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.enc_pos_embed = sincos_embeddings(num_tokens=self.patch_embed.num_patches + 1,
                                               embed_dim=embed_dim,
                                               requires_grad=learnable_positional_encodings)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         fused_attention=fused_attention)
            for _ in range(depth)
        ])

        self.encoder_layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x, output_hidden_states=False):

        batch_size = x.shape[0]

        ### Patch Embedding + Position Embedding (skipping the CLS slot) ###
        x = self.patch_embed(x)
        x = x + self.enc_pos_embed[:, 1:, :]

        ### Concatenate the CLS token with its own positional embedding ###
        cls_token = (self.enc_cls_token + self.enc_pos_embed[:, :1, :]).expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        hidden_states = []
        for block in self.encoder_blocks:
            x = block(x)
            hidden_states.append(x)

        x = self.encoder_layer_norm(x)

        if output_hidden_states:
            return x, hidden_states
        return x


class MAEPatchEncoder(nn.Module):
    """
    Thin wrapper that makes a pretrained MAE encoder behave like our I-JEPA ContextEncoder:
    call it on a batch of images and get back (B, 196, 768) patch tokens.

    The CLS token is dropped so both models are evaluated the exact same way, by mean pooling
    the patch tokens. Pass cls_token=True if you would rather read the CLS embedding, which is
    what MAE itself uses for classification.
    """
    def __init__(self, encoder, cls_token=False):
        super(MAEPatchEncoder, self).__init__()
        self.encoder = encoder
        self.cls_token = cls_token

    def forward(self, x):
        tokens = self.encoder(x)                      # (B, 197, 768)
        if self.cls_token:
            return tokens[:, :1]                      # (B, 1, 768), mean pooling is then a no-op
        return tokens[:, 1:]                          # (B, 196, 768), drop CLS

    def forward_layers(self, x):
        """Patch tokens after every transformer block, for layerwise probing"""
        _, hidden_states = self.encoder(x, output_hidden_states=True)
        return [h[:, 1:] for h in hidden_states]


def load_pretrained_mae_encoder(path_to_checkpoint, device="cpu", cls_token=False, **encoder_kwargs):
    """
    Load the encoder out of an MAE pretraining checkpoint.
    """
    encoder = VITMAEEncoder(**encoder_kwargs)
    state_dict = load_file(path_to_checkpoint)
    weights = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}

    if len(weights) == 0:
        raise ValueError(f"No 'encoder.' weights found in {path_to_checkpoint}, "
                         f"found prefixes: {sorted({k.split('.')[0] for k in state_dict})}")

    encoder.load_state_dict(weights, strict=True)
    return MAEPatchEncoder(encoder, cls_token=cls_token).to(device).eval()
