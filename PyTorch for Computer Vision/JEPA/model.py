import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

 
### SAME MODULES FROM VIT IMPLEMENTATION ###
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
 
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)
 
class SelfAttentionEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_p=0., proj_p=0., fused_attn=True):
        super().__init__()
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
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p if self.training else 0.)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))
 
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, mlp_p=0.):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(mlp_p)
 
    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
 
class EncoderBlock(nn.Module):
    def __init__(self, fused_attention=True, embed_dim=768, num_heads=12, mlp_ratio=4,
                 proj_p=0., attn_p=0., mlp_p=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim, eps=1e-6)
        self.attn  = SelfAttentionEncoder(embed_dim, num_heads, attn_p, proj_p, fused_attention)
        self.norm2 = norm_layer(embed_dim, eps=1e-6)
        self.mlp   = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, act_layer, mlp_p)
 
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
### SIN/COS POSITIONAL EMBEDDINGS ###
def sincos_1d(embed_dim, pos):
    
    assert embed_dim % 2 == 0 # make sure divisible by 2

    freqs = torch.arange(embed_dim // 2, dtype=torch.float32)
    freqs =  1.0 / (10000 ** (freqs / (embed_dim / 2)))

    # outer product (M,) x (D/2) -> (M,D/2)
    out = pos[:, None] * freqs[None, :]

    return torch.cat([torch.sin(out), torch.cos(out)], dim=1) 

def sincos_2d(embed_dim, grid_size):
    """
    2d sincos embeds that encode 2d space rather than just
    1d, so each position is encoded by position along the height
    and the position along the width
    """

    assert embed_dim % 2 == 0
    
    ### get 2d meshgrid, every location is the index i,j in the grid
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')  # (G, G)
    
    ### flatten it
    pos_h = grid_h.reshape(-1)  # (G*G,)    
    pos_w = grid_w.reshape(-1)  # (G*G,)

    ### Encode with half the embed dim in each grid direction
    emb_h = sincos_1d(embed_dim // 2, pos_h)  # (G*G, D/2)
    emb_w = sincos_1d(embed_dim // 2, pos_w)  # (G*G, D/2)

    ### concat H,W embeds
    return torch.cat([emb_h, emb_w], dim=1).unsqueeze(0) # (1, G*G, D)

### JEPA MODEL ###
class ContextEncoder(nn.Module):
    """
    Encodes a subset of image patches (context region).
    Position embeddings are indexed per-patch so we can pass
    an arbitrary subset of the full patch grid.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        attn_p=0.,
        proj_p=0.,
        mlp_p=0.,
        fused_attention=True,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        grid_size = img_size // patch_size

        # learnable positional embeds
        self.pos_embed = nn.Parameter(
            sincos_2d(embed_dim, grid_size),
            requires_grad=False,
        )

        self.blocks = nn.ModuleList([
            EncoderBlock(fused_attention, embed_dim, num_heads, mlp_ratio,
                         proj_p, attn_p, mlp_p, act_layer, norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim, eps=1e-6)
 
        self.apply(self._init_weights)
        self._fix_init_weight()
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _fix_init_weight(self):
        """
        Rescale attn output proj and MLP fc2 by 1/sqrt(2*layer_id).
        Prevents representation norm from growing through residual additions.
        Matches original I-JEPA / GPT-2 init.
        """
        for layer_id, block in enumerate(self.blocks):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward(self, x, context_ids=None):
        """
        x: (B, 3, H, W) tensor
        context_ids: (B, Nc) if provided, they are the subset of patches
                     the context encoder is allowed to see, otherwise
                     everything is used
        """

        B = x.shape[0]
        tokens = self.patch_embed(x)
        pos = self.pos_embed.expand(B, -1, -1)

        if context_ids is None:
            tokens = tokens + pos
        else:
            idx = context_ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]) # B, Nc, E
            tokens = torch.gather(tokens, 1, idx) + torch.gather(pos, 1, idx)
        
        for block in self.blocks:
            tokens = block(tokens)
        
        return self.norm(tokens)

class Predictor(nn.Module):
    """
    Takes context encoder output and a set of target position
    embeddings, then predicts the representation the target
    encoder would produce at those positions.
    """
    def __init__(
        self,
        num_patches,
        encoder_dim=768,
        predictor_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.,
        attn_p=0.,
        proj_p=0.,
        mlp_p=0.,
        fused_attention=True,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        
        super().__init__()

        # Project encoder tokens into the (smaller) predictor space
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)
 
        # Mask token for target positions (learned)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        grid_size = int(num_patches ** 0.5)
        self.pos_embed = nn.Parameter(
            sincos_2d(predictor_dim, grid_size),
            requires_grad=False,
        )
 
        self.blocks = nn.ModuleList([
            EncoderBlock(fused_attention, predictor_dim, num_heads, mlp_ratio,
                         proj_p, attn_p, mlp_p, act_layer, norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(predictor_dim, eps=1e-6)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)
 
        self.apply(self._init_weights)
        self._fix_init_weight()
 
    def _fix_init_weight(self):
        """Same layer-wise rescaling as encoder — original applies this to predictor blocks too."""
        for layer_id, block in enumerate(self.blocks):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
 
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(
        self, 
        context_tokens, 
        context_ids, 
        target_ids
    ):
        """
        context_tokens: (B, Nc, D) output ctx of encoder
        context_ids: (B, Nc) which patches are context
        target_idx: (B, Nt) which patches are targets
        """

        B = context_tokens.shape[0]
        D = self.pos_embed.shape[-1]

        # project from context dim to predictor dim 
        ctx = self.input_proj(context_tokens) # (B, Nc, predictor_dim)

        # add pos embeds to context tokens 
        ctx_pos_idx = context_ids.unsqueeze(-1).expand(-1, -1, D)
        ctx = ctx + torch.gather(self.pos_embed.expand(B, -1, -1), 1, ctx_pos_idx)

        # build target query tokens: mask_token + pos embeds
        tgt_pos_idx = target_ids.unsqueeze(-1).expand(-1,-1,D)
        tgt_pos = torch.gather(self.pos_embed.expand(B, -1, -1), 1, tgt_pos_idx)
        tgt_tokens  = self.mask_token.expand(B, target_ids.shape[1], -1) + tgt_pos

        # concat and run through predicto
        tokens = torch.cat([ctx, tgt_tokens], dim=1) # (B, Nc+Nt, predictor_dim)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # extract only the targets tokens and project back to encoder dim
        Nc = context_ids.shape[1]
        out = tokens[:, Nc:]
        
        return self.output_proj(out)

class IJEPA(nn.Module):
    """
    Image-based Joint Embedding Predictive Architecture (I-JEPA)

    context_encoder trained by gradient descent
    target_encoder EMA copy of context_encoder (no gradients)
    predictor narrow ViT, trained by gradient descent
 
    Forward pass returns the predictor output and target encoder output
    for every target block so the caller can compute the loss.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        # Encoder config
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        # Predictor config (kept narrower than encoder)
        predictor_embed_dim=384,
        predictor_depth=6,
        predictor_num_heads=12,
        # Shared
        mlp_ratio=4.,
        attn_p=0.,
        proj_p=0.,
        mlp_p=0.,
        fused_attention=True,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        # EMA momentum: target_params = m*target + (1-m)*context
        ema_momentum=0.996,
    ):
        super().__init__()
        self.ema_momentum = ema_momentum
        num_patches = (img_size // patch_size) ** 2

        self.context_encoder = ContextEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=encoder_embed_dim, depth=encoder_depth,
            num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
            attn_p=attn_p, proj_p=proj_p, mlp_p=mlp_p,
            fused_attention=fused_attention, act_layer=act_layer,
            norm_layer=norm_layer
        )

        # Create a copy and turn of grads
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # predictor network
        self.predictor = Predictor(
            num_patches=num_patches,
            encoder_dim=encoder_embed_dim,
            predictor_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
            attn_p=attn_p, proj_p=proj_p, mlp_p=mlp_p,
            fused_attention=fused_attention, act_layer=act_layer,
            norm_layer=norm_layer
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None):
        m = momentum if momentum is not None else self.ema_momentum
        for ctx_p, tgt_p in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            tgt_p.data.mul_(m).add_(ctx_p.data, alpha=1.0 - m)

    def forward(
        self, 
        images, 
        context_ids, 
        target_ids
    ):
        
        if context_ids is None:
            return self.context_encoder(images, context_ids=None)
        
        ctx_tokens = self.context_encoder(images, context_ids) 

        ### EMA PATH ###
        with torch.no_grad():   
            # Encode ALL patches with target encoder, then gather target locations
            all_tokens  = self.target_encoder.patch_embed(images) # (B, N, D)
            pos = self.target_encoder.pos_embed.expand(images.shape[0], -1, -1)
            all_tokens = all_tokens + pos
 
            for block in self.target_encoder.blocks:
                all_tokens = block(all_tokens)
            all_tokens = self.target_encoder.norm(all_tokens) # (B, N, D)
 
            # Gather only the target patch representations
            idx = target_ids.unsqueeze(-1).expand(-1, -1, all_tokens.shape[-1])
            tgt_repr = torch.gather(all_tokens, 1, idx) # (B, Nt, D)

        ### Predictor Network 
        pred_repr = self.predictor(ctx_tokens, context_ids, target_ids)

        return F.smooth_l1_loss(pred_repr, tgt_repr.detach())

 