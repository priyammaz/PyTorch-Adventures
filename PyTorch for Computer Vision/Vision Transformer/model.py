import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
    Image to Patch Embeddings via Convolution
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 bias=True):
        
        """
        Args:
            img_size: Expected Image Shape (img_size x img_size)
            patch_size: Wanted size for each patch
            in_chans: Number of channels in image (3 for RGB)
            embed_dim: Transformer embedding dimension
        
        """
        super(PatchEmbed, self).__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size)**2
        
        self.proj = nn.Conv2d(in_channels=in_chans,
                              out_channels=embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size,
                              bias=bias)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x

class SelfAttentionEncoder(nn.Module):
  """
  Self Attention Proposed in `Attention is All  You Need` - https://arxiv.org/abs/1706.03762
  """

  def __init__(self,
               embed_dim=768,
               num_heads=12, 
               attn_p=0,
               proj_p=0,
               fused_attn=True):
    """
    
    Args:
        embed_dim: Transformer Embedding Dimension
        num_heads: Number of heads of computation for Attention 
        attn_p: Probability for Dropout2d on Attention cube
        proj_p: Probability for Dropout on final Projection
    """

    super(SelfAttentionEncoder, self).__init__()
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
  
class MLP(nn.Module):
    """
    Multi Layer Perceptron used in the Vision Transformer Architecture
    """
    def __init__(self, 
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 mlp_p=0):
        
        """
        Args:
            in_features: Transformer Embedding Dimension
            hidden_size: Embedding dimension * mlp_ratio
            out_features: Return back to Transformer Embedding Dimension
            act_layer: Wanted activation for MLP
            mlp_p: Dropout probability for MlP

        """

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(mlp_p)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class EncoderBlock(nn.Module):
    """
    Single Transformer Block consisting of Attention and MLP
    """
    def __init__(self,
                 fused_attention=True,
                 embed_dim=768,
                 num_heads=4, 
                 mlp_ratio=4,
                 proj_p=0,
                 attn_p=0,
                 mlp_p=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        
        """
            Args:
                fused_attention: Flash attention (only for vanilla attention and PyTorch > 2.0)
                embed_dim: Transformer Embedding Dimension
                num_heads: Number of heads of Attention computation
                mlp_ratio: Embedding dimension scaling for MLP
                proj_p: Probability for Dropout on final Projection
                attn_p: Probability for Dropout2d on Attention cube
                mlp_p: Probability of Dropout on MLP layers
                act_layer: Activation function for Attention computation
                norm_layer: Method of normalization

        """
        super(EncoderBlock, self).__init__()
        self.norm1 = norm_layer(embed_dim, eps=1e-6)

        self.attn = SelfAttentionEncoder(embed_dim=embed_dim,
                                         num_heads=num_heads, 
                                         attn_p=attn_p,
                                         proj_p=proj_p,
                                         fused_attn=fused_attention)
    
        self.norm2 = norm_layer(embed_dim, eps=1e-6)
        self.mlp = MLP(in_features=embed_dim,
                       hidden_features=int(embed_dim*mlp_ratio),
                       out_features=embed_dim,
                       act_layer=act_layer,
                       mlp_p=mlp_p)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    """
    Vision Transformer as implemented in `An Image is Worth 16x16 Words: Transformer for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16, 
            in_chans=3,
            num_classes=1000,
            fused_attention=True,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            attn_p=0.0,
            mlp_p=0.0, 
            proj_p=0.0,
            pos_p=0.0,
            head_p=0.0,
            pooling="cls",
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm):
        
        """
        Args:
            img_size: Expected Image Shape (img_size x img_size)
            patch_size: Wanted size for each patch
            in_chans: Number of channels in image (3 for RGB)
            num_classes: Number of output classes 
            attention: Toggle between "vanilla" or "triplet" attention
            fused_attention: Flash attention (only for vanilla attention and PyTorch > 2.0)
            embed_dim: Transformer embedding dimension
            depth: Number of Transformer Blocks
            num_heads: Number of heads of Attention Computation
            mlp_ratio: Embedding dimension scaling for MLP
            use_intersect: If True, will use seperate learning parameters, o.w. Q and K
            pooling: If None, will project on expanded V, otherwise 'max' or 'avg' pooling
            proj_p: Probability for Dropout on final Projection
            attn_p: Probability for Dropout2d on Attention cube
            mlp_p: Probability of Dropout on MLP layers
            head_p: Probability of Dropout on final head prediction layer
            pooling: 
                - "cls": Use a CLS token for sequence aggregation
                - "avg": Use average pooling for sequence aggregation
            act_layer: Activation function for Attention computation
            norm_layer: Method of normalization

        """
        super(VisionTransformer, self).__init__()
        self.pooling = pooling
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        
        assert pooling in ["cls", "avg"]
        if pooling == "cls":
            num_tokens = self.patch_embed.num_patches + 1
        elif pooling == "avg":
            num_tokens = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.randn(1,num_tokens,embed_dim))
        self.pos_drop = nn.Dropout(pos_p)
        
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(fused_attention=fused_attention,
                             embed_dim=embed_dim, 
                             num_heads=num_heads, 
                             mlp_ratio=mlp_ratio,
                             proj_p=proj_p,
                             attn_p=attn_p,
                             mlp_p=mlp_p,
                             act_layer=act_layer,
                             norm_layer=norm_layer)

                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim, eps=1e-6)
        self.head_drop = nn.Dropout(head_p)
        self.head = nn.Linear(embed_dim, num_classes)

        ### Initialize all weights ###
        self.apply(self._init_weights)

    def _cls_pos_embed(self, x):
        if self.pooling == "cls":
            x = torch.cat([self.cls_token.expand(x.shape[0],-1,-1), x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x
    
    def _init_weights(self, module: nn.Module):

        if isinstance(module, VisionTransformer):
            module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data, mean=0, std=0.02)
            module.pos_embed.data = nn.init.trunc_normal_(module.pos_embed.data, mean=0, std=0.02)

        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self._cls_pos_embed(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        if self.pooling == "cls":
            x = x[:,0]
        else:
            x = x.mean(dim=1)
            
        x = self.head_drop(x)
        x = self.head(x)

        return x