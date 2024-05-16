import torch
import torch.nn as nn

class SelfAttentionEncoder(nn.Module):
  def __init__(self,
               embed_dim=768,
               num_heads=12, 
               attn_p=0,
               proj_p=0):

    """
    Args:

        embed_dim: What is the embedding dimension of each embedding vector
        num_heads: How many heads of attention do we want?
        attn_p: Dropout probability on Attention
        proj_p: Dropout probability on projection matrix
    """
    super(SelfAttentionEncoder, self).__init__()
    assert embed_dim % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = int(embed_dim / num_heads)
    self.scale = self.head_dim ** -0.5

    self.qkv = nn.Linear(embed_dim, embed_dim*3)
    self.attn_p = attn_p
    self.attn_drop = nn.Dropout(attn_p)
    self.proj = nn.Linear(embed_dim, embed_dim)
    self.proj_drop = nn.Dropout(proj_p)

  def forward(self, x, attention_mask=None):
    batch_size, seq_len, embed_dim = x.shape
    qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2,0,3,1,4)
    q,k,v = qkv.unbind(0)

    # B x H x S x S
    attn = (q @ k.transpose(-2,-1)) * self.scale
    if attention_mask is not None:
        ### We Need to Unsqueeze Attention Mask to have placeholder dimensions for Num Heads and Seq Len ###

        # B, S -> B, 1, 1, S
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        attn = attn.masked_fill(attention_mask, float('-inf'))

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1,2).reshape(batch_size, seq_len, embed_dim)
    x = self.proj(x)
    x = self.proj_drop(x)
      
    return x
  
class MLP(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 mlp_p=0):


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
    
class Block(nn.Module):
    def __init__(self, 
                 embed_dim=768, 
                 num_heads=12, 
                 mlp_ratio=4, 
                 proj_p=0., 
                 attn_p=0., 
                 mlp_p=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.norm1 = norm_layer(embed_dim, eps=1e-6)
        self.attn = SelfAttentionEncoder(embed_dim=embed_dim,
                                         num_heads=num_heads, 
                                         attn_p=attn_p,
                                         proj_p=proj_p)


        self.norm2 = norm_layer(embed_dim, eps=1e-6)
        self.mlp = MLP(in_features=embed_dim,
                       hidden_features=int(embed_dim*mlp_ratio),
                       out_features=embed_dim,
                       act_layer=act_layer,
                       mlp_p=mlp_p)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x
    
class RoBERTa(nn.Module):
    def __init__(self, 
                 max_seq_len=512,
                 vocab_size=tokenizer.vocab_size,
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4, 
                 attn_p=0., 
                 mlp_p=0., 
                 proj_p=0., 
                 pos_p=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):

        super(RoBERTa, self).__init__()

        self.max_seq_len = max_seq_len
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Embedding(max_seq_len+1, embed_dim)
        self.pos_drop = nn.Dropout(pos_p)

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim=embed_dim, 
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

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)


    def forward(self, x, attention_mask):
        device = x.device

        batch_size, seq_len = x.shape

        ### If we have too long of a sequence length, grab the last chunk ###
        if seq_len > self.max_seq_len:
            x = x[:, -self.max_seq_len:]

        ### We only need the positional information upto the length of data we have plus CLS token ###
        avail_idx = torch.arange(0, seq_len+1, dtype=torch.long, device=device)

        ### Embed all the Tokens ###
        tok_emb = self.embeddings(x)

        ### Concatenate on the CLSL Token ###
        cls_token = self.cls_token.expand(batch_size, -1, -1) # (batch, 1, embed_dim)
        tok_emb = torch.cat((cls_token, tok_emb), dim=1) 

        ### Add positional information ###
        pos_emb = self.pos_embed(avail_idx)
        x = tok_emb + pos_emb
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        ### Slice Off CLS token ###
        cls_token_final = x[:, 0] 

        ### Slice off Remaining Tokens ###
        x = x[:, 1:]

        ### MLM Prediction Head ###
        x = self.head(x)
        
        return x
