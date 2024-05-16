import torch 
import torch.nn as nn

def CausalMasking(seq_len):
    ones = torch.ones((seq_len, seq_len))
    causal_mask = torch.tril(ones)
    causal_mask = causal_mask.reshape(1,1,seq_len,seq_len).bool()
    return causal_mask

class SelfAttentionDecoder(nn.Module):
  def __init__(self,
               seq_len=300,
               embed_dim=768,
               num_heads=12, 
               attn_p=0,
               proj_p=0):

    """
    Args:

        seq_len: What is the expected sequence length to our model?
        embed_dim: What is the embedding dimension of each embedding vector
        num_heads: How many heads of attention do we want?
        attn_p: Dropout probability on Attention
        proj_p: Dropout probability on projection matrix
    """
    super(SelfAttentionDecoder, self).__init__()
    assert embed_dim % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = int(embed_dim / num_heads)
    self.scale = self.head_dim ** -0.5

    self.qkv = nn.Linear(embed_dim, embed_dim*3)
    self.attn_p = attn_p
    self.attn_drop = nn.Dropout(attn_p)
    self.proj = nn.Linear(embed_dim, embed_dim)
    self.proj_drop = nn.Dropout(proj_p)


    ### Define the Causal Mask ###
    self.register_buffer("causal_mask", CausalMasking(seq_len=seq_len).to(torch.bool))

  def forward(self, x):
    batch_size, seq_len, embed_dim = x.shape
    qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2,0,3,1,4)
    q,k,v = qkv.unbind(0)

    attn = (q @ k.transpose(-2,-1)) * self.scale

    attn = attn.masked_fill(self.causal_mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))

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
                 seq_len=300, 
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
        self.attn = SelfAttentionDecoder(seq_len=seq_len,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads, 
                                         attn_p=attn_p,
                                         proj_p=proj_p)


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
    
class GPT(nn.Module):
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

        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.pos_drop = nn.Dropout(pos_p)

        self.blocks = nn.ModuleList(
            [
                Block(seq_len=max_seq_len, 
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

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        ### Weight Sharing ###
        self.embeddings.weight = self.head.weight

        ## Weight Init ###
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, std=0.02, a=-2, b=2)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, x):
        device = x.device

        batch_size, seq_len = x.shape

        ### We only need the positional information upto the length of data we have ###
        avail_idx = torch.arange(0, seq_len, dtype=torch.long, device=device)

        tok_emb = self.embeddings(x)
        pos_emb = self.pos_embed(avail_idx)

        x = tok_emb + pos_emb
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        return x
        
    @torch.no_grad()
    def write(self, input_tokens, max_new_tokens, temperature=1.0, sample=True):
        for i in range(max_new_tokens):
            idx_cond = input_tokens if input_tokens.shape[1] < self.max_seq_len else input_tokens[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            if sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1).unsqueeze(0)
            input_tokens = torch.cat([input_tokens, idx_next], dim=-1)
        return input_tokens.detach().cpu().numpy()