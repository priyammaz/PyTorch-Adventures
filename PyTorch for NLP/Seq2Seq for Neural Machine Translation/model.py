import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:

    embedding_dimension: int = 512
    num_attention_heads: int = 8
    attention_dropout_p: float = 0.0
    hidden_dropout_p: float = 0.0
    mlp_ratio: int = 4
    encoder_depth: int = 6
    decoder_depth: int = 6

    src_vocab_size: int = 30522
    tgt_vocab_size: int = 32000

    max_src_len: int = 512
    max_tgt_len: int = 512
    learn_pos_embed: bool = False

class PositionalEncoding(nn.Module):

    """
    Sin/Cosine (non-learnable) encodings proposed in Attention is All You Need

    Args:
        max_len: Maximum number of tokens possible in a sequence
        embed_dim: Embedding dimension of each token
    """

    def __init__(self, max_len, embed_dim, requires_grad=False):
        super(PositionalEncoding, self).__init__()

        self.max_len = max_len
        self.embed_dim = embed_dim
        self.requires_grad = requires_grad

        self.encodings = self._build_positional_encodings()

    def _build_positional_encodings(self):

        encoding = torch.zeros(self.max_len, self.embed_dim, dtype=torch.float)
        postion_idx = torch.arange(0, self.max_len, dtype=torch.float).reshape(-1,1)
        embed_dim_skip_idx = torch.arange(0, self.embed_dim, step=2, dtype=torch.float)
        
        encoding[:, 0::2] = torch.sin(postion_idx / (10000 ** (embed_dim_skip_idx / self.embed_dim)))
        encoding[:, 1::2] = torch.cos(postion_idx / (10000 ** (embed_dim_skip_idx / self.embed_dim)))

        encoding = nn.Parameter(encoding, requires_grad=self.requires_grad)

        return encoding

    def forward(self, x):

        ### Grab Shape of Tensor ###
        seq_len = x.shape[1]

        ### Clip Encodings to Length Needed ###
        encodings = self.encodings[:seq_len]

        ### Add Positional Embeddings to Data and Return ###
        x = x + encodings

        return x

class Embeddings(nn.Module):

    """
    All the embeddings we need for the source and target langauge. Both source and target need:

    - Token Embeddings
    - Positional Embedings
    """
    
    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.src_embeddings = nn.Embedding(config.src_vocab_size, config.embedding_dimension)
        self.tgt_embeddings = nn.Embedding(config.tgt_vocab_size, config.embedding_dimension)
        
        self.src_positional_encodings = PositionalEncoding(config.max_src_len, 
                                                           config.embedding_dimension, 
                                                           config.learn_pos_embed)
        self.tgt_positional_encodings = PositionalEncoding(config.max_tgt_len, 
                                                           config.embedding_dimension, 
                                                           config.learn_pos_embed)
        
    def forward_src(self, input_ids):
        embeddings = self.src_embeddings(input_ids)
        embeddings = self.src_positional_encodings(embeddings)
        return embeddings
    
    def forward_tgt(self, input_ids):
        embeddings = self.tgt_embeddings(input_ids)
        embeddings = self.tgt_positional_encodings(embeddings)
        return embeddings
        
class Attention(nn.Module):
    """
    Regular Self-Attention but in this case we utilize flash_attention
    incorporated in the F.scaled_dot_product_attention to speed up our training. 
    """
    def __init__(self, config):
        super(Attention, self).__init__()
        
        ### Store Config ###
        self.config = config
        
        ### Sanity Checks ###
        assert config.embedding_dimension % config.num_attention_heads == 0, "Double check embedding dim divisible by number of heads"

        ### Attention Head Dim ###
        self.head_dim = config.embedding_dimension // config.num_attention_heads

        ### Attention Projections ###
        self.q_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.k_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.v_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        

    def forward(self, 
                src, 
                tgt=None, 
                attention_mask=None, 
                causal=False):
        
        """
        This forward function handles all the cases we need. Lets pretend we are doing English to French
        
            - We can provide English as the src along with its padding mask for Encoder self-attention
            - We can provide French as the src along with its padding mask and causal as True for decoder self-attention
            - We can provide English as src and French as tgt along with the src padding_mask for cross attention

        ### ATTENTION MASK FOR SELF-ATTENTION ###

        Attention Mask is in (Batch x Sequence Length) where we have False for tokens we don't want to attend to (from F.SDPA in PyTorch) ###
        F.scaled_dot_product_attention expects a mask of the shape (Batch x ..., x Seq_len x Seq_len) ###
        the "..." in this case is any extra dimensions (such as heads of attention). lets expand our mask to (Batch x 1 x Seq_len x Seq_len) ###
        The 1 in this case refers to the number of heads of attention we want, so it is a dummy index to broadcast over ###
        In each (Seq_len x Seq_len) matrix for every batch, we want False for all columns corresponding to padding tokens ### 

        ### ATTENTION MASK FOR CROSS-ATTENTION ###

        When doing cross attention, our French will be (Batch x french_len x embed_dim) and our English will be (Batch x english_len x embed_dim)
        In typical cross attention fashion, the queries will be the thing we want and Keys/Values will be the thing we are crossing with. In our 
        Decoder Cross Attention, we want to learn how our generated French is related to the encoded english from the Encoder. So our Queries will be
        French and Keys/Values will be the encoded English. 

        Q @ K^T will then give a shape (Batch x ... x french_len x english_len). This means our attention mask also has to have this shape! Just like
        before, we want to mask out the columns of the attention mask, so our french tokens dont attend to any english padding tokens. We can then take
        our english padding mask which is (Batch x english_len), add extra dimensions for head and src_len dimension which will give a 
        (Batch x 1 x 1 x english_len) and then repeat the mask for the source length (batc x 1 x french_len x english_len)

        """

        ### Grab Shapes ###
        batch, src_len, embed_dim = src.shape

        ### If target is not provided, we are doing self attention (with potential causal mask) ###    
        if tgt is None:
            q = self.q_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
            k = self.k_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
            v = self.v_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()

            if attention_mask is not None:

                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,src_len,1)

            attention_out = F.scaled_dot_product_attention(q,k,v, 
                                                           attn_mask=attention_mask, 
                                                           dropout_p=self.config.attention_dropout_p if self.training else 0.0, 
                                                           is_causal=causal)
        
        ### If target is provided then we are doing cross attention ###
        ### Our query will be the target and we will be crossing it with the encoder source (keys and values) ###
        ### The src_attention_mask will still be the mask here, just repeated to the target size ###
        else:
            tgt_len = tgt.shape[1]

            q = self.q_proj(tgt).reshape(batch, tgt_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
            k = self.k_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
            v = self.v_proj(src).reshape(batch, src_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()

            if attention_mask is not None:

                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,tgt_len,1)

            attention_out = F.scaled_dot_product_attention(q,k,v,
                                                           attn_mask=attention_mask, 
                                                           dropout_p=self.config.attention_dropout_p if self.training else 0.0, 
                                                           is_causal=False)

        ### Reshape and Project ###
        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out)

        return attention_out

class FeedForward(nn.Module):
    """
    Regular MLP module after our attention computation. 
    """
    def __init__(self, config):
        super(FeedForward, self).__init__()
        
        hidden_size = config.embedding_dimension * config.mlp_ratio
        self.intermediate_dense = nn.Linear(config.embedding_dimension, hidden_size)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(config.hidden_dropout_p)

        self.output_dense = nn.Linear(hidden_size, config.embedding_dimension)
        self.output_dropout = nn.Dropout(config.hidden_dropout_p)

    def forward(self, x):
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x
    
class TransformerEncoderLayer(nn.Module):

    """
    Stacks together a Self-Attention module and MLP Layer
    """
    
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()

        self.enc_attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_p)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)

    def forward(self, x, attention_mask=None):

        x = x + self.dropout(self.enc_attention(x, attention_mask=attention_mask))
        x = self.layer_norm(x)

        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)

        return x
    
class TransformerDecoderLayer(nn.Module):
    
    """
    Stacks together a Causal-Attention of our target language, Cross Attention with encoded source language, 
    and a MLP layer
    """

    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()

        self.dec_attention = Attention(config)
        self.dec_attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.dec_attention_layernorm = nn.LayerNorm(config.embedding_dimension)

        self.cross_attention = Attention(config)
        self.cross_attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.cross_attention_layernorm = nn.LayerNorm(config.embedding_dimension)

        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):

        tgt = tgt + self.dec_attention_dropout(self.dec_attention(tgt, attention_mask=tgt_mask, causal=True))
        tgt = self.dec_attention_layernorm(tgt)

        tgt = tgt + self.cross_attention_dropout(self.cross_attention(src, tgt, attention_mask=src_mask))
        tgt = self.cross_attention_layernorm(tgt)

        tgt = tgt + self.feed_forward(tgt)
        tgt = self.final_layer_norm(tgt)

        return tgt

class Transformer(nn.Module):

    """
    Final Transformer proposed in Attention is All You Need
    """
    
    def __init__(self, config):
        super(Transformer, self).__init__()
        
        self.config = config

        self.encodings = Embeddings(config)

        self.encoder = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.encoder_depth)]
        )

        self.decoder = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.decoder_depth)]
        )

        self.head = nn.Linear(config.embedding_dimension, config.tgt_vocab_size)

        ### Initialize Weights ###
        self.apply(_init_weights_)

    def forward(self, 
                src_ids, 
                tgt_ids, 
                src_attention_mask=None, 
                tgt_attention_mask=None):

        src_embeddings = self.encodings.forward_src(src_ids)
        tgt_embeddings = self.encodings.forward_tgt(tgt_ids)

        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, 
                                   src_attention_mask)

        for layer in self.decoder:
            tgt_embeddings = layer(src_embeddings, 
                                   tgt_embeddings, 
                                   src_attention_mask, 
                                   tgt_attention_mask)
            
        pred = self.head(tgt_embeddings)

        return pred

    @torch.no_grad()
    def inference(self, 
                  src_ids,
                  tgt_start_id=2,
                  tgt_end_id=3,
                  max_len=512):
    
        tgt_ids = torch.tensor([tgt_start_id], device=src_ids.device).reshape(1,1)

        ### Encode source ###
        src_embeddings = self.encodings.forward_src(src_ids)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings)
        
        ### Generate Target ###
        for i in range(max_len):
            
            tgt_embeddings = self.encodings.forward_tgt(tgt_ids)
            for layer in self.decoder:
                tgt_embeddings = layer(src_embeddings, 
                                       tgt_embeddings)
            
            ### Only Need Last Timestep ###
            tgt_embeddings = tgt_embeddings[:, -1]

            ### Project to Tokens ###
            pred = self.head(tgt_embeddings)
            pred = pred.argmax(axis=-1).unsqueeze(0)
            tgt_ids = torch.cat([tgt_ids,pred], axis=-1)

            if torch.all(pred == tgt_end_id):
                break
        
        return tgt_ids.squeeze().cpu().tolist() 
            
            
def _init_weights_(module):

    """
    Simple weight intialization taken directly from the huggingface
    `modeling_roberta.py` implementation! 
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

if __name__ == "__main__":
    src = torch.randint(0,1000, (1,128)).to("cuda")
    config = TransformerConfig()
    model = Transformer(config)
    model = model.to("cuda")

    model.inference(src)
        
        


