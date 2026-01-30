import torch
import torch.nn as nn
from transformers import CLIPTextModel

class PositionalEncoding(nn.Module):

    """
    Sin/Cosine (non-learnable) encodings proposed in Attention is All You Need

    Args:
        max_len: Maximum number of tokens possible in a sequence
        embed_dim: Embedding dimension of each token
    """

    def __init__(self, max_len, time_embed_start_dim, time_embed_proj_dim, requires_grad=False):
        super(PositionalEncoding, self).__init__()

        self.max_len = max_len
        self.time_embed_start_dim = time_embed_start_dim
        self.scaled_time_embed_dim = time_embed_proj_dim
        self.requires_grad = requires_grad

        self.encodings = self._build_positional_encodings()

        self.time_mlp = nn.Sequential(

            nn.Linear(time_embed_start_dim, time_embed_proj_dim),
            nn.SiLU(),
            nn.Linear(time_embed_proj_dim, time_embed_proj_dim),
            nn.SiLU()
            
        )

    def _build_positional_encodings(self):

        encoding = torch.zeros(self.max_len, self.time_embed_start_dim, dtype=torch.float)
        postion_idx = torch.arange(0, self.max_len, dtype=torch.float).reshape(-1,1)
        embed_dim_skip_idx = torch.arange(0, self.time_embed_start_dim, step=2, dtype=torch.float)
        
        encoding[:, 0::2] = torch.sin(postion_idx / (10000 ** (embed_dim_skip_idx / self.time_embed_start_dim)))
        encoding[:, 1::2] = torch.cos(postion_idx / (10000 ** (embed_dim_skip_idx / self.time_embed_start_dim)))

        sincos_emb = nn.Parameter(encoding, requires_grad=self.requires_grad)
        
        encoding = nn.Embedding(self.max_len, self.time_embed_start_dim)
        encoding.weight = sincos_emb
        
        return encoding

    def forward(self, timestep_idx):

        time_embeddings = self.encodings(timestep_idx)

        time_embeddings = self.time_mlp(time_embeddings)

        return time_embeddings

class TextConditionalEmbeddings(nn.Module):
    
    """
    Text conditional generation (using CLIP as the model)

    Args:
        pre_encoded_text: If the text was already pre-encoded, we don't CLIP anymore
    """

    def __init__(self, 
                 pre_encoded_text=False, 
                 text_conditioning_hf_model="openai/clip-vit-large-patch14",
                 text_embed_dim=768):
        
        super(TextConditionalEmbeddings, self).__init__()

        self.pre_encoded_text = pre_encoded_text

        if not pre_encoded_text:
            self.text_encoder = CLIPTextModel.from_pretrained(
                   text_conditioning_hf_model
            )

            self.text_encoder.eval()

            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        ### Null Token for Classifier Free Guidance ###
        self.null_token = nn.Parameter(torch.randn(1, 1, text_embed_dim))

    def forward(self,
                batch_size,
                text_conditioning=None,
                text_attention_mask=None, 
                cfg_dropout_prob=0):
        
        assert cfg_dropout_prob <= cfg_dropout_prob <= 1, "Ensure dropout is between 0 and 1"

        ### If we have text conditioning passed in ###
        if text_conditioning is not None:

            ### If text is not pre-encoded then encode with CLIP ###
            if not self.pre_encoded_text:
                
                assert (text_conditioning.dtype == torch.long), "CLIP Text Encoder Expects Text Tokens"
                
                with torch.no_grad():
                    text_conditioning = self.text_encoder(
                        input_ids=text_conditioning, 
                        attention_make=text_attention_mask
                    )
            
            ### CFG Dropout ###
            if cfg_dropout_prob > 0:    

                dropout_mask = torch.rand(text_conditioning.shape[0], \
                                        device=text_conditioning.device) < cfg_dropout_prob
                
                text_conditioning[dropout_mask] = self.null_token
                
            ### HACK FOR DDP ###
            ### DDP EXPECTS ALL LEARNABLE PARAMETERS TO HAVE GRADIENTS BUT BECAUSE 
            ### OUR NULL TOKEN IS RANDOM IT COULD ALSO NOT HAVE ANY GRADIENTS BECAUSE
            ### NOTHING WAS SELECTED TO BE NULLED. SO DO A SANITY CHECK, WE CAN ADD
            ### OUR NULL TOKEN TO OUR DATA WITH 0 CONTRIBUTION TO ADD TO GRAPH FOR GRADS
            
            text_conditioning = text_conditioning + 0.0 * self.null_token

        ### If we dont have text conditioning, then just use the Null Token ###
        else:

            text_conditioning = self.null_token.repeat(batch_size, 1, 1)
                
        return text_conditioning

class ClassConditionalEmbeddings(nn.Module):
    
    """
    Class conditional generation (where each class is identified as a index) 
    """
    
    def __init__(self, num_classes, embed_dim):
        
        super(ClassConditionalEmbeddings, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        ### Embedding Matrix for Classes ###
        self.class_embeddings = nn.Embedding(num_classes, embed_dim)

        ### Null Token for Classifier Free Guidance ###
        self.null_token = nn.Parameter(torch.randn(1, embed_dim))

        ### Projection ###
        self.proj = nn.Sequential(  
            
            nn.Linear(embed_dim, embed_dim), 
            nn.SiLU(), 
            nn.Linear(embed_dim, embed_dim), 
            nn.SiLU()

        )

    def forward(self,
                batch_size, 
                class_conditioning=None, 
                cfg_dropout_prob=0):
        
        assert cfg_dropout_prob <= cfg_dropout_prob <= 1, "Ensure dropout is between 0 and 1"

        ### If we have class conditioning ###
        if class_conditioning is not None:

            class_conditioning = self.class_embeddings(class_conditioning)

            if cfg_dropout_prob > 0:

                dropout_mask = torch.rand(class_conditioning.shape[0], \
                                          device=class_conditioning.device) < cfg_dropout_prob
                
                class_conditioning[dropout_mask] = self.null_token

            ### HACK FOR DDP ###
            ### DDP EXPECTS ALL LEARNABLE PARAMETERS TO HAVE GRADIENTS BUT BECAUSE 
            ### OUR NULL TOKEN IS RANDOM IT COULD ALSO NOT HAVE ANY GRADIENTS BECAUSE
            ### NOTHING WAS SELECTED TO BE NULLED. SO DO A SANITY CHECK, WE CAN ADD
            ### OUR NULL TOKEN TO OUR DATA WITH 0 CONTRIBUTION TO ADD TO GRAPH FOR GRADS

            class_conditioning = class_conditioning + 0.0 * self.null_token
        
        ### Otherwise just use Null Token ###
        else:

            class_conditioning = self.null_token.repeat(batch_size, 1)

        class_conditioning = self.proj(class_conditioning)
        
        return class_conditioning
