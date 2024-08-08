import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import RobertaModel as HFRobertaModel

class RobertaEmbeddings(nn.Module):
    """
    Converts our tokens to embedding vectors and then adds positional embeddings (and potentially token type embeddings)
    to our data! We wont need to token type embeddings until we do our QA finetuning. 
    """
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__()

        ### Embeddings for Tokens ###
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dimension, padding_idx=config.pad_token)

        ### Positional Embeddings ###
        self.position_embeddings = nn.Embedding(config.context_length, config.embedding_dimension)

        ### Layernorm and Dropout ###
        self.layernorm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_p)

    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        ### Convert Tokens to Embeddings ###
        x = self.word_embeddings(input_ids)

        ### Add Positional Information ###
        avail_idx = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        pos_embed = self.position_embeddings(avail_idx)
        x = x + pos_embed

        x = self.layernorm(x)
        x = self.dropout(x)

        return x

            
class RobertaAttention(nn.Module):
    """
    Regular Self-Attention but in this case we utilize flash_attention
    incorporated in the F.scaled_dot_product_attention to speed up our training. 
    """
    def __init__(self, config):
        super(RobertaAttention, self).__init__()
        
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
        

    def forward(self, x, attention_mask=None):

        ### Store Shape ###
        batch, seq_len, embed_dim = x.shape

        ### Compute Attention with Flash Attention ###
        q = self.q_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
        k = self.k_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
        v = self.v_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
        
        ### Compute Attention (Attention Mask has shape Batch x Sequence len x Sequence len) ###
        attention_out = F.scaled_dot_product_attention(q, k, v, 
                                                        attn_mask=attention_mask, 
                                                        dropout_p=self.config.attention_dropout_p if self.training else 0.0)


        ### Compute Output Projection ###
        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out)

        return attention_out

class RobertaFeedForward(nn.Module):
    """
    Regular MLP module after our attention computation. 
    """
    def __init__(self, config):
        super(RobertaFeedForward, self).__init__()
        
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
    
class RobertaEncoderLayer(nn.Module):
    """
    Single transformer block stacking together Attention and our FeedForward
    layers, with normalization and residual connections. 
    """
    def __init__(self, config):
        super(RobertaEncoderLayer, self).__init__()

        self.attention = RobertaAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_p)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.feed_forward = RobertaFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask=None):

        x = x + self.dropout(self.attention(x, attention_mask=attention_mask))
        x = self.layer_norm(x)

        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)

        return x
    
class RobertaEncoder(nn.Module):
    """
    This will be the stack of all of our transformer blocks
    """
    def __init__(self, config):
        super(RobertaEncoder, self).__init__()

        self.config = config

        ### Transformer Layers ###
        self.layers = nn.ModuleList(
            [
                RobertaEncoderLayer(config) for _ in range(config.num_transformer_blocks)
            ]
        )
        
    def forward(
        self,
        x,
        attention_mask = None,
    ):

        batch_size, seq_len, embed_dim = x.shape

        if attention_mask is not None:

            ### Make Sure Attention Mask is a Boolean Tensor ###
            attention_mask = attention_mask.bool()

            ### Now our Attention Mask is in (Batch x Sequence Length) where we have 0 for tokens we don't want to attend to ###
            ### F.scaled_dot_product_attention expects a mask of the shape (Batch x ..., x Seq_len x Seq_len) ###
            ### the "..." in this case is any extra dimensions (such as heads of attention). lets expand our mask to (Batch x 1 x Seq_len x Seq_len) ###
            ### The 1 in this case refers to the number of heads of attention we want, so it is a dummy index to broadcast over ###
            ### In each (Seq_len x Seq_len) matrix for every batch, we want False for all columns corresponding to padding tokens ###
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, seq_len, 1)
            
        for layer in self.layers:

            x = layer(x, attention_mask=attention_mask)

        return x

class RobertaMLMHead(nn.Module):

    def __init__(self, config):
        super(RobertaMLMHead, self).__init__()

        self.config = config

        ### Projection Layer for Hidden States ###
        self.dense = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.activation = nn.GELU()

        ### Mapping to Vocabulary ###
        self.decoder = nn.Linear(config.embedding_dimension, config.vocab_size)

    def forward(self, inputs):
        
        ### Pass through Projection/Activation/Norm ###
        x = self.dense(inputs)
        x = self.activation(x)
        x = self.layer_norm(x)

        ### Prediction of Masked Tokens ###
        x = self.decoder(x)

        return x

class RobertaModel(nn.Module):

    def __init__(self, config):
        super(RobertaModel, self).__init__()

        self.config = config

        ### Define all Parts of the Model ###
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        

    def forward(self, input_ids, attention_mask=None):
        
        embeddings = self.embeddings(input_ids)
        output = self.encoder(embeddings, attention_mask)

        return output

class RobertaForMaskedLM(nn.Module):

    def __init__(self, config):
        super(RobertaForMaskedLM, self).__init__()

        self.config = config

        ### Define Model and MLM Head ###
        self.roberta = RobertaModel(config)
        self.mlm_head = RobertaMLMHead(config)

        self.apply(_init_weights_)

    def forward(self,
                input_ids, 
                attention_mask=None, 
                labels=None):
        
        ### Pass data through model ###
        hidden_states = self.roberta(input_ids,
                                     attention_mask)

        preds = self.mlm_head(hidden_states)

        ### Compute Loss if Labels are Available ###
        loss = None
        if labels is not None:
            
            ### Flatten Logits to (B*S x N) and Labels to (B*S) ###
            preds = preds.flatten(end_dim=1)
            labels = labels.flatten()

            loss = F.cross_entropy(preds, labels)

            return hidden_states, preds, loss
        
        else:
            return hidden_states, preds
        
class RobertaForQuestionAnswering(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config

        ### Grab Backbone Based on Config ###
        self.load_backbone()

        ### Initialize Prediction Head ###
        self.qa_head = nn.Linear(config.embedding_dimension, 2)

    def load_backbone(self):
        
        if self.config.pretrained_backbone == "pretrained_huggingface":
            print(f"Loading Huggingface Roberta Backbone: {self.config.hf_model_name}")
            self.roberta = HFRobertaModel.from_pretrained(self.config.hf_model_name)

        else:
            self.roberta = RobertaModel(self.config)

            if self.config.pretrained_backbone == "pretrained":
                if self.config.path_to_pretrained_weights is None:
                    raise Exception("Provide the argument `path_to_pretrained_weights` in the config, else we cant load them!")
                else:
                    
                    if not os.path.isfile(self.config.path_to_pretrained_weights):
                        raise Exception(f"Provided path to safetensors weights {self.config.path_to_pretrained_weights} is invalid!")

                    print(f"Loading RobertaModel Backbone from {self.config.path_to_pretrained_weights}")

                    ### Load Weights with load_file from safetensors ###
                    state_dict = load_file(self.config.path_to_pretrained_weights)

                    ### Cleanup of Weights and keys ###
                    backbone_keys = {}
                    for key in state_dict.keys():

                        ### If Wav2Vec2 is in key name, just remove from the key name ###
                        if "roberta" in key:
                            new_key = key.replace("roberta.", "")
                            backbone_keys[new_key] = state_dict[key]

                        ### If wav2vec2 is not in key name, it isnt a part of the backbone so ignore it ###
                        else:
                            continue

                    ### Load State Dict to Backbone ###
                    self.roberta.load_state_dict(backbone_keys)

    def forward(self,
                input_ids, 
                attention_mask=None, 
                start_positions=None, 
                end_positions=None):

        ### Different returns based on which backbone we are using ###
        if self.config.pretrained_backbone == "pretrained_huggingface":
            outputs = self.roberta(input_ids, attention_mask)

            ### Outputs have shape (Batch x Seq Len x Embedding Dim)
            outputs = outputs.last_hidden_state

        ### Pass Outputs through QA Head, Shape (Batch x Seq Len x 2) ###
        logits = self.qa_head(outputs)

        ### Split Logits by last Dim and sequeeze to make (Batch x Seq Len) ###
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        ### If the True Start/End positions are provided we can compute loss ###
        total_loss = None
        if start_positions is not None and end_positions is not None:
            
            ### Make sure Stard and End positions are just vectors (Batch Size) ###
            if len(start_positions.size()) > 1: 
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ### Any labels with an index value larger than the total sequence length, should be ignored ###
            ### In our case our max length is 512, which includes the question and the context. We need to ensure ###
            ### that the answer is within the context portion of our 512 tokens. Therefore, if the start or end index is ###
            ### beyond 512 tokens, then it isnt in the context and we ignore it. ###
            ### In the `utils.ExtractiveQAPreProcessing` we already handled this so it doesnt matter all that much, I just want to ###
            ### keep this as similar as possible to the Huggingface implementation ###

            # Grab the sequence length as the max tokens we can use to predict the start and end ###
            ignored_index = start_logits.size(1)
            
            # Clamp the labels to this, and then ignore these in the loss computation ###
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            # Compute Loss on logits (Batch x Sequence) and target positions (Batch). Basically, we need to predict which of the 512 in the sequence
            # is the start token and which is the end token
            start_loss = F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = F.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)

            # Average up the losses 
            total_loss = (start_loss + end_loss) / 2

        if total_loss is not None:
            return total_loss, start_logits, end_logits
        else:
            return start_logits, end_logits


        


def _init_weights_(module):
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
