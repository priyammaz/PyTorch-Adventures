"""
Our Wav2Vec2 Model was heavily inspired by modeling_wav2vec2.py from 🤗 Huggingface!

https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py

This implementation is simplified and overly annotated mainly for learning purposes. There were some differences
in the pretrained model offered from Huggingface and the Wav2Vec2 paper, so I tried to keep it as close to the paper 
as possible!

"""
import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model as HFWav2Vec2Model
from safetensors.torch import load_file
from utils import (
    Wav2Vec2ForPreTrainingOutput,
    compute_sub_attention_mask,
    compute_encoded_lengths
)

class Wav2Vec2LayerNormConvLayer(nn.Module):
    """
    Single convolutional block with layernorm and GELU. A few of these will be stacked
    together to create our convolutonal feature extractor. 

    Args:
        in_channels: Number of input channels for the convolution
        out_channels: Number of output channels for the convolution
        kernel_size: Filter size for the convolution
        stride: Stride of the convolution
        bias: If we want to have a bias in our convolution
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 bias: bool): 
        
        super(Wav2Vec2LayerNormConvLayer, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, 
                              stride=stride,
                              bias=bias)
        
        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-5)

        self.activation = nn.GELU()

    def forward(self, x):

        ### Pass X (B x C X L) into Convolution ###
        x = self.conv(x)

        ### Transpose for Channels Last for LayerNorm ###
        x = x.transpose(-2,-1)
        x = self.layer_norm(x)
        x = x.transpose(-2,-1)

        ### GELU Activation ###
        x = self.activation(x)

        return x

class Wav2Vec2FeatureEncoder(nn.Module):
    """
    Stack of our Wav2Vec2LayerNormConvLayers to perform our feature extraction 
    in our audios. For the purposes of pre-training our own Wav2Vec2, the output
    of this module will be normalize and then quantized for our contrastive task. 
    """
    def __init__(self, config):
        super(Wav2Vec2FeatureEncoder, self).__init__()

        self.config = config

        assert len(config.conv_dim) == len(config.conv_stride) == len(config.conv_kernel), \
            "Check Config for same number of convolution components" 
        
        number_of_conv_blocks = len(config.conv_kernel)

        ### Convolution starts with 1 input channel (as our audio is single channel audio) ###
        conv_channels = (1, ) + tuple(config.conv_dim)

        self.conv_layers = nn.ModuleList()
        for conv_idx in range(number_of_conv_blocks):
            self.conv_layers.append(
                
                Wav2Vec2LayerNormConvLayer(in_channels=conv_channels[conv_idx],
                                           out_channels=conv_channels[conv_idx + 1],
                                           kernel_size=config.conv_kernel[conv_idx],
                                           stride=config.conv_stride[conv_idx],
                                           bias=config.conv_bias)
            )

    def forward(self, x):
        
        ### Double Check that Inputs in the Shape (batch, audio_input_channels, audio_length)
        if x.dim() != 3: # If data is only (Batch x Length) assume single channel 
            x = x.unsqueeze(1)
        
        assert x.shape[1] == 1, f"Number of input channels {x.shape[1]} doesn't match config {1}"

        for layer in self.conv_layers:
            x = layer(x)

        return x

class Wav2Vec2PositionalConvEmbedding(nn.Module):
    """
    Grouped Convolution that will encode relative positional information. This
    is actually pretty cool, as normally we have a positional encoding matrix that
    we add to our tokens, but in this case we opt for a different method as described
    in the Wav2Vec2 paper. 
    """
    def __init__(self, config):
        super(Wav2Vec2PositionalConvEmbedding, self).__init__()
        
        ### Define Convolution that computes Positional Encoding ###
        self.conv = nn.Conv1d(
            config.embedding_dimension,
            config.embedding_dimension,
            kernel_size=config.conv_positional_emb_kernel_size,
            padding=config.conv_positional_emb_kernel_size // 2,
            groups=config.conv_positional_emb_groups,
        )
                
        self.activation = nn.GELU()

    def forward(self, x):

        batch, sequence_length, embedding = x.shape
        
        ### x has the shape (Batch x Sequence x Embeddings) but Convolution wants sequence last ###
        x = x.transpose(1, 2)

        ### Compute Positional Encodings ###
        positional_embeddings = self.conv(x)

        ### Clip any excess positional encodings ###
        positional_embeddings = positional_embeddings[:, :, :sequence_length]

        ### Activation and Transpose ###
        positional_embeddings = self.activation(positional_embeddings)
        positional_embeddings = positional_embeddings.transpose(1, 2)

        return positional_embeddings
    

class Wav2Vec2FeatureProjection(nn.Module):
    """
    In the default configuration of Wav2Vec2, our convolutional feature extractor
    will return a tensor with 512 channels. The transformer on the other hand 
    expects an embedding dimension of 768. This linear layer simply maps between
    the two vector spaces.
    """
    def __init__(self, config):

        super(Wav2Vec2FeatureProjection, self).__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=1e-5)
        self.projection = nn.Linear(config.conv_dim[-1], config.embedding_dimension)
        self.dropout = nn.Dropout(config.feature_projection_dropout_p)
    
    def forward(self, x):
        
        ### We need to store Normed X (before projection) for quantization ###
        normed_x = self.layer_norm(x)

        ### Project X to Transformer Embedding Dimension ###
        projected_x = self.projection(normed_x)
        projected_x = self.dropout(projected_x)

        return projected_x, normed_x

class Wav2Vec2Attention(nn.Module):
    """
    Regular Self-Attention but in this case we utilize flash_attention
    incorporated in the F.scaled_dot_product_attention to speed up our training. 
    """
    def __init__(self, config):
        super(Wav2Vec2Attention, self).__init__()
        
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
    
class Wav2Vec2FeedForward(nn.Module):
    """
    Regular MLP module after our attention computation. 
    """
    def __init__(self, config):
        super(Wav2Vec2FeedForward, self).__init__()
        
        hidden_size = config.embedding_dimension * config.mlp_ratio
        self.intermediate_dense = nn.Linear(config.embedding_dimension, hidden_size)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(config.mlp_dropout_p)

        self.output_dense = nn.Linear(hidden_size, config.embedding_dimension)
        self.output_dropout = nn.Dropout(config.mlp_dropout_p)

    def forward(self, x):
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x

class Wav2Vec2EncoderLayer(nn.Module):
    """
    Single transformer block stacking together Attention and our FeedForward
    layers, with normalization and residual connections. 
    """
    def __init__(self, config):
        super(Wav2Vec2EncoderLayer, self).__init__()

        self.attention = Wav2Vec2Attention(config)
        self.dropout = nn.Dropout(config.transformer_encoder_dropout)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask=None):

        x = x + self.dropout(self.attention(x, attention_mask=attention_mask))
        x = self.layer_norm(x)

        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)

        return x
    
class Wav2Vec2Encoder(nn.Module):
    """
    This is all the logic that follows the convolutional feature extractor. 
    The output of our feature extractor is first given positional information
    through the positional conv embeddings, and then are passed through 
    a stack of transformer encoder blocks.

    We also deal with the attention mask here. The attention mask we provide
    from the dataloader is a boolean tensor of shape (Batch x Sequence) where
    False values indicate padding tokens we added in to batch our tensors together. 
    """
    def __init__(self, config):
        super(Wav2Vec2Encoder, self).__init__()

        self.config = config

        ### Convolutional Positional Embedding ###
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.conv_positional_emb_drop_p)

        ### Transformer Layers ###
        self.layers = nn.ModuleList(
            [
                Wav2Vec2EncoderLayer(config) for _ in range(config.num_transformer_layers)
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

            ### So, what we have right now is x is the output of our convolutions. This also means that ###
            ### if we had padded our audios with zeros, those are also included in our output. Now we already ###
            ### computed the output shape of our convolutions and precomputed the sub_attention_mask so ###
            ### we know which tokens are padding tokens. Normally we have a dedicated padding token in our ###
            ### transformer architecture (check RoBERTa) but it doesn't really matter to be honest as we mask it ###
            ### out in the attention computation, so it has no effect. Just as a sanity check that our padding tokens ###
            ### always return 0, we will just convert all our pad tokens to 0 ahead of time ###
            # expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            x[~attention_mask] = 0
        
            ### Now our Attention Mask is in (Batch x Sequence Length) where we have 0 for tokens we don't want to attend to ###
            ### F.scaled_dot_product_attention expects a mask of the shape (Batch x ..., x Seq_len x Seq_len) ###
            ### the "..." in this case is any extra dimensions (such as heads of attention). lets expand our mask to (Batch x 1 x Seq_len x Seq_len) ###
            ### The 1 in this case refers to the number of heads of attention we want, so it is a dummy index to broadcast over ###
            ### In each (Seq_len x Seq_len) matrix for every batch, we want False for all columns corresponding to padding tokens ###
            ### For more details check out my RoBERTa implementation!!! 
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, seq_len, 1)
        

        position_embeddings = self.pos_conv_embed(x)
        x = x + position_embeddings
        x = self.layer_norm(x)
        x = self.dropout(x)

    
        for layer in self.layers:
    
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand(1)

            ### If we are in training mode, and our uniformly sampeld number less than layer_drop, we skip the layer ###
            ### If we are not in training mode, then use all layers ###
            ### CAVEAT: If we train on multiple GPUs with DDP, this causes an unused_parameter issue, 
            ### basically, different gpus will randomly drop different layers, and when it accumulated the gradients
            ### across GPUs, it sees some some layers on a GPU dont have any. So for learning purposes, 
            ### ive included this but we wont use it!
            if (not self.training) or (self.training and (dropout_probability >= self.config.layer_dropout)):
                x = layer(x, attention_mask=attention_mask)

        return x
    
class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    This is the implementation of differentiable indexing as proposed in the Gumbel Softmax paper
    (https://arxiv.org/pdf/1611.01144)

    This quantizer can have multiple codebooks and multiple codes per codebook. In the case of the 
    default Wav2Vec2 config, there are 2 codebooks and 320 codes per book, which leads to a theoretical
    possibility of 320 x 320 unique code vector combinations. 

    We also compute perplexity, which is used in our Diversity loss metric, to ensure that the model is 
    adequately exploring the codebook rather than utilizing a small subset of it. 
    """

    def __init__(self, config):
        super(Wav2Vec2GumbelVectorQuantizer, self).__init__()
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group

        assert config.codevector_dim % config.num_codevector_groups == 0, \
            "Ensure codevector dim {config.codevector_dim} is divisible by the number of groups {config.num_codevector_groups}"

        ### storage for codebook variables (codewords)
        ### Note: im not really sure why, but when i used torch.rand(), the model will not converge at all...
        ### Fairseq uses torch.FloatTensor and then intializes it at uniform and then it works just fine, 
        ### Explore this later for sure! 
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )

        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        # can be decayed for training
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        ### First to understand: What is Perplexity?? ###
        ### The formula for perplexity is basically Entropy (\sum[{p} * log{p}])
        ### Now entropy is a measure of uncertainty or randomness (i.e. to increase entropy is to also increase uncertaintiy/randomness)
        
        ### Now why do we care about perplexity? We need to ensure that the model is making use of the entire codebook! If we dont, we will fall into an issue
        ### known as index collapse, or the model will only make use of a few, or even only 1, of the codes from our codebook. Our perplexity will be a heuristic that
        ### will measure our codebook usage.

        ### Ideally, all codes should be equally likely to be selected and the model will explore all of them! In a distribution, if everything is equally likely, then 
        ### that is known as a uniform distribution. Luckily for us, a uniform distribution also has the maximum entropy compared to any other distribution (as long as they
        ### are bounded by the same support). Therefore, if we can maximize our perplexity (or entropy) that is the same as our model equally selecting across all our codes
        ### in our codebooks. 
        ### You can see some more discussion about this here: https://stats.stackexchange.com/questions/66108/why-is-entropy-maximised-when-the-probability-distribution-is-uniform

        ### So lets tie this back to the Wav2Vec2 Paper equation 4. In Wav2Vec2 they talk about the Diversity loss, but really, in their implementation, they will maximize the 
        ### perplexity (subtext at the bottom of page 4). What we have right now is the distribution of which code each input vector should be assigned to. If we average across
        ### all of our samples and all of the vectors to quantize, then we hope the distribution is overall uniform. IMPORTANT NOTE THOUGH!!! Each probability vector, as we train
        ### should definitely not be uniform, as we need our model to confidently select which code to pick, if it were uniform then its just random guessing codes. But across the 
        ### batch, different samples should pick different codes, so the overall distribution should be uniform. We will be maximizing perplexity of course for across the batch!!
        
        ### This additional diversity (or perplexity) loss has a weight attached to it. So our overall loss will be Contrastive Loss + lambda * Perplexiy Loss, The larger we pick
        ### lambda to be, the more the model will force the batch code selection distribution to be uniform, and this will cause unstable selection as the model is never confident 
        ### in which codes it picks. On the other hand, if lambda is too small, then our model will not explore all the codes and may fall into index collapse. Like everything in life, 
        ### there is a tradeoff!!
        
        ### If we are passing in a span mask, we only need to compute the perplexity across the masked tokens (as that is all we are training to predict anyway) ###
        if mask is not None:

            ### Mask Token Index is (Batch x Sequence) where we have True values on the masked sequence values ###
            ### Probs is (Batch*Sequence x num_codebook_groups x codes_per_group) ###
            ### We just need to filter out for the sequence values that are actually masked in our probs! ###
            marginal_probs = probs[mask.flatten()]
            marginal_probs = marginal_probs.sum(dim=0) / mask.sum()

        else:

            marginal_probs = probs.mean(dim=0)

        ### Compute the pereplxity per codebook ###
        perplexity_per_codebook = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1))

        ### Add up the perplexity across codebooks ###
        perplexity = perplexity_per_codebook.sum()

        return perplexity

    def forward(self, hidden_states, mask_time_indices=None):

        batch_size, sequence_length, hidden_size = hidden_states.shape

        ### OK so this is fun, lets say we have 2 codebooks, each with 300 vectors, and each vector is of size 128 ###
        ### For each token along our sequence  in x, we will first map it to our 600 posibilities (2 * 300 total vectors) ###
        hidden_states = self.weight_proj(hidden_states)

        ### Reshape x so we have all the dimensions by the num_codes_per_group ###
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            
            ### GumbelSoftmax will give us a probability vector across the 300 possible codes. ###
            ### These probabilities are scaled by the temperature parameter ###
            ### Setting hard to true will only return back a value of 1 at the index of the most likely code! ###
            codevector_probs = nn.functional.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True)

            ### Compute Perplexity ###
            codevector_soft_dist = hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float().softmax(dim=-1)
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        
        ### If we are not training (no need for differentiable indexing) then we can just use argmax! ###
        else:
            ### For each input vector in the sequence, just grab the index of the most likely code! ##
            codevector_idx = hidden_states.argmax(dim=-1)

            ### Codevector Probs will just be a 1 for the indexes of the most likely codes and zeros elsewhere ###
            codevector_probs = torch.zeros_like(hidden_states)
            codevector_probs[torch.arange(hidden_states.shape[0]), codevector_idx] = 1

            ### Reshape Codevector Probs to compute Perplexity ###
            ### In this case, during inference, we have basically 100% for the codevector selected and 0 percent for everything else ###
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            ### Compute Perplexity ###
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        ### We will now reshape the codevector probs to be (batch*sequence, num_groups*codes_per_group) ###
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        
        ### Our codevectors have shape (1 x num_groups*codes_per_group * vq_dim//num_groups) ###
        ### We can add an extra dimension to our codevector_probs so it is (batch*sequence, num_groups*codes_per_group, 1) ###
        ### We can then multiply out codevector probs (which is just 1 for the selected code and 0 otherwise) ###
        ### In our codevector probs, if we have 2 groups then there are 2 indexes as 1 and the rest are zeros ###
        ### So then if we multiply our codevector probs by the the codevectors, then it will keep the 2 codevectors that were indexed ###
        ### as 1, and make everything else 0! ###
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors.type_as(codevector_probs)

        ### We now will seperate out the 600 codes to be 2 x 300, and then add across the 300 dimension (as only one of those 300 codes are actually not zero!)###
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(dim=-2)
        
        ### Now for the cool part, currently we have 2 codes for each intial input token, and each are of length 128 (assuming the final VQ dim is 256 and we had 2 groups) ###
        ### In total, we have 600 codes, which is not enough to represent a large diverse set of quantized latents, so instead, we will concatenate them together! This way ###
        ### We will have a total of 300 * 300 possibilities! ###
        
        codevectors = codevectors.view(batch_size, sequence_length, -1)

        return codevectors, perplexity

class Wav2Vec2Model(nn.Module):
    """
    Backbone to our Wav2Vec2 implementation! This basically has everything we need in our model, except the 
    few extra layers needed for Pretraining or fine-tuning. We also include a masking token in the case of
    Pretraining so we can replace embeddings with this learnable token. 
    """
    def __init__(self, config):
        super(Wav2Vec2Model, self).__init__()

        ### Store Config ###
        self.config = config

        ### Load Convolutional Feature Extractor and Conv2Transformer Projection Layer ###
        self.feature_extractor = Wav2Vec2FeatureEncoder(config) 
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        ### If we are masking, we need a learnable masking token ####
        if config.masking_probability > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.embedding_dimension))
            torch.nn.init.uniform_(self.masked_spec_embed)

        self.encoder = Wav2Vec2Encoder(config)

    def forward(self,
                input_values,
                attention_mask = None,
                sub_attention_mask = None,
                mask_time_indices = None, 
                return_features_to_quantize=False):

        ### Extract Features with Convolutions and transpose to channels last ###
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        ### Sanity Check! We have two attention masks, the first (attention_mask) is at the waveform level ###
        ### that tell us the zeros we padded onto the audios. The second attention mask is the sub_attention_mask ###
        ### which is at the encoded features level. This ensures that all the zeros we appended on to the original audios ###
        ### when compressed down into features, are also masked out! If the sub_attention_mask is not provided in the ###
        ### forward function, but the attention mask is provided, we can compute the sub_attention_mask ###
        ### The sub_attention_mask is already being computed in the DataCollatorForWav2Vec2Pretraining ###
        ### but just incase this option is here as well!! If no attention_mask or sub_attention_mask is provided ###
        ### the attention computation will attend to everything. ###
        if (sub_attention_mask is None and attention_mask is not None):
            # compute reduced attention_mask corresponding to feature vectors
            sub_attention_mask = compute_sub_attention_mask(
                self.config, attention_mask
            ).to(input_values.device)

        ### Project all the features to Transformer dim ###
        hidden_states, extract_features = self.feature_projection(extract_features)

        ### If a span mask is provided, replace the masked hidden states with mask token ###
        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=sub_attention_mask,
        )

        if return_features_to_quantize:
            return encoder_outputs, extract_features
        else:
            return encoder_outputs

class Wav2Vec2ForPreTraining(nn.Module):
    """
    Wav2Vec2 definition for pretraining. This includes our Wav2Vec2 model, but also
    our Gumbel Softmax Quantizer and some projection layers. 
    """
    def __init__(self, config):
        super(Wav2Vec2ForPreTraining, self).__init__()

        self.config = config 

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.pre_quantizer_dropout)

        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)

        self.project_hid = nn.Linear(config.embedding_dimension, config.codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.codevector_dim)

        # Initialize weights and apply final processing
        self.apply(weight_init_strategy(config))

    def set_gumbel_temperature(self, temperature):
        self.quantizer.temperature = temperature

    @staticmethod
    def _compute_cosine_similarity(target_features, 
                                   negative_features, 
                                   predicted_features, 
                                   temperature=0.1):
        
        ### Just a reminder of tensor shapes:
        ### true_quantized: (Batch x Sequence Length x VQ_dim)
        ### negative_quantized: (Num Negatives x Batch Size x Sequence Length x VQ_dim)
        ### transformer_output: (Batch x Sequence Length x VQ_dim)

        ### So, what we want to do is compute the cosine similarity between each token in the transformer output
        ### against the Num Negatives + 1 Positive quantized tokens! So lets first concatenate the true quantized
        ### to our negatives. To do this, we need to add a dimension to our true_quantized features so its becomes
        ### (1 x Batch x Sequence Length x VQ_dim). This will create our quantized targets in the shape of 
        ### (Num_negatives + 1 x Batch x Sequence Length x VQ_dim)

        target_features = target_features.unsqueeze(0)
        targets = torch.cat([target_features, negative_features], dim=0) 

        ### Compute Cosine Similarity between our transformer output and targets, along the VQ_dimension (which is the last one) ###
        ### If you take a quick look at the PyTorch cosine similarity function (https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html)
        ### we just needt to make sure our shapes are broadcastable to a common shape, but we already made this happen because our
        ### transformer output is (batch x sequence x vq_dim) and the targets are (Num_negatives + 1 x Batch x Sequence Length x VQ_dim)
        ### So each sample in our Batch x sequence will compute its cosine similarity across all Num negatives + 1 quantized tokens. 
        ### This operation will return the cosine similarity in the shape of (Num_negatives + 1 x Batch x Sequence Length)
        cosine_sim = torch.cosine_similarity(predicted_features, targets, dim=-1)

        ### Now in formula 4, we see that there is a softmax involved. We can actually reformulate this problem easily in terms of CrossEntropyLoss that I explain below ###
        ### Torch CrossEntropyLoss expects logits, so we wont do any softmax now. But, we can scale our cosine similarity by our temperature parameter (kappa in the formula)! 
        cosine_sim = cosine_sim / temperature

        return cosine_sim

    def forward(self,
                input_values,
                attention_mask=None,
                sub_attention_mask=None,
                mask_time_indices=None,
                sampled_negative_indices=None):
        
        ### Make sure our Tokens Mask is Boolean ###
        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        ### Grab Outputs and Features to Quantize from Wav2Vec2 ###
        transformer_outputs, features_to_quantize = self.wav2vec2(input_values, 
                                                                  attention_mask=attention_mask,
                                                                  sub_attention_mask=sub_attention_mask, 
                                                                  mask_time_indices=mask_time_indices, 
                                                                  return_features_to_quantize=True)
         
        ### Project Outputs to VQ Dim ###
        transformer_vq = self.project_hid(transformer_outputs)

        ### Quantize the Masked Features ###
        quantized_codes, perplexity = self.quantizer(features_to_quantize, mask_time_indices=mask_time_indices)
    
        ### Project Quantized Features ###
        quantized_codes = self.project_q(quantized_codes)

        ### Compute Loss!! ###
        loss = None
        diversity_loss = None
        contrastive_loss = None

        if sampled_negative_indices is not None:
            
            batch_size, sequence_length, vq_size = quantized_codes.shape
            _, _, num_negatives = sampled_negative_indices.shape

            ### We have all our quantized codes, and indexes of the negative samples, we need to actually index now! ###
            ### Just to remind ourselves of the shapes of our data right now 
            ### quantized_codes: (Batch x Sequence Length x VQ_dimension)
            ### sampled_negative_indexes: (Batch x Sequence Length x Num Negatives)

            ### All we have to do is for each negative in our sampled_negative_indexes, that is also a masked token that we want to compute contrastive loss on,
            ### we just need to grab its quantized latent! Remeber, when we implemented utils.sample_negatives we "adjusted for batch size". This is so we can just
            ### directly index our quantized_codes. Our first dimension of quantized_codes is (Batch * Sequence lenght). Lets pretend the sequence length is 100. Then 
            ### the first batch goes from 0 to 99, and then the second batch goes from 100 to 199, and so on. So our negatives are indexed by both which batch its in 
            ### times which location in the sequence for that batch its in. So we can do an easy index to grab all of the corresponding quantized negative vectors! 
            negative_quantized_codes = quantized_codes.reshape(-1, vq_size)[sampled_negative_indices.flatten()]
            
            ### Now we have all the quantized features for all of our sampled_negative_indexes! But, in this implementation, we also quantized non-masked indexes. ###
            ### We will deal with this later as you will see! For now, lets just reshape this to be (Num Negatives x Batch x Sequence Length x VQ_dim) ###
            negative_quantized_codes = negative_quantized_codes.reshape(batch_size, sequence_length, num_negatives, vq_size).permute(2,0,1,3)
            
            ### Now that we have our Negative Quantized features, its time to compute contrastive loss! ###
            ### We will basically follow equation 3 from the paper! First we need to compute the Cosine Similarity ###
            ### between a masked output from our transformer (transformer_vq), its true quantized latent from our codebook and the K distractors ###
            ### The goal of the contrastive loss is to increase the similarity between a transformer output and its quanitzed feature, and decrease the ###
            ### similarity to the distractors.
            cosine_sim = self._compute_cosine_similarity(target_features=quantized_codes, 
                                                         negative_features=negative_quantized_codes,
                                                         predicted_features=transformer_vq,
                                                         temperature=self.config.contrastive_logits_temperature)
            
            ### Now a quick sanity check. In the case of having low codebook usage (i.e. lots of the codes are the same) and our negative quantized vectors are 
            ### identical to the positive quantized vectors, this will hurt training quite a bit! Remember, we want our transformer outputs to be close to the 
            ### positive quantized tokens and far away from the negative quantized tokens. If these positive tokens are identical to the negatives, then that
            ### contrastive task doesn't really work. So in those cases, when we compute the cosine similarity between our transformer output and our codes, 
            ### we will get the same cosine value for the positive and negative quantized latent. 

            ### Remember, the cosine similarity is of shape (Num_negatives + 1 x Batch x Sequence Length), where the first column (cosine_sim[0]) is the positive 
            ### class and the remaining columns (cosine_sim[1:]) are all the negatives. So in the case of an equal value between the positive and negative class, 
            ### we will convert the corresponding cosine_sim values of the negatives to -infinity (so when we do crossentropy loss and it takes a softmax it becomes 0)
            ### To create this mask, we will just check equality between the quantized codes and negative quantized codes along the last (VQ_dim) dimension and use that
            ### mask to fill our cosine_sim with the -infinity values 
            neg_equals_pos_mask = (quantized_codes == negative_quantized_codes).all(dim=-1)
            if neg_equals_pos_mask.any():
                cosine_sim[1:][neg_equals_pos_mask] = float("-inf")
            
            ### Finally we can compute our contrastive loss!! ###
            ### Lets take a close look at our contrastive loss objective again! We want to minimize the -log(softmax(e^cosine_sim)). Well, that kind of exactly cross entropy loss 
            ### does. Lets pretend we are doing classification of 10 things. That means we have 10 logits, which then get softmaxed and the prediction is the highest probability
            ### value. Cross entropy learns to maximize the probability of the correct answer and minimize the probability of the incorrect ones. 

            ### In our case there is only one thing, the correct answer is always 0! Remember, we have 1 correct + N distractors concatenated together. N is set to 100 by default, so
            ### we have a vector of cosine similarities of length 101. The correct answer is always 0 (the first cosine_similarity), because that is how we concatenated it together. 
            ### Remember, our mask_time_indices that we passed in has the shape (Batch x Sequence Length) and is a boolean that is True on masked tokens. We only want to compute the loss
            ### on these masked tokens specifically. PyTorch CrossEntropyLoss will ignore by default any label that is set to -100 so we can set up a label vector that is -100 for non-masked
            ### tokens and 0 (the correct answer) for all the masked tokens 

            ### So summary: We will take our (Batch x Sequence Length) mask_time_indices, and create a matrix of the same shape that is -100 where we are not masked and 0 for where it is masked.
            ### This matrix will then be flattened so we have our final vector of -100 and 0 values. The logits that go with every target value is of length 1 + Num_negatives, where the the index 0
            ### is the correct answer (as its the positive class we want to maximize the cosine similarity with) and the remaining num_negatives are incorrect. 

            ### Currently, we have our num_negatives + 1 as our first dimension, we need to make this the last dimension for cross entropy. This means we need to convert:
            ### cosine_sim -> (Num_negatives + 1 x Batch x Sequence Length) ---> (Batch x Sequence_length x Num_negatives + 1) --->   (Batch*Sequence_length x Num_negatives + 1)   
            ### target -> (Batch x Sequence Length) ---> (Batch*Sequence Length)
            ### The target will be -100 everywhere (so we dont compute loss for it) and 0 where we have a span mask

            cosine_sim = cosine_sim.permute(1,2,0).reshape(batch_size*sequence_length, num_negatives+1)
            labels = torch.ones(len(cosine_sim), dtype=torch.long, device=cosine_sim.device) * -100
            labels[mask_time_indices.flatten()] = 0

            ### Compute Contrastive Loss ###
            contrastive_loss = F.cross_entropy(cosine_sim, labels, reduction="sum")
            
            ### Compute Diversity Loss (Really the Perplexity Formula at the bottom of page 4) for only the Masked Tokens ###
            GV = self.config.num_codevector_groups * self.config.num_codevectors_per_group
            diversity_loss = ((GV - perplexity) / GV) * mask_time_indices.sum()

            ### This Loss Value is very Large (as we never took an average across the number of masked tokens, only the sums) ###
            ### This will be ok. Because we are randomly sampling the masked tokens, and because we will be most likely training ###
            ### on multiple GPUs, we dont really know ahead of time the number of tokens there are. So in this case, we can just ###
            ### Scale the gradients (divide the gradients by the total number of masked tokens across all GPUs) during training!! ###
            ### Im pretty sure we would have the same effect if we just averaged here, im just trying to stay close to the Huggingface ###
            ### implementation!!! ###
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_outputs,
            projected_quantized_states=quantized_codes,
            codevector_perplexity=perplexity,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )


class Wav2Vec2ForCTC(nn.Module):

    """
    The toughest part for Automatic Speech Recognition is learning the alignment between 
    two arbritarily long sequences: The raw audio and the characters in the sentence. CTCLoss
    is one such method that learns this alignment so we can actually do ASR. 

    This model can be initialized with our own pretrained backbone, but if you dont have a ton of
    GPU resources or time, you can use the Huggingface backbone as well!
    """

    def __init__(self, config):
        super().__init__()
        
        self.config = config

        ### Grab Backbone Based on Config ###
        self.load_backbone()

        ### Initialize Prediction Head ###
        self.dropout = nn.Dropout(config.asr_head_dropout_p)
        self.lm_head = nn.Linear(config.embedding_dimension, config.vocab_size)

    def load_backbone(self):
        
        if self.config.pretrained_backbone == "pretrained_huggingface":
            print(f"Loading Huggingface Wav2Vec2 Backbone: {self.config.hf_model_name}")
            self.wav2vec2 = HFWav2Vec2Model.from_pretrained(self.config.hf_model_name)
        else:
            self.wav2vec2 = Wav2Vec2Model(self.config)

            if self.config.pretrained_backbone == "pretrained":
                if self.config.path_to_pretrained_weights is None:
                    raise Exception("Provide the argument `path_to_pretrained_weights` in the config, else we cant load them!")
                else:
                    
                    if not os.path.isfile(self.config.path_to_pretrained_weights):
                        raise Exception(f"Provided path to safetensors weights {self.config.path_to_pretrained_weights} is invalid!")

                    print(f"Loading Wav2Vec2Model Backbone from {self.config.path_to_pretrained_weights}")

                    ### Load Weights with load_file from safetensors ###
                    state_dict = load_file(self.config.path_to_pretrained_weights)

                    ### Cleanup of Weights and keys ###
                    backbone_keys = {}
                    for key in state_dict.keys():

                        ### If Wav2Vec2 is in key name, just remove from the key name ###
                        if "wav2vec2" in key:
                            new_key = key.replace("wav2vec2.", "")
                            backbone_keys[new_key] = state_dict[key]

                        ### If wav2vec2 is not in key name, it isnt a part of the backbone so ignore it ###
                        else:
                            continue

                    ### Load State Dict to Backbone ###
                    self.wav2vec2.load_state_dict(backbone_keys)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        print("Freezing Convolutional Feature Encoder")
        if self.config.pretrained_backbone == "pretrained_huggingface":
            ### Huggingface already has a method to freeze model parameters ###
            self.wav2vec2.feature_extractor._freeze_parameters()
        elif self.config.pretrained_backbone == "pretrained":
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        elif self.config.pretrained_backbone == "random":
            raise Exception("Feature Encoder is Randomly initialized!!! You are disabling gradients, you need to train them!")
        else:
            raise ValueError(f"Inputed pretrained_backbone {self.config.pretrained_backbone} not in (pretrained, pretrained_huggingface, random)")
        
    def forward(
        self,
        input_values,
        attention_mask = None,
        labels = None):

        ### Our Pretrained model and Huggingface Wav2Vec2Model have slightly different forwards and returns ###
        if self.config.pretrained_backbone == "pretrained_huggingface":
            outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
            )

            hidden_states = outputs.last_hidden_state
        
        else:

            hidden_states = self.wav2vec2(input_values, 
                                          attention_mask, 
                                          return_features_to_quantize=False)

        ### Pass through Dropout and Compute Logits ### 
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        ### If labels are provided (already tokenized) we can compute our CTC Loss as well ###
        loss = None
        if labels is not None:

            ### If our Attention Mask is None, then attend to all tokens ###
            if attention_mask is None:
                attention_mask = torch.ones_like(input_values, dtype=torch.long)

            ### Compute Input Sizes of feature extracted audio via sub_attention_mask ###
            input_lengths = compute_encoded_lengths(attention_mask.sum(-1), self.config.conv_kernel, self.config.conv_stride).to(torch.long)

            ### Labels are -100 for padding tokens (as per our collate function), no need to keep for loss ###
            labels_mask = (labels >= 0)

            ### Add up nonpad tokens to see the number of tokens per sequence in batch for target sizes ###
            target_lengths = labels_mask.sum(-1)

            ### Grab nonpadded labels (CTC Loss can take flatten vector of (unpadded) inputs of shape (sum(target_lengths)) ###
            flattened_targets = labels.masked_select(labels_mask)

            ### CTC Loss takes log probs and doesnt work in mixed precision, make sure its float32 ###
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)

            ### CTC Loss expects (Sequence x Batch x Vocab Size) but we have (Batch x Sequence x Vocab size) ###
            log_probs = log_probs.transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction="mean",
                    zero_infinity=False,
                )

        return loss, logits

    
############################
### Weight Init Strategy ###
############################

def weight_init_strategy(config):

    def _init_weights(module):

        if isinstance(module, Wav2Vec2ForPreTraining):
            module.project_hid.reset_parameters()
            module.project_q.reset_parameters()

        elif isinstance(module, Wav2Vec2GumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)

        elif isinstance(module, Wav2Vec2PositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)

        elif isinstance(module, Wav2Vec2FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)

        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
    
    return _init_weights