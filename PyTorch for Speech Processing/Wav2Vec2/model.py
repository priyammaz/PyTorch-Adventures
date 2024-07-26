import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class Wav2Vec2Config:

    ### FEATURE ENCODER CONVOLUTION CONFIG ###
    audio_input_channels: int = 1
    conv_dimensions: tuple = (512, 512, 512, 512, 512, 512, 512)
    conv_strides: tuple = (5, 2, 2, 2, 2, 2, 2)
    conv_kernels: tuple = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = False
    feature_projection_dropout_p: float = 0.0

    ### POSITIONAL CONVOLUTIONAL EMBEDDING ###
    conv_positional_emb_drop: float = 0.1
    conv_positional_emb_groups: int = 16
    conv_positional_emb_kernel_size: int = 128

    ### TRANSFORMER CONFIG ###
    num_transformer_layers: int = 12
    num_attention_heads: int = 12
    embedding_dimension: int = 768
    attention_type: str = "flash"
    mlp_ratio: int = 4
    mlp_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    transformer_encoder_dropout: float = 0.1
    layer_dropout: float = 0.1
    initializer_range: float = 0.02

    ### GUMBEL SOFTMAX CONFIG ###
    num_codevector_groups: int = 2
    num_codevectors_per_group: int = 320
    codevector_dim: int = 256
    pre_quantizer_dropout: float = 0.0

    ### MASKING CONFIG ###
    masking_probability: float = 0.065
    masking_span_length: int = 10 
    minimum_spans: int = 2

    ### LOSS CONFIG ###
    contrastive_loss_temperature: float = 0.1
    diversity_loss_weight: float = 0.1

    ### TRAINING CONFIG ###
    num_negatives: int = 100

@dataclass
class Wav2Vec2PreTrainingOutput:
    
    loss: torch.FloatTensor = None
    contrastive_loss: torch.FloatTensor = None
    diversity_loss: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    quantized_state: torch.FloatTensor = None

class Wav2Vec2ConvLayerNormBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 bias: bool): 
        
        super(Wav2Vec2ConvLayerNormBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, 
                              stride=stride,
                              bias=True)
        
        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-6)

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
    def __init__(self, config):
        super(Wav2Vec2FeatureEncoder, self).__init__()

        self.config = config

        assert len(config.conv_dimensions) == len(config.conv_strides) == len(config.conv_kernels), \
            "Check Config for same number of convolution components" 
        
        number_of_conv_blocks = len(config.conv_kernels)
        conv_channels = (config.audio_input_channels, ) + config.conv_dimensions

        self.conv_layers = nn.ModuleList()
        for conv_idx in range(number_of_conv_blocks):
            self.conv_layers.append(
                
                Wav2Vec2ConvLayerNormBlock(in_channels=conv_channels[conv_idx],
                                           out_channels=conv_channels[conv_idx + 1],
                                           kernel_size=config.conv_kernels[conv_idx],
                                           stride=config.conv_strides[conv_idx],
                                           bias=config.conv_bias)
            )

    def forward(self, x):

        ### Inputs in the Shape (batch, 1 channel, audio_length)

        for layer in self.conv_layers:
            x = layer(x)

        return x
        

class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self, config):

        super(Wav2Vec2FeatureProjection, self).__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dimensions[-1])
        self.projection = nn.Linear(config.conv_dimensions[-1], config.embedding_dimension)
        self.dropout = nn.Dropout(config.feature_projection_dropout_p)
    
    def forward(self, x):
        
        ### We need to store Normed X (before projection) for quantization ###
        normed_x = self.layer_norm(x)

        ### Project X to Transformer Embedding Dimension ###
        projected_x = self.projection(normed_x)
        projected_x = self.dropout(projected_x)

        return projected_x, normed_x
    
class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super(Wav2Vec2PositionalConvEmbedding, self).__init__()

        ### Define Convolution that computes Positional Encoding ###
        self.conv = nn.Conv1d(
            in_channels=config.embedding_dimension, 
            out_channels=config.embedding_dimension, 
            kernel_size=config.conv_positional_emb_kernel_size, 
            padding=config.conv_positional_emb_kernel_size // 2,
            groups=config.conv_positional_emb_groups
        )

        ### Use Weight Normalization (https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html) ###
        ### The paper doesnt mention anything about this, but both Huggingface and Fairseq have it so ill have it too! ###
        ### This was introduced in a paper (https://arxiv.org/pdf/1602.07868) from OpenAI, where instead of learning the weight vectors directly
        ### we will instead decouple the vector into its magnitude and direction and learn them seperately. 
        ### We are using a Grouped 1D Convolution, so the weight tensor shape is (Batch x out_channels//num_groups x kernel_size)
        ### We want to compute ther weight norm over the kernel values for each convolutional filter, so we will do it over dim=2 
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)      

        self.activation = nn.GELU()

    def forward(self, x):

        ### x has the shape (Batch x Sequence x Embeddings) but Convolution wants sequence last ###
        x = x.transpose(1,2)

        ### Compute Positional Encodings ###
        positional_embeddings = self.conv(x)

        ### Clip any excess positional encodings ###
        positional_embeddings = positional_embeddings[:, :, :-x.shape[-1]]

        ### Activation and Transpose ###
        positional_embeddings = self.activation(positional_embeddings)
        positional_embeddings = positional_embeddings.transpose(1,2)

        return positional_embeddings

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        
        ### Store Config ###
        self.config = config
        
        ### Sanity Checks ###
        assert config.embedding_dimension % config.num_attention_heads == 0, "Double check embedding dim divisible by number of heads"
        assert config.attention_type in ["flash", "vanilla"], "Attention computations limited to either 'flash' or 'vanilla' attention"

        if config.attention_type == "vanilla":
            self.attn_drop = nn.Dropout(config.attention_dropout_p)

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
        q = self.q_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2)
        
        ### Compute Attention ###
        if self.config.attention_type == "flash":
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1)
            attention_out = F.scaled_dot_product_attention(q, k, v, 
                                                           attn_mask=attention_mask, 
                                                           dropout_p=self.config.attention_dropout_p if self.training else 0.0)

        elif self.config.attention_type == "vanilla":
            attention_out = (q @ k.transpose(-2,-1)) * (self.head_dim ** -0.5)
            
            if attention_mask is not None:

                # Attention Mask is in shape (B x L x L), need to add an extra dimension for the heads
                attention_mask = attention_mask.unsqueeze(1)
                attention_out = attention_out.masked_fill(~attention_mask.bool(), float("-inf"))
            
            attention_out = attention_out.softmax(dim=-1)
            attention_out = self.attn_drop(attention_out)
            attention_out = attention_out @ v

        ### Compute Output Projection ###
        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out)

        return attention_out

class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config):
        super(Wav2Vec2FeedForward, self).__init__()

        hidden_dimension = config.embedding_dimension * config.mlp_ratio

        self.fc1 = nn.Linear(config.embedding_dimension, hidden_dimension)
        self.activation = nn.GELU()
        self.dropout_1 = nn.Dropout(config.mlp_dropout_p)

        self.fc2 = nn.Linear(hidden_dimension, config.embedding_dimension)
        self.dropout_2 = nn.Dropout(config.mlp_dropout_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.fc2(x)
        x = self.dropout_2(x)
        return x
    
class Wav2Vec2TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(Wav2Vec2TransformerEncoderLayer, self).__init__()
        
        self.attention = Attention(config)
        self.dropout = nn.Dropout(config.transformer_encoder_dropout)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)

    def forward(self, x, attention_mask=None):
        
        ### Attention + Residual ###
        x_residual = x
        x = self.attention(x, attention_mask=attention_mask)
        x = self.dropout(x)
        x = x + x_residual
        x = self.layer_norm(x)

        ### Feed Forward + Residual ###
        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)

        return x

class Wav2Vec2TransformerEncoder(nn.Module):
    def __init__(self, config):

        super(Wav2Vec2TransformerEncoder, self).__init__()

        self.config = config 

        ### Convolutional Positional Embedding ###
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)
        self.dropout = nn.Dropout(config.conv_positional_emb_drop)

        ### Transformer Layers ###
        self.layers = nn.ModuleList(
            [
                Wav2Vec2TransformerEncoderLayer(config) for _ in range(config.num_transformer_layers)
            ]
        )

    def forward(self, x, attention_mask=None):
        
        ### Compute Positional Embeddings on Data ###
        positional_encoding = self.pos_conv_embed(x)
        x = x + positional_encoding
        x = self.layer_norm(x)
        x = self.dropout(x)

        ### Pass Data through all Transformer Encoder Layers ###
        for layer in self.layers:

            ### Random Sample for Layer Dropout ###
            dropout_prob = torch.rand(1)

            ### If we are in training mode, and our uniformly sampeld number less than layer_drop, we skip the layer ###
            ### If we are not in training mode, then use all layers ###
            if (not self.training) or (self.training and (dropout_prob >= self.config.layer_dropout)):
                x = layer(x, attention_mask)

        return x

class GumbelSoftmaxQuantizer(nn.Module): 
    def __init__(self, config):
        super(GumbelSoftmaxQuantizer, self).__init__()

        self.vq_dim = config.codevector_dim
        self.num_groups = config.num_codevector_groups
        self.num_codes_per_group = config.num_codevectors_per_group

        assert self.vq_dim % self.num_groups == 0 , "Make sure your VQ Dim is divisible by the number of Codevector Groups!"

        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups*self.num_codes_per_group, self.vq_dim // self.num_groups).uniform_()
        )

        ### Projection from output of Convolutions to num_groups*num_codes_per_group
        self.q_proj = nn.Linear(config.conv_dimensions[-1], self.num_groups*self.num_codes_per_group)
    
        ### Set a GumbelSoftmax Temperature ###
        self.temperature = 2.0

    @staticmethod
    def _compute_perplexity(probs, masked_token_idx=None):
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
        if masked_token_idx is not None:

            ### Mask Token Index is (Batch x Sequence) where we have True values on the masked sequence values ###
            ### Probs is (Batch*Sequence x num_codebook_groups x codes_per_group) ###
            ### We just need to filter out for the sequence values that are actually masked in our probs! ###
            probs = probs[masked_token_idx.flatten()]
            probs = probs.mean(axis=0)
        
        else:

            probs = probs.mean(axis=0)

        ### This is the main part of the equation at the bottom of page 4 of the paper! ###
        ### First we want to compute the perplexity per codebook ###
        perplexity_per_codebook = torch.exp(-torch.sum(probs * torch.log(probs + 1e-7), dim=-1))
        
        ### Add together perplexities from each codebook ###
        perplexity = perplexity_per_codebook.sum()

        return perplexity


    def forward(self, 
                x, 
                masked_token_idx=None):

        batch_size, sequence_length, hidden_size = x.shape

        ### OK so this is fun, lets say we have 2 codebooks, each with 300 vectors, and each vector is of size 128 ###
        ### For each token along our sequence  in x, we will first map it to our 600 posibilities (2 * 300 total vectors) ###
        x = self.q_proj(x)
        
        ### Reshape x so we have all the dimensions by the num_codes_per_group ###
        x = x.reshape(batch_size*sequence_length*self.num_groups, self.num_codes_per_group)

        ### If we are in Training model then we will use GumbelSoftmax (differentiable indexing) ###
        if self.training:

            ### GumbelSoftmax will give us a probability vector across the 300 possible codes. ###
            ### These probabilities are scaled by the temperature parameter ###
            ### Setting hard to true will only return back a value of 1 at the index of the most likely code! ###
            codevector_probs = nn.functional.gumbel_softmax(
                x, tau=self.temperature, hard=True
            ).type_as(x)

            ### Compute Perplexity ###
            codevector_softmax_probs = x.reshape(batch_size * sequence_length, self.num_groups, self.num_codes_per_group).softmax(axis=-1)
            perplexity = self._compute_perplexity(codevector_softmax_probs, masked_token_idx=masked_token_idx)
            
        ### If we are not training (no need for differentiable indexing) then we can just use argmax! ###
        else:
            ### For each input vector in the sequence, just grab the index of the most likely code! ##
            codevector_idx = x.argmax(dim=-1)

            ### Codevector Probs will just be a 1 for the indexes of the most likely codes and zeros elsewhere ###
            codevector_probs = torch.zeros_like(x)
            codevector_probs[torch.arange(x.shape[0]), codevector_idx] = 1

            ### Reshape Codevector Probs to compute Perplexity ###
            ### In this case, during inference, we have basically 100% for the codevector selected and 0 percent for everything else ###
            codevector_probs = codevector_probs.reshape(batch_size*sequence_length, self.num_groups, self.num_codes_per_group)
            
            ### Compute Perplexity ###
            perplexity = self._compute_perplexity(codevector_probs, masked_token_idx=masked_token_idx)

        ### We will now reshape the codevector probs to be (batch*sequence, num_groups*codes_per_group) ###
        codevector_probs = codevector_probs.reshape(batch_size*sequence_length, -1)

        ### Our codevectors have shape (1 x num_groups*codes_per_group * vq_dim//num_groups) ###
        ### We can add an extra dimension to our codevector_probs so it is (batch*sequence, num_groups*codes_per_group, 1) ###
        ### We can then multiply out codevector probs (which is just 1 for the selected code and 0 otherwise) ###
        ### In our codevector probs, if we have 2 groups then there are 2 indexes as 1 and the rest are zeros ###
        ### So then if we multiply our codevector probs by the the codevectors, then it will keep the 2 codevectors that were indexed ###
        ### as 1, and make everything else 0! ###
        selected_codes = codevector_probs.unsqueeze(-1) * self.codevectors
        
        ### We now will seperate out the 600 codes to be 2 x 300, and then add across the 300 dimension (as only one of those 300 codes are actually not zero!)###
        selected_codes = selected_codes.reshape(batch_size * sequence_length, self.num_groups, self.num_codes_per_group, self.vq_dim // self.num_groups)
        selected_codes = selected_codes.sum(axis=2)

        ### Now for the cool part, currently we have 2 codes for each intial input token, and each are of length 128 (assuming the final VQ dim is 256 and we had 2 groups) ###
        ### In total, we have 600 codes, which is not enough to represent a large diverse set of quantized latents, so instead, we will concatenate them together! This way ###
        ### We will have a total of 300 * 300 possibilities! ###
        selected_codes = selected_codes.reshape(batch_size, sequence_length, -1)

        return selected_codes, perplexity


class Wav2Vec2Model(nn.Module):
    def __init__(self, config):
        super(Wav2Vec2Model, self).__init__()

        ### Store Config ###
        self.config = config

        ### Load Convolutional Feature Extractor and Conv2Transformer Projection Layer ###
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        ### If we are masking, we need a learnable masking token ####
        if self.config.masking_probability > 0: 
            ### So apparently torch.tensor and torch.Tensor are different, so just noting this!
            ### torch.tensor infers the dtype automatically, while torch.Tensor returns a torch.FloatTensor
            ### We want a float tensor so we will use torch.Tensor
            self.mask_token = nn.Parameter(torch.Tensor(config.embedding_dimension).uniform_())
        
        ### Define Transformer Layers ###
        self.transformer_encoder = Wav2Vec2TransformerEncoder(config)

    def forward(self, 
                input_values, 
                attention_mask=None, 
                masked_token_idx=None,
                return_features_to_quantize=False):
        
        ### Extract Features w/ Convolutions and then transpose to channels last ###
        extracted_features = self.feature_extractor(input_values)
        extracted_features = extracted_features.transpose(1,2)

        ### Project Extracted Features to Transformer Embedding Dimension ###
        ### This is also where we get the normed output of our convolutions for future quantization ###
        extracted_features, features_to_quantize = self.feature_projection(extracted_features)

        ### Replace Tokens with self.mask_token where we have our span masks ###
        if masked_token_idx is not None:
            extracted_features[masked_token_idx] = self.mask_token.to(dtype=extracted_features.dtype)

        ### Pass through transformer ###
        output = self.transformer_encoder(extracted_features, attention_mask=attention_mask)

        if return_features_to_quantize:
            return output, features_to_quantize        
        else:
            return output
    
class Wav2Vec2ForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        ### Define Full Wav2Vec2 Model ###
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.pre_quantizer_dropout)

        ### Define Quantizer ###
        self.quantizer = GumbelSoftmaxQuantizer(config)

        ### Projection from Transformer Embedding Size to VQ Dimension ###
        self.transformer_to_vq = nn.Linear(config.embedding_dimension, config.codevector_dim)

        ### Quantizer Projection ###
        self.vq_proj = nn.Linear(config.codevector_dim, config.codevector_dim)

        ### Initialize Weights ###
        self.apply(self._init_weights)

    def _init_weights(self, module):
        
        if isinstance(module, Wav2Vec2ForPreTraining):
            ### Default Initialization for Projections ###
            self.transformer_to_vq.reset_parameters()
            self.vq_proj.reset_parameters()

        elif isinstance(module, Wav2Vec2Model):
            ### Uniform Initialization of Mask Token ###
            module.mask_token.data.uniform_()

        elif isinstance(module, GumbelSoftmaxQuantizer):
            module.q_proj.weight.data.normal_(mean=0.0, std=1.0)
            module.q_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)

        elif isinstance(module, Wav2Vec2PositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight, 
                mean=0, 
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels))
            )
            nn.init.constant_(module.conv.bias, 0)

        elif isinstance(module, Wav2Vec2FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)

        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.kernel_size[0] * module.in_channels))
                nn.init.uniform_(module.bias, a=-k, b=k)

    @staticmethod
    def _compute_cosine_similarity(true_quantized, 
                                   negative_quantized, 
                                   transformer_output, 
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

        true_quantized = true_quantized.unsqueeze(0)
        targets = torch.cat([true_quantized, negative_quantized], dim=0) 

        ### Compute Cosine Similarity between our transformer output and targets, along the VQ_dimension (which is the last one) ###
        ### If you take a quick look at the PyTorch cosine similarity function (https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html)
        ### we just needt to make sure our shapes are broadcastable to a common shape, but we already made this happen because our
        ### transformer output is (batch x sequence x vq_dim) and the targets are (Num_negatives + 1 x Batch x Sequence Length x VQ_dim)
        ### So each sample in our Batch x sequence will compute its cosine similarity across all Num negatives + 1 quantized tokens. 
        ### This operation will return the cosine similarity in the shape of (Num_negatives + 1 x Batch x Sequence Length)
        cosine_sim = torch.cosine_similarity(transformer_output, targets, dim=-1)

        ### Now in formula 4, we see that there is a softmax involved. We can actually reformulate this problem easily in terms of CrossEntropyLoss that I explain below ###
        ### Torch CrossEntropyLoss expects logits, so we wont do any softmax now. But, we can scale our cosine similarity by our temperature parameter! 
        cosine_sim = cosine_sim / temperature

        return cosine_sim
    
    def update_gumbel_temperature(self, temperature):
        self.quantizer.temperature = temperature
        
    def forward(self, 
                input_values, 
                attention_mask=None, 
                masked_token_idx=None, 
                sampled_negative_indices=None):

        ### Grab Outputs and Features to Quantize from Wav2Vec2 ###
        outputs, features_to_quantize = self.wav2vec2(input_values, 
                                                      attention_mask=attention_mask, 
                                                      masked_token_idx=masked_token_idx, 
                                                      return_features_to_quantize=True)
        
        ### Project Outputs to VQ Dim ###
        transformer_vq = self.transformer_to_vq(outputs)

        ### Quantize the Masked Features ###
        quantized_codes, perplexity = self.quantizer(features_to_quantize, masked_token_idx=masked_token_idx)
    
        ### Project Quantized Features ###
        quantized_codes = self.vq_proj(quantized_codes)

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
            negative_quantized_codes = negative_quantized_codes.reshape(
                batch_size, sequence_length, num_negatives, vq_size
                ).permute(2,0,1,3)
            
            ### Now that we have our Negative Quantized features, its time to compute contrastive loss! ###
            ### We will basically follow equation 3 from the paper! First we need to compute the Cosine Similarity ###
            ### between a masked output from our transformer (transformer_vq), its true quantized latent from our codebook and the K distractors ###
            ### The goal of the contrastive loss is to increase the similarity between a transformer output and its quanitzed feature, and decrease the ###
            ### similarity to the distractors.
            cosine_sim = self._compute_cosine_similarity(true_quantized=quantized_codes, 
                                                         negative_quantized=negative_quantized_codes,
                                                         transformer_output=transformer_vq,
                                                         temperature=self.config.contrastive_loss_temperature)
            
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
            ### Remember, our masked_token_idx that we passed in has the shape (Batch x Sequence Length) and is a boolean that is True on masked tokens. We only want to compute the loss
            ### on these masked tokens specifically. PyTorch CrossEntropyLoss will ignore by default any label that is set to -100 so we can set up a label vector that is -100 for non-masked
            ### tokens and 0 (the correct answer) for all the masked tokens 

            ### So summary: We will take our (Batch x Sequence Length) masked_token_idx, and create a matrix of the same shape that is -100 where we are not masked and 0 for where it is masked.
            ### This matrix will then be flattened so we have our final vector of -100 and 0 values. The logits that go with every target value is of length 1 + Num_negatives, where the the index 0
            ### is the correct answer (as its the positive class we want to maximize the cosine similarity with) and the remaining num_negatives are incorrect. 

            ### Currently, we have our num_negatives + 1 as our first dimension, we need to make this the last dimension for cross entropy. This means we need to convert:
            ### cosine_sim -> (Num_negatives + 1 x Batch x Sequence Length) ---> (Batch x Sequence_length x Num_negatives + 1) --->   (Batch*Sequence_length x Num_negatives + 1)   
            ### target -> (Batch x Sequence Length) ---> (Batch*Sequence Length)
            ### The target will be -100 everywhere (so we dont compute loss for it) and 0 where we have a span mask

            cosine_sim = cosine_sim.permute(1,2,0).reshape(batch_size*sequence_length, num_negatives+1)
            labels = torch.ones(len(cosine_sim), dtype=torch.long, device=cosine_sim.device) * -100
            labels[masked_token_idx.flatten()] = 0

            ### Compute Contrastive Loss ###
            contrastive_loss = F.cross_entropy(cosine_sim, labels, reduction="sum")
            
            ### Compute Diversity Loss (Really the Perplexity Formula at the bottom of page 4) for only the Masked Tokens ###
            GV = self.config.num_codevector_groups * self.config.num_codevectors_per_group
            diversity_loss = ((GV - perplexity) / GV) * masked_token_idx.sum()

            ### This Loss Value is very Large (as we never took an average across the number of masked tokens, only the sums) ###
            ### This will be ok. Because we are randomly sampling the masked tokens, and because we will be most likely training ###
            ### on multiple GPUs, we dont really know ahead of time the number of tokens there are. So in this case, we can just ###
            ### Scale the gradients (divide the gradients by the total number of masked tokens across all GPUs) during training!! ###
            ### Im pretty sure we would have the same effect if we just averaged here, im just trying to stay close to the Huggingface ###
            ### implementation!!! ###
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        return Wav2Vec2PreTrainingOutput(
            loss=loss, 
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
            last_hidden_state=outputs,
            quantized_state=quantized_codes
        )


        
        
        


        



        

