"""
Implementation of RoBERTa! This is as close as possible (but much more simplified) 
version of the RoBERTa implementation from 🤗 Huggingface!

https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py


"""
import math
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
    
class RobertaLocalAttention(nn.Module):
    def __init__(self, config):
        super(RobertaLocalAttention, self).__init__()

        self.window_size = config.window_size
        self.causal = config.causal
        self.look_backward = config.look_backward
        self.look_forward = config.look_forward if not self.causal else 0

        self.qkv = nn.Linear(config.embedding_dimension, config.embedding_dimension*3)
        self.dropout = nn.Dropout(config.attention_dropout_p)

        self.embed_dim = config.embedding_dimension
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

    def collect_windows(self, x, backward=1, forward=1, pad_value=-1):
        
        if x.dim() == 4: ### Collection on K,V ###
            batch_heads, windows, window_len, embed_dim = x.shape

            ### Now a problem, if we want for every window in Q, we want to attend to not only the
            ### cooresponding window in K,V but also the previous window, then when we are on the very 
            ### first window, we have nothing to grab beforehand, therefore, we need to pad our data by
            ### The number of windows we want to grab before and after 
    
            ### in practice our tensor has n windows, if we want to grab one before and one after for each of them, 
            ### we need to have n+2 windows, where we have an entire padded window on the left and another on the right ###
            pad = (0,0,0,0,backward,forward)

        elif x.dim() == 3:

            ### This is incase our tensor doesnt have an embed dim (like for our index tensor we use) ###
            batch_head_dim, windows, window_len = x.shape
            pad = (0,0,backward,forward)

        ### Now we can go ahead and pad our tensor and add in those extra pad buckets ###
        x = torch.nn.functional.pad(x, pad=pad, value=-1)

        ### Lastly we have to now put it all together! Lets pretend we split our sequence in to 4 buckets and each bucket is of length 512. ###
        ### If we want to attend to the previous bucket, the corresponding bucket, and the future bucket, that means the 512 tokens in our Query ###
        ### Needs to attend to 3*512 tokens in our Keys. So how can we do this? Just some simple indexing will solve this!

        ### Lets allow our 4 buckets in our queries to have indexes [0,1,2,3]. If we are looking backward and forward by 1, then we have a single pad bucket
        ### on either side of our original buckets, so now we will have the buckets that look like [<PAD>,0,1,2,3,<PAD>] and with padded indexes [0,1,2,3,4,5].
        ### And again the data in the slice of our padded indexes [1,2,3,4] is identical to the original indexes [0,1,2,3]

        ### So in our original indexes [0,1,2,3], we want this pattern:
        ### bucket 0 attends to [<PAD>,0,1]  which have our padded indexes of [0,1,2]
        ### bucket 1 attends to [0,1,2] which have our padded indexes of [1,2,3]
        ### bucket 2 attends to [1,2,3] which have our padded indexes of [2,3,4]
        ### bucket 3 attends to [2,3,<PAD>] which have our padded indexes of [3,4,5]

        ### So we see the pattern then, for every index in our original buckets [0,1,2,3], we want
        ### the index to index + forward + backward
        
        ### At original index 0, we want indexes 0 to 0+1+1=2 in our padded indexes
        ### At original index 1, we want indexes 1 to 1+1+1=3 in our padded indexes
        ### At original index 2, we want indexes 2 to 2+1+1=4 in our padded indexes
        ### At original index 3, we want indexes 3 to 3+1+1=5 in our padded indexes
        
        ### Lastly, if we want to grab a slice of our buckets from bucket 0 to bucket 2, we actually need to
        ### grab bucket 0 to bucket 3, as python is not right inclusive, that is why you see the extra end_windows+1 term

        gathered = []
        for i in range(windows):

            ### Grab the starting and end window index ###
            start_window = i
            end_window = i + forward + backward

            ### Slice the windows (with the extra +1 term because python isnt right inclusive ###
            grabbed_windows = x[:, start_window:end_window+1]

            ### Each grabbed_windows is of size (Batch_heads x forward+backward+1 x window_size x embed_dim ###
            ### forward+backward+1 is the number of consective buckets we want to concat together ###
            ### If we are looking forward and backward by 1, then we only have 3 consective buckets each of size 512 in our case ###
            ### We can put this all together and flatten to create a single sequence of length 3*512=1536 ###
            ### We then need to unsqueeze to add the bucket dimension back to concat on in the future, so the final shape of a single
            ### grabbed_window is (Batch x 1 x num_buckets*window_length x embed_dim)
            grabbed_windows = grabbed_windows.flatten(1,2).unsqueeze(1)
            gathered.append(grabbed_windows)

        ### What we have now is for each input bucket (we have 4 in the example), we grabbed the consecutive buckets and stuck them together ###
        ### so we have a list of 4 tensors, each of which are (Batch x 1 x num_buckets*window_length x embed_dim). We can then concat them all ###
        ### together on the first dimension to give us a final (Batch x 4 x num_buckets*window_length x embed_dim). ###
        ### This gives us exactly what we want then! For each input bucket, we grabbed backward number of buckets, forward number of buckets, and ###
        ### stacked it all together so when we compute attention, our input queries have a larger receptive field looking into neighboring buckets ###
        
        gathered = torch.cat(gathered, dim=1)

        return gathered
        
        
    def forward(
        self,
        x,
        attention_mask = None
    ):

        ### Project Q,K,V ###
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        ### Grab Info on tensors ###
        batch, orig_seq_len, embed_dim = q.shape
        device = q.device

        
        ### Reshape ###
        q = q.reshape(batch, orig_seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.reshape(batch, orig_seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(batch, orig_seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)


        ### Merge together Head/Batch Dimension ###
        q = q.flatten(0,1)
        k = k.flatten(0,1)
        v = v.flatten(0,1)

        if attention_mask is not None:
            ### We have a mask per sample, lets repeat it so we have it once per batch_head ###
            attention_mask = attention_mask.repeat(self.num_heads, 1)

        ### Need to make sure our sequence length is divisible by the window size ###
        if orig_seq_len % self.window_size != 0:
            diff = self.window_size * math.ceil(orig_seq_len/self.window_size) - orig_seq_len
            q = torch.nn.functional.pad(q, pad=(0,0,0,diff))
            k = torch.nn.functional.pad(k, pad=(0,0,0,diff))
            v = torch.nn.functional.pad(v, pad=(0,0,0,diff))

        ### Store final seq len (same as orig_seq_len if no padded was needed) ###
        seq_len = q.shape[1]

        ### Compute Number of Windows ###
        num_windows = seq_len // self.window_size

        ### Create Index for Sequence so we can chop it up into our windows (and add dummy batch_head dim) ###
        t = torch.arange(seq_len, device=device)
        bucketed_t = t.reshape(1, num_windows, self.window_size) 

        ### Bucket our Q,K,V into the Chunked Windows ###
        bucketed_q = q.reshape(batch*self.num_heads, num_windows, self.window_size, self.head_dim)
        bucketed_k = k.reshape(batch*self.num_heads, num_windows, self.window_size, self.head_dim)
        bucketed_v = v.reshape(batch*self.num_heads, num_windows, self.window_size, self.head_dim)

        ### Now for the fun part!!! What we have is our queries as (B*num_heads x num_windows x window_size x head_dim). ###
        ### Our Keys and Values are also the same shape currently ###
        ### So if we did attention directly, on a per-window basis, then we will repeat a window_size ** 2 operation, num_windows times ###
        ### Although, we can expand the receptive field a little as well, what if we want one window in our query to look at not only the ###
        ### cooresponding window in the keys, but also the window before and after? Or two windows before and after? Or only the window before to keep causality? ###
        ### Therefore, we need to expand our keys and values from (B*num_heads x num_windows x window_size x head_dim) to (B*num_heads x num_windows x k*window_size x head_dim)
        ### where k is how many consective windows we want to compute attention against. 
        bucketed_k = self.collect_windows(bucketed_k, self.look_backward, self.look_forward)
        bucketed_v = self.collect_windows(bucketed_v, self.look_backward, self.look_forward)

        ### Lets also apply this collection to our indexes so we know where everything is, along with where our pad tokens are ###
        collected_bucketed_t = self.collect_windows(bucketed_t, self.look_backward, self.look_forward)

        ### Create Padding Mask where collected_bucketed_t is -1 ### 
        bucket_pad_mask = (collected_bucketed_t == -1)

        ### Compute Raw Scores between our Q and K ###
        ### Again, remember Q has 4 buckets in our example and each bucket is of length 512 ###
        ### It is being attended to by the previous bucket and the future bucket, so we have 512 * 3 token in our keys ###
        ### This means our attention matrix should be of size 512 x 1536 ###
        attention_scores = bucketed_q @ bucketed_k.transpose(-1,-2)

        ### Our Bucket mask has shape (1 x num_buckets (4) x key length (1536) ###
        ### Our attention scores are (batch_heads x num_buckets (4) x query_length (512) x key_length (1536) ###
        ### The pad masks needs to repeat for all the queries as well, as we dont need the padded key for any of queries ###
        ### So lets add a dimension and repeat our mask so we have a final mask of (1 x num_buckets (4) x query_length (512) x key_length (1536) ###
        ### Because this mask can be broadcasted across our attention mask we can use it to fill with -inf before softmax ###
        bucket_pad_mask = bucket_pad_mask.unsqueeze(-2).repeat(1,1,self.window_size,1)

        ### For the same reason as above, lets expand and repeat the collected_bucketed_t so we know for every query index, which keys were computed against ###
        collected_bucketed_t = collected_bucketed_t.unsqueeze(-2).repeat(1,1,self.window_size,1)

        ### If we are Non Causal then we need to deal with looking forward and backward ###
        if not self.causal:
     
            ### Now this is optional, but technically we are overcomputing right now ###
            ### We are attending by blocks, but if our window size is 3 and we are looking forward and backward by a single block ###
            ### this mean the first token (index 0) is attending to 3 pad tokens before and 6 tokens going forward. If we want our window ###
            ### to be exactly 3 before and 3 after in each index, then we are overcomputing too many! In this figure lets also say the input lenght is 
            ### 12, so the 12 tokens were chunked into 4 chunks each of size 3 
    
            ### Our collected_bucketed_t is of the shape (1 x Bucket x window_size x (forward+backward+1)*window_size
            ### This is what it looks like:
            # tensor([[[[-1, -1, -1,  0,  1,  2,  3,  4,  5],
            #           [-1, -1, -1,  0,  1,  2,  3,  4,  5],
            #           [-1, -1, -1,  0,  1,  2,  3,  4,  5]],
            
            #          [[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
            #           [ 0,  1,  2,  3,  4,  5,  6,  7,  8],
            #           [ 0,  1,  2,  3,  4,  5,  6,  7,  8]],
            
            #          [[ 3,  4,  5,  6,  7,  8,  9, 10, 11],
            #           [ 3,  4,  5,  6,  7,  8,  9, 10, 11],
            #           [ 3,  4,  5,  6,  7,  8,  9, 10, 11]],
            
            #          [[ 6,  7,  8,  9, 10, 11, -1, -1, -1],
            #           [ 6,  7,  8,  9, 10, 11, -1, -1, -1],
            #           [ 6,  7,  8,  9, 10, 11, -1, -1, -1]]]])
    
            ### The above tensor shows that:
            ### Query index [0,1,2] was mulitplied by key index [-1-1-1,0,1,2,3,4,5]
            ### Query index [3,4,5] was mulitplied by key index [0,1,2,3,4,5,6,7,8]
            ### Query index [6,7,8] was mulitplied by key index [3,4,5,6,7,8,9,10,11]
            ### Query index [9,10,11] was mulitplied by key index [6,7,8,9,10,11,-1,-1,-1]
            ### Where -1 is just our chunk padding explained `collect_windows`
            
            ### So for query index 0 we compute again against these key indexes:
            ### [-1,-1,-1,0,1,2,3,4,5]
            ### This means query 0 is computing against 3 tokens before it, but 5 tokens after it
            ### technically all we want to do is compute 3 tokens before (-1,-1,-1) and 3 tokens after (1,2,3), so we need to mask token 4 and 5
    
            ### For query index 1, we again compute against the same key indexes as query index 0:
            ### [-1,-1,-1,0,1,2,3,4,5]
            ### What we really want is to attend to 3 tokens before (-1,-1,0) and 3 tokens after 1 (2,3,4), so we need to mask token 5 only
    
            ### So how can we do this operation quickly? We can create a repeated query index first, so for every row in in every bucket, we can ###
            ### have a row of the index of which query was being multiplied ###
            
            ### This is what the repeated_query_index looks like
            # tensor([[[[ 0,  0,  0,  0,  0,  0,  0,  0,  0],
            #           [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
            #           [ 2,  2,  2,  2,  2,  2,  2,  2,  2]],
            
            #          [[ 3,  3,  3,  3,  3,  3,  3,  3,  3],
            #           [ 4,  4,  4,  4,  4,  4,  4,  4,  4],
            #           [ 5,  5,  5,  5,  5,  5,  5,  5,  5]],
            
            #          [[ 6,  6,  6,  6,  6,  6,  6,  6,  6],
            #           [ 7,  7,  7,  7,  7,  7,  7,  7,  7],
            #           [ 8,  8,  8,  8,  8,  8,  8,  8,  8]],
            
            #          [[ 9,  9,  9,  9,  9,  9,  9,  9,  9],
            #           [10, 10, 10, 10, 10, 10, 10, 10, 10],
            #           [11, 11, 11, 11, 11, 11, 11, 11, 11]]]])
            
            num_concat_windows = (self.look_backward + self.look_forward + 1) 
            repeated_query_index = t.reshape(1,-1,self.window_size,1).repeat(1,1,1,self.window_size*num_concat_windows)
            
            ### Next we can compute the number of tokens we want to look forward and backward ###
            total_look_backward = (self.window_size * self.look_backward)
            total_look_forward = (self.window_size * self.look_forward)
    
            ### We can then add the total we want to look forward and backward to our repeated query index. This will produce the ###
            ### the max index of every row, and any index larger than that must be masked out 
            ### and the min index of every row, and any index smaller than that must be masked out 
            max_index = repeated_query_index + total_look_forward
            min_index = repeated_query_index - total_look_backward
    
            ### When creating these masks, the only extra thing is we need to ignore the -1 as those arent really indexes, just placeholders ###
            ### and they are already masked out anyway from the bucket_pad_mask, so we can ignore it for our overcompute_mask 
            upper_mask = ((collected_bucketed_t>max_index) & (collected_bucketed_t != -1))
            lower_mask = ((collected_bucketed_t<min_index) & (collected_bucketed_t != -1))
            overcompute_mask = upper_mask | lower_mask

            ### Create a Casual Mask with all False (as no causality in encoder attention ###
            causal_mask = torch.zeros_like(attention_scores, device=device).bool()

        ### If we are causal then we only need to look backward! ###
        else:

            ### As explained above, we need our repeated query index ###
            num_concat_windows = (self.look_backward + self.look_forward + 1) 
            repeated_query_index = t.reshape(1,-1,self.window_size,1).repeat(1,1,1,self.window_size*num_concat_windows)

            ### Now, we dont need any key value that is greater than our repeated query index to maintain causality ###
            causal_mask = (collected_bucketed_t>repeated_query_index)

            ### Everything below is exactly what we did above, just only for looking backward because we only are looking back ###
            total_look_backward = (self.window_size * self.look_backward)
            min_index = repeated_query_index - total_look_backward
            overcompute_mask = ((collected_bucketed_t<min_index) & (collected_bucketed_t != -1))


        ### We also may have pad tokens in our data, this is our regular attention mask, we need to handle this too! ####
        if attention_mask is not None:

            ### First Pad to Multiple if needed ###
            if orig_seq_len % self.window_size != 0:
                diff = self.window_size * math.ceil(orig_seq_len/self.window_size) - orig_seq_len
                attention_mask = torch.nn.functional.pad(attention_mask, pad=(0,diff), value=-1)

            ### Chunk into the Buckets ###
            attention_mask = attention_mask.reshape(batch*self.num_heads, num_windows, self.window_size)

            ### Collect Windows ###
            attention_mask = self.collect_windows(attention_mask, self.look_backward, self.look_forward)

            ### Expand to window size and Repeat ###
            attention_mask = attention_mask.unsqueeze(-2).repeat(1,1,self.window_size,1)

            ### Mask out wherever we have a 0 value in our attention mask ###
            attention_mask = (attention_mask == 0)

        else:

            attention_mask = torch.zeros_like(attention_scores, device=device).bool()

        ### Lets put all of our masks together! ###
  
        combined_mask = (overcompute_mask | attention_mask)
        if self.causal:
            combined_mask = (combined_mask | causal_mask)

        ### Fill with -inf for softmax computation ###
        attention_scores = attention_scores.masked_fill(combined_mask, float("-inf"))

        ### Compute Softmax ###
        attention_scores = attention_scores.softmax(dim=-1)

        ### Incase a full row was zeroed out, it will return nan during softmax, we can just set them to 0 ###
        if torch.any(torch.isnan(attention_scores)):
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=0, neginf=0) 

        ### Dropout on Attention ###
        attention_scores = self.dropout(attention_scores)
        
        ### Multiply by the Values ###
        output = attention_scores @ bucketed_v

        ### Return back to original shape ###
        output = output.reshape(batch, self.num_heads, -1, self.head_dim)

        ### Clip extra tokens if we had to pad to multiple and return back to original shape ###
        output = output[:,:,:orig_seq_len].permute(0,2,1,3).flatten(2)
    
        return output

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

        self.attention = RobertaLocalAttention(config)
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
            
        for layer in self.layers:

            x = layer(x, attention_mask=attention_mask)

        return x

class RobertaMLMHead(nn.Module):

    """
    The Masked Language model head is a stack of two linear layers with an activation in between!
    """

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

    """
    Backbone of our model, has to be pretrained via MLM on a ton of data!
    """

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

    """
    This model will perform the masked language modeling task. 
    """

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

    """
    RoBERTa model for Extractive Question answering. The inputs to this are sequence of tokens
    that are the question you want to ask and the context for the question the answer from. The
    goal of this model is to predict the start and end token of context that includes the answer
    to the question we have. 
    """

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

                        ### If roberta is in key name, just remove from the key name ###
                        if "roberta" in key:
                            new_key = key.replace("roberta.", "")
                            backbone_keys[new_key] = state_dict[key]

                        ### If roberta is not in key name, it isnt a part of the backbone so ignore it ###
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

        else:
            outputs = self.roberta(input_ids, attention_mask)

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
