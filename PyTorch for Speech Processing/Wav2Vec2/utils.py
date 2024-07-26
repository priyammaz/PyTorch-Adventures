import numpy as np
import torch
import torch.nn as nn

def feature_encoder_output_length(lengths: torch.Tensor, kernel_sizes: tuple, strides: tuple):
    """
    To perform masking AFTER the encoding 1D Convolutions, we need to 
    compute what the shape of the output tensor is after each successive convolutions
    is applied.

    Convolution formula can be found in PyTorch Docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

    Args:

        lengths: Tensor of the lengths of each audio in batch
        kernel_sizes: Tuple of all the kernel size for each convolution
        strides: Tuple of all the strides for each convolution
    """

    assert len(kernel_sizes) == len(strides), "Double check that you have defined a kernel size and stride for each convolution!"

    ### Convolution Ouput Formula ###
    def _compute_conv_out(lengths, kernel_size, stride):
        return torch.floor((lengths - (kernel_size - 1) - 1)/stride) + 1
    
    for k, s in zip(kernel_sizes, strides):
        lengths = _compute_conv_out(lengths, k, s)

    ### Make Sure lengths are of dtype int ###
    lengths = lengths.type(torch.int)

    return lengths

def padding_attention_masking(raw_audio: list, 
                              kernel_sizes: tuple, 
                              strides: tuple):
    """
    Given some audio, compute the padding attention mask after encoding! Basically, we have to add a bunch of 0.0 values
    to each audio sequence to make sure that we have the same sequence length for each of them. After performing the convolutions, 
    we will have our encoded audio, but the encodings from the padded portions of audios need to be ignored in our Transformer. To
    do this we need to create our attention mask that will indicate all the tokens that are a part of that 0.0 padding values!

    In the huggingface implementation this is called the "sub_attention_mask"

    To do this, we will compute the length of each audio using our function 'utils.feature_encoder_output' and then pad to the longest
    encoded length!

    Args:
        raw_audio: List of the raw audio files, each of a different length
        kernel_sizes: Tuple of all the kernel size for each convolution
        strides: Tuple of all the strides for each convolution
    """

    ### Get Batch Size ###
    batch_size = len(raw_audio)

    ### Get Raw Audio Lengths ###
    raw_lengths = torch.tensor([len(audio.flatten()) for audio in raw_audio])

    ### Compute The Length of Encoded Tokens After Convolutions ###
    encoded_lengths = feature_encoder_output_length(raw_lengths, kernel_sizes, strides)

    ### Build Attention Mask (F.SDPA wants a Batch x S x S Attention Matrix) ###
    max_length = encoded_lengths.max()
    feature_attention_mask = torch.zeros(size=(batch_size, max_length, max_length))

    ### We only need to mask out the columns (the Keys), the Value paddings can remain ###
    ### CHECK my RoBERTa implementation for more details!!
    for idx, length in enumerate(encoded_lengths):
        feature_attention_mask[idx, :, :length] = 1
    
    return feature_attention_mask
 
    

def span_masking(input_values_shape: tuple, 
                 masking_probability: float = 0.065, 
                 masking_span_length: int = 10, 
                 minimum_spans: int = 2,
                 attention_mask: torch.Tensor = None
                 ):

    """
    Wav2Vec2 utilizes span masking for its pre-training task. (i.e. we randomly select the index of start tokens with our
    masking_probability, and then mask the next consecutive tokens of length masking_span_length). This span masking is applied
    to the output of the encoder Convolutions!

    Args:
        input_values_shape: Tuple of the Batch size and Output Shape of the convolutional encoder
        masking_probability: With what likelihood should a token be the start of a spanned mask
        masking_span_length: What should be the length of every span
        minimum_spans: Minumum number of spans to have in each audio
        attention_mask: Attention mask indicating padding tokens on the encoded audio 
    """

    ### Grab Batch Size and Padded Sequence Length from Input Values ###
    batch_size, total_sequence_length = input_values_shape

    ### Compute Lengths of Each Audio in Batch ###
    if attention_mask is None:
        sequence_lengths = [total_sequence_length] * batch_size
    else:
        ### Attention mask is (B,L,L), we just need to add up the number of nonmasked tokens in each row, once per batch! ###
        sequence_lengths = attention_mask[:, 0].sum(axis=-1).to(torch.int).tolist()

    ### Sample The Starting Indexes of Spans ###
    sequence_masks = []
    for length in sequence_lengths:
        
        ### Define Mask for Full (Padded) Sequence ###
        mask = torch.zeros(total_sequence_length).bool()

        ### Sample Potential Starting Indexes for Span Masking ###
        sampled_starting_idx = (torch.rand(length) <= masking_probability).nonzero()

        ### Sanity Check that we have atleast minimum_spans ###
        if len(sampled_starting_idx) < minimum_spans:
            sampled_starting_idx = torch.randint(low=0, high=length, size=(minimum_spans,1))
        
        ### Compute Full Span Indexes ###
        span_offsets = torch.arange(masking_span_length)
        spans = sampled_starting_idx + span_offsets

        ### Convert Each Span to A Single Vector of Indexes to Mask ###
        spans = spans.flatten()

        ### Any Span Index Longer than Sequence Length Should be removed (We dont want to mask any Padding Tokens) ###
        spans = spans[spans <= length - 1]

        ### Set All Span Indexes to True in our mask ###
        mask[spans.flatten()] = True
        sequence_masks.append(mask.unsqueeze(0))

    sequence_masks = torch.concatenate(sequence_masks)
    
    return sequence_masks


def sample_negatives(features_shape, num_negatives, span_mask):

    """
    This is kind of finiky, im sure there is a better way to do this but here goes!

    What we need to do is for every masked sample, we need to generated 'num_negative' number of negative samples
    for our contrastive loss from the other masked samples. There are a few steps as far as I can tell to do this:
    
    We start with the masked_indexes, a boolean vector for each sequence indicating if that index is being masked.
     
    To be clear, there are two indexes going on:
        
        (1) Masked Index which are the actual indexes of the masked tokens in a sample of encoded/masked data -> [42, 58, 64, ...]
        (2) Enumerated Index which is from 0 to the number of masked tokens -> [0, 1, 2, ...]
    
    So, for every sample in the batch we need to:
    
        (1) Grab the masked indexes and number of masked tokens in the sample
        (2) Uniformly sample 'num_negative' number of enumerated indexes ranging from 0 to the number of masked tokens
        (3) We need to ensure that anything we sample is NOT THE POSITIVE SAMPLE. So if we are on enumerated index 0, and
            have 100 masked tokens, we can sample any enumerated index from 1 to 99, but NOT 0. So we check for this and then just add
            1 to any case. Therefore if there is an overlap where for index 0 we are sampling the negative 0, we will now
            just be sampling 1!
        (4) If we are adding 1 to everything, there is now a chance that our enumerated indexes may go above the number of mask tokens, 
            which also isn't good... So if we have 100 tokens, the only case this happens is if in the sampling for token 99, 
            we get the index 99 for the negative. This will make this value 100, as per the previous step. Therefore in this case
            we will just resample a new value for anything greater, with any value between 0 and 1 less than the max possible, in this case
            98. 
        (5) Use these computed enumerated indexes to index the original masked token indexes to get our negatives


    Whew... these arent very fun manipulations at all...

    Caveat: Not totally sure, but if we want 100 negatives, but only 50 were masked, there will be repeated negatives in this case, probably fine?

    Args:
        masked_indexes: Tensor of shape (Batch x Sequence Length) indicating the location of masked tokens
        num_negatives: Number of negatives we want to sample

    """


    ### Pass in the Data Shape (Post Convolutional Encoding) ###
    batch_size, sequence_length = features_shape
    
    ### Get Indexes for sequence of features ###
    sequence_index = np.arange(sequence_length)
    
    ### Empty Tensor to fill with sampled negatives ###
    sampled_negatives = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)
    
    ### Create Span Mask if not supplied (Nothing to sample though in this case) ###
    if span_mask is None:
        span_mask = np.ones((batch_size, sequence_length), dtype=bool)

    for idx in range(batch_size):
        
        ### Grab Span Mask for Sample ###
        batch_span_mask = span_mask[idx]

        ### Grab the Corresponding Mask Index for Sequence in this Batch ###
        masked_indexes = sequence_index[batch_span_mask]

        ### Create Matrix of feature indices to avoid sampling positive tensor ###
        num_masked = batch_span_mask.sum()
        feature_index = np.expand_dims(np.arange(num_masked),-1)
        feature_index = np.repeat(feature_index, num_negatives, axis=-1)

        ### Sample Indicies (Notice, we will sample index 0 to num_masked - 1) ###
        ### This is so if there is an overlap between sampled index and the positive (true) index ###
        ### We can just add 1, but keep the highest index to num_masked ###
        sample_index = np.random.randint(0, num_masked-1, size=(num_masked, num_negatives))

        ### If our Sampled Index is Equal to our Feature index, that is a repeat of a positive class, so just add 1 to make it different! ###
        sample_index[(sample_index == feature_index)] += 1

        ### Store these Sample Indexes in our sampled_negatives array with the corresponding masked index ###
        ### Break this down: 
        ### sampled_negatives[idx] -> Index the batch dimension in our sampled_negatives (starts out as all zeros)
        ### sampled_negatives[idx][batch_span_mask] -> this indexes our batch of sampled_negatives only for the indexes that we have a mask (we only sample negatives for masked locations)
        ### masked_indexes: if our sequence length is 20, the masked_indexes is which indexes from 0 to 19 are masked
        ### sampled_index: if we have 8 masked things in our sequence then the sampled_index goes from 0 to 7. This tensor randomly samples those indexes to have num_negative negatives for each masked position
        ### masked_indexes[sample_index]: We dont care about the sampled_index from 0 to 7, we want the negative indexes in terms of the original index 0 to 19, so this converts the sampled index to our sequence index!
        sampled_negatives[idx][batch_span_mask] = masked_indexes[sample_index]

        ### In the future, we will flatten all this, so if we have a sequence length of 20, the first sample should go from 0 to 19, but the second sample should go from 20 to 39. 
        ### So we need to just adjust for batch size. Everything starts at the 0 index right now, so we just ned to add what sample in the batch are we on, times the sequence length 
        sampled_negatives[idx] += idx * sequence_length

    ### Convert to PyTorch Tensor ###
    sampled_negatives = torch.tensor(sampled_negatives, dtype=torch.long)

    return sampled_negatives

if __name__ == "__main__":
    
    ### Test Feature Encoder Output ###
    lengths = torch.tensor([1000, 500, 600])
    kernel_sizes = (3, 3, 3, 3, 3, 3)
    strides = (2, 2, 2, 2, 2, 2)

    rand_tensors = [torch.randn(1, l) for l in lengths]

    ### Compute output shape with Convolutions ###
    conv_output_lengths = []
    convs = nn.Sequential(*[nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, stride=s) 
                             for (k,s) in zip(kernel_sizes, strides)])
    for tensor in rand_tensors:
        with torch.no_grad():
            output = convs(tensor)
        conv_output_lengths.append(output.shape[-1])
    
    ### Compute Output Shape with Formula ###
    computed_output_shape = feature_encoder_output_length(lengths, kernel_sizes, strides).tolist()
    assert computed_output_shape == conv_output_lengths

    ## Test Encoded Attention Mask ###
    attention_mask = padding_attention_masking(rand_tensors, kernel_sizes, strides)
 
    ### Test Span Masking ###
    batch_size = len(lengths)
    sequence_length = max(conv_output_lengths)
    span_masked = span_masking(input_values_shape=(batch_size, sequence_length), attention_mask=attention_mask)

    ### Get Negatives for Spans ###
    negatives = sample_negatives(features_shape=(batch_size, sequence_length), 
                                 num_negatives=10, 
                                 span_mask=span_masked)
    

    