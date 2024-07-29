import torch
import torch.nn as nn
from dataset import LibriSpeechDataset, Wav2Vec2CollateFunctionForPreTraining
from transformers import Wav2Vec2Config
from torch.utils.data import Dataset, DataLoader

class Wav2Vec2GumbelVectorQuantizer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group

        assert config.codevector_dim % self.num_groups == 0, \
            f"Make sure Codevector Dim {config.codevector_dim} is divisible by number of groups {config.num_codevector_groups}"

        # storage for codebook variables (codewords)
        self.codevectors = nn.Parameter(
            torch.randn(1,self.num_groups*self.num_vars, config.codevector_dim // self.num_groups)
        )

        ### Projection from Output convolution channels to total quantized vectors ###
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
    
    @staticmethod
    def _compute_perplexity_2(probs, mask=None):
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
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
            codevector_probs = nn.functional.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True).type_as(hidden_states)

            ### Compute Perplexity ###
            codevector_soft_dist = hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float().softmax(dim=-1)
            perplexity1 = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
            perplexity2 = self._compute_perplexity_2(codevector_soft_dist, torch.tensor(mask_time_indices))

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
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors

        ### We now will seperate out the 600 codes to be 2 x 300, and then add across the 300 dimension (as only one of those 300 codes are actually not zero!)###
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(dim=-2)
        
        ### Now for the cool part, currently we have 2 codes for each intial input token, and each are of length 128 (assuming the final VQ dim is 256 and we had 2 groups) ###
        ### In total, we have 600 codes, which is not enough to represent a large diverse set of quantized latents, so instead, we will concatenate them together! This way ###
        ### We will have a total of 300 * 300 possibilities! ###
        
        codevectors = codevectors.view(batch_size, sequence_length, -1)

        return codevectors, perplexity

if __name__ == "__main__":
    
    config = Wav2Vec2Config()
    w = Wav2Vec2GumbelVectorQuantizer(config)
    dataset = LibriSpeechDataset("/mnt/datadrive/data/LibriSpeech/", include_splits=["dev-clean", "test-clean"], return_transcripts=False, max_audio_duration=20)
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=4, collate_fn=Wav2Vec2CollateFunctionForPreTraining(config), num_workers=24)

    for batch in loader:
        mask = batch["mask_time_indices"]
        x = torch.randn((*batch["sub_attention_mask"].shape, 512))
        w(x, mask)
        break
# class Wav2Vec2GumbelVectorQuantizer(nn.Module):
#     """
#     Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
#     GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
#     """

#     def __init__(self, config):
#         super().__init__()
#         self.num_groups = config.num_codevector_groups
#         self.num_vars = config.num_codevectors_per_group

#         if config.codevector_dim % self.num_groups != 0:
#             raise ValueError(
#                 f"`config.codevector_dim {config.codevector_dim} must be divisible "
#                 f"by `config.num_codevector_groups` {self.num_groups} for concatenation"
#             )

#         # storage for codebook variables (codewords)
#         self.codevectors = nn.Parameter(
#             torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
#         )
#         self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

#         # can be decayed for training
#         self.temperature = 2

#     @staticmethod
#     def _compute_perplexity(probs, mask=None):
#         if mask is not None:
#             mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
#             probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
#             marginal_probs = probs.sum(dim=0) / mask.sum()
#         else:
#             marginal_probs = probs.mean(dim=0)

#         perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
#         return perplexity

#     def forward(self, hidden_states, mask_time_indices=None):
#         batch_size, sequence_length, hidden_size = hidden_states.shape

#         # project to codevector dim
#         hidden_states = self.weight_proj(hidden_states)
#         hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

#         if self.training:
#             # sample code vector probs via gumbel in differentiateable way
#             codevector_probs = nn.functional.gumbel_softmax(
#                 hidden_states.float(), tau=self.temperature, hard=True
#             ).type_as(hidden_states)

#             # compute perplexity
#             codevector_soft_dist = torch.softmax(
#                 hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
#             )
#             perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
#         else:
#             # take argmax in non-differentiable way
#             # comptute hard codevector distribution (one hot)
#             codevector_idx = hidden_states.argmax(dim=-1)
#             codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
#                 -1, codevector_idx.view(-1, 1), 1.0
#             )
#             codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

#             perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

#         codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
#         # use probs to retrieve codevectors
#         codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
#         codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
#         codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

#         return codevectors, perplexity