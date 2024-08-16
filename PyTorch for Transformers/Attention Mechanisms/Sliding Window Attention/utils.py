import torch
import random
from typing import Literal
from transformers import RobertaTokenizerFast
from dataclasses import dataclass, asdict


@dataclass
class RobertaConfig:
    
    ### Tokenizer Config
    vocab_size: int = 50265
    start_token: int = 0
    end_token: int = 2
    pad_token: int = 2
    mask_token: int = 50264

    ### Transformer Config ###
    embedding_dimension: int = 768
    num_transformer_blocks: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    layer_norm_eps: float = 1e-6
    hidden_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    context_length: int = 2048
    attention_type: Literal["windowed", "full"] = "windowed"
    window_size: int = 128
    causal: bool = False
    look_backward: int = 1
    look_forward: int = 1

    ### Masking Config ###
    masking_prob: float = 0.15

    ### Huggingface Config ###
    hf_model_name: str = "FacebookAI/roberta-base"

    ### Model Config ###
    pretrained_backbone: Literal["pretrained", "pretrained_huggingface", "random"] = "pretrained"
    path_to_pretrained_weights: str = None

    ### Added in to_dict() method so this Config is compatible with Huggingface Trainer!!! ###
    def to_dict(self):
        return asdict(self)

def random_masking_text(tokens, 
                        special_tokens_mask, 
                        vocab_size=50264,
                        special_ids=(0,1,2,3,50264),
                        mask_ratio=0.15, 
                        mask_token=50264):
    
    """
    Function for our random masking of tokens (excluding special tokens). This follow the logic provided 
    by BERT/RoBERTa:

        - Select 15% of the tokens for masking
            - 80% of the selected tokens are replaced with a mask token
            - 10% of the selected tokens are replaced with another random token
            - 10% of the selected tokens are left alone

    This is almost identical to the masking function in our introductory jupyter notebook walkthrough of 
    masked language modeling, but some minor changes are made to apply masking to batches of tokens
    rather than just one sequence at a time!
    """

    ### Create Random Uniform Sample Tensor ###
    random_masking = torch.rand(*tokens.shape)

    ### Set Value of Special Tokens to 1 so we DONT MASK THEM ###
    random_masking[special_tokens_mask==1] = 1

    ### Get Boolean of Words under Masking Threshold ###
    random_masking = (random_masking < mask_ratio)

    ### Create Labels ###
    labels = torch.full((tokens.shape), -100)
    labels[random_masking] = tokens[random_masking]

    ### Get Indexes of True ###
    random_selected_idx = random_masking.nonzero()

    ### 80% Of the Time Replace with Mask Token ###
    masking_flag = torch.rand(len(random_selected_idx))
    masking_flag = (masking_flag<0.8)
    selected_idx_for_masking = random_selected_idx[masking_flag]

    ### Seperate out remaining indexes to be assigned ###
    unselected_idx_for_masking = random_selected_idx[~masking_flag]

    ### 10% of the time (or 50 percent of the remaining 20%) we fill with random token ###
    ### The remaining times, leave the text as is ###
    masking_flag = torch.rand(len(unselected_idx_for_masking))
    masking_flag = (masking_flag<0.5)
    selected_idx_for_random_filling = unselected_idx_for_masking[masking_flag]
    selected_idx_to_be_left_alone = unselected_idx_for_masking[~masking_flag]

    ### Fill Mask Tokens ###
    if len(selected_idx_for_masking) > 0:
        tokens[selected_idx_for_masking[:, 0], selected_idx_for_masking[:, 1]] = mask_token
    
    ### Fill Random Tokens ###
    if len(selected_idx_for_random_filling) > 0:
        non_special_ids = list(set(range(vocab_size)) - set(special_ids))
        randomly_selected_tokens = torch.tensor(random.sample(non_special_ids, len(selected_idx_for_random_filling)))
        tokens[selected_idx_for_random_filling[:,0], selected_idx_for_random_filling[:,1]] = randomly_selected_tokens
        
    return tokens, labels

def RobertaMaskedLMCollateFunction(config):

    """
    Simply collation function that grabs batches of samples, pads them, and then random masks them for the 
    Masked Language Modeling task!
    """

    tokenizer = RobertaTokenizerFast.from_pretrained(config.hf_model_name)

    def collate_fn(batch):

        ### Grab Stuff from Batch ###
        input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]

        ### Pad and Concatenate Batch ###
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, padding_value=tokenizer.pad_token_id, batch_first=True)

        ### Create Padding Attention Mask ###
        attention_mask = [torch.ones(len(sample)) for sample in input_ids]
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)

        ### Get the Special Tokens Mask (so we dont mask them out in MLM Task) ###
        special_tokens_mask = torch.isin(input_ids, torch.tensor(tokenizer.all_special_ids))

        ### Random Masking for MLM ###
        tokens, labels = random_masking_text(input_ids, 
                                            special_tokens_mask, 
                                            vocab_size=tokenizer.vocab_size,
                                            special_ids=tokenizer.all_special_ids,
                                            mask_ratio=config.masking_prob, 
                                            mask_token=tokenizer.mask_token_id)
        
        batch = {"input_ids": tokens,
                "attention_mask": attention_mask,  
                "labels": labels}
        
        return batch

    return collate_fn
