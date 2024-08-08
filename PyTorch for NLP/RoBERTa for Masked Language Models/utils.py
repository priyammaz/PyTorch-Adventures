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
    context_length: int = 512

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


def ExtractiveQAPreProcessing(config):

    tokenizer = RobertaTokenizerFast.from_pretrained(config.hf_model_name)

    def char2token_mapping(examples):

        """
        Taken basically from the 🤗 Huggingface Q&A Example provided here:
        https://huggingface.co/docs/transformers/en/tasks/question_answering. 
        
        I just made some changes (that probably hurt efficiency) just so the 
        code it is a bit more readable!

        Basically we have a problem, in our Squad Dataset, we are provided:
            - Question
            - Context
            - Answer (a portion of the context)
        
        So in the answers we are provided the actual text of the answer and the
        starting character index of the answer in the context. Once we tokenize
        our data, a single token can be multiple characters, so now we need to 
        convert our answer from the starting and ending character to the starting
        and ending token! This function basically does that in a somewhat annoying
        way.

        """

        ### Grab Questions ###
        questions = [q.strip() for q in examples["question"]]
        
        ### Tokenize Questions and Context together as Pairs (concatenates them together with some sep tokens) ###
        ### only_second means we will only truncate context! ###
        ### return_offset_mapping will provide (0,0) for all 
        ### special characters, and then start and end character 
        ### for all other tokens (as tokens can be multiple characters)
        inputs = tokenizer(
            text=questions,
            text_pair=examples["context"],
            max_length=config.context_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        ### Grab the Offset Mappings and Answers ###
        ### These offsets reset, so it starts from character index 0 for the question, 
        ### and then starts from character index 0 again for the answer ###
        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]

        ### Create some lists to hold the start and end tokens for our answers, we are about to do our character to token conversion ###
        starting_token_idxs = []
        ending_token_idxs = []

        ### Iterate through every sample in the batch ###
        for i, offset in enumerate(offset_mapping):

            ### Grab Answer for this iteration ###
            answer = answers[i]

            ### The start character index is given, and if we have the anwer we have the number 
            ### of characters in the answer, so we can add it to find the end character index 
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            ### inputs may look like dictionary but it is actually of type: transformers.BatchEncoding
            ### This means it has some methods we can use, what we need is some identifiers of which 
            ### Remember, we concatenated the context onto the question with some seperation tokens
            ### so the `sequence_ids` method will reveal to us which tokens in our sequence were the
            ### question (with index 0) and which tokens are context (with index 1)

            sequence_ids = inputs.sequence_ids(i)

            ### An example of the sequence ids can be:
            ### [None, 0, 0, 0, 0, 0, None, None, 1, 1, 1, 1, 1, None, None, ...]
            ### The None indicates special tokens (start, sep, pad) ###
            ### We want the starting and ending index of all the 1 indexes so lets grab those!
            
            ### Initialize start and end of context with None ###
            context_start = None
            context_end = None

            ### Iterate through the sequence ids ###
            for idx, id in enumerate(sequence_ids):

                ### If our context start has not been set yet, and we see a 1, then that is the start ###
                if context_start is None and id == 1:
                    context_start = idx

                ### once the start is set, once we see something that isnt 1, then the token before must be the end ###
                elif context_start is not None and id != 1:
                    context_end = idx - 1
                    break

                ### If we get to the end of the sequence and it is still 1, then there must be no padding, so the context goes till the end! ###
                elif context_start is not None and idx == len(sequence_ids) - 1:
                    context_end = idx

            ### Lets Grab the Offsets so we know the start and end character of the start and end context tokens ###
            context_start_char = offset[context_start][0]
            context_end_char = offset[context_end][1]

            ### We have to make sure that our entire answer is within the context for training (could have gotten clipped if context is too long) ###
            if (start_char >= context_start_char) and (end_char <= context_end_char): 
            
                start_token_idx = None
                end_token_idx = None
                for token_idx, (offset, seq_id) in enumerate(zip(offset, sequence_ids)):
                    
                    ### Only want to look at tokens that are context tokens (index 1) ###
                    if seq_id == 1:

                        ### If our start_char or end_char index of the answer 
                        ### is within the range of the start and end char of each token
                        ### then that tells us the start and end token!
                        ### We add 1 because range is NOT RIGHT INCLUSIVE ###

                        if start_char in range(offset[0], offset[1]+1):
                            start_token_idx = token_idx
                        if end_char in range(offset[0], offset[1]+1):
                            end_token_idx = token_idx

                starting_token_idxs.append(start_token_idx)
                ending_token_idxs.append(end_token_idx)
            
            ### If the answer is not fully inside, then we will just return (0,0) as a placeholder for the start and end, we will not compute loss on this later ###
            else:
                starting_token_idxs.append(0)
                ending_token_idxs.append(0)

        
        inputs["start_positions"] = starting_token_idxs
        inputs["end_positions"] = ending_token_idxs

        return inputs

    return char2token_mapping
