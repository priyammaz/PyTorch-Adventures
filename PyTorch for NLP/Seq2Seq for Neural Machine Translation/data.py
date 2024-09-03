import pandas as pd
import torch
from torch.utils.data import DataLoader
from tokenizer import FrenchTokenizer
from transformers import AutoTokenizer
from model import TransformerConfig
from datasets import load_from_disk

def TranslationCollator(src_tokenizer, tgt_tokenizer):

    def _collate_fn(batch):

        src_ids = [torch.tensor(i["src_ids"]) for i in batch]
        tgt_ids = [torch.tensor(i["tgt_ids"]) for i in batch]

        src_pad_token = src_tokenizer.pad_token_id
        src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=src_pad_token)
        src_pad_mask = (src_padded!=src_pad_token)
        
        tgt_pad_token = tgt_tokenizer.special_tokens_dict["[PAD]"]
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=tgt_pad_token)
        input_tgt = tgt_padded[:, :-1].clone()
        output_tgt = tgt_padded[:, 1:].clone()

        input_tgt_mask = (input_tgt != tgt_pad_token)
        output_tgt[output_tgt==tgt_pad_token] = -100
        
        batch = {"src_input_ids": src_padded, 
                 "src_pad_mask": src_pad_mask, 
                 "tgt_input_ids": input_tgt, 
                 "tgt_pad_mask": input_tgt_mask, 
                 "tgt_outputs": output_tgt}
        
        return batch
        
    return _collate_fn

if __name__ == "__main__":

    path_to_data = "/mnt/datadrive/data/machine_translation/english2mutilingual_train.csv"
    path_to_vocab = "trained_tokenizer/sentencepiece_tokenizer.json"

    tgt_tokenizer =  FrenchTokenizer("trained_tokenizer/french_wp.json")
    src_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    path_to_data = "/mnt/datadrive/data/machine_translation/english2french/tokenized_french2english_corpus"
    dataset = load_from_disk(path_to_data)

    collate_fn = TranslationCollator(src_tokenizer, tgt_tokenizer)

    loader = DataLoader(dataset["train"], batch_size=64, collate_fn=collate_fn, num_workers=32)
    
    from tqdm import tqdm
    for batch in tqdm(loader):
        pass