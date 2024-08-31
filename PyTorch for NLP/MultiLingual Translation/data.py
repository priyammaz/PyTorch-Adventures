import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import MultilingualTokenizer
from transformers import AutoTokenizer
from model import TransformerConfig

class TranslationDataset(Dataset):

    def __init__(self, 
                 config,
                 path_to_data, 
                 src_tokenizer, 
                 tgt_tokenizer):

        self.config = config
        self.path_to_data = path_to_data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.data = pd.read_csv(path_to_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sample = self.data.iloc[idx]

        src_text = sample.english
        tgt_text = sample.target
        lang = sample.target_language
        
        src_text = f"Translate from English to {lang}: " + src_text
        
        src_ids = self.src_tokenizer.encode(src_text)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)
        
        batch = {"src_ids": src_ids, 
                 "tgt_ids": tgt_ids}
        
        return batch 

def TranslationCollator(config, src_tokenizer, tgt_tokenizer):

    def _collate_fn(batch):
        
        src_ids = [i["src_ids"] for i in batch]
        tgt_ids = [i["tgt_ids"] for i in batch]

        ### Handle src_ids greater than max length ###
        for idx, src_id in enumerate(src_ids):
            if len(src_id) > config.max_src_len:
                diff = len(src_id) - config.max_src_len + 1
                src_id = src_id[:-diff]
                src_id.append(src_tokenizer.eos_token)
                src_ids[idx] = torch.tensor(src_id)
            else:
                src_ids[idx] = torch.tensor(src_id)

        ### Handle tgt_ids greater than max length ###
        for idx, tgt_id in enumerate(tgt_ids):
            if len(tgt_id) > config.max_src_len+1:
                diff = len(tgt_id) - config.max_tgt_len + 2
                tgt_id = tgt_id[:-diff]
                tgt_id.append(tgt_tokenizer.special_tokens_dict["<eos>"])
                tgt_ids[idx] = torch.tensor(tgt_id)
            else:
                tgt_ids[idx] = torch.tensor(tgt_id)

        src_pad_token = src_tokenizer.pad_token_id
        src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=src_pad_token)
        src_pad_mask = (src_padded!=src_pad_token)
        
        tgt_pad_token = tgt_tokenizer.special_tokens_dict["<pad>"]
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=tgt_pad_token)
        input_tgt = tgt_padded[:, :-1]
        output_tgt = tgt_padded[:, 1:]

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
    config = TransformerConfig()
    tgt_tokenizer = MultilingualTokenizer(path_to_vocab)
    src_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    dataset = TranslationDataset(config, path_to_data, src_tokenizer, tgt_tokenizer)

    collate_fn = TranslationCollator(config, src_tokenizer, tgt_tokenizer)

    loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn, num_workers=32)
    
    from tqdm import tqdm
    for batch in tqdm(loader):
        print(batch)
        break