import os
from datasets import load_dataset, concatenate_datasets, load_from_disk

from tokenizer import FrenchTokenizer
from transformers import AutoTokenizer

def build_english2french_dataset(path_to_data_root):

    hf_dataset = []
 
    for dir in os.listdir(path_to_data_root):
        print("Processing:", dir)

        path_to_dir = os.path.join(path_to_data_root, dir)

        french_text = english_text = None

        for txt in os.listdir(path_to_dir):
            if txt.endswith(".fr"):
                french_text = os.path.join(path_to_dir, txt)
            elif txt.endswith(".en"):
                english_text = os.path.join(path_to_dir, txt)

        if french_text is not None and english_text is not None:
            french_dataset = load_dataset("text", data_files=french_text)["train"]
            english_dataset = load_dataset("text", data_files=english_text)["train"]

            english_dataset = english_dataset.rename_column("text", "english_src")
            english_dataset = english_dataset.add_column("french_tgt", french_dataset["text"])
            
            hf_dataset.append(english_dataset)

    hf_dataset = concatenate_datasets(hf_dataset)
    
    hf_dataset = hf_dataset.train_test_split(test_size=0.005)

    path_to_save = os.path.join(path_to_data_root, "hf_french2english_corpus")

    hf_dataset.save_to_disk(path_to_save)

def tokenize_english2french_dataset(path_to_hf_data, path_to_save, num_workers=24, truncate=False, max_length=512, min_length=5):

    french_tokenizer = FrenchTokenizer("trained_tokenizer/french_wp.json", truncate=truncate, max_length=max_length)
    english_tokenzer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    raw_dataset = load_from_disk(path_to_hf_data)
    
    def _tokenize_text(examples):

        english_text = examples["english_src"]
        french_text = examples["french_tgt"]
        src_ids = english_tokenzer(english_text, truncation=True, max_length=512)["input_ids"]
        tgt_ids = french_tokenizer.encode(french_text)

        batch = {"src_ids": src_ids, 
                 "tgt_ids": tgt_ids}
        
        return batch
    
    tokenized_dataset = raw_dataset.map(_tokenize_text, batched=True, num_proc=num_workers)
    tokenized_dataset = tokenized_dataset.remove_columns(["english_src", "french_tgt"])

    filter_func = lambda example: (len(example["tgt_ids"]) > min_length)
    tokenized_dataset = tokenized_dataset.filter(filter_func)
    print(tokenized_dataset)

    tokenized_dataset.save_to_disk(path_to_save)


if __name__ == "__main__":
    # path_to_data_root = "/mnt/datadrive/data/machine_translation/english2french/"
    # build_english2french_dataset(path_to_data_root)

    path_to_data = "/mnt/datadrive/data/machine_translation/english2french/hf_french2english_corpus"
    path_to_save = "/mnt/datadrive/data/machine_translation/english2french/tokenized_french2english_corpus"
    tokenize_english2french_dataset(path_to_data, path_to_save, truncate=True)