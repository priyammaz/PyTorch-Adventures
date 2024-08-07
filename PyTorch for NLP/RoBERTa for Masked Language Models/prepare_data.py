from transformers import RobertaTokenizerFast
from datasets import load_dataset, concatenate_datasets, load_from_disk
import time
import argparse

parser = argparse.ArgumentParser(description="RoBERTA Data Prep")

parser.add_argument(
    "--test_split_pct", 
    default=0.005, 
    help="What percent of data do you want to use for Train/Test split",
    type=float
)

parser.add_argument(
    "--context_length", 
    default=512, 
    help="Pass in argument to override the default in Config, but then make sure config \
        reflects this when training",
    type=int
)

parser.add_argument(
    "--path_to_data_store", 
    required=True, 
    help="Path to where you want to save the final tokenized dataset",
    type=str
)

parser.add_argument(
    "--huggingface_cache_dir",
    default=None,
    help="path to huggingface cache directory if different from default",
    type=str
)

parser.add_argument(
    "--dataset_split_seed",
    default=42, 
    help="Seed to ensure reproducible split of dataset",
    type=int
)

parser.add_argument(
    "--num_workers",
    default=16, 
    help="Number of workers you want to use to process dataset",
    type=int
)

parser.add_argument(
    "--hf_model_name",
    default="FacebookAI/roberta-base",
    help="Name of model so we can use the huggingface tokenizer", 
    type=str
)

def prepare_data(args):

    context_length = args.context_length
    path_to_save = args.path_to_data_store
    cache_dir = args.huggingface_cache_dir

    ### Load tokenizer ###
    tokenizer = RobertaTokenizerFast.from_pretrained(args.hf_model_name)

    ### Load Datasets ###
    wikidataset = load_dataset("wikipedia", "20220301.en", cache_dir=cache_dir)
    booksdataset = load_dataset("bookcorpus/bookcorpus", cache_dir=cache_dir)

    ### Grab only text columns ###
    wikidataset = wikidataset.select_columns(["text"])
    booksdataset = booksdataset.select_columns(["text"])

    ### Concatenate Datasets Together ###
    dataset = concatenate_datasets([wikidataset["train"], booksdataset["train"]])

    ### Train/Test Split Dataset ###
    dataset = dataset.train_test_split(test_size=0.005, seed=42)

    ### Tokenize Dataset ###
    def compute_tokens(examples):
        """
        Function takes text, removes any new-line/tab characters and then tokenizes
        """
        sentences = [" ".join(example.replace("\n", " ").replace("\t", " ").split()) \
                    for example in examples["text"]]
        return tokenizer(sentences, return_attention_mask=False)
        
    tokenized_data = dataset.map(
        compute_tokens, 
        batched=True, 
        num_proc=args.num_workers, 
        remove_columns="text"
    )

    ### Group Texts ###
    def group_texts(batch):

        """
        Function takes batches of tokens, concats all of them together, and then cuts
        into the desired sizes of the context length we want. Make sure whatever context
        size you use here is the minimum context length used in the model
        """

        concatenated_texts = []
        for sentence in batch["input_ids"]:
            concatenated_texts.extend(sentence)

        concatenated_texts = [
            concatenated_texts[i:i+context_length] \
                for i in range(0, len(concatenated_texts), context_length)
        ]

        data = {"input_ids": concatenated_texts}

        return data

    tokenized_data = tokenized_data.map(group_texts, batched=True, num_proc=args.num_workers)

    ### Save Data ###
    tokenized_data.save_to_disk(path_to_save)


if __name__ == "__main__":

    args = parser.parse_args()
    prepare_data(args)

    ### Test that it worked ###
    start = time.time()
    data = load_from_disk(args.path_to_data_store)
    end = time.time()

    print("Time to Load Dataset", end-start)
    print(data)