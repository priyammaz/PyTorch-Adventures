import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk
from safetensors.torch import load_file
from transformers import AutoTokenizer
import evaluate

from model import Transformer, TransformerConfig
from tokenizer import FrenchTokenizer

def compute_bleu(path_to_raw_data, 
                 path_to_model_safetensors,
                 rand_selected_for_scoring=None):

    """
    A typical metric to evaluate translation performance if BLEU. Here we 
    iterate through the testing portion of our prepared dataset to see
    the score we get. 

    This takes some time (as I never created a batch_inference script) so
    you can take a random sample of the testing data by specifying how many 
    samples you want in rand_selected_for_scoring.

    """
    ### Load Model ###
    config = TransformerConfig()
    model = Transformer(config)

    ### Load Pretrained Weights ###
    weight_dict = load_file(path_to_model_safetensors)
    model.load_state_dict(weight_dict)
    model.eval()
    model = model.to("cuda")

    ### Load Tokenizers ###
    tgt_tokenizer =  FrenchTokenizer("trained_tokenizer/french_wp.json")
    src_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    ### Load Testing Dataset to Compute BLEU Score (Random Sample so we dont iterate through all the data) ###
    testing_data = load_from_disk(path_to_raw_data)["test"]
    if rand_selected_for_scoring is not None:
        testing_data = testing_data.shuffle().select(range(rand_selected_for_scoring))

    ### Load BLEU Score Evaluation ###
    bleu = evaluate.load("bleu")
    
    predictions = []
    references = []

    for sample in tqdm(testing_data):
        english = sample["english_src"]
        french = sample["french_tgt"]

        ### Clean Up French for Comparison ###
        clean_french = tgt_tokenizer.tokenizer.normalizer.normalize_str(french)

        ### Predict Translation from English ###
        src_ids = torch.tensor(src_tokenizer(english)["input_ids"][:config.max_src_len]).unsqueeze(0).to("cuda")
        translated = model.inference(src_ids, 
                                     tgt_start_id=tgt_tokenizer.special_tokens_dict["[BOS]"],
                                     tgt_end_id=tgt_tokenizer.special_tokens_dict["[EOS]"])
        prediction = tgt_tokenizer.decode(translated, skip_special_tokens=True)

        predictions.append(prediction)
        references.append(clean_french)
        
    ### Compute BLEU Score ###
    score = bleu.compute(predictions=predictions, references=references)

    print("Testset BLEU Score: ", score["bleu"])


if __name__ == "__main__":
    path_to_raw_data = "/mnt/datadrive/data/machine_translation/english2french/raw_english2french_corpus"
    path_to_model_safetensor = "work_dir/Seq2Seq_Neural_Machine_Translation/checkpoint_150000/model.safetensors"
    compute_bleu(path_to_raw_data, path_to_model_safetensor)