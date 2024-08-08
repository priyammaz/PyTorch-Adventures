"""
Quick function to compute our Squad score for the validation set of the Squad dataset! Probably could be better
as I compute everything here sample by sample instead of batches, but its good enough for learning!
"""

from datasets import load_dataset
from torchmetrics.text import SQuAD
from inference import InferenceModel
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Inference Script on SQUAD Validation Dataset")

parser.add_argument(
    "--path_to_model_weights",
    required=True, 
    help="Path to RobertaForQuestionAnsweing weights",
    type=str
)

parser.add_argument(
    "--path_to_store",
    required=True,
    help="Where to store final results json of the squad score",
    type=str
)

parser.add_argument(
    "--huggingface_model",
    action=argparse.BooleanOptionalAction,
    default=False,
    type=bool
)

parser.add_argument(
    "--cache_dir",
    default=None,
    help="Path to cache directory if it is different from the huggingface default",
    type=str
)


def compute_squad_metrics(path_to_weights, 
                          huggingface_model, 
                          path_to_store,
                          cache_dir):

    ### Load Squad Validation Set ###
    squad_dataset = load_dataset("squad", cache_dir=cache_dir)
    validation_dataset = squad_dataset["validation"]

    ### Load Squad Metric Computation ###
    squad = SQuAD()

    ### Load Inference Class ###
    inferencer = InferenceModel(path_to_weights=path_to_weights, huggingface_model=huggingface_model)

    ### Loop through data for computing ###
    results = {"exact_match": 0, "f1": 0}
    for sample in tqdm(validation_dataset):

        question = sample["question"]
        context = sample["context"]
        prediction = inferencer.inference_model(question, context)
        
        ### Put Target into the Form SQuAD Wants it In ###
        pred = [{"prediction_text": prediction["answer"], "id": sample["id"]}]
        
        ### Compute Squad Score ###
        result = squad(pred, sample)

        exact_match = result["exact_match"].item()
        f1_score = result["f1"].item()

        results["exact_match"] += exact_match
        results["f1"] += f1_score

    results = {k:v/len(validation_dataset) for (k,v) in results.items()}

    print("Saving results to:", path_to_store)
    with open(path_to_store, "w") as f:
        json.dump(results, f)
    
    return results

if __name__ == "__main__":

    args = parser.parse_args()

    results = compute_squad_metrics(args.path_to_model_weights, 
                                    args.huggingface_model,
                                    args.path_to_store, 
                                    args.cache_dir)

    print("Squad Results:")
    print(results)