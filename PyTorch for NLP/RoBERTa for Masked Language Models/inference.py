import torch
from transformers import RobertaTokenizerFast
from utils import RobertaConfig
from model import RobertaForQuestionAnswering
from safetensors.torch import load_file

class InferenceModel:

    """
        Quick inference function that works with the models we have trained!
    """

    def __init__(self, path_to_weights, huggingface_model=True):

        ### Init Config with either Huggingface Backbone or our own ###
        self.config = RobertaConfig(pretrained_backbone="pretrained_huggingface" if huggingface_model else "random")

        ### Load Tokenizer ###
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.config.hf_model_name)

        ### Load Model ###
        self.model = RobertaForQuestionAnswering(self.config)
        weights = load_file(path_to_weights)
        self.model.load_state_dict(weights)
        self.model.eval()


    def inference_model(self, 
                        question, 
                        context):
                
        ### Tokenize Text
        inputs = self.tokenizer(text=question,
                                text_pair=context,
                                max_length=self.config.context_length,
                                truncation="only_second",
                                return_tensors="pt")
        
        ### Pass through Model ####
        with torch.no_grad():
            start_token_logits, end_token_logits = self.model(**inputs)

        ### Grab Start and End Token Idx ###
        start_token_idx = start_token_logits.squeeze().argmax().item()
        end_token_idx = end_token_logits.squeeze().argmax().item()

        ### Slice Tokens and then Decode with Tokenizer (+1 because slice is not right inclusive) ###
        tokens = inputs["input_ids"].squeeze()[start_token_idx:end_token_idx+1]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        
        prediction = {"start_token_idx": start_token_idx, 
                      "end_token_idx": end_token_idx, 
                      "answer": answer}
        
        return prediction

if __name__ == "__main__":

    ### Sample Text ###
    context = "Veronica favorite perfume scents are rose and sandlewood, but sometimes she likes other ones as well"
    question = "What is Veronica's favorite scent for her perfumes?"

    ### Inference Model ###
    path_to_weights = "work_dir/finetune_qa_hf_roberta_backbone/model.safetensors"
    inferencer = InferenceModel(path_to_weights=path_to_weights, huggingface_model=True)
    prediction = inferencer.inference_model(question, context)
    print(prediction)