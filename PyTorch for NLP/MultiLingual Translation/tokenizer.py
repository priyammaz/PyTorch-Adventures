import pandas as pd
from tokenizers import Tokenizer, SentencePieceBPETokenizer
from tokenizers.processors import TemplateProcessing 

def train_tokenizer(path_to_data, path_to_store):
    dataset = pd.read_csv(path_to_data)
    target_text = list(dataset.target)

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train_from_iterator(
        target_text,
        vocab_size=32000,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens
    )

    tokenizer.save(path_to_store)

class MultilingualTokenizer:
    
    def __init__(self, path_to_vocab):
        
        self.path_to_vocab = path_to_vocab
        self.tokenizer = self.prepare_tokenizer()
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.special_tokens_dict = {"<unk>": self.tokenizer.token_to_id("<unk>"),
                                    "<eos>": self.tokenizer.token_to_id("<eos>"),
                                    "<pad>": self.tokenizer.token_to_id("<pad>"),
                                    "<bos>": self.tokenizer.token_to_id("<bos>")}


    def prepare_tokenizer(self):
        tokenizer = Tokenizer.from_file(self.path_to_vocab)
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<eos>", tokenizer.token_to_id("<eos>")),
                ("<bos>", tokenizer.token_to_id("<bos>"))
            ]
         )
        
        return tokenizer


    def encode(self, input):
        
        def _parse_tokenized(tokenized):
            return tokenized.ids
        
        if isinstance(input, str):
            tokenized = self.tokenizer.encode(input)
            tokenized = _parse_tokenized(tokenized)

        elif isinstance(input, (list, tuple)):
            tokenized = self.tokenizer.encode_batch(input)
            tokenized = [_parse_tokenized(t) for t in tokenized]
        
        return tokenized
    
    def decode(self, input, skip_special_tokens=True):

        if isinstance(input, list):
            
            if all(isinstance(item, list) for item in input):
                decoded = self.tokenizer.decode_batch(input, skip_special_tokens=skip_special_tokens)
            elif all(isinstance(item, int) for item in input):
                decoded = self.tokenizer.decode(input, skip_special_tokens=skip_special_tokens)
            
        return decoded
        

        

if __name__ == "__main__":
    path_to_data = "/mnt/datadrive/data/machine_translation/english2mutilingual_train.csv"
    path_to_store = "trained_tokenizer/sentencepiece_tokenizer.json"
    train_tokenizer(path_to_data, path_to_store)

    path_to_vocab = "trained_tokenizer/sentencepiece_tokenizer.json"
    tokenizer = MultilingualTokenizer(path_to_vocab)

    sample = "hello world"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode([encoded, encoded, encoded], skip_special_tokens=False)
    print(tokenizer.vocab_size)


    
