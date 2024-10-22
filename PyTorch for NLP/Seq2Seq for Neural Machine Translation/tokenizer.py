import os
import glob
import argparse
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing

special_token_dict = {"unknown_token": "[UNK]",
                      "pad_token": "[PAD]", 
                      "start_token": "[BOS]",
                      "end_token": "[EOS]"}

def train_tokenizer(path_to_data_root):

    """
    We need to train a WordPiece tokenizer on our french data (as our regular tokenizers are mostly for English!)
    I set all the special tokens we need above:
        unkown_token: Most important incase tokenizer sees a token not a part of our original token set
        pad_token: Padding for the french text
        start_token: Prepend all french text with start token so the decoder has an input to start generating from
        end_token: Append all french text with end token so decoder knowsn when to stop generating anymore. 

    The only thing in here to keep in mind is the normalizers. There are some issues with how the same letter can 
    be represented in Unicode, so we have to do unicode normalization.

    For example: 

    "é" can be written as either (\u00E9) as a single unicode
    "é" can also be written as "e" + ' where we break the accents off of the e and write as a sequence of 2 unicode characters \u0065\u0301

    We want all our data to be in one or the either for some consistency, so we will be using NMC which tries to represent these characters
    with just a single unicode
    """
    
    ### Prepare Tokenizer Definition ###
    tokenizer = Tokenizer(WordPiece(unk_token=special_token_dict["unknown_token"]))
    tokenizer.normalizer = normalizers.Sequence([NFC(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()
    
    ### Find all Target Language Files (ours are french ending with fr) ###
    french_files = glob.glob(os.path.join(path_to_data_root, "**/*.fr"))
    
    ### Train Tokenizer ###
    trainer = WordPieceTrainer(vocab_size=32000, special_tokens=list(special_token_dict.values()))
    tokenizer.train(french_files, trainer)
    tokenizer.save("trained_tokenizer/french_wp.json")

class FrenchTokenizer:

    """
    This is just a wrapper on top of the trained tokenizer to put together all the functionality we need 
    for encoding and decoding
    """
    
    def __init__(self, path_to_vocab, truncate=False, max_length=512):
        
        self.path_to_vocab = path_to_vocab
        self.tokenizer = self.prepare_tokenizer()
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.special_tokens_dict = {"[UNK]": self.tokenizer.token_to_id("[UNK]"),
                                    "[PAD]": self.tokenizer.token_to_id("[PAD]"),
                                    "[BOS]": self.tokenizer.token_to_id("[BOS]"),
                                    "[EOS]": self.tokenizer.token_to_id("[EOS]")}

        self.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
                ("[BOS]", self.tokenizer.token_to_id("[BOS]"))
            ]
        )
        
        self.truncate = truncate
        if self.truncate:
            self.max_len = max_length - self.post_processor.num_special_tokens_to_add(is_pair=False)

    def prepare_tokenizer(self):
        tokenizer = Tokenizer.from_file(self.path_to_vocab)
        tokenizer.decoder = decoders.WordPiece()
        return tokenizer

    def encode(self, input):
        
        def _parse_process_tokenized(tokenized):
            if self.truncate:
                tokenized.truncate(self.max_len, direction="right")
            tokenized = self.post_processor.process(tokenized)
            return tokenized.ids
        
        if isinstance(input, str):
            tokenized = self.tokenizer.encode(input)
            tokenized = _parse_process_tokenized(tokenized)

        elif isinstance(input, (list, tuple)):
            tokenized = self.tokenizer.encode_batch(input)
            tokenized = [_parse_process_tokenized(t) for t in tokenized]
        
        return tokenized
    
    def decode(self, input, skip_special_tokens=True):

        if isinstance(input, list):
            
            if all(isinstance(item, list) for item in input):
                decoded = self.tokenizer.decode_batch(input, skip_special_tokens=skip_special_tokens)
            elif all(isinstance(item, int) for item in input):
                decoded = self.tokenizer.decode(input, skip_special_tokens=skip_special_tokens)
            
        return decoded
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizer Prep")

    parser.add_argument(
        "--path_to_data_root", 
        required=True, 
        help="Path to where you want to save the final tokenized dataset",
        type=str
    )

    args = parser.parse_args()

    path_to_data_root = "/mnt/datadrive/data/machine_translation/english2french/"
    train_tokenizer(args.path_to_data_root)

    tokenizer = FrenchTokenizer("trained_tokenizer/french_wp.json")
    sentence = "Héllo world!"
    enc = tokenizer.encode(sentence)
    print(enc)
    dec = tokenizer.decode(enc, skip_special_tokens=False)
    print(dec)