import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2CTCTokenizer
from dataclasses import dataclass
from transformers import Wav2Vec2Processor

from utils import (
    compute_span_mask,
    sample_negative_indices, 
    compute_sub_attention_mask, 
    Wav2Vec2Config
)


class LibriSpeechDataset(Dataset):

    """
    LibriSpeechDataset downloaded from OpenSLR: https://www.openslr.org/12

    There are 5 splits downloaded, 3 which are for training and 3 for testing:

        Training: ["train-clean-100", "train-clean-360", "train-other-500"]
        Validation: ["dev-clean", "test-clean"]

    Makes sure to run the dataset.precompute_audio_durations before hand so you have those duration 
    files available
    """
    def __init__(self, 
                 path_to_data_root, 
                 include_splits=["train-clean-100", "train-clean-360", "train-other-500"],
                 max_audio_duration=20.0, 
                 min_audio_duration=2.0,
                 sampling_rate=16000,
                 num_audio_channels=1, 
                 truncate_audio=True,
                 return_transcripts=True,
                 hf_model_name="facebook/wav2vec2-base"):
        
        if isinstance(include_splits, str):
            include_splits = [include_splits]

        self.sampling_rate = sampling_rate
        self.return_transcripts = return_transcripts
        self.truncate_audio = truncate_audio
        self.num_audio_channels = num_audio_channels
        self.min_audio_samples = int(min_audio_duration * sampling_rate)
        self.max_audio_samples = int(max_audio_duration * sampling_rate)

        ### GET PATH TO ALL AUDIO/TEXT FILES ###
        self.librispeech_data = []
        for split in include_splits:
            path_to_split = os.path.join(path_to_data_root, split)

            for speaker in os.listdir(path_to_split):
                path_to_speaker = os.path.join(path_to_split, speaker)

                for section in os.listdir(path_to_speaker):
                    path_to_section = os.path.join(path_to_speaker, section)

                    ### Grab Files and Split FLAC Audios and Text Transcripts ###
                    files = os.listdir(path_to_section)
                    transcript_file = [path for path in files if ".txt" in path][0]

                    ### Grab Audio Durations (from dataset.precompute_audio_duration)
                    audio_durations = pd.read_csv(os.path.join(path_to_section, "audio_durations.csv"))
                    audio_durations_dict = audio_durations.set_index("root")["duration"].to_dict()


                    ### Load Transcripts ###
                    with open(os.path.join(path_to_section, transcript_file), "r") as f:
                        transcripts = f.readlines()

                    ### Split Transcripts by Audio Filename and Transcript ###
                    for line in transcripts:
                        split_line = line.split()
                        audio_root = split_line[0]
                        audio_file = audio_root + ".flac"
                        full_path_to_audio_file = os.path.join(path_to_section, audio_file)
                        transcript = " ".join(split_line[1:]).strip()
                        duration = audio_durations_dict[audio_root]

                        ### If the audio duration is greater that the minimum and duration less than maximum or we enable truncation then save sample ###
                        if (duration >= min_audio_duration) and (duration <= max_audio_duration or truncate_audio):
                            self.librispeech_data.append((full_path_to_audio_file, transcript))
                        
        if return_transcripts:
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(hf_model_name)

    def __len__(self):
        return len(self.librispeech_data)
    
    def __getitem__(self, idx):
        
        ### Grab Path to Audio and Transcript ###
        path_to_audio, transcript = self.librispeech_data[idx]

        ### Load Audio ###
        audio, orig_sr = torchaudio.load(path_to_audio, num_frames=self.max_audio_samples)

        ### Resample Audio ###
        if orig_sr is not self.sampling_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=self.sampling_rate)

        ### If only a single channel squeeze out the channel dimension ###
        if self.num_audio_channels == 1:
            audio = audio.squeeze()

        ### Normalize to Zero Mean Unit Variance ###
        normed_audio = ((audio - audio.mean()) / np.sqrt(audio.var() + 1e-7))

        if self.return_transcripts:
            tokenized_transcript = torch.tensor(self.tokenizer.encode(transcript))
            batch = {"input_values": normed_audio, 
                     "labels": tokenized_transcript}
        else:
            batch = {"input_values": normed_audio}

        return batch

def Wav2Vec2CollateFunctionForPreTraining(config):

    """
    Just a simple wrapper on a collate function so I can pass config information
    """
    def collate_fn(batch):

        """
        This collate function is basically the heart of our implementation! It includes everything we need for training
        such as attention masks, sub_attention_masks, span_masks and our sampled negatives!
        """
        
        ### Grab Audios from our Batch Dictionary ###
        batch_audios = [sample["input_values"] for sample in batch]

        ### Pad Audios to the Longest Audio and Create Attention Mask ###
        attention_mask = [torch.ones(len(audio)) for audio in batch_audios]
        audios = torch.nn.utils.rnn.pad_sequence(batch_audios, batch_first=True, padding_value=0.0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        ### Compute Sub Attention Mask (Post Convolutional Encoder) ###
        sub_attention_mask = compute_sub_attention_mask(config, attention_mask)

        ### Compute Span Masks on the Encoded Features ###
        span_mask = compute_span_mask(shape=tuple(sub_attention_mask.shape),
                                      mask_prob=config.masking_probability, 
                                      mask_length=config.masking_span_length,
                                      min_masks=config.minimum_spans,
                                      attention_mask=sub_attention_mask)
        

        ### Sample Negatives ###
        sampled_negatives = sample_negative_indices(features_shape=tuple(sub_attention_mask.shape),
                                                    num_negatives=config.num_negatives,
                                                    mask_time_indices=span_mask)
        

        ### Store Batch ###
        batch = {"input_values": audios, 
                 "attention_mask": attention_mask.bool(), 
                 "sub_attention_mask": sub_attention_mask.bool(),
                 "mask_time_indices": span_mask,
                 "sampled_negative_indices": sampled_negatives}
    
        return batch

    return collate_fn

@dataclass
class Wav2Vec2CollateFunctionForCTC:
    """
    This collate function was taken directly from the 🤗 Huggingface Wav2Vec2 Finetuning example!!!
    https://huggingface.co/blog/fine-tune-wav2vec2-english

    """

    processor: Wav2Vec2Processor

    def __call__(self, features):
  
        ### Create a list of dictionaries containing our audio and label_ids ###
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        ### Pads audio with zeros to complete the batch ###
        ### When we define the wav2vec2processor that we pass in, we will not include any attention mask on the audio ###
        ### Theres really no need in finetuning, but to do this correctly we will group by audio lenghts in the trainer so we dont have ###
        ### too much extra padding being added on ###
        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        ### When used under the `as_target_processor()` context, it uses the tokenizer by default ###
        ### You can get more information here: https://huggingface.co/docs/transformers/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor ###
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors="pt",
            )

        ### When using the target processor, it will also return an attention mask! Use it to fill -100 into labels so we ignore them ###
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

if __name__ == "__main__":

    path_to_data = "/mnt/datadrive/data/LibriSpeech/"
    config = Wav2Vec2Config()

    ### Test Pretraining Loader ###    
    dataset = LibriSpeechDataset(path_to_data, include_splits=["dev-clean", "test-clean"], return_transcripts=False, max_audio_duration=20)
    
    loader = DataLoader(dataset, batch_size=4, collate_fn=Wav2Vec2CollateFunctionForPreTraining(config), num_workers=0)
    for data in loader:
        print(data)
        break
