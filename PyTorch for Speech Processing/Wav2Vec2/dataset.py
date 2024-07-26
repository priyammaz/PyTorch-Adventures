import os
import numpy as np
import librosa
import torchaudio
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import padding_attention_masking, span_masking, sample_negatives
from model import Wav2Vec2Config

def precompute_audio_durations(path_to_data_root: str):

    splits = ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "test-clean"]

    for split in splits:
        
        path_to_split = os.path.join(path_to_data_root, split)

        if os.path.isdir(path_to_split):
            print(f"Computing Durations of {split}")
            for speaker in tqdm(os.listdir(path_to_split)):
                path_to_speaker = os.path.join(path_to_split, speaker)

                for section in os.listdir(path_to_speaker):
                    path_to_section = os.path.join(path_to_speaker, section)

                    ### Grab Files and Split FLAC Audios and Text Transcripts ###
                    audio_files = [file for file in os.listdir(path_to_section) if ".flac" in file]              
                
                    root_duration_dict = {"root": [], "duration": []}
                    for file in audio_files:
                        file_root = file.split(".")[0]
                        audio_duration = librosa.get_duration(path=os.path.join(path_to_section, file))
                        root_duration_dict["root"].append(file_root)
                        root_duration_dict["duration"].append(audio_duration)

                    data = pd.DataFrame(root_duration_dict)
                    path_to_section_duration = os.path.join(path_to_section, "audio_durations.csv")
                    data.to_csv(path_to_section_duration, index=False)


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
                 return_transcripts=True):
        
        if isinstance(include_splits, str):
            include_splits = [include_splits]

        self.sampling_rate = sampling_rate
        self.return_transcripts = return_transcripts

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

                        if (min_audio_duration <= duration <= max_audio_duration):                            
                            self.librispeech_data.append((full_path_to_audio_file, transcript))

    def __len__(self):
        return len(self.librispeech_data)
    
    def __getitem__(self, idx):
        
        ### Grab Path to Audio and Transcript ###
        path_to_audio, transcript = self.librispeech_data[idx]

        ### Load Audio ###
        audio, sr = torchaudio.load(path_to_audio)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sampling_rate)

        ### Normalize to Zero Mean Unit Variance ###
        normed_audio = ((audio - audio.mean()) / np.sqrt(audio.var() + 1e-7))
        
        if self.return_transcripts:
            return normed_audio, transcript
    
        else:
            return normed_audio

def Wav2Vec2CollateFunctionForPreTraining(config):

    """
    Just a simple wrapper on a collate function so I can pass config information
    """
    def collate_fn(batch):
        
        ### Pad Audios to Longest Audio ###
        audio_lengths = torch.tensor([audio.shape[-1] for audio in batch])
        max_audio_length = torch.max(audio_lengths)
        batch_size = len(batch)

        audio_batch = torch.zeros((batch_size, config.audio_input_channels, max_audio_length))
        for idx, (audio, length) in enumerate(zip(batch, audio_lengths)):
            audio_batch[idx, :, :length] = audio

        ### Compute Attention Mask (Post Convolutional Encoder) ###
        attention_mask = padding_attention_masking(batch, 
                                                   kernel_sizes=config.conv_kernels, 
                                                   strides=config.conv_strides)
        
        convolutional_encoder_output_shape = attention_mask.shape[-1]
        
        ### Compute Span Masks on the Encoded Features ###
        span_mask = span_masking(input_values_shape=(batch_size, convolutional_encoder_output_shape),
                                 masking_probability=config.masking_probability, 
                                 masking_span_length=config.masking_span_length,
                                 minimum_spans=config.minimum_spans,
                                 attention_mask=attention_mask)
        
        
        ### Sample Positives and Negatives ###
        sampled_negatives = sample_negatives(features_shape=(batch_size, convolutional_encoder_output_shape),
                                             num_negatives=config.num_negatives,
                                             span_mask=span_mask)
        

        ### Store Batch ###
        batch = {"input_values": audio_batch, 
                 "attention_mask": attention_mask, 
                 "span_mask": span_mask,
                 "sampled_negatives": sampled_negatives,
                 "audio_durations": audio_lengths}

        return batch

    return collate_fn


if __name__ == "__main__":
    path_to_data = "/mnt/datadrive/data/LibriSpeech/"

    ### Pre Compute Audio Durations ###
    # precompute_audio_durations(path_to_data)

    ### Define Dataset ###    
    dataset = LibriSpeechDataset(path_to_data, include_splits=["train-clean-100", "train-clean-360", "train-other-500"], return_transcripts=False)
    config = Wav2Vec2Config()
    loader = DataLoader(dataset, batch_size=64, collate_fn=Wav2Vec2CollateFunctionForPreTraining(config), num_workers=24)
    for data in tqdm(loader):
        pass
        
