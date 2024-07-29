import os
import numpy as np
import librosa
import torchaudio
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import (
    compute_span_mask,
    sample_negative_indices,
    compute_encoded_lengths, 
    compute_sub_attention_mask, 
    Wav2Vec2Config
)


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
                 truncate_audio=True,
                 return_transcripts=True):
        
        if isinstance(include_splits, str):
            include_splits = [include_splits]

        self.sampling_rate = sampling_rate
        self.return_transcripts = return_transcripts
        self.truncate_audio = truncate_audio
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
                        

    def __len__(self):
        return len(self.librispeech_data)
    
    def __getitem__(self, idx):
        
        ### Grab Path to Audio and Transcript ###
        path_to_audio, transcript = self.librispeech_data[idx]

        ### Load Audio ###
        audio, sr = librosa.load(path_to_audio, sr=self.sampling_rate)

        ### Truncate Audio ###
        if (len(audio) > self.max_audio_samples) and self.truncate_audio:
            audio = audio[:self.max_audio_samples]

        ### Normalize to Zero Mean Unit Variance ###
        normed_audio = ((audio - audio.mean()) / np.sqrt(audio.var() + 1e-7))

        ### Convert to Tensor ###
        normed_audio = torch.from_numpy(normed_audio)

        if self.return_transcripts:
            return normed_audio, transcript
    
        else:
            return normed_audio

def Wav2Vec2CollateFunctionForPreTraining(config):

    """
    Just a simple wrapper on a collate function so I can pass config information
    """
    def collate_fn(batch_audios):
        
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


if __name__ == "__main__":
    path_to_data = "/mnt/datadrive/data/LibriSpeech/"

    ### Pre Compute Audio Durations ###
    # precompute_audio_durations(path_to_data)

    ### Define Dataset ###    
    dataset = LibriSpeechDataset(path_to_data, include_splits=["dev-clean", "test-clean"], return_transcripts=False, max_audio_duration=20)
    config = Wav2Vec2Config()
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=4, collate_fn=Wav2Vec2CollateFunctionForPreTraining(config), num_workers=24)
    for data in tqdm(loader):
        print(data)
        break
        
