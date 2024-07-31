import os
import librosa
import pandas as pd
from tqdm import tqdm
import argparse

def precompute_audio_durations(path_to_data_root: str):
   
    """
    This is just a quick script to create some CSV files that contain the duration of the audio files 
    in our Librispeech datasets. This will just be helpful to prefilter audio that is too short or too long.
    """

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Quick duration parsing of audio")
    parser.add_argument(
        "--path_to_librispeech_data", 
        default="data/LibriSpeech",
        type=str
    )

    args = parser.parse_args()

    precompute_audio_durations(args.path_to_librispeech_data)