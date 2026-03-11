import yaml
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from modules.encodec import EncodecModel, EnCodecConfig

def load_yaml(path_to_yaml):
    with open(path_to_yaml, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_audio(path_to_audio, sr):
    waveform, audio_sr = torchaudio.load(path_to_audio)

    if audio_sr != sr:
        resampler = torchaudio.transforms.Resample(audio_sr, sr)
        waveform = resampler(waveform)    
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.unsqueeze(0)


def main():
    
    PATH_TO_CONFIG = "configs/config.yaml"
    PATH_TO_WEIGHTS = "work_dir/EnCodecTrainerLibriTTSLocal/final_checkpoint/pytorch_model.bin"

    with open("data/test.txt", "r") as f:
        audio_files = f.readlines()

    path_to_audio = audio_files[0].strip()

    config = load_yaml(PATH_TO_CONFIG)
    encodec_config = EnCodecConfig(**config["generator_config"])
    model = EncodecModel(encodec_config)
    state_dict = torch.load(PATH_TO_WEIGHTS)
    model.load_state_dict(state_dict)
    model.eval()

    audio = load_audio(path_to_audio, config["training_config"]["sampling_rate"])
    
    with torch.no_grad():
        tokens, scale = model.tokenize(audio)

    print(tokens)


if __name__ == "__main__":
    main()