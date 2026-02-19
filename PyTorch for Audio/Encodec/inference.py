import yaml
import torch
import torchaudio.transforms as T
import torchaudio
import warnings
import matplotlib.pyplot as plt
from utils import save_audios
warnings.filterwarnings("ignore")

from modules.encodec import EncodecModel, EnCodecConfig

def load_yaml(path_to_yaml):
    with open(path_to_yaml, 'r') as file:
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

def compute_mel(audio, sr, n_fft=1024, hop_length=256, n_mels=80):
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0
    )
    mel = mel_transform(audio)  # (B, n_mels, T)
    mel = torch.log(mel + 1e-5)
    return mel

def plot_mel_comparison(original, reconstruction, sr):
    mel_orig = compute_mel(original.squeeze(0), sr)
    mel_recon = compute_mel(reconstruction.detach().cpu().squeeze(0), sr)

    mel_orig = mel_orig.squeeze(0).numpy()
    mel_recon = mel_recon.squeeze(0).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    im0 = axes[0].imshow(mel_orig, aspect="auto", origin="lower")
    axes[0].set_title("Original Audio Mel Spectrogram")
    axes[0].set_ylabel("Mel bins")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mel_recon, aspect="auto", origin="lower")
    axes[1].set_title("Reconstructed Audio Mel Spectrogram")
    axes[1].set_ylabel("Mel bins")
    axes[1].set_xlabel("Time")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig("figs/french_reconstruction.png")

if __name__ == "__main__":

    PATH_TO_CONFIG = "configs/config.yaml"
    PATH_TO_WEIGHTS = "work_dir/EnCodecTrainerLibriTTSLocal/final_checkpoint/pytorch_model.bin"
    PATH_TO_AUDIO = "ood_samples/french_sample.mp3"
    config = load_yaml(PATH_TO_CONFIG)

    encodec_config = EnCodecConfig(**config["generator_config"]
    )
    model = EncodecModel(encodec_config)
    state_dict = torch.load(PATH_TO_WEIGHTS)
    model.load_state_dict(state_dict)   

    audio = load_audio(PATH_TO_AUDIO, config["training_config"]["sampling_rate"])
    
    tokens, scale = model.tokenize(audio)
    print("Predicted Tokens")
    print(tokens)

    reconstruction = model.decode(tokens, scale)

    plot_mel_comparison(
        audio.cpu(),
        reconstruction.cpu(),
        config["training_config"]["sampling_rate"]
    )

    save_audios(reconstruction, "ood_samples/french_reconstruction.wav", 24000)
    

    
    
