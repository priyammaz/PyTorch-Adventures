import yaml
import torch
import torch.nn
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
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

def main():

    PATH_TO_CONFIG = "configs/config.yaml"
    PATH_TO_WEIGHTS = "work_dir/EnCodecTrainerLibriTTSLocal/final_checkpoint/pytorch_model.bin"

    with open("data/test.txt", "r") as f:
        audio_files = f.readlines()
    
    config = load_yaml(PATH_TO_CONFIG)

    encodec_config = EnCodecConfig(**config["generator_config"])
    model = EncodecModel(encodec_config)
    state_dict = torch.load(PATH_TO_WEIGHTS)
    model.load_state_dict(state_dict)   
    
    num_codebooks = encodec_config.num_quantizers
    codebook_size = encodec_config.codebook_size

    usage = torch.zeros(num_codebooks, codebook_size, dtype=torch.long)


    for path_to_audio in tqdm(audio_files):

        audio = load_audio(
            path_to_audio.strip(),
            config["training_config"]["sampling_rate"]
        )

        with torch.no_grad():
            tokens, scale = model.tokenize(audio)

        # tokens: (num_codebooks, 1, T)
        tokens = tokens.squeeze(1)  # (num_codebooks, T)

        for q in range(num_codebooks):
            inds = tokens[q]  # (T,)
            usage[q] += torch.bincount(
                inds,
                minlength=codebook_size
            )

    print("\nCodebook Usage Statistics\n")

    for q in range(num_codebooks):

        counts = usage[q].float()

        probs = counts / counts.sum()

        perplexity = torch.exp(
            -(probs * torch.log(probs + 1e-10)).sum()
        )

        dead = (counts == 0).sum().item()

        print(f"Codebook {q}")
        print(f"  Perplexity: {perplexity:.2f} / {codebook_size}")
        print(f"  Used codes: {(counts>0).sum().item()} / {codebook_size}")
        print(f"  Dead codes: {dead}")
        print(f"  Usage %: {(counts>0).float().mean()*100:.2f}%")
        print()

    global_counts = usage.sum(dim=0)
    global_probs = global_counts / global_counts.sum()

    global_perplexity = torch.exp(
        -(global_probs * torch.log(global_probs + 1e-10)).sum()
    )

    print("Global perplexity:", global_perplexity.item())

    num_codebooks = usage.shape[0]

    import matplotlib.pyplot as plt
    import numpy as np
    
    usage_np = usage.numpy()

    usage_sorted = np.sort(usage_np, axis=1)[:, ::-1]

    plt.figure(figsize=(12, 6))

    plt.imshow(
        np.log(usage_sorted + 1),  # +1 avoids log(0)
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(label="log(Count)")

    plt.xlabel("Code Rank (most used → least used)")
    plt.ylabel("Codebook Index")
    plt.title("Sorted Codebook Usage Heatmap")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    main()
    

    
    
