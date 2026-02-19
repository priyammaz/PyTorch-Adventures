import os
import torch
import torchaudio
import matplotlib.pyplot as plt

root = "work_dir/EnCodecTrainerLibriTTSLocal"
original_folder = os.path.join(root, "gen_inputs")

# order matters here
iters_to_plot = [2000, 100000, 200000]

n_mels = 80
device = "cpu"

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=24000,
    n_fft=512,
    hop_length=128,
    n_mels=n_mels,
).to(device)

to_db = torchaudio.transforms.AmplitudeToDB().to(device)

# pick one file
filename = sorted(os.listdir(original_folder))[2]
print("Using file:", filename)
print(filename)

def load_mel(path):
    wav, sr = torchaudio.load(path)
    wav = wav.to(device)
    mel = mel_transform(wav)
    mel = to_db(mel)
    return mel.squeeze(0).cpu()

ordered_paths = []
ordered_labels = []

# original FIRST
ordered_paths.append(os.path.join(original_folder, filename))
ordered_labels.append("Original")

# then generations in specified order
for i in iters_to_plot:
    path = os.path.join(root, f"gens_iter_{i}", filename)
    if os.path.exists(path):
        ordered_paths.append(path)
        ordered_labels.append(f"Generated (iter {i})")

mels = [load_mel(p) for p in ordered_paths]


plt.figure(figsize=(12, 3 * len(mels)))

for idx, (mel, label) in enumerate(zip(mels, ordered_labels)):
    ax = plt.subplot(len(mels), 1, idx + 1)
    im = ax.imshow(mel, origin="lower", aspect="auto")
    ax.set_title(label)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("figs/gen_spectrogram.png")