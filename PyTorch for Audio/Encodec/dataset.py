"""
Quick dataset script!
"""
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

class AudioDataset(Dataset):

    """
    The dataset is build from a text file of paths to 
    audio files we want to train on. From each audio 
    file a segment will be randomly sampled and then 
    passed to the model
    """
    def __init__(self, 
                 path_to_txt, 
                 segment_length=24000, 
                 sample_rate=24000):

        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        # Read all audio file paths from the text file
        with open(path_to_txt, 'r') as f:
            self.audio_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.audio_paths)} audio files from {path_to_txt}")

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):

        audio_path = self.audio_paths[idx]

        waveform, sr = torchaudio.load(audio_path)
            
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)    

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Get audio duration
        audio_length = waveform.shape[1]

        # Randomly sample a segment
        if audio_length > self.segment_length:
            # Random start position
            start = torch.randint(0, audio_length - self.segment_length - 1, (1,)).item()
            waveform = waveform[:, start:start + self.segment_length]
        elif audio_length < self.segment_length:
            # Pad if audio is shorter than segment_length
            padding = self.segment_length - audio_length
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)

        return waveform
