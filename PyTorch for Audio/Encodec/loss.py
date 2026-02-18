import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        device='cuda'
    ):
        super().__init__()

        window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(sr=sampling_rate,n_fft=n_fft,n_mels=n_mel_channels,fmin=mel_fmin,fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).cuda().float()

        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, x):
        shape = x.shape
        
        ### If more than 1 channel, then we will run analysis per channel ###
        ### but this doesnt apply to the mono audio case! ###
        if len(shape) > 2:
            x = x.reshape(shape[0] * shape[1], -1)

        ### Reflect padding for even windowing ###
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(x, (p, p), "reflect")
        
        ### Run STFT ###
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=False,
        )

        ### Proj to Mel ###
        mel_output = torch.matmul(self.mel_basis, torch.sum(torch.pow(fft, 2), dim=[-1]))
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))

        # restore original shape [B, H, T]
        if len(shape) > 2:
            log_mel_spec = log_mel_spec.reshape(shape[0], shape[1], -1)

        return log_mel_spec
    
def generator_loss(
        fmap_real, # list[list[tensor]] -> K discriminators, L layers each
        fmap_fake, # list[list[tensor]]
        logits_fake, # list[tensor] -> K discriminator outputs
        input_wav,
        output_wav, 
        sample_rate=24000,
        num_mels=80
):
    
    ### Time Domain Loss ###
    time_loss = F.l1_loss(input_wav, output_wav)

    ### Frequency Domain Loss ###
    ### What we do here is try different window size/hop size combinations and aggregate together! ###
    ### It is pretty typical to have hop_size be 1/4 of the window. 
    ### The idea is from the uncertainty principle. Larger windows give better frequency resolution and ###
    ### poor time resolution. Small windows are opposite. So lets just do a bunch and add them together! ###
    frequency_loss = 0.0
    for i in range(5,12): # 2**5 -> 2**11 : 32 -> 2048
        window_size = 2 ** i
        hop_size = window_size // 4
        fft = Audio2Mel(n_fft=window_size, win_length=window_size, 
                        hop_length=hop_size, sampling_rate=sample_rate,
                        n_mel_channels=num_mels)
        input_mel = fft(input_wav)
        output_mel = fft(output_wav)

        ### Now we do both an L1 and and L2 Loss between these ###
        frequency_loss = frequency_loss + F.l1_loss(input_mel, output_mel) + F.mse_loss(input_mel, output_mel)
    
    if (fmap_real is not None) and (fmap_fake is not None) and (logits_fake is not None):

        K = len(logits_fake)
        L = len(fmap_fake[0])
    
        ### Generator Loss (hinge loss) ###
        generator_loss = 0.0
        for lfake in logits_fake:
            generator_loss = generator_loss + torch.mean(F.relu(1 - lfake))
        generator_loss = generator_loss / K

        ### Disc Feature Loss ###
        feature_loss = 0.0
        for freals, ffakes in zip(fmap_real, fmap_fake):
            for freal, ffake in zip(freals, ffakes):
                ### normalize by torch.mean(torch.abs(freal)) so layers with large magnitudes dont dominate
                feature_loss = feature_loss + F.l1_loss(freal, ffake) / torch.mean(torch.abs(freal))

        feature_loss = feature_loss / (K*L)
    
    else:
        generator_loss = torch.tensor([0.0], dtype=output_wav.dtype, device=output_wav.device, requires_grad=True)
        feature_loss = torch.tensor([0.0], dtype=output_wav.dtype, device=output_wav.device, requires_grad=True)

    return {
        "time_loss": time_loss, 
        "frequency_loss": frequency_loss,
        "generator_loss": generator_loss, 
        "feature_loss": feature_loss
    }

def discriminator_loss(logits_real, logits_fake):
    """hinge loss to train discriminator"""
    disc_loss = 0.0
    for lreal, lfake in zip(logits_real, logits_fake):
        disc_loss = disc_loss + torch.mean(F.relu(1-lreal)) + torch.mean(F.relu(1+lfake))
    disc_loss = disc_loss / len(logits_real)
    return disc_loss