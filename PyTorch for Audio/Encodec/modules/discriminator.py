import torch
import torch.nn as nn
import torchaudio
import einops

from .conv import NormConv2d

def get_2d_padding(kernel_size, dilation = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)

class DiscriminatorSTFT(nn.Module):
    def __init__(self, filters=32, in_channels=1, out_channels=1, 
                 n_fft=1024, hop_length=256, win_length=1024, 
                 max_filters=1024, filters_scale=1, kernel_size=(3,9),
                 dilations=[1,2,4], stride=(1,2), normalized=True, 
                 norm="weight_norm", activation="LeakyReLU",
                 activation_params: dict = {'negative_slope': 0.2}):
        super().__init__()

        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)

        ### Spectrogram Operation ###
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, 
            window_fn=torch.hann_window, normalized=self.normalized, center=False, 
            pad_mode=None, power=None)
        
        ### Build Convs ###
        self.convs = nn.ModuleList()

        ### Initial input from our concatenated spectrogram (real and imaginary) ###
        spec_channels = 2 * self.in_channels
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
        )

        in_channels = min(filters_scale * self.filters, max_filters) # increase until we reach max
        
        ### for every conv (num convs set by dilations)
        
        for i, dilation in enumerate(dilations):
            out_channels = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(NormConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)),
                                         norm=norm))
            in_channels = out_channels # update for next operation

        ### Get the next out_channels after the dilations (+1) for a non-dilated convolution
        out_channels = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(NormConv2d(in_channels, out_channels, kernel_size=(kernel_size[0], kernel_size[0]),
                                     padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                     norm=norm))
        
        ### final conv to out_channels
        self.conv_post = NormConv2d(out_channels, self.out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                    norm=norm)
        
    def forward(self, x):

        fmap = []
        z = self.spec_transform(x) # returns both real and complex parts
        z = torch.cat([z.real, z.imag], dim=1) # concat together real and complex
        z = einops.rearrange(z, 'b c w t -> b c t w') # make it the standard image shape
     
        ### loop through convs while keeping intermediate feature maps
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        
        z = self.conv_post(z)

        return z, fmap
    
class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Multi-Scale STFT (MS-STFT) discriminator.
    """
    def __init__(self, filters=32, in_channels=1, out_channels=1,
                 n_ffts=[1024, 2048, 512], hop_lengths=[256, 512, 128],
                 win_lengths=[1024, 2048, 512], **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x):
        
        logits = []
        fmaps = []

        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)

        return logits, fmaps
