"""
EnCodec only uses the MultiScaleSTFTDiscriminator, but we additionally add on
MultiPeriod and MultiScale (time domain) discriminators from the HIFIGAN so we 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
import torchaudio
import einops

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

from .conv import NormConv2d

@dataclass
class DisciminatorConfig:

    in_channels: int = 1
    out_channels: int = 1

    ### Filter Control ###
    filters: int = 32
    max_filters: int = 1024
    filter_scale: int = 1

    ### Multiscale STFT Discriminator ###
    use_multiscale_freq_discrim: bool = True
    n_ffts: Tuple[int, ...] = (2048, 1024, 512, 256, 128)
    hop_lengths: Tuple[int, ...] = (512, 256, 128, 64, 32)
    win_lengths: Tuple[int, ...] = (2048, 1024, 512, 256, 128)
    kernel_size: Tuple[int, ...] = (3,9)
    dilations: Tuple[int, ...] = (1,2,4)
    stride: Tuple[int, ...] = (1,2)
    normalized: bool = True
    norm: str = "weight_norm"
    activation: str = "LeakyReLU"
    activation_params: Dict[str, Any] = field(default_factory=lambda: {'negative_slope': 0.2})
    
    ### Multiscale (time) Discriminator ###
    use_multiscale_time_discrim: bool = True
    msd_num_downsamples: int = 2
    msd_kernel_size: int = 5
    msd_stride: int = 3

    ### Multiperiod Discriminator ###
    use_multiperiod_time_discrim: bool = True
    mpd_periods: Tuple[int, ...] = (2,3,5,7,11)
    
def init_weights(m, mean=0.0, std=0.01):
    """standard weight init for all Conv modules"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # NormConv2d nests the Conv inside the module so we have to 
        # go one more in!
        if hasattr(m, "conv"):
            m = m.conv
        m.weight.data.normal_(mean, std)

def get_1d_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_2d_padding(kernel_size, dilation = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)

class DiscriminatorSTFT(nn.Module):
    def __init__(self, filters=32, in_channels=1, out_channels=1, 
                 n_fft=1024, hop_length=256, win_length=1024, 
                 max_filters=1024, filter_scale=1, kernel_size=(3,9),
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

        in_channels = min(filter_scale * self.filters, max_filters) # increase until we reach max
        
        ### for every conv (num convs set by dilations)
        
        for i, dilation in enumerate(dilations):
            out_channels = min((filter_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(NormConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)),
                                         norm=norm))
            in_channels = out_channels # update for next operation

        ### Get the next out_channels after the dilations (+1) for a non-dilated convolution
        out_channels = min((filter_scale ** (len(dilations) + 1)) * self.filters, max_filters)
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
    def __init__(self, config):
        
        super().__init__()
        
        assert len(config.n_ffts) == len(config.hop_lengths) == len(config.win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(config.filters, in_channels=config.in_channels, out_channels=config.out_channels,
                              n_fft=config.n_ffts[i], win_length=config.win_lengths[i], hop_length=config.hop_lengths[i],
                              max_filters=config.max_filters, filter_scale=config.filter_scale, kernel_size=config.kernel_size, 
                              dilations=config.dilations, stride=config.stride, normalized=config.normalized,
                              norm=config.norm, activation=config.activation, activation_params=config.activation_params)
            for i in range(len(config.n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, real, gen):

        real_outs = []
        gen_outs = []
        real_feat_maps = []
        gen_feat_maps = []

        for discrim in self.discriminators:

            real_out, real_feat_map = discrim(real)
            gen_out, gen_feat_map = discrim(gen)

            real_outs.append(real_out)
            gen_outs.append(gen_out)
            real_feat_maps.append(real_feat_map)
            gen_feat_maps.append(gen_feat_map)

        return real_outs, gen_outs, real_feat_maps, gen_feat_maps
            
class PeriodicDiscriminator(nn.Module):
    """A smaller version of the original PeriodicDiscriminator to reduce
    parameter count to make it more similar to the MSSTFT Discriminator"""
    def __init__(self, 
                 period, 
                 in_channels=1, 
                 kernel_size=5, 
                 stride=3, 
                 filters=32, 
                 filter_scale=1, 
                 max_filters=1024):
        
        super(PeriodicDiscriminator, self).__init__()
        
        self.period = period
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_block = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(in_channels, min(filters, max_filters), (kernel_size, 1), (stride, 1), padding=(get_1d_padding(kernel_size),0))), 
                weight_norm(nn.Conv2d(min(filters, max_filters), min(filters * filter_scale, max_filters), (kernel_size, 1), (stride, 1), (get_1d_padding(kernel_size),0))), 
                weight_norm(nn.Conv2d(min(filters * filter_scale, max_filters), min(filters * filter_scale**2, max_filters), (kernel_size, 1), (stride, 1), (get_1d_padding(kernel_size),0))), 
                weight_norm(nn.Conv2d(min(filters * filter_scale**2, max_filters), min(filters * filter_scale**3, max_filters), (kernel_size, 1), (stride, 1), (get_1d_padding(kernel_size),0))), 
                weight_norm(nn.Conv2d(min(filters * filter_scale**3, max_filters), min(filters * filter_scale**4, max_filters), (kernel_size, 1), 1, padding=(get_1d_padding(kernel_size),0)))
            ]
        )

        self.output_proj = weight_norm(nn.Conv2d(min(filters * filter_scale**4, max_filters), 1, (3,1), 1, (1,0)))

        self.apply(init_weights)

    def forward(self, x):

        feature_maps = []
   
        batch_size, channels, seq_len = x.shape

        if seq_len % self.period != 0:
            n_pad = self.period - (seq_len % self.period)
            x = F.pad(x, (0,n_pad), "reflect")
            seq_len += n_pad

        x = x.reshape(batch_size, channels, seq_len//self.period, self.period)

        for conv in self.conv_block:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        x = self.output_proj(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_maps

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, config):
        super(MultiPeriodDiscriminator, self).__init__()

        self.config = config

        self.discriminators = nn.ModuleList(
            [
                PeriodicDiscriminator(p, 
                                      in_channels=config.in_channels,
                                      kernel_size=config.msd_kernel_size, 
                                      stride=config.msd_stride, 
                                      filters=config.filters, 
                                      filter_scale=config.filter_scale,
                                      max_filters=config.max_filters)

                for p in config.mpd_periods
            ]
        )

    def forward(self, real, gen):

        real_outs = []
        gen_outs = []
        real_feat_maps = []
        gen_feat_maps = []

        for discrim in self.discriminators:
            real_out, real_feat_map = discrim(real)
            gen_out, gen_feat_map = discrim(gen)

            real_outs.append(real_out)
            gen_outs.append(gen_out)
            real_feat_maps.append(real_feat_map)
            gen_feat_maps.append(gen_feat_map)

        return real_outs, gen_outs, real_feat_maps, gen_feat_maps

class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False, in_channels=1, filters=32, filter_scale=1, max_filters=1024):
        super(ScaleDiscriminator, self).__init__()

        norm = weight_norm if not use_spectral_norm else spectral_norm

        self.conv_block = nn.ModuleList(
            [
                norm(nn.Conv1d(in_channels, min(filters, max_filters), 15, 1, padding=7)), 
                norm(nn.Conv1d(min(filters, max_filters), min(filters * filter_scale, max_filters), 41, 2, groups=4, padding=20)), 
                norm(nn.Conv1d(min(filters * filter_scale, max_filters), min(filters * filter_scale**2, max_filters), 41, 2, groups=16, padding=20)), 
                norm(nn.Conv1d(min(filters * filter_scale**2, max_filters), min(filters * filter_scale**3, max_filters), 41, 4, groups=16, padding=20)), 
                norm(nn.Conv1d(min(filters * filter_scale**3, max_filters), min(filters * filter_scale**4, max_filters), 41, 4, groups=16, padding=20)), 
                norm(nn.Conv1d(min(filters * filter_scale**4, max_filters), min(filters * filter_scale**5, max_filters), 41, 1, groups=16, padding=20)), 
                norm(nn.Conv1d(min(filters * filter_scale**5, max_filters), min(filters * filter_scale**6, max_filters), 5, 1, padding=2))
            ]
        )

        self.output_proj = norm(nn.Conv1d(min(filters * filter_scale**6, max_filters), 1, 3, 1, padding=1))

        self.apply(init_weights)
    
    def forward(self, x):

        feature_maps = []
        
        for conv in self.conv_block:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        x = self.output_proj(x)
        feature_maps.append(x)

        return x, feature_maps

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, config):
        super(MultiScaleDiscriminator, self).__init__()

        self.config = config

        args = {"filters": config.filters, 
                "filter_scale": config.filter_scale, 
                "max_filters": config.max_filters, 
                "in_channels": config.in_channels}
        self.discriminator = nn.ModuleList(
            [ScaleDiscriminator(use_spectral_norm=True, **args)] + [ScaleDiscriminator(use_spectral_norm=False, **args) for _ in range(config.msd_num_downsamples)]
        )

        self.meanpools = nn.ModuleList(
            [
                nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(config.msd_num_downsamples)
            ]
        )

    def forward(self, real, gen):

        real_outs = []
        gen_outs = []
        real_feat_maps = []
        gen_feat_maps = []

        for i, discrim in enumerate(self.discriminator):

            if i != 0:
                real = self.meanpools[i-1](real)
                gen = self.meanpools[i-1](gen)

            real_out, real_feat_map = discrim(real)
            gen_out, gen_feat_map = discrim(gen)
        
            real_outs.append(real_out)
            real_feat_maps.append(real_feat_map)
            gen_outs.append(gen_out)
            gen_feat_maps.append(gen_feat_map)

        return real_outs, gen_outs, real_feat_maps, gen_feat_maps
    
class Discriminator(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.config = config 
        
        self.use_multiscale_freq_discrim = False
        self.use_multiscale_time_discrim = False
        self.use_multiperiod_time_discrim = False

        if config.use_multiscale_freq_discrim:
            self.use_multiscale_freq_discrim = True
            self.stft_discrim = MultiScaleSTFTDiscriminator(config)

        if config.use_multiscale_time_discrim:
            self.use_multiscale_time_discrim = True
            self.time_discrim = MultiScaleDiscriminator(config)
        
        if config.use_multiperiod_time_discrim:
            self.use_multiperiod_time_discrim = True
            self.period_discrim = MultiPeriodDiscriminator(config)

    def forward(self, real, gen):

        real_outs = []
        gen_outs = []
        real_feat_maps = []
        gen_feat_maps = []

        if self.use_multiscale_freq_discrim:
            _real_outs, _gen_outs, _real_feat_maps, _gen_feat_maps = self.stft_discrim(real, gen)
            real_outs.extend(_real_outs)
            gen_outs.extend(_gen_outs)
            real_feat_maps.extend(_real_feat_maps)
            gen_feat_maps.extend(_gen_feat_maps)
        
        if self.use_multiscale_time_discrim:
            _real_outs, _gen_outs, _real_feat_maps, _gen_feat_maps = self.time_discrim(real, gen)
            real_outs.extend(_real_outs)
            gen_outs.extend(_gen_outs)
            real_feat_maps.extend(_real_feat_maps)
            gen_feat_maps.extend(_gen_feat_maps)
        
        if self.use_multiperiod_time_discrim:
            _real_outs, _gen_outs, _real_feat_maps, _gen_feat_maps = self.period_discrim(real, gen)
            real_outs.extend(_real_outs)
            gen_outs.extend(_gen_outs)
            real_feat_maps.extend(_real_feat_maps)
            gen_feat_maps.extend(_gen_feat_maps)

        return real_outs, gen_outs, real_feat_maps, gen_feat_maps




    
        