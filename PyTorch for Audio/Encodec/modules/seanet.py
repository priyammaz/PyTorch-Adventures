import torch
import torch.nn as nn
import numpy as np

from .conv import SConv1d, SConvTranspose1d
from .lstm import SLSTM
from .snake import Snake

class SEANetResnetBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 kernel_sizes=[3,1],
                 dilations=[1,1],
                 activation="ELU",
                 activation_params={"alpha": 1.0},
                 norm="weight_norm",
                 norm_params={}, 
                 pad_mode="reflect",
                 compress=2, 
                 true_skip=True):
        
        super().__init__()

        ### Output channel dimensions 
        hidden = dim // compress
        act = getattr(nn, activation) if activation != "Snake" else Snake

        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_channels = dim if i == 0 else hidden # input with dim, everything else after input hidden
            out_channels = dim if i == len(kernel_sizes) - 1 else hidden # output is first hidden, at the end back to dim
            
            block += [
                act(**activation_params) if activation != "Snake" else act(in_channels),
                SConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, 
                        norm=norm, norm_kwargs=norm_params, pad_mode="reflect")
            ]

        self.block = nn.Sequential(*block)
        
        ### If true_skip we will just add input to output ###
        if true_skip:
            self.shortcut = nn.Identity()

        ### Otherwise we project the input and then add to output ###
        else:
            self.shortcut = SConv1d(dim, dim, kernel_size=1,
                                    norm=norm, norm_kwargs=norm_params, 
                                    pad_mode=pad_mode)
    
    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class SEANetEncoder(nn.Module):
    def __init__(self, 
                 channels=1, 
                 dimension=128, 
                 n_filters=32, 
                 n_residual_layers=1, 
                 ratios=[8,5,4,2],
                 activation="ELU", 
                 activation_params={"alpha": 1.0},
                 norm="weight_norm",
                 norm_params={},
                 kernel_size=7, 
                 last_kernel_size=7, 
                 residual_kernel_size=3, 
                 dilation_base=2, 
                 pad_mode="reflect", 
                 true_skip=False, 
                 compress=2, 
                 lstm=2,
                 lstm_bidirectional=True):

        super().__init__()
        
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        ### Get activation function 
        act = getattr(nn, activation) if activation != "Snake" else Snake

        ### Initialize multiplier ###
        mult = 1

        ### Start model from input channels to starting n_filters channels ###
        model = [
            SConv1d(channels, mult * n_filters, kernel_size,
                    norm=norm, norm_kwargs=norm_params, 
                    pad_mode=pad_mode)
        ]

        ### For each downsample block ###
        for i, ratio in enumerate(self.ratios):

            ### We have n_residual_layers first
            for j in range(n_residual_layers):

                model += [
                    SEANetResnetBlock(
                        mult * n_filters, kernel_sizes=[residual_kernel_size, 1], 
                        dilations=[dilation_base ** j, 1], 
                        norm=norm, norm_params=norm_params, 
                        activation=activation, activation_params=activation_params, 
                        pad_mode=pad_mode, compress=compress, true_skip=true_skip
                    )
                ]

            ### Followed by the downsample ###
            model += [
                act(**activation_params) if activation != "Snake" else act(mult * n_filters),
                SConv1d(mult * n_filters, mult * n_filters * 2, 
                        kernel_size=ratio * 2, stride=ratio, # stride=ratio will downsample by that factor
                        norm=norm, norm_kwargs=norm_params, 
                        pad_mode=pad_mode)
            ]

            ### update mult for the next iteration
            mult *= 2

        ### Add on LSTM layers
        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm, bidirectional=lstm_bidirectional)]
        
        ### Post process with a final convolution ###
        model += [
            act(**activation_params) if activation != "Snake" else act(mult * n_filters),
            SConv1d(mult * n_filters, dimension, last_kernel_size, 
                    norm=norm, norm_kwargs=norm_params, 
                    pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class SEANetDecoder(nn.Module):
    def __init__(self, 
                 channels=1, 
                 dimension=128, 
                 n_filters=32, 
                 n_residual_layers=1, 
                 ratios=[8,5,4,2],
                 activation="ELU",
                 activation_params={"alpha": 1.0},
                 final_activation=None,
                 final_activation_params={},
                 norm="weight_norm",
                 norm_params={}, 
                 kernel_size=7, 
                 last_kernel_size=7,
                 residual_kernel_size=3, 
                 dilation_base=2,
                 pad_mode="reflect", 
                 true_skip=False,
                 compress=2, 
                 lstm=2,
                 lstm_bidirectional=True):

        super().__init__()

        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation) if activation != "Snake" else Snake
        mult = int(2 ** len(self.ratios))

        ### This will basically be opposite of the Encoder 
        model = [
            SConv1d(dimension, mult * n_filters, kernel_size, 
                    norm=norm, norm_kwargs=norm_params, 
                    pad_mode=pad_mode)
        ]

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm, bidirectional=lstm_bidirectional)]

        for i, ratio in enumerate(self.ratios):

            model += [
                act(**activation_params) if activation != "Snake" else act(mult * n_filters),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2, 
                                 kernel_size=ratio * 2, stride=ratio, 
                                 norm=norm, norm_kwargs=norm_params)
            ]

            for j in range(n_residual_layers):

                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1], 
                        dilations=[dilation_base ** j, 1], 
                        norm=norm, norm_params=norm_params, 
                        activation=activation, activation_params=activation_params, 
                        pad_mode=pad_mode, compress=compress, true_skip=true_skip
                    )
                ]
            
            mult //= 2

        model += [
            act(**activation_params) if activation != "Snake" else act(mult * n_filters),
            SConv1d(n_filters, channels, last_kernel_size, 
                    norm=norm, norm_kwargs=norm_params, 
                    pad_mode=pad_mode)
        ]

        ### Add an optional final activation (like tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        
        self.model = nn.Sequential(*model)

    def forward(self, z):
        return self.model(z)
    
if __name__ == "__main__":

    encoder = SEANetEncoder()
    decoder = SEANetDecoder()

    rand = torch.randn(1,1,24000)
    print("Input Shape:", rand.shape)
    z = encoder(rand)
    print("Latent Shape:", z.shape)
    y = decoder(z)
    print("Decoded Shape:", y.shape)

    assert y.shape == rand.shape
