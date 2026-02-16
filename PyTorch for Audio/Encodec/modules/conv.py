import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

CONV_NORMALIZATIONS = ['none', # No normalization
                       'weight_norm', 'spectral_norm', # Weight parameterization normalization 
                       'time_layer_norm', 'layer_norm', 'time_group_norm' # standard normalization
                       ]

class ConvLayerNorm(nn.LayerNorm):

    """
    Typical data shape for layernorm is (B x L x E), and we normalize
    over the embedding dimension E. But if our data is images 
    like (B x C x H x W), or if its a sequence (B x C x L) we will permute our
    data to have Channels last and norm along that dimension, and then return it back!
    """

    def __init__(self, 
                 normalized_shape, 
                 **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return

def apply_parameterization_norm(module, norm):

    """
    If we are using a reparameterization norm this is 
    where we apply it to the weights of our convolutions
    """
    
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # Dont apply anything as we are using either "none" or a flavor of layernorm
        return module
    
def get_norm_module(module, norm, **norm_kwargs):
    """
    If we are doing a layernorm, this is where we define it
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        # Groupnorm with 1 group is not exactly the same as 
        # layernorm and you can see some details here:
        # https://discuss.pytorch.org/t/groupnorm-num-groups-1-and-layernorm-are-not-equivalent/145468
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()

class NormConv1d(nn.Module):
    """
    Putting together conv1d and norm
    """
    def __init__(self, *args, norm="none", norm_kwargs={}, **kwargs):
        super().__init__()
        self.conv = apply_parameterization_norm(
            nn.Conv1d(*args, **kwargs), norm
        )

        self.norm = get_norm_module(self.conv, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.conv(x))
    
class NormConv2d(nn.Module):
    """
    Putting together conv2d and norm
    """
    def __init__(self, *args, norm="none", norm_kwargs={}, **kwargs):
        super().__init__()
        self.conv = apply_parameterization_norm(
            nn.Conv2d(*args, **kwargs), norm
        )

        self.norm = get_norm_module(self.conv, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.conv(x))
    
class NormTransposeConv1d(nn.Module):
    """
    Putting together transposeconv1d and norm
    """
    def __init__(self, *args, norm="none", norm_kwargs={}, **kwargs):
        super().__init__()
        self.convtr = apply_parameterization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )

        self.norm = get_norm_module(self.convtr, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.convtr(x))

class NormTransposeConv2d(nn.Module):
    """
    Putting together transposeconv2d and norm
    """
    def __init__(self, *args, norm="none", norm_kwargs={}, **kwargs):
        super().__init__()
        self.convtr = apply_parameterization_norm(
            nn.ConvTranspose2d(*args, **kwargs), norm
        )

        self.norm = get_norm_module(self.convtr, norm, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.convtr(x))

def get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total=0):

    """
    length=10, kernel_size=3, stride=2, padding_total=2
    n_frames = (10 - 3 + 2) / 2 + 1 = 5.5
    ideal_length = (6 - 1) * 2 + (3 - 2) = 5 * 2 + 1 = 11


    """
    
    length = x.shape[-1]

    ### What is our expected output size (this is fractional if there is an incomplete window)
    n_frames = (length - kernel_size + padding_total) / stride + 1

    ### If n_frame is a fraction, then the question is how how many samples do we need to add to 
    ### the input data to make sure n_frames is no longer fractional
    ### math.ceil rounds up to get full frames
    ### - 1 gives us he number of stride steps we need
    ### multiply by strides to get distance covered
    ### kernel_size - padding_total adjusts for the overhang of the final kernel window
    ### The last frame starts at position `(n_frames - 1) * stride`, but the kernel extends
    ### beyond this starting position by `kernel_size - 1` positions. However, we already added 
    ### padding_total which will be added to the input, so we subtract it out.
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)

    return ideal_length - length

def pad1d(x, paddings, mode="zero", value=0):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[1]
    padding_left, padding_right = paddings

    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad: # cant do reflect padding if we dont have enough to reflect
            extra_pad = max_pad - length + 1 # compute number of samples to add on so we can reflect
            x = F.pad(x,(0, extra_pad)) # Pad with zeros
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)

def unpad1d(x, paddings):
    padding_left, padding_right = paddings
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]

class SConv1d(nn.Module):
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 norm="none",
                 norm_kwargs={}, 
                 pad_mode="reflect"):
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm = norm 
        self.pad_mode = pad_mode

        ### Define the Normalized Convolution
        self.conv = NormConv1d(
            in_channels, out_channels, kernel_size, stride, 
            dilation=dilation, groups=groups, bias=bias, 
            norm=norm, norm_kwargs=norm_kwargs
        )

    def forward(self, x):

        ### Input (sequence) data shape 
        batch, channels, seq_len = x.shape

        ### If our convolution is dilated, then the kernel size
        ### will effectively be larger than what it really is. 
        ### for example if we have a kernel size of 3, with a dilation
        ### of 2, every conv operation will see over a range of 5
        ### values (with holes in between)
        kernel_size = (self.kernel_size - 1) * self.dilation + 1 # compute the effective kernel size

        ### We want to make sure we add enough padding to ensure the output = input / stride
        ### the standard conv output formula is:
        ### output_length = floor((input_length + padding_total - kernel_size) / stride) + 1
        ### For simplicity lets assume that things divide evenly so we dont have a floor function (we handle this later)
        ### output_length = ((input_length + padding_total - kernel_size) / stride) + 1
        ### If padding_total = kernel_size - stride, we get:
        ### output_length = ((input_length + kernel_size - stride - kernel_size) / stride) + 1
        ### output_length = ((input_length - stride) / stride) + 1
        ### output_length = (input_length/stride) - (stride / stride) + 1
        ### output_length = (input_length/stride) - 1 + 1
        ### output_length = (input_length/stride)
        padding_total = kernel_size - self.stride # the total padding we have to add to our signal to get expected downsampling

        ### Now we had that assumption that things divide evenly. This only works if every convolution
        ### window is full. For example:
        ### For instance, with total padding = 4, kernel size = 4, stride = 2:
        ### 0 0 1 2 3 4 5 0 0   # (0s are padding)
        ### 1   2   3           # (output frames of a convolution, last 0 is never used)

        ### We then can compute extra_padding = 1 and we add an extra 0 padding to our data
        ### 0 0 1 2 3 4 5 0 0 0   # (0s are padding)
        ### 1   2   3   4         # (output frames of a convolution, last 0 is now used)
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, self.stride, padding_total)
 
        ### Now we add padding to both sides of the data 
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)

        return self.conv(x)

class SConvTranspose1d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 norm="none", 
                 norm_kwargs={}):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm = norm

        self.convtr = NormTransposeConv1d(
            in_channels, out_channels, kernel_size, stride,
            norm=norm, norm_kwargs=norm_kwargs
        )

    def forward(self, x):

        padding_total = self.kernel_size - self.stride

        y = self.convtr(x)

        ### We added two kinds of padding in the encoder
        ### the padding_total and the extra padding. the
        ### padding_total is deterministic, computed based on 
        ### the convolution configuration. On the other hand 
        ### the extra_padding is dynamic, based on the input 
        ### length to ensure complete frame coverage. 
        ### when decoding, we will remove the padding 
        ### that was added in by the encoding, but we will
        ### only remove the padding_total. We dont know exactly
        ### the extra_padding from the encoder, as removing it here would 
        ### require also passing the length at the matching layer
        ### in the encoder. So for simplicity we will remove what we know
        ### and then remove anything leftover at the end of the model!
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        y = unpad1d(y, (padding_left, padding_right))

        return y

if __name__ == "__main__":

    sconv = SConv1d(in_channels=16, 
                    out_channels=32,
                    kernel_size=11,
                    stride=2)
    
    rand = torch.randn(4,16,100) # expect with a stride of 2 to half L
    
    print(rand.shape)
    conv_out = sconv(rand)
    print(conv_out.shape)

    tsconv = SConvTranspose1d(in_channels=32, 
                              out_channels=16, 
                              kernel_size=11, 
                              stride=2)
    tsconv_out = tsconv(conv_out)
    print(tsconv_out.shape)

