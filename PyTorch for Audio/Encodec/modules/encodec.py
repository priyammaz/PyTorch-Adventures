import torch
import torch.nn as nn
import random
import accelerate 

from .seanet import SEANetEncoder, SEANetDecoder
from .quantizer import ResidualVectorQuantization

class EncodecModel(nn.Module):
    def __init__(self, channels=1, dimension=128, n_filters=32, n_residual_layers=1, 
                 ratios=[8,5,4,2], activation="ELU", activation_params={"alpha": 1.0},
                 final_activation=None, final_activation_params={}, norm="weight_norm",
                 norm_params={}, kernel_size=7, last_kernel_size=7, residual_kernel_size=3, 
                 dilation_base=2, pad_mode="reflect", true_skip=False, compress=2, 
                 lstm=2, num_quantizers=8, codebook_dim=None, codebook_size=1024, decay=0.99, 
                 kmeans_init=True, kmeans_iters=50, threshold_ema_dead_code=2, 
                 commit_weight=1, accelerator=None):
        
        super().__init__()

        self.num_quantizers = num_quantizers
        self.accelerator = accelerator

        self.encoder = SEANetEncoder(
            channels=channels, dimension=dimension, n_filters=n_filters, 
            n_residual_layers=n_residual_layers, ratios=ratios,
            activation=activation, activation_params=activation_params,
            norm=norm, norm_params=norm_params,
            kernel_size=kernel_size, last_kernel_size=last_kernel_size, 
            residual_kernel_size=residual_kernel_size, dilation_base=dilation_base, 
            pad_mode=pad_mode, true_skip=true_skip, compress=compress, lstm=lstm
        )

        self.quantizer = ResidualVectorQuantization(
            num_quantizers=num_quantizers, dim=self.encoder.dimension, 
            codebook_size=codebook_size, codebook_dim=codebook_dim, 
            decay=decay, kmeans_init=kmeans_init, 
            kmeans_iters=kmeans_iters, threshold_ema_dead_code=threshold_ema_dead_code, 
            commitment_weight=commit_weight, accelerator=accelerator
        )

        self.decoder = SEANetDecoder(
            channels=channels, dimension=dimension, n_filters=n_filters, 
            n_residual_layers=n_residual_layers, ratios=ratios,
            activation=activation, activation_params=activation_params,
            final_activation=final_activation, final_activation_params=final_activation_params,
            norm=norm, norm_params=norm_params, 
            kernel_size=kernel_size, last_kernel_size=last_kernel_size,
            residual_kernel_size=residual_kernel_size, dilation_base=dilation_base,
            pad_mode=pad_mode, true_skip=true_skip, compress=compress, lstm=lstm
        )

    def _is_distributed(self):
        return self.accelerator is not None and self.accelerator.num_processes > 1

    def normalize(self, x):

        """
        Per sample loudness normalization, we rescale the signal so each sample 
        has unit RMS (root mean squared)
        """
        
        batch_size, channels, seq_len = x.shape

        mono = x
        if channels > 1:
            mono = x.mean(dim=1, keepdim=True)

        volume = mono.pow(2).mean(dim=1, keepdim=True).sqrt()
        scale = 1e-8 + volume
        
        x = x / scale

        return x, scale

    def denormalize(self, x, scale):
        return x * scale

    def forward(self, x):

        batch_size, channels, seq_len = x.shape

        ### Normalize input data ###
        x, scale = self.normalize(x)
     
        ### Encode frames ###
        encoded = self.encoder(x)

        ### Select how many codebooks we want to use randomly (variable codebook training) ###
        num_books_to_use = torch.tensor(random.randint(0, self.num_quantizers), device=x.device)

        ### Make sure its the same across all GPUs ###
        if self._is_distributed():
            num_books_to_use = accelerate.utils.broadcast(num_books_to_use, from_process=0)
        
        ### Pass through quantizer ###
        quantized_out, _, out_losses = self.quantizer(encoded, num_books_to_use)

        ### quantized_out has only summed upto num_books_to_use books ###
        ### so lets pass a potential partial approx to our decoder ###
        decoded = self.decoder(quantized_out)

        ### denormalize decoded back to original audio magnitude ###
        decoded = self.denormalize(decoded, scale)

        ### Clip off any extra samples if decoded is longer than the original input ###
        ### remember our decoder already removes the known padding, but some dynamic ###
        ### padding still may remain! ###
        decoded = decoded[:, :, :seq_len]

        return {"encoder_out": encoded, 
                "quantized": quantized_out, 
                "quantizer_loss": torch.mean(out_losses), 
                "decoded": decoded}

    @torch.no_grad()
    def tokenize(self, x):
        pass

    @torch.no_grad()
    def decode(self, x):
        pass

if __name__ == "__main__":
    rand = torch.randn(2,1,24000)
    model = EncodecModel()
    model(rand)

