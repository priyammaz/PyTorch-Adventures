import torch
import torch.nn as nn

########################################
######### VANILLA AUTOENCODER ##########
########################################

class LinearVanillaAutoEncoder(nn.Module):
    def __init__(self, bottleneck_size=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32*32),
            nn.Sigmoid()
        )

    def forward(self, x):

        batch, channels, height, width = x.shape
        
        ### Flatten Image to Vector ###
        x = x.flatten(1)

        ### Pass Through Encoder ###
        enc = self.encoder(x)

        ### Pass Through Decoder ###
        dec = self.decoder(enc)

        ### Put Decoded Image Back to Original Shape ###
        dec = dec.reshape(batch, channels, height, width)

        return enc, dec
        
class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, in_channels=1, channels_bottleneck=4):
        super().__init__()

        self.bottleneck = channels_bottleneck
        self.in_channels = in_channels 
        
        self.encoder_conv = nn.Sequential(

            ### Convolutional Encoding ###
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(8),
            nn.ReLU(), 

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(16),
            nn.ReLU(), 

            nn.Conv2d(in_channels=16, out_channels=self.bottleneck, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(self.bottleneck),
            nn.ReLU(),

            
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.bottleneck, out_channels=16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=8, out_channels=in_channels, kernel_size=2, stride=2, padding=1),
            nn.Sigmoid()
        )
    

    def forward_enc(self, x):
        batch_size, num_channels, height, width = x.shape
        conv_enc = self.encoder_conv(x)
        return conv_enc

    def forward_dec(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.bottleneck, 4, 4))
        conv_dec = self.decoder_conv(x)
        return conv_dec
        
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        enc = self.forward_enc(x)
        dec = self.forward_dec(enc)
        return enc, dec
        
########################################
####### VARIATIONAL AUTOENCODER ########
########################################

class LinearVariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        #########################################################
        ### The New Layers Added in from Original AutoEncoder ###
        self.fn_mu =  nn.Linear(32, latent_dim)
        self.fn_logvar = nn.Linear(32, latent_dim)
        #########################################################
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32*32),
            nn.Sigmoid()
        )


    def forward_enc(self, x):

        x = self.encoder(x)
        
        mu = self.fn_mu(x)
        logvar = self.fn_logvar(x)

        #############################################
        ### Sample with Reparamaterization Trick ###
        sigma = torch.exp(0.5*logvar)
        noise = torch.rand_like(sigma, device=sigma.device)
        z = mu + sigma*noise
        ############################################
        
        return z, mu, logvar

    def forward_dec(self, x):
        return self.decoder(x)        
        
    def forward(self, x):

        batch, channels, height, width = x.shape
        
        ### Flatten Image to Vector ###
        x = x.flatten(1)

        ### Pass Through Encoder ###
        z, mu, logvar = self.forward_enc(x)

        ### Pass Sampled Data Through Decoder ###
        dec = self.decoder(z)

        ### Put Decoded Image Back to Original Shape ###
        dec = dec.reshape(batch, channels, height, width)

        return z, dec, mu, logvar

class ConvolutionalVartiationalAutoEncoder(nn.Module):
    def __init__(self, in_channels=1, channels_bottleneck=4):
        super().__init__()

        self.bottleneck = channels_bottleneck
        self.in_channels = in_channels 
        
        self.encoder_conv = nn.Sequential(

            ### Convolutional Encoding ###
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(8),
            nn.ReLU(), 

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(16),
            nn.ReLU(), 

            nn.Conv2d(in_channels=16, out_channels=self.bottleneck, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(self.bottleneck),
            nn.ReLU(),

            
        )

        #########################################################
        ### The New Layers Added in from Original AutoEncoder ###
        self.conv_mu =  nn.Conv2d(in_channels=self.bottleneck, out_channels=self.bottleneck, kernel_size=3, stride=1, padding="same")
        self.conv_logvar = nn.Conv2d(in_channels=self.bottleneck, out_channels=self.bottleneck, kernel_size=3, stride=1, padding="same")
        #########################################################

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.bottleneck, out_channels=16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=8, out_channels=in_channels, kernel_size=2, stride=2, padding=1),
            nn.Sigmoid()
        )
    

    def forward_enc(self, x):
        batch_size, num_channels, height, width = x.shape
        conv_enc = self.encoder_conv(x)

        #############################################
        ### Compute Mu and Sigma ###
        mu = self.conv_mu(conv_enc)
        logvar = self.conv_logvar(conv_enc)
        
        ### Sample with Reparamaterization Trick ###
        sigma = torch.exp(0.5*logvar)
        noise = torch.rand_like(sigma, device=sigma.device)
        z = mu + sigma*noise
        ############################################
        
        return z, mu, logvar

    def forward_dec(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.bottleneck, 4, 4))
        conv_dec = self.decoder_conv(x)
        return conv_dec
        
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        z, mu, logvar = self.forward_enc(x)
        dec = self.forward_dec(z)
        return z, dec, mu, logvar

############################################
# VECTOR QUANTIZED VARIATIONAL AUTOENCODER #
############################################

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=1024, latent_dim=2):
        super().__init__()
        
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

    def forward(self, x, efficient=True):

        batch_size = x.shape[0]
        
        ### Bad Implementation That Requires Matrix Expansion ###
        if not efficient:

            # C: Codebook Size, L: Latent Dim
            
            ### Embedding: [C, L] -> [B, C, L]
            emb = self.embedding.weight.unsqueeze(0).repeat(batch_size,1,1)

            ### X: [B, L] -> [B, 1, L]
            x = x.unsqueeze(1)

            ### [B, C]
            distances = torch.sum(((x - emb)**2), dim=-1)

        ### Alternative more Efficient Implementation ###
        else:
            ### Distance btwn every Latent and Code: (L-C)**2 = (L**2 - 2LC + C**2 ) ###

            ### L2: [B, L] -> [B, 1]
            L2 = torch.sum(x**2, dim=1, keepdim=True)

            ### C2: [C, L] -> [C]
            C2 = torch.sum(self.embedding.weight**2, dim=1).unsqueeze(0)

            ### CL: [B,L]@[L,C] -> [B, C]
            CL = x@self.embedding.weight.t()

            ### [B, 1] - 2 * [B, C] + [C] -> [B, C]
            distances = L2 - 2*CL + C2
        
        ### Grab Closest Indexes, create matrix of corresponding vectors ###
        
        ### Closest: [B, 1]
        closest = torch.argmin(distances, dim=-1)

        ### Create Empty Quantized Latents Embedding ###
        # latents_idx: [B, C]
        quantized_latents_idx = torch.zeros(batch_size, self.codebook_size, device=x.device)

        ### Place a 1 at the Indexes for each sample for the codebook we want ###
        batch_idx = torch.arange(batch_size)
        quantized_latents_idx[batch_idx,closest] = 1

        ### Matrix Multiplication to Grab Indexed Latents from Embeddings ###

        # quantized_latents: [B, C] @ [C, L] -> [B, L]
        quantized_latents = quantized_latents_idx @ self.embedding.weight

        return quantized_latents 


class LinearVectorQuantizedVAE(nn.Module):
    def __init__(self, latent_dim=2, codebook_size=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        #########################################################
        ###  The New Layers Added in from Original VAE Model  ###
        self.vq = VectorQuantizer(codebook_size, latent_dim)
        
        #########################################################
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32*32),
            nn.Sigmoid()
        )


    def forward_enc(self, x):

        x = self.encoder(x)
    
        return x

    def quantize(self, x):
        
        #############################################
        ## Quantize the Latent Space Representation #

        codes = self.vq(x)

        ### Compute VQ Loss ###
        codebook_loss = torch.mean((codes - x.detach())**2)
        commitment_loss = torch.mean((codes.detach() - x)**2)

        ### Straight Through ###
        codes = x + (codes - x).detach()
        
        #############################################

        return codes, codebook_loss, commitment_loss

    def forward_dec(self, x):
        codes, codebook_loss, commitment_loss = self.quantize(x)
        decoded = self.decoder(codes)
        
        return codes, decoded, codebook_loss, commitment_loss
        
    def forward(self, x):
        
        batch, channels, height, width = x.shape
        
        ### Flatten Image to Vector ###
        x = x.flatten(1)

        ### Pass Through Encoder ###
        latents = self.forward_enc(x)
        
        ### Pass Sampled Data Through Decoder ###
        quantized_latents, decoded, codebook_loss, commitment_loss = self.forward_dec(latents)

        ### Put Decoded Image Back to Original Shape ###
        decoded = decoded.reshape(batch, channels, height, width)

        return latents, quantized_latents, decoded, codebook_loss, commitment_loss


class ConvolutionalVectorQuantizedVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4, codebook_size=512):
        super().__init__()

        self.bottleneck = latent_dim
        self.in_channels = in_channels 
        self.codebook_size = codebook_size
        
        self.encoder_conv = nn.Sequential(

            ### Convolutional Encoding ###
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(8),
            nn.ReLU(), 

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(16),
            nn.ReLU(), 

            nn.Conv2d(in_channels=16, out_channels=self.bottleneck, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(self.bottleneck),
            nn.ReLU(),

            
        )

        #########################################################
        ###  The New Layers Added in from Original VAE Model  ###
        self.vq = VectorQuantizer(codebook_size, latent_dim)
        
        #########################################################

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.bottleneck, out_channels=16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=8, out_channels=in_channels, kernel_size=2, stride=2, padding=1),
            nn.Sigmoid()
        )
    

    def forward_enc(self, x):
        batch_size, num_channels, height, width = x.shape
        conv_enc = self.encoder_conv(x)
        return conv_enc

    def quantize(self, z):
        
        #############################################
        ## Quantize the Latent Space Representation #

        codes = self.vq(z)

        ### Compute VQ Loss ###
        codebook_loss = torch.mean((codes - z.detach())**2)
        commitment_loss = torch.mean((codes.detach() - z)**2)

        ### Straight Through ###
        codes = z + (codes - z).detach()
        
        #############################################

        return codes, codebook_loss, commitment_loss

    def forward_dec(self, x):
        batch_size, channels, height, width = x.shape

        ##########################################
        ### Reshape B x C x H X W -> B*H*W x C ###
        x = x.permute(0,2,3,1)
        x = torch.flatten(x, start_dim=0, end_dim=-2)

        ### Quantize Data ###
        codes, codebook_loss, commitment_loss = self.quantize(x)

        ### Reshape Back to B x C x H X W ###
        codes = codes.reshape(batch_size, height, width, channels)
        codes = codes.permute(0,3,1,2)

        ##########################################
        
        conv_dec = self.decoder_conv(codes)
        
        return codes, conv_dec, codebook_loss, commitment_loss
        
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        latents = self.forward_enc(x)
        quantized_latents, decoded, codebook_loss, commitment_loss = self.forward_dec(latents)
        return latents, quantized_latents, decoded, codebook_loss, commitment_loss