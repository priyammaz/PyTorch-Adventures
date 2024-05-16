import torch
import torch.nn as nn

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