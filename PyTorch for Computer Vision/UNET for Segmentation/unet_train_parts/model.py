import torch
import torch.nn as nn 
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groupnorm_num_groups):
        super().__init__()

        ### Input Convolutions + GroupNorm ###
        self.groupnorm_1 = nn.GroupNorm(groupnorm_num_groups, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

        ### Input + Time Embedding Convolutions + GroupNorm ###
        self.groupnorm_2 = nn.GroupNorm(groupnorm_num_groups, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")

        ### Residual Layer ###
        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):

        ### Store Copy for Residual ###
        residual_connection = x
        
        ### Input GroupNorm and Convolutions ###
        x = self.groupnorm_1(x)
        x = F.relu(x)
        x = self.conv_1(x)

        ### Group Norm and Conv Again! ###
        x = self.groupnorm_2(x)
        x = F.relu(x)
        x = self.conv_2(x)

        ### Add Residual and Return ###
        x = x + self.residual_connection(residual_connection)
        
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate=False):
        super().__init__()

        if interpolate:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
            )

        else:
            self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)

    def forward(self, inputs):
        batch, channels, height, width = inputs.shape
        upsampled = self.upsample(inputs)
        assert (upsampled.shape == (batch, channels, height*2, width*2))
        return upsampled
        
class UNET(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=150, 
                 start_dim=64, 
                 dim_mults=(1,2,4,8), 
                 residual_blocks_per_group=1, 
                 groupnorm_num_groups=16, 
                 interpolated_upsample=False,
                 skip_connection=True):
        
        super().__init__()
  
        self.input_image_channels = in_channels
        self.interpolate = interpolated_upsample
        self.skip_connection = skip_connection
        
        #######################################
        ### COMPUTE ALL OF THE CONVOLUTIONS ###
        #######################################

        ### Get Number of Channels at Each Block ###
        channel_sizes = [start_dim*i for i in dim_mults]
        starting_channel_size, ending_channel_size = channel_sizes[0], channel_sizes[-1]

        ### Compute the Input/Output Channel Sizes for Every Convolution of Encoder ###
        self.encoder_config = []
        
        for idx, d in enumerate(channel_sizes):
            ### For Every Channel Size add "residual_blocks_per_group" number of Residual Blocks that DONT Change the number of channels ###
            for _ in range(residual_blocks_per_group):
                self.encoder_config.append(((d, d), "residual")) # Shape: (Batch x Channels x Height x Width) -> (Batch x Channels x Height x Width)

            ### After Residual Blocks include Downsampling (by factor of 2) but dont change number of channels ###
            self.encoder_config.append(((d,d), "downsample")) # Shape: (Batch x Channels x Height x Width) -> (Batch x Channels x Height/2 x Width/2)
            
            ### If we are not at the last channel size include a channel upsample (typically by factor of 2) ###
            if idx < len(channel_sizes) - 1:
                self.encoder_config.append(((d,channel_sizes[idx+1]), "residual")) # Shape: (Batch x Channels x Height x Width) -> (Batch x Channels*2 x Height x Width)
            
        ### The Bottleneck will have "residual_blocks_per_group" number of ResidualBlocks each with the input/output of our final channel size###
        self.bottleneck_config = []
        for _ in range(residual_blocks_per_group):
            self.bottleneck_config.append(((ending_channel_size, ending_channel_size), "residual"))

        ### Store a variable of the final Output Shape of our Encoder + Bottleneck so we can compute Decoder Shapes ###
        out_dim = ending_channel_size

        ### Reverse our Encoder config to compute the Decoder ###
        reversed_encoder_config = self.encoder_config[::-1]

        ### The output of our reversed encoder will be the number of channels added for residual connections ###
        self.decoder_config = []
        for idx, (metadata, type) in enumerate(reversed_encoder_config):
            ### Flip in_channels, out_channels with the previous out_dim added on ###
            enc_in_channels, enc_out_channels = metadata

            ### Compute Number of Input Channels (if we want concatenated skip connection or not) ###
            concat_num_channels = out_dim + enc_out_channels if self.skip_connection else out_dim
            self.decoder_config.append(((concat_num_channels, enc_in_channels), "residual"))
                    
            if type == "downsample":
                ### If we did a downsample in our encoder, we need to upsample in our decoder ###
                self.decoder_config.append(((enc_in_channels, enc_in_channels), "upsample"))

            ### The new out_dim will be the number of output channels from our block (or the cooresponding encoder input channels) ###
            out_dim = enc_in_channels

        ### Add Extra Residual Block for residual from input convolution ###
        # hint: We know that the initial convolution will have starting_channel_size
        # and the output of our decoder will also have starting_channel_size, so the
        # final ResidualBlock we need will need to go from starting_channel_size*2 to starting_channel_size

        concat_num_channels = starting_channel_size*2 if self.skip_connection else starting_channel_size
        self.decoder_config.append(((concat_num_channels, starting_channel_size), "residual"))
        
        
        #######################################
        ### ACTUALLY BUILD THE CONVOLUTIONS ###
        #######################################

        ### Intial Convolution Block ###
        self.conv_in_proj = nn.Conv2d(self.input_image_channels, 
                                      starting_channel_size, 
                                      kernel_size=3, 
                                      padding="same")
        
        self.encoder = nn.ModuleList()
        for metadata, type in self.encoder_config:
            if type == "residual":
                in_channels, out_channels = metadata
                self.encoder.append(ResidualBlock(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  groupnorm_num_groups=groupnorm_num_groups))
            elif type == "downsample":
                in_channels, out_channels = metadata
                self.encoder.append(
                    nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=3, 
                              stride=2, 
                              padding=1)
                    )

        
        ### Build Encoder Blocks ###
        self.bottleneck = nn.ModuleList()
        
        for (in_channels, out_channels), _ in self.bottleneck_config:
            self.bottleneck.append(ResidualBlock(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 groupnorm_num_groups=groupnorm_num_groups))

        ### Build Decoder Blocks ###
        self.decoder = nn.ModuleList()
        for metadata, type in self.decoder_config:
            if type == "residual":
                in_channels, out_channels = metadata
                self.decoder.append(ResidualBlock(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  groupnorm_num_groups=groupnorm_num_groups))
            elif type == "upsample":
                in_channels, out_channels = metadata
                self.decoder.append(UpsampleBlock(in_channels=in_channels, 
                                                  out_channels=out_channels,
                                                  interpolate=self.interpolate))

        ### Output Convolution ###
        self.conv_out_proj = nn.Conv2d(in_channels=starting_channel_size, 
                                       out_channels=num_classes,
                                       kernel_size=3, 
                                       padding="same")


    def forward(self, x):
        residuals = []

        ### Pass Through Projection and Store Residual ###
        x = self.conv_in_proj(x)
        residuals.append(x)

        ### Pass through encoder and store residuals ##
        for module in self.encoder:
            x = module(x)
            residuals.append(x)
           

        ### Pass Through BottleNeck ###
        for module in self.bottleneck:
            x = module(x)

        ### Pass through Decoder while Concatenating Residuals ###
        for module in self.decoder:

            if isinstance(module, ResidualBlock):
                
                ### Pass through Convs with Skip Connections ###
                residual_tensor = residuals.pop()

                if self.skip_connection:
                    x  = torch.cat([x, residual_tensor], axis=1)

                x = module(x)
                
            else:
                ### Pass Through Upsample Block ###
                x = module(x)


        ### Map back to num_channels for final output ###
        x = self.conv_out_proj(x)

        return x