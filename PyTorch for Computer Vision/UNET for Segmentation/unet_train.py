import os
import random
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator

path_to_data = "../../data/ADE20K"

class ADE20KDataset(Dataset):
    def __init__(self, path_to_data, train=True, image_size=128, random_crop_ratio=(0.2, 1)):
        self.path_to_data = path_to_data
        self.train = train
        self.image_size = image_size
        self.min_ratio, self.max_ratio = random_crop_ratio

        if train:
            split = "training"
        else:
            split = "validation"

        ### Get Path to Images and Segmentations ###
        self.path_to_images = os.path.join(self.path_to_data, "images", split)
        self.path_to_annotations = os.path.join(self.path_to_data, "annotations", split)

        ### Get All Unique File Roots ###
        self.file_roots = [path.split(".")[0] for path in os.listdir(self.path_to_images)]

        ### Store all Transforms we want ###
        self.resize = transforms.Resize(size=(self.image_size, self.image_size))
        self.normalize = transforms.Normalize(mean=(0.48897059, 0.46548275, 0.4294), 
                                              std=(0.22861765, 0.22948039, 0.24054667))
        self.random_resize = transforms.RandomResizedCrop(size=(self.image_size, self.image_size))
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_roots)

    def __getitem__(self, idx):

        ### Grab File Root ###
        file_root = self.file_roots[idx]

        ### Get Paths to Image and Annotation ###
        image = os.path.join(self.path_to_images, f"{file_root}.jpg")
        annot = os.path.join(self.path_to_annotations, f"{file_root}.png")

        ### Load Image and Annotation ###
        image = Image.open(image).convert("RGB")
        annot = Image.open(annot)

        ### Train Image Transforms ###
        if self.train:

            ### Resize Image and Annotation ###
            if random.random() < 0.5:
                
                image = self.resize(image)
                annot = self.resize(annot)

            ### Random Resized Crop ###
            else:

                ### Get Smaller Side ###
                min_side = min(image.size)
    
                ### Get a Random Crop Size with Ratio ###
                random_ratio = random.uniform(self.min_ratio, self.max_ratio)

                ### Compute Crop Size ###
                crop_size = int(random_ratio * min_side)

                ### Get Parameters of Random Crop ###
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(crop_size, crop_size))

                ### Crop Image and Annotation ###
                image = TF.crop(image, i, j, h, w)
                annot = TF.crop(annot, i, j, h, w)

                ### Resize Image to Desired Image Size ###
                image = self.resize(image)
                annot = self.resize(annot)
            

            ### Random Horizontal Flip ###
            if random.random() < 0.5:
                image = self.horizontal_flip(image)
                annot = self.horizontal_flip(annot)

        ### Validation Image Transforms ###
        else:

            image = self.resize(image)
            annot = self.resize(annot)
                
        ### Convert Everything to Tensors ###
        image = self.totensor(image)
        annot = torch.tensor(np.array(annot), dtype=torch.long)

        ### Update Annotations as class 0 is other and not needed ###
        annot = annot - 1 # Make it from [0-150] to [-1-149]

        ### Normalize Image ###
        image = self.normalize(image)

        return image, annot

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

### Define Simple Training Logger ###
class LocalLogger:
    def __init__(self, 
                 path_to_log_folder, 
                 filename="train_log.pkl"):
        
        self.path_to_log_folder = path_to_log_folder
        self.path_to_file = os.path.join(path_to_log_folder, filename)

        self.log_exists = os.path.isfile(self.path_to_file)

        if self.log_exists:
            with open(self.path_to_file, "rb") as f:
                self.logger = pickle.load(f)
            
        else:
            self.logger = {"epoch": [], 
                           "train_loss": [], 
                           "train_acc": [], 
                           "test_loss": [], 
                           "test_acc": []}
            
    def log(self, epoch, train_loss, train_acc, test_loss, test_acc):
        self.logger["epoch"].append(epoch)
        self.logger["train_loss"].append(train_loss)
        self.logger["train_acc"].append(train_acc)
        self.logger["test_loss"].append(test_loss)
        self.logger["test_acc"].append(test_acc)

        with open(self.path_to_file, "wb") as f:
            pickle.dump(self.logger, f)


### Write Training Function ###
def train(batch_size=64, 
          gradient_accumulation_steps=2,
          learning_rate=0.001, 
          num_epochs=150,
          image_size=256,
          experiment_name="unet_w_skip_ade20k",
          skip_connection=True):

    ### Define Accelerator ###
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    ## Create Working Directory ###
    path_to_experiment = os.path.join("work_dir", experiment_name)
    if not os.path.exists(path_to_experiment):
        os.mkdir(path_to_experiment)

    ### Instantiate Logger ###
    logger = LocalLogger(path_to_experiment, f"{experiment_name}_log.pkl")

    ### Load Dataset ###
    micro_batchsize = batch_size // gradient_accumulation_steps
    train_data = ADE20KDataset(path_to_data, train=True, image_size=image_size)
    test_data = ADE20KDataset(path_to_data, train=False, image_size=image_size)
    train_dataloader = DataLoader(train_data, batch_size=micro_batchsize, shuffle=True, num_workers=16, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=micro_batchsize, shuffle=False, num_workers=16, pin_memory=True)

    ### Define Loss Function ###
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    ### Load Model ###
    model = UNET(in_channels=3, 
                 num_classes=150, 
                 start_dim=64, 
                 dim_mults=(1,2,4,8),
                 skip_connection=skip_connection)

    ### Load Optimizer ###
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    ### Load Scheduler ###
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=500, 
                                                num_training_steps=(len(train_dataloader) * num_epochs))
    
    ### Prepare Everything ###
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )
    
    ### Train Model ###
    best_test_loss = np.inf

    for epoch in range(1,num_epochs+1):

        accelerator.print(f"Training Epoch [{epoch}/{num_epochs}]")
        
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []

        ### Train Loop ###
        accumulated_loss = 0 
        accumulated_accuracy = 0
        progress_bar = tqdm(range(len(train_dataloader)//gradient_accumulation_steps), disable = not accelerator.is_main_process)
        
        model.train()
        for images, targets in train_dataloader:
            
            with accelerator.accumulate(model):
                
                ### Pass Through Model ###
                pred = model(images)

                ### Compute and Store Loss (Scaled by Grad Accumulations) ##
                loss = loss_fn(pred, targets)
                accumulated_loss += loss / gradient_accumulation_steps

                ### Compute and Store Accuracy ###
                predicted = pred.argmax(axis=1)
                accuracy = (predicted == targets).sum() / torch.numel(predicted)
                accumulated_accuracy += accuracy / gradient_accumulation_steps

                ### Compute Gradients ###
                accelerator.backward(loss)

                ### Gradient Clipping and Logging ###
                if accelerator.sync_gradients:
                    
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    ### Gather Metrics Across GPUs ###
                    loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
                    accuracy_gathered = accelerator.gather_for_metrics(accumulated_accuracy)

                    ### Store Current Iteration Loss and Accuracy ###
                    train_loss.append(torch.mean(loss_gathered).item())
                    train_acc.append(torch.mean(accuracy_gathered).item())

                    ### Reset Accumulated for Next Accumulation ###
                    accumulated_loss, accumulated_accuracy = 0, 0

                    ### Iterate Progress Bar ###
                    progress_bar.update(1)

                ### Update Model ###
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ### Gather Training Loss Metrics ###
            # if accelerator.sync_gradients:

            #     ### Gather Metrics Across GPUs ###
            #     loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
            #     accuracy_gathered = accelerator.gather_for_metrics(accumulated_accuracy)

            #     ### Store Current Iteration Loss and Accuracy ###
            #     train_loss.append(torch.mean(loss_gathered).item())
            #     train_acc.append(torch.mean(accuracy_gathered).item())

            #     ### Reset Accumulated for Next Accumulation ###
            #     accumulated_loss, accumulated_accuracy = 0, 0

            #     ### Iterate Progress Bar ###
            #     progress_bar.update(1)
        
        ### Testing Loop ###
        model.eval()
        for images, targets in test_dataloader:

            with torch.no_grad():
                pred = model(images)

            ### Compute Loss ###
            loss = loss_fn(pred, targets)

            ### Compute Accuracy ###
            predicted = pred.argmax(axis=1)
            accuracy = (predicted == targets).sum() / torch.numel(predicted)

            ### Gather Losses and Accuracy ###
            loss_gathered = accelerator.gather_for_metrics(loss)
            accuracy_gathered = accelerator.gather_for_metrics(accuracy)

            ### Store Current Iteration Error ###
            test_loss.append(torch.mean(loss_gathered).item())
            test_acc.append(torch.mean(accuracy_gathered).item())

        ### Average Loss and Acc for Epoch ###
        epoch_train_loss = np.mean(train_loss)
        epoch_test_loss = np.mean(test_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_test_acc = np.mean(test_acc)

        accelerator.print(f"Training Accuracy: {epoch_train_acc}, Training Loss: {epoch_train_loss}")
        accelerator.print(f"Testing Accuracy: {epoch_test_acc}, Testing Loss: {epoch_test_loss}")

        ### Log Training ###
        logger.log(epoch=epoch, 
                   train_loss=epoch_train_loss, 
                   train_acc=epoch_train_acc, 
                   test_loss=epoch_test_loss, 
                   test_acc=epoch_test_acc)
        
        ### Save Model ###
        if epoch_test_loss < best_test_loss:
            accelerator.print("---SAVING---")

            best_test_loss = epoch_test_loss
            accelerator.save_model(model, os.path.join(path_to_experiment, "best_checkpoint"), safe_serialization=False)

        accelerator.save_model(model, os.path.join(path_to_experiment, "last_checkpoint"), safe_serialization=False)


if __name__ == "__main__":
    train(batch_size=128, 
          gradient_accumulation_steps=4,
          learning_rate=0.001, 
          num_epochs=150,
          image_size=256,
          experiment_name="UNET_w_skip_ADE20K",
          skip_connection=True)
                




    

 