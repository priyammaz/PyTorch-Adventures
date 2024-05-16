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

### Load in My UNET and DataLoader ###
from unet_train_parts import ADE20KDataset, UNET

### Path to ADE20K Data ###
path_to_data = "../../data/ADE20K"

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

    ### Define Loss Function (ignore index -1 as its unlabeled background) ###
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    ### Load Model ###
    model = UNET(in_channels=3, 
                 num_classes=150, 
                 start_dim=64, 
                 dim_mults=(1,2,4,8),
                 skip_connection=skip_connection)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    accelerator.print("Number of Parameters:", params)
    
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
          experiment_name="UNET_wo_skip_ADE20K_test",
          skip_connection=True)
                




    

 