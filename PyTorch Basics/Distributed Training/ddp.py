import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

### Import In DDP Related Packages ###
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")


class VanillaAlexNet(nn.Module):
    def __init__(self, classes=2, dropout_p=0.5):
        super().__init__()
        self.classes = classes

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=64),  # ADDED IN BATCHNORM

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=192),  # ADDED IN BATCHNORM

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=384),  # ADDED IN BATCHNORM

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),  # ADDED IN BATCHNORM

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=256),  # ADDED IN BATCHNORM
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.reshape(batch_size, -1)
        x = self.head(x)
        return x


### SETUP ENVIRONMENT ###
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


### SETUP DISTRIBUTED SAMPLER ###
def build_distributed_sampler(path_to_data, batch_size, world_size, rank):
    ############### OLD CODE ##############
    dataset = ImageFolder(path_to_data)
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(),
        ToTensor(),
        normalizer])

    val_transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        normalizer])

    train_samples, test_samples = int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[train_samples, test_samples])

    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    #######################################

    ############### NEW CODE ##############
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    return trainloader, valloader


def main(rank, world_size):
    ### Setup The Environmental Variables and Initialize ###
    setup(rank, world_size)

    ### Build our Distributed DataLoaders ###
    trainloader, valloader = build_distributed_sampler("../data/PetImages/", batch_size=64, world_size=world_size,
                                                       rank=rank)

    ### Define Model and Convert to SyncBatchNorm ###
    model = VanillaAlexNet().to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    ### Set Optimizer and Loss Function ###
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    ### Start Training loop ###
    epochs = 5

    ### Standard Trainin Loop ###
    for epoch in range(1, epochs + 1):
        if rank == 0:  # Only print when we are using the default node 0 (otherwise everything will print once for each GPU)
            print(f"Starting Epoch {epoch}")

        ### Set Epoch for train and valloader for every epoch. This is necessary for proper shuffling in every iteration
        trainloader.sampler.set_epoch(epoch)
        valloader.sampler.set_epoch(epoch)

        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []

        model.train()

        for image, label in trainloader:
            image, label = image.to(rank), label.to(rank)
            optimizer.zero_grad()
            out = model.forward(image)
            loss = loss_fn(out, label)
            training_losses.append(loss.item())

            ### CALCULATE ACCURACY ###
            predictions = torch.argmax(out, axis=1)
            accuracy = (predictions == label).sum() / len(predictions)
            training_accuracies.append(accuracy.item())

            loss.backward()
            optimizer.step()

        model.eval()
        for image, label in valloader:
            image, label = image.to(rank), label.to(rank)
            with torch.no_grad():
                out = model.forward(image)
                loss = loss_fn(out, label)
                validation_losses.append(loss.item())

                ### CALCULATE ACCURACY ###
                predictions = torch.argmax(out, axis=1)
                accuracy = (predictions == label).sum() / len(predictions)
                validation_accuracies.append(accuracy.item())


        training_loss_mean = torch.mean(torch.tensor(training_losses, dtype=torch.float)).to(rank)
        valid_loss_mean = torch.mean(torch.tensor(validation_losses, dtype=torch.float)).to(rank)
        training_acc_mean = torch.mean(torch.tensor(training_accuracies, dtype=torch.float)).to(rank)
        valid_acc_mean = torch.mean(torch.tensor(validation_accuracies, dtype=torch.float)).to(rank)

        ### AGGREGATE LOSSES AND MEANS ACROSS ALL GPUS ###
        torch.distributed.all_reduce(training_loss_mean, op=dist.ReduceOp.SUM)
        torch.distributed.all_reduce(valid_loss_mean, op=dist.ReduceOp.SUM)
        torch.distributed.all_reduce(training_acc_mean, op=dist.ReduceOp.SUM)
        torch.distributed.all_reduce(valid_acc_mean, op=dist.ReduceOp.SUM)

        ### DIVIDE THE SUM BY NUMBER OF GPUS (WORLD SIZE)
        training_loss_mean = training_loss_mean / world_size
        valid_loss_mean = valid_loss_mean / world_size
        training_acc_mean = training_acc_mean / world_size
        valid_acc_mean = valid_acc_mean / world_size

        if rank == 0:  # Only print when we are using the default node 0 (otherwise everything will print once for each GPU)
            print("Training Loss:", training_loss_mean.item())
            print("Training Accuracy:", training_acc_mean.item())
            print("Validation Loss:", valid_loss_mean.item())
            print("Validation Accuracy:", valid_acc_mean.item())

    dist.destroy_process_group()  ## End Training and Remove Everything


if __name__ == "__main__":
    world_size = 2
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size
        )