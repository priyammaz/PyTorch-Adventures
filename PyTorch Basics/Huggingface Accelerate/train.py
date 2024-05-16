import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.models import resnet18
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from torchmetrics import Accuracy

EPOCHS = 50
GRADIENT_ACCUM_STEPS = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.0005

### Init Accelerator ###
accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUM_STEPS)

accuracy_fn = Accuracy(task="multiclass", num_classes=10).to(accelerator.device)

### Load Model ###
model = resnet18()
model.fc = nn.Linear(512,10)
model = model.to(accelerator.device)

### Load Dataset ###
transform = transforms.Compose(
    [transforms.Resize((64,64)),
    transforms.ToTensor()]
)

mini_batchsize = BATCH_SIZE // GRADIENT_ACCUM_STEPS 
trainset = datasets.CIFAR10(root=".", train=True, transform=transform, download=True)
testset = datasets.CIFAR10(root=".", train=False, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=mini_batchsize, shuffle=True, num_workers=8, pin_memory=True)
testloader = DataLoader(testset, batch_size=mini_batchsize, shuffle=True, num_workers=8, pin_memory=True)

### Define Loss Function ###
loss_fn = nn.CrossEntropyLoss()

### Define Optimizer ###
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

### Define Scheduler ###
total_training_steps = len(trainloader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=500, 
                                            num_training_steps=total_training_steps)

### Prepare Everything ###
model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
    model, optimizer, trainloader, testloader, scheduler
)

all_train_losses, all_test_losses = [], []
all_train_accs, all_test_accs = [], []

for epoch in range(EPOCHS):

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    model.train()
    accumulated_loss = 0 
    accumulated_accuracy = 0
    for images, targets in trainloader:

        ### Move Data to Correct GPU ###
        images, targets = images.to(accelerator.device), targets.to(accelerator.device)
        
        with accelerator.accumulate(model):
            
            ### Pass Through Model ###
            pred = model(images)

            ### Compute and Store Loss ##
            loss = loss_fn(pred, targets)
            accumulated_loss += loss / GRADIENT_ACCUM_STEPS

            ### Compute and Store Accuracy ###
            predicted = pred.argmax(axis=1)
            accuracy = accuracy_fn(predicted, targets)
            accumulated_accuracy += accuracy / GRADIENT_ACCUM_STEPS

            ### Update Model ###
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:

            ### Gather Metrics Across GPUs ###
            loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
            accuracy_gathered = accelerator.gather_for_metrics(accumulated_accuracy)

            ### Store Current Iteration Error ###
            train_loss.append(torch.mean(loss_gathered).item())
            train_acc.append(torch.mean(accuracy_gathered).item())

            ### Reset Accumulated for next Accumulation ###
            accumulated_loss, accumulated_accuracy = 0, 0


    model.eval()

    for images, targets in testloader:
        images, targets = images.to(accelerator.device), targets.to(accelerator.device)
        with torch.no_grad():
            pred = model(images)

            #### Compute Loss ###
            loss = loss_fn(pred, targets)

            ### Computed Accuracy ###
            predicted = pred.argmax(axis=1)
            accuracy = accuracy_fn(predicted, targets)

            ### Gather across GPUs ###
            loss_gathered = accelerator.gather_for_metrics(loss)
            accuracy_gathered = accelerator.gather_for_metrics(accuracy)

            ### Store Current Iteration Error ###
            test_loss.append(torch.mean(loss_gathered).item())
            test_acc.append(torch.mean(accuracy_gathered).item())
        
    epoch_train_loss = np.mean(train_loss)
    epoch_test_loss = np.mean(test_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_test_acc = np.mean(test_acc)

    all_train_losses.append(epoch_train_loss)
    all_test_losses.append(epoch_test_loss)
    all_train_accs.append(epoch_train_acc)
    all_test_accs.append(epoch_test_acc)

    accelerator.print(f"Training Accuracy: ", epoch_train_acc, "Training Loss:", epoch_train_loss)
    accelerator.print(f"Testing Accuracy: ", epoch_test_acc, "Testing Loss:", epoch_test_loss)
    