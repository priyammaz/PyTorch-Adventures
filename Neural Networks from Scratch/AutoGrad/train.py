import numpy as np
from tqdm import tqdm
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

### Prep Model ###
class MyTorchMNIST(nn.Module):

    def __init__(self):

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)

        return x
    
model = MyTorchMNIST()
### Prep Dataset ###
train = MNIST("../../data", train=True, download=True)
test = MNIST("../../data", train=False, download=True)

def collate_fn(batch):

    ### Prep and Scale Images ###
    images = np.concatenate([np.array(i[0]).reshape(1,784)for i in batch]) / 255

    ### One Hot Encode Label (MNIST only has 10 classes) ###
    labels = [i[1] for i in batch]
    labels = np.eye(10)[labels]
    
    images = mytorch.Tensor(images)
    labels = mytorch.Tensor(labels)

    return images, labels

trainloader = DataLoader(train, batch_size=16, collate_fn=collate_fn)
testloader = DataLoader(test, batch_size=16, collate_fn=collate_fn)

### Prep Optimizer ###
optimizer = optim.SGD(model.parameters(), lr=0.1)

### Prep Loss Function ###
loss_fn = nn.CrossEntropyLoss()

### Train Model for 10 Epochs ###
for epoch in range(10):

    print(f"Training Epoch {epoch}")

    train_loss, train_acc = [], []
    eval_loss, eval_acc = [], []

    for images, ohe_labels in tqdm(trainloader):

        ### Pass Through Model ###
        pred = model(images)
        
        ### Compute Loss ###
        loss = loss_fn(pred, ohe_labels)

        ### Update Model ###
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ### Compute Accuracy ###
        predicted = pred.data.argmax(axis=-1)
        labels = ohe_labels.data.argmax(axis=-1)
        accuracy = np.sum(predicted == labels) / len(predicted)

        train_loss.append(loss.data)
        train_acc.append(accuracy)

    for images, ohe_labels in tqdm(testloader):

        ### Pass Through Model ###
        pred = model(images)
        
        ### Compute Loss ###
        loss = loss_fn(pred, ohe_labels)

        ### Compute Accuracy ###
        predicted = pred.data.argmax(axis=-1)
        labels = ohe_labels.data.argmax(axis=-1)
        accuracy = np.sum(predicted == labels) / len(predicted)

        eval_loss.append(loss.data)
        eval_acc.append(accuracy)
    
    print(f"Training Loss: {np.mean(train_loss)}")
    print(f"Eval Loss: {np.mean(eval_loss)}")
    print(f"Training Acc: {np.mean(train_acc)}")
    print(f"Eval Acc: {np.mean(eval_acc)}")