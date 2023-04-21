"""
Code Heavily Inspired by the Following Repositories!!!

Huggingface Pytorch Image Models
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

Lucidrains ViT Pytorch
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/local_vit.py

Jankrepl
https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/vision_transformer/custom.py

Karpathy
https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor
from tqdm import tqdm

class PatchEmbed(nn.Module):
    """
    PatchEmbed module will take an input image in the shape (C, H, W), patch the image into
    patches of size patch_size and embed each patch into embedding dim of embed_dim
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=786):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        ### Calculate Number of Patches ###
        self.n_patches = (img_size // patch_size) ** 2

        ### Use Convolution to Patch the Images ###
        self.proj = nn.Conv2d(in_channels=in_chans,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # (batch , embed_dim , sqrt(n_patches) , sqrt(n_patches))
        x = x.flatten(2) # (batch , embed_dim , n_patches)
        x = x.transpose(1,2) # (batch, n_patches, embed_dim)
        return x


class Head(nn.Module):
    """
    Single Attention Head to calculate the Q, K, V and return the weighted average matrix via 3 linear layers
    """
    def __init__(self, embed_dim=768, head_dim=768, attn_p=0):
        super(Head, self).__init__()
        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.attn_dropout = nn.Dropout(attn_p)

    def forward(self, x):
        batch_size, n_patch, embed_dim = x.shape
        q = self.query(x) # (batch, n_patches+1, head_dim)
        k = self.key(x) # (batch, n_patches+1, head_dim)
        v = self.value(x) # (batch, n_patches+1, head_dim)

        sam = (q @ k.transpose(-2,-1)) * embed_dim**-0.5 # (batch , n_patches+1, n_patches+1)
        attn = sam.softmax(dim=-1) # (batch , n_patches+1, n_patches+1)
        attn = self.attn_dropout(attn)
        weighted_average = attn @ v # (batch , n_patches+1, head_dim)
        return weighted_average


class MultiHeadedAttention(nn.Module):
    """
    Multiple Attention Head to repeat Head module num_heads times and concatenate outputs of heads together.
    """
    def __init__(self, embed_dim=768, num_heads=12, attn_p=0, proj_p=0):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([Head(embed_dim=embed_dim, head_dim=self.head_size, attn_p=attn_p) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (batch, n_patches+1, embed_dim)
        out = self.proj_drop(self.proj(out)) # (batch, n_patches+1, embed_dim)
        return out



class EfficientAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_p, proj_p):
        super(EfficientAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = int(self.embed_dim / num_heads)

        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.attn_dropout = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        batch, patches, embed_dim = x.shape # (batch, n_patches+1, embed_dim)
        qkv = self.qkv(x) # (batch, n_patches+1, 3*embed_dim)
        qkv = qkv.reshape(batch, patches, 3, self.num_heads, self.head_size) # (batch, patch+1, 3, num_heads, head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, patches+1, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2] # Each of shape (batch, num_heads, patches+1, head_size)

        ### SAME AS BEFORE NOW ###
        sam = (q @ k.transpose(-2,-1)) * self.head_size**-0.5 # (batch, num_heads, patches+1, patches+1)
        attn = sam.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        weighted_average = attn @ v # (batch, num_heads, patches+1, head_size)
        weighted_average = weighted_average.transpose(1,2) # (batch, patches+1, num_heads, head_size)
        weighted_average = weighted_average.flatten(2) # (batch, patches+1, embed_dim)
        out = self.proj_drop(self.proj(weighted_average))
        return out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, mlp_p=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(mlp_p)

    def forward(self, x):
        x = self.act(self.fc1(x)) # (batch, n_patches+1, embed_dim * mlp_ratio)
        x = self.drop(x)
        x = self.fc2(x) # (batch, n_patches+1, embed_dim)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Full Transformer block with Attention and Linear Layers
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 mlp_p=0, attn_p=0, proj_p=0, efficient=True):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)

        if efficient:
            self.attn = EfficientAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           attn_p=attn_p,
                                           proj_p=proj_p)
        else:
            self.attn = MultiHeadedAttention(embed_dim=embed_dim,
                                             num_heads=num_heads,
                                             attn_p=attn_p,
                                             proj_p=proj_p)

        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_features = int(embed_dim*mlp_ratio)
        self.mlp = MLP(in_features=embed_dim,
                       hidden_features=hidden_features,
                       out_features=embed_dim,
                       mlp_p=mlp_p)

    def forward(self, x):
        """
        Residual connections to avoid vanishing gradients
        """
        x = x + self.attn(self.norm1(x)) # (batch, n_patches+1, embed_dim)
        x = x + self.mlp(self.norm2(x)) # (batch, n_patches+1, embed_dim)
        return x

class VisionTransformer(nn.Module):
    """
    VisionTransfomrer put together. Main parameters to change are:

    img_size: Size of input image
    patch_size:  Size of individual patches (Smaller patches lead to more patches)
    n_classes: Number of outputs for classification
    embed_dim: Length of embedding vector for each patch
    depth: Number of wanted transformer blocks
    num_heads: Number of wanted attention heads per block
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_classes=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, attn_p=0.2,
                 mlp_p=0.2, proj_p=0.2, pos_drop=0.2, efficient=True):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter((torch.zeros(1, 1+self.patch_embed.n_patches, embed_dim)))
        self.pos_drop = nn.Dropout(pos_drop)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim=embed_dim,
                                 num_heads=num_heads,
                                 mlp_ratio=mlp_ratio,
                                 mlp_p=mlp_p,
                                 attn_p=attn_p,
                                 proj_p=proj_p,
                                 efficient=efficient)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x) # (batch, n_patches, embed_dim)
        cls_token = self.cls_token.expand(batch_size, -1, -1) # (batch, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (batch, n_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0] # (batch, embed_dim)
        x = self.head(cls_token_final) # (batch, n_classes)
        return x, cls_token_final


def train(model, device, epochs, optimizer,
          scheduler, loss_fn, trainloader,
          valloader, savepath="ViTDogsvCatsSmall.pt"):
    log_training = {"epoch": [],
                    "training_loss": [],
                    "training_acc": [],
                    "validation_loss": [],
                    "validation_acc": []}

    best_val_loss = np.inf
    for epoch in range(1, epochs + 1):
        print(f"Starting Epoch {epoch}")
        training_losses, training_accuracies = [], []
        validation_losses, validation_accuracies = [], []

        for image, label in tqdm(trainloader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            out, _ = model.forward(image)

            ### CALCULATE LOSS ##
            loss = loss_fn(out, label)
            training_losses.append(loss.item())

            ### CALCULATE ACCURACY ###
            predictions = torch.argmax(out, axis=1)
            accuracy = (predictions == label).sum() / len(predictions)
            training_accuracies.append(accuracy.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

        for image, label in tqdm(valloader):
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                out, _ = model.forward(image)

                ### CALCULATE LOSS ##
                loss = loss_fn(out, label)
                validation_losses.append(loss.item())

                ### CALCULATE ACCURACY ###
                predictions = torch.argmax(out, axis=1)
                accuracy = (predictions == label).sum() / len(predictions)
                validation_accuracies.append(accuracy.item())

        training_loss_mean, training_acc_mean = np.mean(training_losses), np.mean(training_accuracies)
        valid_loss_mean, valid_acc_mean = np.mean(validation_losses), np.mean(validation_accuracies)

        ### Save Model If Val Loss Decreases ###
        if valid_loss_mean < best_val_loss:
            print("---Saving Model---")
            torch.save(model.state_dict(), savepath)
            best_val_loss = valid_loss_mean

        log_training["epoch"].append(epoch)
        log_training["training_loss"].append(training_loss_mean)
        log_training["training_acc"].append(training_acc_mean)
        log_training["validation_loss"].append(valid_loss_mean)
        log_training["validation_acc"].append(valid_acc_mean)


        print("Training Loss:", training_loss_mean)
        print("Training Acc:", training_acc_mean)
        print("Validation Loss:", valid_loss_mean)
        print("Validation Acc:", valid_acc_mean)

    return log_training, model


if __name__ == "__main__":
    ViT = VisionTransformer(embed_dim=384,
                            depth=6,
                            num_heads=6,
                            efficient=True)

    params = sum([np.prod(p.size()) for p in ViT.parameters()])
    print(f"Total Number of Parameters: {params}")

    ### SETUP DATASET ###
    PATH_TO_DATA = "../data/dogsvcats"
    dataset = ImageFolder(PATH_TO_DATA)

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

    ### SETUP TRAINING LOOP ###
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    print("Training on Device {}".format(DEVICE))

    ViT = ViT.to(DEVICE)
    EPOCHS = 100
    optimizer = optim.AdamW(params=ViT.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5,
                                                  max_lr=1e-3, mode='exp_range',
                                                  gamma=0.98, cycle_momentum=False)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 128

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    logs, model = train(model=ViT,
                        device=DEVICE,
                        epochs=EPOCHS,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        trainloader=trainloader,
                        valloader=valloader)

    logs = pd.DataFrame(logs)
    logs.to_csv("dogsvcats.csv", index=False)









































