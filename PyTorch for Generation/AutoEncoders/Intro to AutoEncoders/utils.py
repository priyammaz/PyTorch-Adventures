import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from celluloid import Camera
from sklearn.decomposition import PCA

def VAELoss(x, x_hat, mean, log_var, kl_weight=1, reconstruction_weight=1):

    ### Compute the MSE For Every Pixel [B, C, H, W] ###
    pixel_mse = ((x-x_hat)**2)

    ### Flatten Each Image in Batch to Vector [B, C*H*W] ###
    pixel_mse = pixel_mse.flatten(1)

    ### Sum  Up Pixel Loss Per Image and Average Across Batch ###
    reconstruction_loss = pixel_mse.sum(axis=-1).mean()

    ### Compute KL Per Image and Sum Across Flattened Latent###
    kl = (1 + log_var - mean**2 - torch.exp(log_var)).flatten(1)
    kl_per_image = - 0.5 * torch.sum(kl, dim=-1)

    ### Average KL Across the Batch ###
    kl_loss = torch.mean(kl_per_image)
    
    return reconstruction_weight*reconstruction_loss + kl_weight*kl_loss
    
def build_embedding_plot(encoding, title):

    encoding = pd.DataFrame(encoding, columns=["x", "y", "class"])
    encoding = encoding.sort_values(by="class")
    encoding["class"] = encoding["class"].astype(int).astype(str)

    for grouper, group in encoding.groupby("class"):
        plt.scatter(x=group["x"], y=group["y"], label=grouper, alpha=0.8, s=5)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    

def plot_embedding_visual(encoded_data_per_eval, iterations_per_eval=250, path_to_save="encoding_vis"):
    
    fig, ax = plt.subplots()
    
    for idx, encoding in enumerate(encoded_data_per_eval):
        
        encoding = pd.DataFrame(encoding, columns=["x", "y", "class"])
        encoding = encoding.sort_values(by="class")
        encoding["class"] = encoding["class"].astype(int).astype(str)
    
        for grouper, group in encoding.groupby("class"):
            plt.scatter(x=group["x"], y=group["y"], label=grouper, alpha=0.8, s=5)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.axis('off')
        plt.title("Linear AutoEncoder Embeddings")
        plt.savefig(os.path.join(path_to_save, f"step_{idx*iterations_per_eval}.png"), dpi=300)

        plt.close()

def build_embedding_animation(encoded_data_per_eval, iterations_per_eval=100):

    fig, ax = plt.subplots()

    camera = Camera(fig)
    
    for idx, encoding in enumerate(encoded_data_per_eval):
        
        encoding = pd.DataFrame(encoding, columns=["x", "y", "class"])
        encoding = encoding.sort_values(by="class")
        encoding["class"] = encoding["class"].astype(int).astype(str)
    
        for grouper, group in encoding.groupby("class"):
            plt.scatter(x=group["x"], y=group["y"], label=grouper, alpha=0.8, s=5)
    
        ax.text(0.4, 1.01, f"Step {idx*iterations_per_eval}", transform=ax.transAxes, fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        camera.snap()
        
    plt.close()
    anim = camera.animate(blit=True)
    
    return anim
    


def interpolate_space(model, x_range=(-3,3), y_range=(-3,3), num_steps=25):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    x_space = np.linspace(x_range[0], x_range[1], num_steps)
    y_space = np.linspace(y_range[0], y_range[1], num_steps)

    points = []
    for x in x_space:
        for y in y_space:
            points.append([x,y])

    points = torch.tensor(points, dtype=torch.float32).to(device)

    ### Pass Through Model Decoder and Reshape ###
    dec = model.forward_dec(points).detach().cpu()
    dec = dec.reshape(-1, 1, 32, 32)
    dec = dec.reshape((num_steps,num_steps, *dec.shape[1:]))

    fig, ax = plt.subplots(num_steps,num_steps, figsize=(10,10))

    for x in range(num_steps):
        for y in range(num_steps):
            
            img = np.array(dec[x,y].permute(1,2,0))
            ax[x,y].imshow(img, cmap="gray")
            ax[x,y].set_xticklabels([])
            ax[x,y].set_yticklabels([])
            ax[x,y].axis("off")
            
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()