import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from celluloid import Camera
from sklearn.decomposition import PCA

def VAELoss(x, x_hat, mean, log_var, kl_weight=0.0005):
    reproduction_loss = torch.mean((x-x_hat)**2)
    kl = torch.mean(- 0.5 * torch.sum(1+ log_var - mean**2 - torch.exp(log_var), dim=-1))
    return reproduction_loss + kl * kl_weight
    
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