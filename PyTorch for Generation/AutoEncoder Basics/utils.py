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
        