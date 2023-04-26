## Dive into Attention with the Vision Transformer &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mh-yaSWwfTs1UcOdRQjRIvLuj6PU6liZ?usp=sharing)

![ViT](https://github.com/google-research/vision_transformer/raw/main/vit_figure.png)

Until a few years ago, Convolutions have been the default method of all Deep Learning for Vision tasks. But there was a limitation
of the Convolution mechanic that prompted the creation of the Vision Transformer: Lack of Spatial Attention. More specifically, 
this means that Convolutions are able to model local features within the kernel size very well, but there was no way to explain
how the top left part of an image is related to the bottom right. 

The Transformer architecture was able to solve this for Sequence data, as the main goal is to learn the relationships between
different pairs of words. In the same way, the goal of the Vision Transformer will be to learn how different parts of an image
are related to one another. 

This will be our first exploration of the Transformer architecture and we will be implementing everything from scratch!
This means we want to learn the properties of the Attention mechanism and how to actually implement them using only PyTorch!
The main ideas we will cover are:

- Why Attention and not Convolutions?
- Patch Embeddings (Conversion of Images to "sequences")
- Understanding CLS tokens and Positional Embeddings
- How to build a single Attention Head (Q, K, V)
- Expanding Single Headed Attention to MultiHeaded Attention
- The purpose of LayerNormalization
