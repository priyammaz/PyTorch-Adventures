# Deep Learning from End to End

![banner](src/visuals/banner.png)

---
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/) &nbsp; 
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) &nbsp; 
[![](https://img.shields.io/badge/contributors-welcome-informational?style=for-the-badge)](https://github.com/priyammaz/HAL-DL-From-Scratch/graphs/contributors) &nbsp;
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)

### Open Source Learning and the Democratization of AI

The National Center for Supercomputing Applications (NCSA) Center of AI Innovation (CAII) at the University of Illinois
Urbana-Champaign has a focus on driving and enabling student-driven research. Their goal is to give access to the state-of-the-art
tools and hardware so people can find novel ways to solve unanswered problems. I am one such student that has learned an incredible amount 
of knowledge and gained a lot of intuition on Artificial Intelligence through my mentors here. 

Along with the NCSA, I also want to acknowledge all the amazing open source materials that I have found and learned 
from over the years. These resources often filled the gap for me between the theory that justified the model architecture and the actual implementation of them.
I will do my best to reference all that work throughout, so you can see where I had learned it from initially!
The purpose of this repository is to bring together that wealth of knowledge to truly be a one-stop-shop for everyone from beginners
to researchers to gain something from and continue to push the boundaries of Open Source Research!!

**If you want to contribute**, (and please do it you want!!) go ahead an submit a PR and I can review it!
### Getting Started
All of these tutorials can easily be run on the [HAL](https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster)
system at the NCSA and you can follow the new users instructions [here](https://wiki.ncsa.illinois.edu/display/ISL20/New+User+Guide+for+HAL+System)
to setup an account. 

If you prefer to use [Google Colaboratory](https://colab.research.google.com/), that will also work fine! You will
just need to setup the environment for specific packages needed (Easy pip installs to get those). For the datasets, you 
can save them in your Google Drive and access them from there!


### Data Prep ###
We will be using a couple of datasets in our Deep Learning Adventures!!
- [Cats Vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- [IMBD Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
- [MNIST Handwritten Digits](https://pytorch.org/vision/stable/datasets.html)
- [Harry Potter Corups](https://github.com/formcept/whiteboard/tree/master/nbviewer/notebooks/data/harrypotter)

Ensure you have a */data* folder in your root directory of the git repo and run the following to install all datasets
```
bash download_data.sh 
```

## Foundations
- [**Intro to PyTorch**](Intro%20to%20PyTorch) &nbsp;[![button](src/visuals/play_button_small.png)](https://www.youtube.com/watch?v=QzJql9AOGt4)  &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YQanR0ME7ThsU9YwLzXhGvYGOdH2ErSa?usp=sharing)


- [**PyTorch DataLoaders**](PyTorch%20DataLoaders) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nurV-kJmoPYlXP-qNAGGLsFXuS3lpNil?usp=sharing)


- [**Basics of Transfer Learning**](Basics%20of%20Transfer%20Learning) &nbsp; [![button](src/visuals/play_button_small.png)](https://www.youtube.com/watch?v=QzJql9AOGt4) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KYCINwxq-y8QOMCRylsxDaP9RCUHz-bV?usp=sharing)


- [**Intro to PyTorch for Vision**](PyTorch%20for%20Computer%20Vision/Intro%20to%20Vision) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BE-y1876znMeijFu4AX4qcZdt-fs8o7a?usp=sharing)


- [**Going Deeper with Residual Connections**](PyTorch%20for%20Computer%20Vision/ResNet) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OPnOApHCcZFFWkw-zfhNvfyQeswQxgea?usp=sharing)


- **PyTorch for NLP: RNN/LSTM**


- **Distributed Data/Model Parallelism**

## HuggingFace 🤗
- **Intro to Huggingface**
- **Training a Custom Tokenizer**
- **Finetuning Pre-trained Language Models**
- **Finetuning Pre-trained Image Models**
- **Accelerate for Multi-GPU Training**

## Attention From Scratch
- [**Dive into Attention with the Vision Transformer**](Dive%20Into%20Attention%20with%20Vision%20Transformers)
- **Dive into Attention with the Masked AutoEncoder: Masked Image Modeling**
- **Dive into Attention with GPT: Causal Language Modeling**
- **Dive into Attention with BERT: Masked Language Modeling**

## Encoder/Decoder From Scratch
- **Seq2Seq for Language Translation**
- **CNN/RNN for Image Captioning**
- **Attention is All You Need for Language Translation**

## Creative AI From Scratch
- **Variational AutoEncoders**
- **Vector-Quantized Variational AutoEncoder**
- **Generative Adversarial Networks**
- **Stable Diffusion**

## Vision Tasks from Scratch
- **UNet for Image Segmentation**
- **YOLO for Object Detection**

## Reinforcement Learning From Scratch
- **Q-Learning**
- **Deep-Q Learning**

## Utilities
- **RayTune for Hyperparameter Sweep**
- **Weights and Biases for Model Logging**
- **Deep Lake for MultiModal Data Storage**