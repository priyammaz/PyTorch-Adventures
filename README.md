# Exploratory Data Adventures!

![banner](src/visuals/banner.png)

---
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/) &nbsp; 
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) &nbsp; 
[![](https://img.shields.io/badge/contributors-welcome-informational?style=for-the-badge)](https://github.com/priyammaz/HAL-DL-From-Scratch/graphs/contributors) &nbsp;
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)

### Open Source Learning and the Democratization of AI

The greatest aspect of Artificial Intelligence today is its open source nature. Very rarely in history have key technologies, especially those as powerful as these models developed today, have been made completely available to the public. Without the incredible online resources made by all of the contributing scientists and engineers, I never would have learned as much as I know today. This repository acts as a documentation of my own exploration as well as an attempt to teach everything I learn to others. I will try my best to cite everyone and everything I reference that helped me learn everything as well!

The main limitation for researchers in this field today is we normally don't have buckets of GPU's just sitting around to train with! Every example I do will be a proof of concept, but to the best of my ability I will attempt to reproduce any model that is feasible!

#### Contributions

I am typically more wrong than I am right! If you find any errors in my work, that means there is an error in my knowledge. Please let me know as I want this to be as accurate as possible, but also, I want to learn as much as I can! If you want to contribute anything yourself, just submit a PR and I will review it!

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
#### Extra Datasets ####
There are a few other datasets that we will use but are inconsistent to automatically download and are used in the more advanced architectures! Just download them from the link and save them in the */data* folder! These datasets may also be too large to train in Google Drive so keep that in mind!
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- [MS-COCO](https://cocodataset.org/#download)
- [ADE20K](http://sceneparsing.csail.mit.edu/)

## Foundations
- [**Intro to PyTorch: Exploring the Mechanics**](PyTorch%20Basics/Intro%20to%20PyTorch/) &nbsp; [![button](src/visuals/play_button_small.png)](https://www.youtube.com/watch?v=zQ-OQXBJcyw)  &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YQanR0ME7ThsU9YwLzXhGvYGOdH2ErSa?usp=sharing)


- [**PyTorch Datasets and DataLoaders**](PyTorch%20Basics/PyTorch%20DataLoaders/) &nbsp; [![button](src/visuals/play_button_small.png)](https://www.youtube.com/watch?v=IkjmZI817ko)  &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nurV-kJmoPYlXP-qNAGGLsFXuS3lpNil?usp=sharing)


- [**Leveraging Pre-Trained Models for Transfer Learning**](PyTorch%20Basics/Basics%20of%20Transfer%20Learning/)  &nbsp; [![button](src/visuals/play_button_small.png)](https://www.youtube.com/watch?v=QvPCiHr6eLU)  &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KYCINwxq-y8QOMCRylsxDaP9RCUHz-bV?usp=sharing)


- [**Intro to PyTorch for Vision: Digging into Convolutions**](PyTorch%20for%20Computer%20Vision/Intro%20to%20Vision/) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BE-y1876znMeijFu4AX4qcZdt-fs8o7a?usp=sharing)


- [**Going Deeper with Residual Connections**](PyTorch%20for%20Computer%20Vision/ResNet/) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OPnOApHCcZFFWkw-zfhNvfyQeswQxgea?usp=sharing)


- [**Digging into the LSTM: Sequence Classification**](PyTorch%20for%20NLP/LSTM/LSTM%20IMDB%20Classification/) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c98opjQt1w-HTp10U1myjSWU9acDsaV4?usp=sharing)


- [**Lets Write a Story: Sequence Models for Text Generation**](PyTorch%20for%20NLP/LSTM/LSTM%20Harry%20Potter%20Generation/) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KO4JeIHRiKxiRJdK7gY-B9bZGfDSvCt_?usp=sharing)


- [**Large Model Considerations: Distributed Training**](PyTorch%20Basics/Distributed%20Training/) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cyxgaWonj-FrfEbZvTwAVepkZhaF_sda?usp=sharing)

- **Distributed Training with Huggingface 🤗 Accelerate**

## Computer Vision ##
- [**Going Deeper with ResNet**](PyTorch%20for%20Computer%20Vision/ResNet/) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OPnOApHCcZFFWkw-zfhNvfyQeswQxgea?usp=sharing)
- [**UNet for Image Segmentation**](PyTorch%20for%20Computer%20Vision/UNET%20for%20Segmentation/)
- [**Moving from Convolutions: Vision Transformer**](PyTorch%20for%20Computer%20Vision/Vision%20Transformer) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mh-yaSWwfTs1UcOdRQjRIvLuj6PU6liZ?usp=sharing)
- **Masked Image Modeling with Masked Autoencoders**
- **Self-Supervised Learning with DINO**
- **Hierarchical Vision Transformers with Swin Transformer**
  
## Natural Language Processing ##
- [**Causal Language Modeling: GPT**](PyTorch%20for%20NLP/GPT%20for%20Causal%20Language%20Models)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DZ406Ytb-ls1jDI1BovARwYq__ptr1Tx?usp=sharing)
- [**Masked Language Modeling: RoBERTa**](PyTorch%20for%20NLP/RoBERTa%20for%20Masked%20Language%20Models)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MChQ84-1VKBbjNCmzPQL02hxl-gckEYh?usp=sharing)
- **MLP to Mixture of Experts**

## Speech Processing ##
- **Intro to Audio Processing in PyTorch**
- **Connectionist Temporal Classification Loss**
- **Intro to Automatic Speech Recognition**
- **ASR through Self-Supervised Learning: Wav2Vec2**
- **RNN Transducer as an Alternative to CTC**

## Generative AI
- ### AutoEncoders ##
  - [**Intro to AutoEncoders**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Intro_To_AutoEncoders.ipynb)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DldfPN9q1uSA4UkZYHV-3Ms5be333EKN?usp=sharing)
  - [**Variational AutoEncoders**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Variational_AutoEncoders.ipynb)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_NLc6g5UJ-tmRUXZbF5r1FgWoEApaLmH?usp=sharing)
  - [**Vector-Quantized Variational Autoencoders**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Vector_Quantized_Variational_AutoEncoders.ipynb)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QqdHlnfJV5BATUymrXy-wi3F8YUIQFpl?usp=sharing)
  - [**Scaling Up AutoEncoders**](PyTorch%20for%20Generation/AutoEncoders/Scaling%20up%20AutoEncoders/)

- ### Autoregressive Generation ##
  - **PixelCNN**
  - **WaveNet**
- ### Generative Adversarial Networks ##
  - **Intro to Generative Adversarial Networks**
  - **SuperResolution with SRGAN**
  - **Image2Image Translation with CycleGAN**

- ### Diffusion ##
  - [**Intro to Diffusion**](PyTorch%20for%20Generation/Diffusion/Intro%20to%20Diffusion/) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KBupTiAId1LO67IcM-yn3xkK81aj06sG?usp=sharing)
  - **Text-Conditional Diffusion with Classifier Free Guidance**
  - **Latent-Space Diffusion**

## MultiModal AI ##
- **Building Vision/Language Representations: CLIP**
- **Automatic Image Captioning**
- **Visual Question Answering**

## Dive into Attention ##
- **Barebones Attention Mechanism**
- **Sparse Windowed Attention**
- **Linear Attention**

## Sequence to Sequence Modeling ##
- **Seq2Seq for Language Translation**
- **CNN/RNN for Image Captioning**
- **Attention is All You Need for Language Translation**

## Reinforcement Learning
- **Q-Learning**
- **Deep-Q Learning**
