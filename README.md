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
- [**Intro to PyTorch: Exploring the Mechanics**](PyTorch%20Basics/Intro%20to%20PyTorch/) &nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/d86lJxKInYg?feature=shared) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YQanR0ME7ThsU9YwLzXhGvYGOdH2ErSa?usp=sharing)


- [**PyTorch Datasets and DataLoaders**](PyTorch%20Basics/PyTorch%20DataLoaders/) &nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/S8X6qcColBY?feature=shared)  &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nurV-kJmoPYlXP-qNAGGLsFXuS3lpNil?usp=sharing)


- [**Leveraging Pre-Trained Models for Transfer Learning**](PyTorch%20Basics/Basics%20of%20Transfer%20Learning/) &nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://www.youtube.com/watch?v=c6VTUx0EdqM)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KYCINwxq-y8QOMCRylsxDaP9RCUHz-bV?usp=sharing)


- [**Intro to Vision: Digging into Convolutions & AlexNet**](PyTorch%20for%20Computer%20Vision/Intro%20to%20Vision/) &nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/WoIxtSBYyYA)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BE-y1876znMeijFu4AX4qcZdt-fs8o7a?usp=sharing)


- [**The Curse of Backprop: Residual Connections & Implementing ResNet**](PyTorch%20for%20Computer%20Vision/ResNet/) [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://www.youtube.com/watch?v=TqIU9K8nNhs)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OPnOApHCcZFFWkw-zfhNvfyQeswQxgea?usp=sharing)


- [**Digging into the Sequences: Classification with Recurrence**](PyTorch%20for%20NLP/Recurrent%20Neural%20Networks/IMDB%20Classification/) [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://www.youtube.com/watch?v=UBjmWHX8xlI)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c98opjQt1w-HTp10U1myjSWU9acDsaV4?usp=sharing)


- [**Lets Write a Story: Recurrence for Text Generation**](PyTorch%20for%20NLP/Recurrent%20Neural%20Networks/Harry%20Potter%20Generation/) [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://www.youtube.com/watch?v=f8qoaeF2kzY)&nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KO4JeIHRiKxiRJdK7gY-B9bZGfDSvCt_?usp=sharing)

- [**Distributed Training with Huggingface 🤗 Accelerate**](PyTorch%20Basics/Huggingface%20Accelerate/) [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/cHKyhHu6WW0)&nbsp;

## Neural Networks from Scratch ##
- [**ManualGrad: Simple Neural Network Implementation**](Neural%20Networks%20from%20Scratch/ManualGrad/)
- [**MyTorch: Building a Simple AutoGrad Engine**](Neural%20Networks%20from%20Scratch/AutoGrad/)
  
## Computer Vision ##

- [**UNet for Image Segmentation**](PyTorch%20for%20Computer%20Vision/UNET%20for%20Segmentation/) [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/fBlLsugz6Q8) &nbsp;
- [**Moving from Convolutions: Vision Transformer**](PyTorch%20for%20Computer%20Vision/Vision%20Transformer) [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/SjCwOKhzFno)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mh-yaSWwfTs1UcOdRQjRIvLuj6PU6liZ?usp=sharing)
- [**Masked Image Modeling with Masked Autoencoders**](PyTorch%20for%20Computer%20Vision/Masked%20AutoEncoder/)
- **JEPA**
  
## Natural Language Processing ##
- [**Causal Language Modeling: GPT**](PyTorch%20for%20NLP/GPT%20for%20Causal%20Language%20Models)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DZ406Ytb-ls1jDI1BovARwYq__ptr1Tx?usp=sharing)
  
- [**Masked Language Modeling: RoBERTa**](PyTorch%20for%20NLP/RoBERTa%20for%20Masked%20Language%20Models)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtube.com/playlist?list=PL16vydMdqFg9kpCd_kj92bUOA5P0STduv&feature=shared) &nbsp;&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MChQ84-1VKBbjNCmzPQL02hxl-gckEYh?usp=sharing)
  
- [**Building Vision/Language Representations: CLIP**](PyTorch%20for%20NLP/CLIP/)
  
- [**Attention is All You Need for Language Translation**](PyTorch%20for%20NLP/Seq2Seq%20for%20Neural%20Machine%20Translation/)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://www.youtube.com/playlist?list=PL16vydMdqFg9g9xevVEl-MVVaK9jcqyC5) &nbsp;
  
- [**Llama 4**](PyTorch%20for%20NLP/Llama4/)&nbsp;  [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/yXbF-1n9wxs?feature=shared) &nbsp;

- [**Language Diffusion Models**](PyTorch%20for%20NLP/Language%20Diffusion%20Model/)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/yXbF-1n9wxs?feature=shared) &nbsp;
  
## Audio Processing ##
- [**Intro to Audio Processing in PyTorch**](PyTorch%20for%20Audio/Intro%20to%20Audio%20Processing/)&nbsp;  [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/55QWsm1itKo) &nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qjE2YCRvSI3JtH6To-fCvtZNiegfnFoT?usp=sharing)
- **Audio Processing From Scratch**
- [**Connectionist Temporal Classification Loss**](PyTorch%20for%20Audio/Connectionist%20Temporal%20Classification/)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/qvTDRUDDK30)
- [**Intro to Automatic Speech Recognition**](PyTorch%20for%20Audio/Intro%20to%20Automatic%20Speech%20Recognition/) &nbsp;  [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/725Og6eyWCc) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XmHksCjSuTVbfl-URTNgiuDs2s3_jnd9?usp=sharing)
- [**Quantized Audio Pre-Training: Wav2Vec2**](PyTorch%20for%20Audio/Wav2Vec2/)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/TMjKUHSFRbw)
- [**Intro to TTS: Tacotron2**](PyTorch%20for%20Audio/Tacotron2/)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/5WxgTVDf_v8)
- [**Neural Vocoder: HifiGAN**](PyTorch%20for%20Audio/HIFIGAN/)&nbsp;  [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/725Og6eyWCc)
- [**EnCodec**](PyTorch%20for%20Audio/Encodec/)&nbsp;  [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/725Og6eyWCc)
- **WaveNet**

## Generative AI
- ### AutoEncoders ##
  - [**Intro to AutoEncoders**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Intro_To_AutoEncoders.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/p7yUWIySj8o)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DldfPN9q1uSA4UkZYHV-3Ms5be333EKN?usp=sharing)
  - [**Variational AutoEncoders**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Variational_AutoEncoders.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/9NgC0sh9Msc)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_NLc6g5UJ-tmRUXZbF5r1FgWoEApaLmH?usp=sharing)
  - [**Vector-Quantized Variational Autoencoders**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Vector_Quantized_Variational_AutoEncoders.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/1mi2MSvigcc)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QqdHlnfJV5BATUymrXy-wi3F8YUIQFpl?usp=sharing)

- ### Quantization Methods
  - [**Gumbel Softmax Vector Quantization**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/gumbel_softmax_quantizer.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/c65nlNgfFrE)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nlHZlrFz0WV93k7n_VmdeUvAegJEoGay?usp=sharing)
  - [**Residual Vector Quantization**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Residual_Vector_Quantizer.ipynb)&nbsp;  [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/ZnyfaQRQ8GI)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ljFb-z3sICEewYdp6Dmm8jFmlqIru_nf?usp=sharing)
  - **Lookup-Free Quantization**

- ### Generative Adversarial Networks ##
  - [**Intro to Generative Adversarial Networks**](PyTorch%20for%20Generation/Generative%20Adversarial%20Network/Intro%20to%20GANs/)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/R9VOZnKEBE0)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1s2zEnDKo0zr4hqzSmsDzQvWHs40KPkgz?usp=sharing)
  - [**Deep Convolutional GAN**](PyTorch%20for%20Generation/Generative%20Adversarial%20Network/DCGAN/)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/tRSdyRJKVB4)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kB-6KvR0IVsXtWI6MElNd3m02mpX7S9u?usp=sharing)
  - [**Wasserstein GAN**](PyTorch%20for%20Generation/Generative%20Adversarial%20Network/WGAN/)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HneZAIokGAZW8Eq0ussFBoaw3BmG3imm?usp=sharing)
  - **SuperResolution with SRGAN**
  - [**Image2Image Translation with CycleGAN**](PyTorch%20for%20Generation/Generative%20Adversarial%20Network/CycleGAN/)

- ### Diffusion ##
  - [**Intro to Diffusion**](PyTorch%20for%20Generation/Diffusion/Intro%20to%20Diffusion/) &nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/_otkRnYaozY)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KBupTiAId1LO67IcM-yn3xkK81aj06sG?usp=sharing)
  - [**Text-Conditional Diffusion with Classifier Free Guidance**](PyTorch%20for%20Generation/Diffusion/Conditional%20Diffusion/)
  - [**Latent-Space Diffusion**](PyTorch%20for%20Generation/Diffusion/Latent%20Diffusion/)
  - **DDIM**
  - **Diffusion Transformers**

- ### Autoregressive Generation

  - [**PixelRNN**](PyTorch%20for%20Generation/Autoregressive%20Generation/PixelRNN/)
  - [**PixelCNN**](PyTorch%20for%20Generation/Autoregressive%20Generation/PixelCNN/)
 
## Dive into Transformers ##

- **Attention Mechanisms**
  - [**Attention is All You Need**](PyTorch%20for%20Transformers/Attention%20Mechanisms/Attention/) &nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/JXY5CmiK3LI)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vJAlxRehKG7yzAFJTTPa4Ecew0cxc1fL?usp=sharing)
  - [**Windowed Attention**](PyTorch%20for%20Transformers/Attention%20Mechanisms/Sliding%20Window%20Attention/)
  - **LinFormer**
  - **ConFormer**

- **Positional Embeddings**
  - **Sinusoidal Encodings**
  - **Rotary Positional Encoding**
  - **ALiBi**

- **Feed Forward**
  - **Mixture of Experts**
  
- **Normalization**
  - **Layer Norm**
  - **RMS Norm**
  
- **Inference Optimization**
  - **KV Cache**
  - **Multi-Head Latent Attention**

## Reinforcement Learning
- ### Model-Based Learning 
  - [**Policy Iteration**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Reinforcement%20Learning/Model-Based%20Learning/intro_rl_and_policy_iter.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/tRSdyRJKVB4)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Up4FwHlwhh8o2961Ka1D3Kf7EOKgPp6x?usp=sharing)
  - [**Value Iteration**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Reinforcement%20Learning/Model-Based%20Learning/value_iteration.ipynb)&nbsp;  [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/tRSdyRJKVB4)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1H6ElFJXbRFQtmUst9IPlxiLw-aZkg0ho?usp=sharing)
- ### Model-Free Learning 
  - [**Monte Carlo**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Reinforcement%20Learning/Model-Free%20Learning/monte_carlo.ipynb)&nbsp;  [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/7j4krL1QQqY)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SttDsqUIZcZUZzh8p5Ir_rKdY9kcIK6O?usp=sharing)
  - [**SARSA**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Reinforcement%20Learning/Model-Free%20Learning/sarsa.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/nEDBDYDtGIM)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vGk4dck6EAXvlff6RJI9MMWayZ1ixpRb?usp=sharing)
  - [**Q Learning**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Reinforcement%20Learning/Model-Free%20Learning/q_learning.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/Sm2eXcTPd5s)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pH5knVA5Vc4hZSUM_GSv-sp58BeNHsMj?usp=sharing) 
  - [**TD(N)**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Reinforcement%20Learning/Model-Free%20Learning/td_n.ipynb)&nbsp;  [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/FenWuNaKNyQ)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GdPHVH8zIElgWtFP4y-4sm1HsNcakjCa?usp=sharing) 
  - [**TD(Lambda) + Eligibility Traces**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Reinforcement%20Learning/Model-Free%20Learning/td_lambda.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/sMfJADpFca8)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fb0olJX8DR-fODhXRG6v3p4e_YiRsdbl?usp=sharing) 

- ### Intro to Deep Reinforcement Learning
  #### A. Value Based Methods
  - [**Deep Q Learning**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Deep%20Reinforcement%20Learning/Deep%20Q-Learning/deep_q_learning.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/WgLgFdtXNvQ)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LcGwNDl2sE3ZPcJmCE2XjHHyADvZflhd?usp=sharing) 
  - [**Double Deep Q Learning**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Deep%20Reinforcement%20Learning/Double%20Deep-Q%20Learning/double_deep_q_learning.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/xyrDkAhG9EY)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LzXqDgpN-POGw6LUwyubTWqEOyAz07Tm?usp=sharing) 
  - [**Dueling Deep Q Learning**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Deep%20Reinforcement%20Learning/Dueling%20Deep-Q%20Learning/dueling_deep_q_learning.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/pkMJus7_reY)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18I1Oc65jJcSexteWvfTUZLfhTRNT3wy_?usp=sharing) 
  - [**Prioritized Experience Replay**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Deep%20Reinforcement%20Learning/Prioritized%20Experience%20Replay/prioritized_experience_replay.ipynb)
  - [**PER With SumTree**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Deep%20Reinforcement%20Learning/Prioritized%20Experience%20Replay/sumtree_per.ipynb)

  #### B. Policy and Critic Based Methods
  - [**Policy Networks (REINFORCE)**](PyTorch%20for%20Reinforcement%20Learning/Deep%20RL%20Policy%20Models/Policy%20Networks%20\(REINFORCE\)/policy_networks.ipynb)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/CF1g4CstVkY)&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1opyG_iBgmNnDG_28AFjiRR6LV5_Vj49C?usp=sharing) 
  - [**REINFORCE with Baseline**](PyTorch%20for%20Reinforcement%20Learning/Deep%20RL%20Policy%20Models/REINFORCE%20with%20Baseline/reinforce_with_baseline.ipynb)
  - [**Generalized Advantage Estimation (GAE)**](PyTorch%20for%20Reinforcement%20Learning/Deep%20RL%20Policy%20Models/Generalized%20Advantage%20Estimate/generalized_advantage_estimation.ipynb)
  - [**Vanilla Actor-Critic**](PyTorch%20for%20Reinforcement%20Learning/Deep%20RL%20Policy%20Models/Vanilla%20Actor%20Critic/actor_critic.ipynb)

- ### Advanced Deep Reinforcement Learning
  - [**Synchronous Actor-Critic (A2C)**](PyTorch%20for%20Reinforcement%20Learning/Deep%20RL%20Policy%20Models/A2C/a2c.ipynb)
  - [**Trust Region Policy Optimzation  (TRPO)**](PyTorch%20for%20Reinforcement%20Learning/Deep%20RL%20Policy%20Models/Trust%20Region%20Policy%20Optimization/trpo.ipynb)
  - [**Proximal Policy Optimization (PPO)**](PyTorch%20for%20Reinforcement%20Learning/Deep%20RL%20Policy%20Models/Proximal%20Policy%20Optimization/ppo.ipynb)
  - [**Deep Deterministic Policy Gradient (DDPG)**](PyTorch%20for%20Reinforcement%20Learning/Continuous%20Control/DDPG/ddpg.ipynb)
  - **Twin Delayed DDPG (TD3)**
  - **Soft Actor Critic (SAC)**

## Tools
- **Gradient Checkpointing**
- [**Low Rank Adaptation**](PyTorch%20Tools/LoRA/)&nbsp; [<img src="src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1905749914002641138)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/nhRYyXGkjSU) &nbsp;
- **Knowledge Distillation**
- [**Quantized Low Rank Adaptation**](PyTorch%20Tools/QLoRA/)
- **TensorRT**
- **DeepSpeed**
- **Triton**