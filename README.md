# Deep Learning from End to End

![banner](src/visuals/banner.png)

---
The National Center for Supercomputing Applications Center of AI Innovation at the University of Illinois
Urbana-Champaign hosts many HAL tutorials dedicated to give newcomers to the 
fields of AI a practical first look at the tools needed to perform research in 
this area. 

### Get Started
All of these tutorials can easily be run on the [HAL](https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster)
system at the NCSA and follow the new users instructions [here](https://wiki.ncsa.illinois.edu/display/ISL20/New+User+Guide+for+HAL+System)
to setup an account. 

If you prefer to use [Google Colaboratory](https://colab.research.google.com/), that will also work fine! You will
just need to setup the environment for specific packages needed (Easy pip installs to get those)


### Data Prep ###
We will be using a couple of datasets in our Deep Learning Adventures!!
- Cats Vs Dogs
- IMBD Movie Reviews
- MNIST Handwritten Digits
- Harry Potter Corups

Ensure you have a */data* folder in your root directory of the git repo and run the following to install all datasets
```angular2html
bash download_data.sh 
```

### Intro to PyTorch

[![button](src/visuals/play_button.png)](https://www.youtube.com/watch?v=QzJql9AOGt4) &nbsp;[Get Started with PyTorch!!](Intro%20to%20PyTorch)

Deep Learning has revolutionized AI architectures and it is vital to learn the tools to build these models! 
We will cover the following ideas to get acquanited with the PyTorch system and review some basic ideas!

- PyTorch Fundamentals
    - Tensors
    - AutoGrad
- Optimization through Gradient Descent
- Linear Regression and Mean Squared Error Loss
- Logistic Regression and Binary Cross Entropy Loss
- MNIST Classification with Dense Neural Network
- Utilizing GPU to accelerate compute

Understanding these tools and techniques should give you a fairly good intuition on how PyTorch functions
and should get you started on most types of problems

### PyTorch DataLoaders
[Get Started with Datasets and Dataloaders!!](PyTorch%20DataLoaders)

There is no way to train a model without first knowing how to work with different data types and load them in an 
efficient manner. For this we will explore the incredible PyTorch Dataset and DataLoader that handles most of the heavy lifting for you!!
We will be focusing on two domains in this part:

1) Computer Vision 
   1) How to Build a custom PyTorch Dataset for the Dogs Vs Cats Data 
   2) Some introduction to the Transforms module in Torchvision 
2) Natural Language 
   1) How to load sequence data for the IMBD dataset
   2) Custom data collator to manage sequences of different lengths

Once we have built custom datasets, we will then look how to wrap it in the DataLoader to enable us to grab minibatches!!

### Intro to Pre-Trained Models
[![button](src/visuals/play_button.png)](https://www.youtube.com/watch?v=QzJql9AOGt4) &nbsp;[Get Started with PreTrained Models!!](Leveraging%20Pre-Trained%20Models)

Most niche problems don't have large datasets so a typical strategy is to start with a Pre-Trained model 
on one task and transfer that knowledge to the new task of interest. To do so we will be attempting to 
classify images of Dogs vs Cats with a few different methods:
- Train entire ResNet18 Model from PyTorch from scratch
- Train only classification head of a Pre-Trained ResNet18 Model
- Leverage the widely popular HuggingFace 🤗 repository to complete the same task

### Vision Transformers from Scratch 

[Get Started with Vision Transformers!!](Dive%20Into%20Attention%20with%20Vision%20Transformers)

Transformers have revolutionized Deep Learning and is the predominant architecture of choice for almost all
state-of-the-art models. The most unique aspect of the model is the ability to process any type of data. Where
previously we would typically use Convolutions for image tasks and RNN/LSTM type models for sequence tasks,
the Transformer can process all types of data with a little work. The common architecture allows us to then experiment
with unique MultiModal and Self-Supervised Pretraining tasks. In this repository we will be looking at the 
Vision Transformer (which adds a few features over the traditional Transformer Architecure) but the majority 
of the ideas hold true regardless of architectures. More specifically we will be looking at:
- What are the benefits of Attention of Convolutions
- What are Patch Embeddings (i.e. how to convert images to "sequence" data as the Transformer expects it)
- Understanding the CLS Token and Positional Embeddings
- How to build a Single Attention Head with Queries, Keys and Values
- Expanding Single Headed Attention to MultiHeaded Attention
- Building an Efficient Attention block that paralleizes all Attention Heads at once rather in sequence
- Understanding why Layer Normalization is prefered over Batch Normalization

### Materials to Come!!!
- [ ] PyTorch DataLoaders
- [ ] Residual and Skip Connections
- [ ] Advanced PyTorch tutorials on Sequence Data
- [ ] Deep Dive into the Huggingface Ecosystem
- [ ] Build and Train a GPT Model From Scratch (using Torch only!!!)
- [ ] PyTorch Distributed Data and Model Parallelism
- [ ] Advanced Model Logging via Weights and Biases
- [ ] RayTune for PyTorch Hyperparameter Sweep
- [ ] Efficient Data Storage/Access via Deep Lake
