# Introduction to Diffusion &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1I8IDHNMURVHgd9hlBEi7cfprGNEbp16J/view?usp=sharing)

Diffusion models have become the most powerful form of Generative models today and are used everywhere! Whether you are talking about Images, Audios, or Videos, diffusion are the best way to learn the generative process. We will be doing a deep dive exploration of this really cool approach. 

### Caveat!
Diffusion models (especially the UNet architecture) can become pretty large pretty quickly, so I will be going through building this model for the [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) dataset that isn't included in the default download for this repo. If you want to follow along for larger models, download the dataset, grab some GPUs and we can start training! If you want to use Colab, no worries! We can train a smaller model on MNIST. 

### Example Generation

![example_gen](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/src/visuals/celeba_diffusion.png)