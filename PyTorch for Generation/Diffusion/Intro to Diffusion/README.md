# Introduction to Diffusion &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1I8IDHNMURVHgd9hlBEi7cfprGNEbp16J/view?usp=sharing)

Diffusion models have become the most powerful form of Generative models today and are used everywhere! Whether you are talking about Images, Audios, or Videos, diffusion are the best way to learn the generative process. We will be doing a deep dive exploration of this really cool approach. 

### Caveat!
Diffusion models (especially the UNet architecture) can become pretty large pretty quickly, so I will be going through building this model for the [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) dataset that isn't included in the default download for this repo. If you want to follow along for larger models, download the dataset, grab some GPUs and we can start training! If you want to use Colab, no worries! We can train a smaller model on MNIST. 

### Example Output ###
![example](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/PyTorch%20for%20Generation/Diffusion/Intro%20to%20Diffusion/generated/124000.png)

## Accelerated Training
I wanted to see how my model would do if I scaled it up a bit and trained it for a few days on a cluster! To do this I used [Huggingface Accelerate](https://huggingface.co/docs/accelerate/en/index) for distributed training and trained on the CELEBA dataset. You can see the results of this training on Weights and Biases [here](https://api.wandb.ai/links/exploratorydataadventure/ndtdvrf1)

### Example Generation from 250K steps of Training
![example_gen](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/src/visuals/celeba_diffusion.png)

### Run it yourself!

If you want to run this script yourself, just make sure you have Huggingface Accelerate setup and the data downloaded! After create some folders for your working directory where checkpoints will be saved and another generation directory where your images will save as it trains. Makes sure to pass those paths to the arguments below and then you can run this to start your own diffusion training!

```
accelerate launch train.py --experiment_name "CELEBA_DIFFUSION" \
                           --path_to_data "<PATH_TO_YOUR_DATA>" \
                           --working_directory "<PATH_TO_YOUR_WORKING_DIRECTORY>" \
                           --generated_directory "<PATH_TO_YOUR_GENERATION_DIRECTORY>" \
                           --num_diffusion_timesteps 1000 \
                           --plot_freq_interval 100 \
                           --num_generations 3 \
                           --num_training_steps 250000 \
                           --warmup_steps 2000 \
                           --evaluation_interval 500 \
                           --batch_size 64 \
                           --gradient_accumulation_steps 2 \
                           --learning_rate 1e-4 \
                           --loss_fn mse \
                           --img_size 128 \
                           --starting_channels 128 \
                           --num_workers 24 \
                           --num_keep_checkpoints 3

```
