# Conditional Diffusion

Previously, in our [Intro to Diffusion](https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20Generation/Diffusion/Intro%20to%20Diffusion) we explored Diffusion as a mechanism for generating images! We trained this model on the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset to see if we could train a model to generate high quality face images. Unfortunately, there was one key limitation: The generated image was totally random. We had no say over any characteristics of the image, so we will be exploring Conditional diffusion this time around! Luckily there isn't too much difference between what we did before and adding text conditioning to our model!

### Components of Conditional Diffusion

Previously, in regular diffusion, our goal was to learn $p(x)$, the data distribution, so we could sample from the distribution. In the case of conditional diffusion, we will be learning $p(x|y)$ where y is some conditioning signal. But to do this what do we need?

- **Conditioning Signal**: We need to provide some embeddings that represent our conditioning signal, and this changes based on what we want to use. If we want to use text, luckily there are a large number of language models out there that we can pass text through and grab the embeddings that represent that text. If we want to use images, we can similarly pass our conditioning image through a Vision model to grab image embeddings. In our case, we will be using embeddings from the [OpenAI CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip) model, but probably any langauge model would work fine, its just CLIP was jointly trained between images and text so it makes the most intuitive sense. 
- **Weaving Modalities**: In our [UNET](https://arxiv.org/abs/1505.04597) implementation, we added Self-Attention: a global computation across the spatial dimension of the image. For combining our UNet with our text conditioning signal, we will add in Cross-Attention, which will learn how different tokens in our text is related to different portions across the spatial dimension of our image.
- **Classifier Free Guidance**: A problem we have is, if we want to learn $p(x|y)$, how will the model know that we are learning the joint distribution, and that the generated samples are representative of $y$? Also, we may learn the joint distribution, but it is important to also learn the data distribution $p(x)$. Well, we could use an additional classification model. Lets say we are generating MNIST images, so we want to generate handwritten digits, and our conditioning signal is the class label 0 through 9. We could pass our generated images into a model that already can predict MNIST well. If the MNIST model can predict our generated images as the correct class as well, then we are good to go! Unfortunately, we dont have class labels, we have text, so its not really clear what the classifier in this case would be predicting. So we will be using [Classifier Free Guidance](https://arxiv.org/pdf/2207.12598). Remember in Diffusion, we pass in noise and the model has to learn to predict the original denoised image, and in this case needs to do so with the additional conditioning signal. What we can do is with some probability (known as the ```cfg_weight``` which we set to 0.1) we will delete the conditioning signal. This means 90% of the time, the model will have the conditioning signal y, and 10% of the time it will have no conditioning signal at all to do the same prediction. This means the model has to jointly learn the $p(x)$ when no signal is provided and also learn $p(x|y)$ when the signal is provided. There is a great [blob post by Sander Dieleman](https://sander.ai/2022/05/26/guidance.html) going into more detail about this!

### Data Preparation 

We will be training our model on the MSCOCO dataset, which is actually really small with only about 200K images, but should be good enough to learn with! To prepare this, just go to their [download page](https://cocodataset.org/#download) and get their ```"2017 Train images"``` and ```"2017 train/Val annotations"```. Download and unzip these to any folder you want!

### Folder Prep 
Makes sure to create a folder for your working directory and another folder for your generated images. Inside the folder for your generated images, create two more folder called ```conditional``` and ```unconditional``` so we can save images from both tasks to see how the model is doing. 


### Train Model ###

Here is my script for the model I trained, but feel free to change it to whatever you want! I again used [🤗 Huggingface Accelerate](https://huggingface.co/docs/accelerate/en/index) for my distributed training.

```bash
accelerate launch train.py --experiment_name "COCO_Conditional Diffusion" \
                           --path_to_data "<PATH_TO_COCO_DATASET>" \
                           --working_directory "<PATH_TO_WORK_DIR>" \
                           --generated_directory "<PATH_TO_WORK_DIR>" \
                           --num_diffusion_timesteps 1000 \
                           --plot_freq_interval 100 \
                           --num_generations 3 \
                           --num_training_steps 200000 \
                           --warmup_steps 2000 \
                           --evaluation_interval 500 \
                           --batch_size 128 \
                           --gradient_accumulation_steps 2 \
                           --learning_rate 1e-4 \
                           --loss_fn mse \
                           --img_size 128 \
                           --starting_channels 128 \
                           --cfg_weight 0.1 \
                           --num_workers 24 \
                           --num_keep_checkpoints 2
```

The model I trained was roughly 114 Million trainable parameters. In total there were 177 million parameters if we include the CLIP text encoder, although the weights on that model were frozen. You can see the overall results of the training run [here](https://api.wandb.ai/links/exploratorydataadventure/cd3qou08)!

### Results

Now to preface this, obviously I cannot match stable diffusion or anything similar as this is a small model trained on a relatively small dataset. In comparison Stable Diffusion has nearly 1 Billion parameters and is trained on the [LAION-5B](https://laion.ai/blog/laion-5b/) dataset, which has 5 Billion Image/Text pairs. The quality of the images of my generations are pretty questionable too, but we are just looking for evidence of learning here. 

#### Prompt 1: A red kite flying in the sky

So I see a sky and a red thing in it in this generation, so not too bad!


![image](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/PyTorch%20for%20Generation/Diffusion/Conditional%20Diffusion/generated/sample_gen_1.png)

#### Prompt 2: A snowy mountain range with a beautiful sunny sky

In this case I am seeing snow and a mountain range, and the sky is nice and blue, so also not too bad!

![image](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/PyTorch%20for%20Generation/Diffusion/Conditional%20Diffusion/generated/sample_gen_2.png)

### Visual of Diffusion Process

Here is a sample of the backward diffusion process for our conditional images. Something to note here that is cool, compared to the CELEBA we did in the Intro to Diffusion, is we actually are getting denoisining at earlier steps of the generation. There is 1000 steps denoising, so every image in the sequence represents 100 steps. We already start to see some of the image being generated in the 400th step, where in CELEBA, most of the image was generated in the last 200 steps. This probably means we dont really need 1000 steps for simple CELEBA diffusion and it was overkill to train with that, just a few hundred would have most likely sufficed.

![image](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/PyTorch%20for%20Generation/Diffusion/Conditional%20Diffusion/generated/conditional_diffusion_process.png)

The prompts used to generate these were the following:
- A red kite flying through a cloudy blue sky
- A sailboat floating on the ocean
- A man with an umbrella walking through the rain