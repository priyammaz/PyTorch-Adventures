## UNet for Semantic Segmentation

![image](https://github.com/priyammaz/HAL-DL-From-Scratch/blob/main/src/visuals/autoencoder_vs_unet.png?raw=true)

UNets were proposed back in 2015, and is one of the earliest, if not the first, example of a fully convolutional end-to-end segmentation system. UNets are not all that different from an AutoEncoder, except for one key thing: Skip Connections. Both AutoEncoders and UNets comprise of an encoder, bottleneck and deocder, but the UNet also connects outputs of the encoder to its counterpart in the decoder. 

If you read the paper you will realize that this paper was ahead of its time. There is no mention of residual connections or resnets anywhere, and thats because this paper was published 6 months before ResNet was! Although this is using concatenation instead of summation for its residual connections, it is effectively the same thing. 

I believe this is one of the most rock-solid architectures that have been proposed and has far outlived most models, as UNets power todays most impressive generative diffusion models despite being almost a decade old. The implementation we do here is basically copy-pasted into my [Intro to Diffusion](https://github.com/priyammaz/HAL-DL-From-Scratch/tree/main/PyTorch%20for%20Generation/Diffusion/Intro%20to%20Diffusion) so you can take a look there!

We will be implementing a slightly more modern variant of the UNet, where not only will be have skip connections, but each block of the UNet will also have residual connections to allow for deeper models. We will train this model on two datasets, Carvana and ADE20K, just to see some of the differences in binary vs multiclass classification, as well as general challenge of segmenting complex scenes. 

Here is an example of how this model performs after some training!

![ADE20K](https://github.com/priyammaz/HAL-DL-From-Scratch/blob/main/src/visuals/ade20k_unet_prediction.png?raw=true)

### Datasets

Download the data and place in the **data** folder in the root directory of this repository! If you save elsewhere, just update the *path_to_data* argument at the start of the notebook. 

#### Carvana

The main jupyter notebook will be using the Carvana dataset which you can download from Kaggle from their [Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge). The zip folder that will download will have a bunch more zip folders inside it. We will only be using the *train.zip* and *train_masks.zip* folders.

#### ADE20K

This is the tougher dataset that you can download from the [MIT Scene Parsing Benchmark](http://sceneparsing.csail.mit.edu/), I only downloaded the **train/val** split of the data.
