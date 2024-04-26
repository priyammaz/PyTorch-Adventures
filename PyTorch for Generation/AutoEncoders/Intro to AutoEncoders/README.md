# Intro to AutoEncoders

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img src="https://github.com/priyammaz/HAL-DL-From-Scratch/blob/main/src/visuals/AE_Embeddings.gif?raw=true"></td>
    <td><img src="https://github.com/priyammaz/HAL-DL-From-Scratch/blob/main/src/visuals/VAE_Embeddings.gif?raw=true"></td>
    <td><img src="https://github.com/priyammaz/HAL-DL-From-Scratch/blob/main/src/visuals/VQVAE_Embeddings.gif?raw=true"></td>
  </tr>
</table>


Autoencoders are an incredible compression tool which are used across the board in all types of tasks. Most notable is in many diffusion models such as Stable Diffusion, a Variational AutoEncoder is employed for Latent Space Diffusion modeling (we will see more of this later!). For now though, I want us to get familiar with how AutoEncoders work and build 3 types: (1) **Vanilla AutoEncoders**, (2) **Variational AutoEncoders** and (3) **Vector Quantized Variational AutoEncoders**. We will build both a Linear and Convolutional model and both the Variational and Vector Quantized models will be changes from the original Vanilla Autoencoder implementation so you can see what goes into it!

**Note!**

All of our implementations we will do going forward is on MNIST, and that dataset is too simple to give appreciable differences between these architectures! I mainly decided to do this so we can focus on the core ideas of each architecture rather than fiddling around with making the model larger to learn the encodings. The next step will be to build much larger AutoEnoder models so we can see better what is going on!

