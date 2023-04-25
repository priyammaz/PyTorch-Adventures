### PyTorch for Vision: Convolutions &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BE-y1876znMeijFu4AX4qcZdt-fs8o7a?usp=sharing)
The convolution mechanic was a huge step forward in Deep Learning for Computer Vision and still remains
one of the dominant methods for image processing today! In this lesson we will be exploring the typical
Convolutional Neural network structure and how to implement [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), one of the earliest examples that won
the ImageNet competition in 2012. 

### Convolution animations
Look at these incredible Convolution Animations from [here](https://github.com/vdumoulin/conv_arithmetic) so you can better visualize
exactly what it is trying to do! You can also further read [this](https://arxiv.org/pdf/1603.07285.pdf) to learn more about the 
mathematics of convolution arthmetic. 

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_no_strides.gif"></td>
    <td><img width="150px" src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/same_padding_no_strides.gif"></td>
    <td><img width="150px" src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/full_padding_no_strides.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
    <td>Half padding, no strides</td>
  </tr>
  <tr>
    <td><img width="150px" src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_strides.gif"></td>
    <td><img width="150px" src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/padding_strides.gif"></td>
    <td><img width="150px" src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/padding_strides_odd.gif"></td>
  </tr>
  <tr>
    <td>No padding, strides</td>
    <td>Padding, strides</td>
    <td>Padding, strides (odd)</td>
  </tr>
</table>




### Main Ideas for Vision
There is obviously a lot involved when it comes to Vision but we will go over some core ideas that allow us to use Deep Learning for this task:
- The problems of flattening an image for a linear layer
- Understanding the Convolution mechanic as a sliding filter approach
- Early examples of Convolutions with the Sobel Filter
- How to use PyTorch nn.Conv2d
- Learn the typical Convolutional Neural Network structure
- Max/Average Pooling as a Downsampling technique
- BatchNormalization to ensure Normality between layers
- Dropout to curtail overfitting 
- Build and Train the AlexNet from scratch 
