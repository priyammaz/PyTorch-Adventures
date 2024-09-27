## Leveraging Pre-Trained Models for Transfer Learning &nbsp; [<img src="../../src/visuals/x_logo.png" alt="drawing" style="width:25px;"/>](https://x.com/data_adventurer/status/1839491569533223011)&nbsp; [<img src="../../src/visuals/play_button.png" alt="drawing" style="width:40px;"/>](https://www.youtube.com/watch?v=c6VTUx0EdqM)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KYCINwxq-y8QOMCRylsxDaP9RCUHz-bV?usp=sharing)

Most niche problems don't have large datasets so a typical strategy is to start with a Pre-Trained model 
on one task and transfer that knowledge to the new task of interest. To do so we will be attempting to 
classify images of Dogs vs Cats with a few different methods:
- Train entire AlexNet Model from PyTorch from scratch
- Train an entire AlexNet model starting from PreTrained weights
- Train only classification head of a Pre-Trained AlexNet Model


**Definitely Read the [AlexNet Paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)!!**

![AlexNet](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-22_at_6.35.45_PM.png)
We will be going into a lot more depth in the next section to understand Convolutions, but for now I just want to 
offer some intuition about Transfer Learning as it's a unique and important idea for modern Deep Learning. AlexNet was
probably the first real evidence we had about Deep Learning being a powerful Pattern Recognition tool and in 2012, won the 
ImageNet challenge. We will be writing our own AlexNet later from scratch but we will be pulling it for now from the PyTorch 
model hub!

