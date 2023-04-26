### PyTorch for NLP: Sequence Classification

We will be shifting gears to sequence data! Until now we have essentially dealt with a Single input 
Single output type of Neural Network (also known as **One to One**). Here is a common visual you wil
see when dealing with Sequence type models:

![sequence](http://karpathy.github.io/assets/rnn/diags.jpeg)

The credit to this image goes to Andrej Karpathy in his incredible blog [The Unreasonable Effectiveness of
Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) which I highly recommend you give a 
read!

In this lesson we will be looking at the **Many to One** type model. Essentially we have a sequence of words
from the IMDB Dataset that we need to aggregate. We want to then take this model and output it to a binary classification
head to predict if a review is positive or negative. Our trick to do this will be the **Long Short Term
Memory** Architecture. We will be doing a few things in this implementation:
- Deep Dive into the RNN/LSTM architecture and why we prefer it over Convolutions
- Explore how Cell states in the LSTM enable long term memory
- Understand Backpropagation through Time for gradient updates on sequential data
- Take a peek under the hood of an LSTM cell and its 3 gates
- Understand how different classification heads enable different prediction types
- Build a Dataset/Dataloader for Text data
- Learn the PyTorch Embeddings module to map words to vector representations
- Build and Train an LSTM Model from scratch!