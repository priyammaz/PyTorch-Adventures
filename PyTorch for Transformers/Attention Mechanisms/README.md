# Attention

The Attention mechanism in Transformers have become ubiquitous in almost every state-of-the-art Neural Network today. Although it has such a strong prevalance now, powering our Langauge Models, Vision Models, Generative Models and more, there are also some key limitations. Attention was introduced first in 2017, and a lot of new approaches and methods of augmenting it have been developed! In this section, we will build some of the most important ones!!

## Attention is All You Need

<div>
<img src="https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/src/visuals/attention_is_all_you_need.png" width="500"/>
</div>

The [Attention is All You Need](https://arxiv.org/pdf/1706.03762) paper was the first to introduce to us the attention mechanism in the form of the Transformer! This was originally a sequence to sequence model that included all forms of attention we use today: Self-Attention Encoder, Self-Attention Decoder and Cross Attention. Cross Attention is just a fancier version of the Self-Attention encoder when we are doing cross-modality modeling, so we will focus on the main pieces, the Encoder and Decoder!


### Sliding Window Attention 

<div>
<img src="https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/src/visuals/sliding_window_attention.png" width="500"/>
</div>

Although the Attention mechanism has found its way into every modality, it has the clear problem of a quadratic complexity with respect to sequence length. Remember, attention computes how every pair of tokens are related to each other, so if you have a sequence length of 10, there will be 100 computations. Buf if we have long sequence lengths like 10000, then that will attention will do 10 Million computations! This cost is too high for both training and inference, so we need to find a way to reduce it. One such attempt at this has been Sparse Sliding Window Attention (AKA Local Attention).

