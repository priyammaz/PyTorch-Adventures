# EnCodec: Introduction to Speech Tokenization

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/encodec_model.png?raw=true" width="600" />

## Text Tokenization is Easy!
Natural language is relatively easy to work with. Although many algorithms for text tokenization exists (Wordpiece, Bytepair Encoding, etc...), they all kind of do the same thing: map discrete symbols (words, characters, or subwords) into integer IDs drawn from a fixed vocabulary.

## Speech Tokenization is Not!
Speech however is fundamentally different. Unlike text, speech is not composed of a sequence of discrete symbols. It is a continuous and high dimensional waveform. 

At a typical sampling rate of 24kHz, you have 24,000 real valued floating point numbers per every second of audio. There is no predifined vocabulary, no natural segmentation into symbols, and no obvious way to assign discrete token IDs. 

If we want to apply modern sequence models like Transformers to speech in the same way we do text, we must first convert this continuous signal into a sequence of discrete tokens. 

## EnCodec

Many architectures exist today for this, but EnCodec is one of the first significant attempts. EnCodec is a neural audio codec that learns to compress speech down into a sequence of discrete tokens and then reconstruct the waveform from those tokens, just like an AutoEncoder! 

### Residual Vector Quantization

The main part of this architecture is the quantizer, and for most speech applications, Residual Vector Quantization works really well! This mainly comes down to the fact that speech has a nice structure of hierarchical information. You can imagine audio having high-level semantic information (speech content) and low-level acoustic details (timbre, prosody, etc..). Early VQs can capture those higher level features, while progressive deepder VQs capture the lower ones. 

EnCodec has upto 32 codebooks in its implementation as it is trained on a wide variety of audio types. Because we will mainly be focusing on Speech, most architectures today for that task typically use only 8 codebooks, each with 1024 codes

Each codebook in the RVQ is initialized with KMeans and then updated with Exponential Moving Averages. 

### Losses

As you can imagine there will be a variety of loss functions being used to train this!

1) **Time Domain Loss**: Simple MSE between the input and reconstruted audio
2) **Frequency Domain Loss**: MSE and L1 between the Mel spectrograms of the input and reconstructed audios. To additionally help with the uncertainty principle (time vs frequency resolutions), we compute this loss on a variety of spectrograms with different window sizes
3) **GAN Loss**: A discriminator looks at the real and imaginary components of spectrograms (with different window sizes) for real and generated audio. The loss function used is the Hinge loss. 
4) **Commitment Loss**: We want to ensure that the output of the encoder is close to the codes in the codebook. 

### Dynamic Loss Balancing

As you can imagine, making sure the effective gradient contribution from each of these losses can be hard to balance. So we use the proposed method from the EnCodec paper for dynamic loss balancing! 

The main idea is that we rescale the gradients from each of the losses to match the proportion of contributions we want. 

## Lets Train a Model!

