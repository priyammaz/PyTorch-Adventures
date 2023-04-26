## Large Model Considerations: Distributed Training &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cyxgaWonj-FrfEbZvTwAVepkZhaF_sda?usp=sharing)

Models these days have gotten unreasonably big! If we look at the current set of state of the art models across the board:
- GPT 3: 175 Billion Parameters
- ViT-22B: 22 Billion Parameters
- OpenAI Whisper: 1.6 Billion Parameters

How do we train something like this because it couldn't fit on a GPU, much less an entire computer?

**Distributed Training**

![modelparallel](https://fairscale.readthedocs.io/en/latest/_images/pipe.png)

[credit](https://fairscale.readthedocs.io/en/latest/deep_dive/pipeline_parallelism.html)

Distributed training is a technique to be able to share the computational cost of training a model 
across multiple GPUs or even multiple Compute Nodes! We will be exploring 3 ways you can deal with larger
models and when you can use each one:

- Gradient Accumulation
- Distributed Data Parallelism
- Model Parallelism

We will be applying the above principles on our simple [AlexNet](../PyTorch%20for%20Computer%20Vision/Intro%20to%20Vision) model 
from before but these principles are identical regardless!