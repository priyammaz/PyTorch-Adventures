# Sliding Window Attention

Although the Attention mechanism has found its way into every modality, it has the clear problem of a quadratic complexity with respect to sequence length. Remember, attention computes how every pair of tokens are related to each other, so if you have a sequence length of 10, there will be 100 computations. Buf if we have long sequence lengths like 10000, then that will attention will do 10 Million computations! This cost is too high for both training and inference, so we need to find a way to reduce it. One such attempt at this has been Sparse Sliding Window Attention (AKA Local Attention).

![image](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/src/visuals/sliding_window_attention.png)

[Image Source](https://paperswithcode.com/method/sliding-window-attention)

### Shououts
- [local-attention](https://github.com/lucidrains/local-attention) from [lucidrains](https://github.com/lucidrains) is what this code was mostly based off of! There are some differences where I tried to do the same operations less efficiently but hopefully more readable/explainable as a learning tool, and I removed all the extra features that weren't necessary for understanding!
- [LongFormer](https://arxiv.org/pdf/2004.05150) will give you a ton of information about this mechianism, they also mention they built a very efficient [custom cuda kernel](https://github.com/allenai/longformer)
- [Video](https://www.youtube.com/watch?v=_8KNb5iqblE) by [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) was super helpful for me to wrap my head around all the ideas presented

### Why Use Sliding Window?

Sliding Window Attention should remind you of convolutions actually. We pick some window size and slide it across our sequence, and only within those windows do we compute attention. If our window size is $w$ and our sequence length is $n$ our best-case-scenario complexity of $w*n$ versus the $n^2$ we would otherwise get with full attention. Couple things though:

- This mechanism is really only helpful if our $w$ is meaningfully smaller than $n$
- We assume that the most important information about a secific timestep is within the vicinity of it, as we have lost full global attention. (Longformer adds in global attention as well but I stick to just sliding window here)


### PyTorch and OverComputing Attention
The best-case-scenario of $w*n$ is actually pretty hard to obtain in PyTorch. Think about it this way, lets pretend we computed the full attention and masked the top and bottom triangles, this would also give us the same *sliding window* effect. But this defeats the purpose, as we dont want to compute any combination of tokens that are not in that central diagonal band to begin with. The blocked algorithm that I will be using (based on the implementation from [local-attention](https://github.com/lucidrains/local-attention)) breaks our data into chunks and then we have the option to attend to however many previous chunks and future chunks we want. Lets pretend we have a sequence of length 8, each chunk is of size 2, and we want to look forward and backward one chunk. This is what it will look like:

![image](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/src/visuals/block_window_attention.png)

The sequence is broken into chunks of:
- Chunk 1: [1,2]
- Chunk 2: [3,4]
- Chunk 3: [5,6]
- Chunk 4: [7,8]
  
We can then compute attention between the current chunk and then one chunk before/after it. This would be:

- Chunk 1 attends to Chunk 2 (no chunk before it), so tokens [1,2] attend to [1,2,3,4]
- Chunk 2 attends to Chunk 1 and 3, so tokens [3,4] attend to [1,2,3,4,5,6]
- Chunk 3 attends to Chunk 2 and 4, so tokens [5,6] attend to [3,4,5,6,7,8]
- Chunk 4 attends to Chunk 3 (no chunk after it) so tokens [7,8] attend to [5,6,7,8]

The problem is, we want to look forward and backward one chunk per token, not per chunk. What this means is, if our chunk is of size two, for every token, we want to look forward 2 tokens and backwards 2 tokens. What we have right now is for all tokens in a chunk, we look forward 2 and backwards 2. More specifically, lets look at token 1. We cannot go back, so we only want to look forward 2 tokens, to tokens [2,3]. But in our block windowed attention setup, we also computed attention between token 1 and token 4, which is 3 tokens away. Therefore, we have overcomputed. Similarly, for token 4, we only want to look forward 2 tokens which are [5,6] and backward to tokens [2,3], therefore attention between token 4 and token 1 was overcomputed. So after doing our blocked attention computation, we need to go back and mask out the attention values again that were extra. 

There may be more methods to implement this that I dont know (if you do please let me know!), but PyTorch is actually not very good on its own to do these sparse matrix multiplications. To have the most optimized windowed attention that doesn't have any of these extra computations would require some custom cuda kernels to pull it off! Luckily, [FlashAttention](https://github.com/Dao-AILab/flash-attention) already supports sliding window attention. 

### FLOPS Comparison ###

A good test for our mechanism is comparing our implemented Sliding Window Attention and a regular Self Attention in the total number of floating point operations (FLOPS). We would expect a quadratic growth in the number of operations in regular attention, but a linear growth in windowed attention. To create a figure comparing flops at different sequence lengths just run the following script!

```
python profile_attention.py --window_size 512 \
                            --look_backward 1 \
                            --look_forward 1 \
                            --multiplier 40 
```

For every sequence length from the set window size (512) to every multiple upto 40 (512\*1, 512
\*2, ... 512\*40), we will generate a random tensor and pass it through both a regular attention mechanism and our own windowed attention mechanism. We will then use [fvcore](https://github.com/facebookresearch/fvcore/tree/main) to compute the number of flops. This will generate the following figure: 

![image](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/PyTorch%20for%20Transformers/Attention%20Mechanisms/Sliding%20Window%20Attention/flops_comparison.png)

As we can clearly see, Self Attention has a quadratic growth in the number of computations, whereas Windowed attention grows linearly!


## Train RoBERTa with Windowed Attention

We can also test our new attention mechanism by training it on something. I decided to use my [RoBERTa implementation](https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20NLP/RoBERTa%20for%20Masked%20Language%20Models) and just drop in the new Windowed Attention! 

### Prepare Data

First step is to prepare some longer context data. I initially trained RoBERTa with a context size of 512, but now we will boost it 2048. Like before we will be training on Wikipedia + BookCorpus. To prepare the data you can run the following:

```bash
python prepare_data.py \
    --test_split_pct 0.005 \
    --context_length 2048 \
    --path_to_data_store <PATH_TO_DATA_STORE> \
    --huggingface_cache_dir <PATH_TO_HUGGINGFACE_CACHE> \
    --dataset_split_seed 42 \
    --num_workers 24 \
    --hf_model_name "FacebookAI/roberta-base"
```
All you need to provide is, where do you want to save the data and the huggingface cache if its different from the default. 

### Train Model 

We can now train our model on our prepped data!

```bash
accelerate launch pretrain_roberta.py \
    --experiment_name "RoBERTa_Pretraining_localattention" \
    --working_directory <PATH_TO_WORK_DIR> \
    --hf_model_name "FacebookAI/roberta-base" \
    --path_to_prepped_data <PATH_TO_PREPPED_DATA> \
    --context_length 2048 \
    --attention_type "windowed" \
    --window_size 128 \
    --look_backward 1 \
    --look_forward 1 \
    --max_grad_norm 1.0 \
    --layer_norm_eps 1e-5 \
    --per_gpu_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --num_training_steps 100000 \
    --num_warmup_steps 10000 \
    --lr_scheduler linear \
    --logging_steps 1 \
    --evaluation_interval 2500 \
    --checkpoint_interval 2500 \
    --learning_rate 6e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --seed 42 \
    --log_wandb
```







