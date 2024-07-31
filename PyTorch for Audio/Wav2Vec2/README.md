# Wav2Vec2: Self-Supervised Quantized Audio Pre-Training

Wav2Vec was an extremely important speech pretraining architecture introduced in 2019. At a high level, how this architecture worked was to use some convolutional layers to first encode an audio waveform into a  sequence of feature vectors $z_{t}$. The transformer is then trained via contrastive learning, to distinguish a feature vector k steps in the future $z_{i+k}$ (k is set to 12 in the implementation) from 10 other uniformly selected distractor feature vectors from the same audio sequence (as stated in the paper, sampling from different sequences seems to hurt performance). 

Wav2Vec 2.0 basically takes this idea from before but makes a few importance changes. 
- **Quantization**: The original Wav2Vec performed a contrastive loss for a future timestep output of the convolutional encoder against 10 distractor timesteps from the same sequence that were also output from the same encoder. Wav2Vec 2.0 explores using discrete speech units which are quantized encodings of the direct continuous outputs of the convolutional encoder. The output of the convolutions are still $z_t$ and each timestep in $z$ is quantized via Gumbel-Softmax to create $q_t$. We then randomly mask $z_t$, pass it to our transformer encoder, to create our sequence of context representations $c_t$. For the tokens that were masked before the transformer, we will use their corresponding $c_t$ to compute a contrastive loss between the corresponding $q_t$ and 100 other quantized distractors from the same sequence (caveat, in this implementation and in the paper, when we sample 100 quantized distractors, these 100 will come only from the quantized latents of the masked tokens). 
  - For more exploration into Quantization, I would recommend taking a look at my [implementation of VQ-VAE](https://github.com/priyammaz/PyTorch-Adventures/blob/main/PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Vector_Quantized_Variational_AutoEncoders.ipynb). Something you will notice is, quantization has a problem of not being differentiable. To quantize a vector, we take our continuous vector, compute the distance between it and a finite set of codevectors, and then use the closest one. Unfortunately, the *min* operation in finding the closest vector is not differentiable, so the VQ-VAE paper uses stop gradients and manually passes the model gradient around the non-differentiable operations. Wav2Vec 2.0 on the other hand uses somethign known as [Gumbel-Softmax](https://arxiv.org/pdf/1611.01144) which is a differentiable approximation of a discrete distribution!
 
- **Span Masking**: I mentioned that we will randomly mask $z_t$ before passing it to our transformer, but we will actually be performing something known as span masking. Basically, we will randomly sample 6.5% of the tokens to mask, and then mask a total of 10 consecutive tokens from there. This gives roughly a 49% masking ratio of the entire audio sequence!

![architecture](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/src/visuals/wav2vec2_architecture.png)


## Shoutouts

- [🤗 Huggingface Transformers](https://huggingface.co/) The code was largely based off of their ```modeling_wav2vec2.py```, this was basically a dumbed down and simplified version of the work they did!
- [patrickvonplaten](https://github.com/patrickvonplaten) The training script was largely inspired by the ```run_wav2vec2_pretraining_no_trainer.py```
- [Awni Hannun](https://awnihannun.com/) for their really great explanation of how [CTC Loss](https://distill.pub/2017/ctc/) works! Although I was just using the PyTorch CTC Loss, it was helpful to understand anyway!

## Preparing Data ###

### Download Data
First step is we need to download our LibriSpeech data (which is about 960 Hours of english audio and corresponding transcripts). We can easily do this from [OpenSLR](https://www.openslr.org/12). I included the following script ```prepare_data.sh``` in the ```data``` folder that you can run via ```bash data/prepare_data.sh``` which will download all the data to the data folder and create a LibriSpeech folder that will contain all the subfolders with audios and transcripts. 

```bash
path_to_store="data/"

train_clean_100="train-clean-100"
train_clean_360="train-clean-360"
train_clean_500="train-clean-500"
dev_clean="dev-clean"
test_clean="test-clean"

wget https://www.openslr.org/resources/12/$train_clean_100.tar.gz -P $path_to_store
wget https://www.openslr.org/resources/12/$train_clean_360.tar.gz -P $path_to_store
wget https://www.openslr.org/resources/12/$train_clean_500.tar.gz -P $path_to_store
wget https://www.openslr.org/resources/12/$dev_clean.tar.gz -P $path_to_store
wget https://www.openslr.org/resources/12/$test_clean.tar.gz -P $path_to_store

tar -xvzf $path_to_store$train_clean_100.tar.gz -C $path_to_store
tar -xvzf $path_to_store$train_clean_360.tar.gz -C $path_to_store
tar -xvzf $path_to_store$train_clean_500.tar.gz -C $path_to_store
tar -xvzf $path_to_store$dev_clean.tar.gz -C $path_to_store
tar -xvzf $path_to_store$test_clean.tar.gz -C $path_to_store
```

If you want to change the download directory, just update the path in the ```path_to_store```! Wherever you pick your path to be, it will create a folder inside it called ```LibriSpeech```. Once ready, just run 

```bash
bash data/prepare_data.sh
```

### Compute Durations
To be able to pre-filter our data by duration (i.e. we only have enough GPU memory to train upto 15 second audio clips) it would be convenient to just have the durations upfront. So I created a quick script that will create a bunch of CSV files in the audio directories that store the name of the audio file and the duration in seconds! We can do this with the quick helper script ```compute_durations.py```.

```bash
python compute_durations.py --path_to_librispeech_data "data/LibriSpeech"
```

## Pre-Training Wav2Vec2 Model ###

To train our model we can use the following command! All you need to do is provide the path to the working directory (where you want to save your checkpoints) and your path to the data root for librispeech (the folder which includes ```train-clean-100```, ```train-clean-360```, ```train-other-500```, ```dev-clean```, ```test-clean```). A more complete command is found in ```pretrain.sh``` and includes all the arguments that we can pass in! If you need any information on the arguments you can just run ```python pretrain_wav2vec2.py --help```.

Also, if you are running on an HPC, there can be a wallclock, so you need to resume training and we can do that too. The script will automatically create checkpoints, and you can indicate the interval of those checkpoints with the ```--checkpoint_interval``` argument. You can then resume from checkpoint by passing in ```--resume_from_checkpoint "checkpoint_{step}"``` where **step** is what training step the model was on when checkpointed. 

```bash
accelerate launch train.py --experiment_name "Wav2Vec2_Pretraining" \
                           --working_directory "<PATH_TO_WORK_DIR>" \
                           --path_to_data_root "<PATH_TO_DATA_ROOT>" \
                           --train_splits train-clean-100 train-clean-360 train-other-500 \
                           --test_splits dev-clean test-clean \
                           --minimum_audio_duration 2.0 \
                           --maximum_audio_duration 15.0 \
                           --masking_probability 0.065 \
                           --masking_span_length 10 \
                           --minimum_spans 2 \
                           --num_negatives 100 \
                           --per_gpu_batch_size 64 \
                           --gradient_accumulation_steps 4 \
                           --num_training_steps 200000 \
                           --num_warmup_steps 32000 \
                           --lr_scheduler_type polynomial \
                           --logging_steps 1 \
                           --evaluation_interval 500 \
                           --checkpoint_interval 1000 \
                           --learning_rate 0.001 \
                           --num_workers 8 \
                           --log_wandb \
                           --seed 0 \
                           --resume_from_checkpoint "checkpoint_{step}"
```
