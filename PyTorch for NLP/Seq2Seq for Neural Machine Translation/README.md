# Attention is All You Need

It is finally time to implement the paper that started this entire explosion of transformer based models!

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/attention_is_all_you_need.png?raw=true" alt="drawing" width="500"/>

This transformer architecture is typically called Seq2Seq or Encoder-Decoder. In our case we will be training it on language translation from English to French. 

The main idea behind this architecture is we first use a Transformer encoder to encode our English tokens (and grab meaningful representations from them). This encoder is almost identical to the [RoBERTa](https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20NLP/RoBERTa%20for%20Masked%20Language%20Models) implementation I did. On the other hand to generate the french language, we have to use a Decoder (with a causal mask) that looks a lot like my implementation of [GPT](https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20NLP/GPT%20for%20Causal%20Language%20Models)

We are missing one important piece though. We have a mechanism to encode english and another mechanism to generate french, but what we dont have is a way to mesh how the english is related to the french! This is Cross Attention!

### Cross Attention
Remember, Attention is a computation of Queries, Keys and Values. The Queries is what we want, the the Keys and Values is what we are comparing to. Normally, the Q,K,V are all computed on the same tensor as we are doing self-attention. Now, our Queries will be French (as that is what we are trying to generate) and the Keys and Values will be the encoded english from the encoder so we learn how the english is related to the french to inform the generation correctly!

## Training Your Own Translation Model!

### Step 1: Download you Data
We will be using the English-French dataset from the [WMT Translation Task](https://www.statmt.org/wmt14/translation-task.html). I downloaded the data and created the following file structure:

```
    └── english2french/
        ├── common_crawl/
        │   ├── commoncrawl.fr-en.en
        │   └── commoncrawl.fr-en.fr
        ├── europarl/
        │   ├── europarl-v7.fr-en.en
        │   └── europarl-v7.fr-en.fr
        ├── giga_french/
        │   ├── giga-fren.release2.fixed.en
        │   └── giga-fren.release2.fixed.fr
        └── un_corpus/
            ├── undoc.2000.fr-en.en
            └── undoc.2000.fr-en.ft
```

When you download, it'll give more languages, we are just going to be focusing on English/French though. 

### Step 2: Traing A WordPiece Tokenizer 

Our next step is to train a tokenizer for the French language (for the english input we can just use regular BERT tokenizer). We can run our ```tokenizer.py``` just point it to the ```english2french``` folder where we downloaded all the data previously

```bash
python tokenizer.py --path_to_data_root "<PATH_TO_DATA_ROOT>"
```

This will create a tokenizer json stored in ```trained_tokenizer/french_wp.json```. 

### Process and Tokenize Data 

You can then run the following to prepare and tokenize all the data!

```bash
python prepare_data.py --path_to_data_root "<PATH_TO_DATA_ROOT>" \
                       --huggingface_cache_dir "<PATH_TO_HUGGINGFACE_CACHE_DIR>" \
                       --test_split_pct 0.005 \
                       --max_length 512 \
                       --min_length 5 \
                       --num_workers 24
```

This will create in your ```data_root``` two folders: ```raw_english2french_corpus``` will contain the raw english/french text pairs and ```tokenized_english2french_corpus``` will be its processed and tokenized form for training. 

### Train Model 
Now that all our data is ready to go, we can train our model! We will be using Huggingface Accelerate for multi-GPU training. At the start of the training script, there are some arguments you can fill in to scale the model however you like! The main thing you need to change is the ```path_to_data```, where you just want to provide the path to the ```tokenized_english2french_corpus``` created earlier. 

```
accelerate launch train.py
```
The results of this training script for 150K Steps can be found [here](https://api.wandb.ai/links/exploratorydataadventure/nv5i9c3z)

### Compute BLEU Score
A common metric to evaluate translation models are BLEU. We can compute that as well with the ```compute_bleu.py``` where you need to provide the path_to_raw_data which should be the path to the ```raw_english2french_corpus``` we created earlier and a path to some trained weights that were created during training (these should be called ```model.safetensors```)

Our model achieves a BLEU Score of 30.8! This is pretty good for a base Transformer, although the original implementation from the paper reported 38.1. Not a bad start though for a totally functional Translation model!