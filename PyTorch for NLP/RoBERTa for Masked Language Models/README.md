# Robustly Optimized BERT (RoBERTa)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MChQ84-1VKBbjNCmzPQL02hxl-gckEYh?usp=sharing)


RoBERTa is an updated variant of the original BERT model! The main changes made are:

- Removal of the Next Sentence Prediction Task
- Dynamic Masking 
- Training on Much Larger datasets with larger batches. 

Today we will be exploring a dummy example of RoBERTa first to make sure we can replicate some of its functionality on a small dataset, and then attempt to reproduce the full RoBERTa-Base model as much as possible!

![Image](https://github.com/priyammaz/HAL-DL-From-Scratch/blob/main/src/visuals/masked_language_modeling_vis.png?raw=true)

## Introduction 

First things first, go through the [RoBERTa Jupyter Notebook](https://github.com/priyammaz/PyTorch-Adventures/blob/main/PyTorch%20for%20NLP/RoBERTa%20for%20Masked%20Language%20Models/RoBERTa%20Masked%20Language%20Modeling.ipynb) that will give you all the background you need to understand the implementation of the architecture. In this example we will try to train a masked language model on the Harry Potter dataset to see how well we can play fill-in-the-blank!

## Reproduce RoBERTa

Now that you have an idea of how the achitecture works we will attempt to reproduce RoBERTa as close as we can! This will involve a few steps!

### Prepare Dataset
RoBERTa was trained on the [Wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia) and [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) datasets. We will start by pretokenizing and grouping this dataset, and then saving it so we can load it during training! To run the prep script you can run the following!

```bash
python prepare_data.py \
    --test_split_pct 0.005 \
    --context_length 512 \
    --path_to_data_store "<PATH_TO_DATA_STORE>" \
    --huggingface_cache_dir "<HUGGINGFACE_CACHE_DIR>" \
    --dataset_split_seed 42 \
    --num_workers 16 \
    --hf_model_name "FacebookAI/roberta-base"
```

All you need to provide is the `path_to_data_store` where you want the data to save, and also the `huggingface_cache_dir` if you want to use a different cache directory that the default one that huggingface provides! The `seed` will also make sure you have the same split that I do!

### Pre-Train RoBERTa with Masked Language Modeling 
```bash
accelerate launch train.py \
    --experiment_name "RoBERTa_Pretraining" \
    --working_directory "<PATH_TO_WORK_DIR>" \
    --hf_model_name "FacebookAI/roberta-base" \
    --path_to_prepped_data "<PATH_TO_DATA_STORE>" \
    --context_length 512 \
    --masking_probability 0.15 \
    --num_workers 24 \
    --hidden_dropout_p 0.1 \
    --attention_dropout_p 0.1 \
    --num_transformer_blocks 12 \
    --num_attention_heads 12 \
    --embedding_dimension 768 \
    --mlp_ratio 4 \
    --layer_norm_eps 1e-5 \
    --initializer_range 0.02 \
    --per_gpu_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_training_steps 300000 \
    --num_warmup_steps 20000 \
    --lr_scheduler linear \
    --logging_steps 1 \
    --evaluation_interval 2500 \
    --checkpoint_interval 2500 \
    --learning_rate 6e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --seed 42
```

For training, I tried to use the same parameters that the [RoBERTa Paper](https://arxiv.org/pdf/1907.11692) says they used on the last page of their paper! The main things you need to provide are the `working_directory` where the model will create a new folder after the `experiment_name` provided and where it will save all the checkpoints as the model trains! You also need to provide the `path_to_prepped_data` that should be identical to where you saved the dataset in the previous section of preparing the dataset. 

### Fine-Tuning for Question Answering

Now that we have a model that can play fill-in-the-blank, its time to fine-tune it to do something we care about. Although we could have done something a bit easier like Sequence classification, Extractive Question Answering sounded more fun! So we will be using the Huggingface Trainer and borrow some of the code from the [QA Tutorial](https://huggingface.co/docs/transformers/en/tasks/question_answering) provided to do this!

```bash
python finetune_roberta_qa.py \
        --experiment_name "finetune_qa_hf_roberta_backbone" \
        --working_directory "<PATH_TO_WORK_DIR>" \
        --path_to_cache_dir "<HUGGINGFACE_CACHE_DIR>" \
        --num_train_epochs 3 \
        --save_steps 250 \
        --eval_steps 250 \
        --logging_steps 5 \
        --warmup_steps 100 \
        --per_device_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --huggingface_model_name "FacebookAI/roberta-base" \
        --pretrained_backbone "pretrained_huggingface"
```

