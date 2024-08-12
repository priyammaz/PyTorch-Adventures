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
accelerate launch pretrain_roberta.py \
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
    --per_gpu_batch_size 64 \
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

For training, I tried to use the same parameters that the [RoBERTa Paper](https://arxiv.org/pdf/1907.11692) says they used on the last page of their paper! The main things you need to provide are the `working_directory` where the model will create a new folder after the `experiment_name` provided and where it will save all the checkpoints as the model trains! You also need to provide the `path_to_prepped_data` that should be identical to where you saved the dataset in the previous section of preparing the dataset. I don't have a enough GPU resources to fully replicate this, and due to a smaller batch size, I had some instances of unstable training so I killed the training and restarted from the last good checkpoint again. Training for about 48 hours yielded a model trained for 75K steps, and those results can be seen [here](https://api.wandb.ai/links/exploratorydataadventure/ko6u7rwf). 

### Fine-Tuning for Question Answering

Now that we have a model that can play fill-in-the-blank, its time to fine-tune it to do something we care about. Although we could have done something a bit easier like Sequence classification, Extractive Question Answering sounded more fun! So we will be using the Huggingface Trainer and borrow some of the code from the [QA Tutorial](https://huggingface.co/docs/transformers/en/tasks/question_answering) provided to do this! The finetuning will happen on the [SQUAD Dataset](https://arxiv.org/abs/1606.05250) which is a reading comprehension dataset where you have some text and a question about the text. The answer to the question is typically a shorter segment from the input text, so that is what our model will be finetuned to predict!

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
        --pretrained_backbone "pretrained_huggingface" \ # options are pretrained_huggingface, pretrained, random
        --path_to_pretrained_backbone "<PATH_TO_MODEL.SAFETENSORS"> # use if we are using our own backbone
```

The main thing that matters here is what you want to use as your `pretrained_backbone`. If you use *pretrained_huggingface* then it will load an already pretrained RoBERTa model to use. If you just provide *pretrained*, then it will load the weights from our own pretrained backbone, as long as the `path_to_pretrained_backbone` to the model.safetensors you want to use is also provided. We will be finetuning three variants here to compare: (1) Pretrained Huggingface RoBERTa backbone, (2) My own pretrained backbone that was trained for 75K steps and smaller batch size and (3) a randomly initialized backbone to see if our pretrained model learned anything valuable! Examples of this can be seen in `finetune_roberta_qa.sh` The training results can be seen [here](https://api.wandb.ai/links/exploratorydataadventure/u0wgkr3c)

The main results were:

| Variant | Loss |
| -------- | ------- |
| Facebook Pretrained | 0.81 |
| My Pretrained | 1.34 |
| Randomized | 4.31 |

### SQUAD Evaluation of our RoBERTaForQuestionAnswering

Now that we have finetuned our model for Question Answering, a typical evaluation metric is to compute our [SQUAD score](https://rajpurkar.github.io/SQuAD-explorer/)! This score has two values, **Exact Match** (the proportion of our predicted answers that match one of the valid answers) and the **F1** score that computes something similar to the amount of overlap between our predicted answer and true answers. 

For the evaluation we can use the following command:

```bash
python evaluate_squad_score.py \
    --path_to_model_weights "<PATH_TO_MODEL.SAFETENSORS>" \
    --path_to_store "<PATH_TO_JSON_STORE>" \
    --cache_dir "<CACHE_DIR>" \
    --huggingface_model # only need this if we finetuned a model with a huggingface backbone
```
We need to provide the `path_to_model_weights` that is one of the *model.safetensors* files saved during finetuning of the model. We also need to provide a path to a file in `path_to_store` that will be the path to a `.json` file where we will save the final results of our model. If you want to use a finetuned model that used a huggingface pretrained backbone, just provide the flag `--huggingface_model`. 

| Variant | Exact Match | F1 |
| -------- | ------- | ------- |
| Facebook Pretrained | 83.23 | 89.86 |
| My Pretrained | 65.91 | 75.72 |
| Randomized | 1.63 | 5.74 |

As we can clearly see, the Facebook pretrained model is much better than our own pretrained model, but we have clear evidence that our pretrained backbone learned something meaningful as the randomized backbone was basically always wrong!



### Wrap-Up

This is basically how the RoBERTa architecture works! You should have leaned all about masked language modeling and then QA Span classification from this! Honestly its not really the models that are hard to implement, it mainly all the data processing that takes some time to think through!

