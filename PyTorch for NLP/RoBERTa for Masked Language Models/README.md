## Robustly Optimized BERT (RoBERTa)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MChQ84-1VKBbjNCmzPQL02hxl-gckEYh?usp=sharing)

![Image](https://github.com/priyammaz/HAL-DL-From-Scratch/blob/main/src/visuals/masked_language_modeling_vis.png?raw=true)

RoBERTa is an updated variant of the original BERT model! The main changes made are (1) Removal of the Next Sentence Prediction Task, (2) Dynamic Masking, (3) Training on Much Larger datasets with larger batches. 

Although we can't replicate the entire results for RoBERTa, we can implement it and train it on our (small) Harry Potter dataset to see what it does! 

Some of the functions we will write are a bit unecessary, for example, if you need to do masking, just take a look at the Huggingface [DataCollatorForLanguageModeling](https://huggingface.co/docs/transformers/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling), but it'll be good to see and write the masking logic yourself atleast once!

