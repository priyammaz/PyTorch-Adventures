### Finetuen Huggingface RoBERTa Backbone ###
python finetune_roberta_qa.py \
        --experiment_name "finetune_qa_hf_roberta_backbone" \
        --working_directory "work_dir" \
        --path_to_cache_dir "/mnt/datadrive/data/huggingface_cache" \
        --num_train_epochs 3 \
        --save_steps 250 \
        --eval_steps 250 \
        --logging_steps 5 \
        --warmup_steps 100 \
        --per_device_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --save_total_limit 5 \
        --huggingface_model_name "FacebookAI/roberta-base" \
        --pretrained_backbone "pretrained_huggingface"

### Finetune My RoBERTa Backbone ###
python finetune_roberta_qa.py \
        --experiment_name "finetune_qa_my_roberta_backbone" \
        --working_directory "work_dir" \
        --path_to_cache_dir "/mnt/datadrive/data/huggingface_cache" \
        --num_train_epochs 3 \
        --save_steps 250 \
        --eval_steps 250 \
        --logging_steps 5 \
        --warmup_steps 100 \
        --per_device_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --save_total_limit 5 \
        --huggingface_model_name "FacebookAI/roberta-base" \
        --pretrained_backbone "pretrained" \
        --path_to_pretrained_backbone "work_dir/RoBERTa_Pretraining/checkpoint_75000/model.safetensors"

### Finetune Random Initialization RoBERTa Backbone ###
python finetune_roberta_qa.py \
        --experiment_name "finetune_qa_randomized_roberta_backbone" \
        --working_directory "work_dir" \
        --path_to_cache_dir "/mnt/datadrive/data/huggingface_cache" \
        --num_train_epochs 3 \
        --save_steps 250 \
        --eval_steps 250 \
        --logging_steps 5 \
        --warmup_steps 100 \
        --per_device_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --save_total_limit 5 \
        --huggingface_model_name "FacebookAI/roberta-base" \
        --pretrained_backbone "random"