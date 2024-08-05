### Train with a Huggingface Backbone ###
CUDA_VISIBLE_DEVICES=0 python finetune_wav2vec2.py \
        --experiment_name "finetune_huggingface_backbone" \
        --working_directory "work_dir" \
        --path_to_dataset_root "/mnt/datadrive/data/LibriSpeech/" \
        --num_train_epochs 50 \
        --save_steps 250 \
        --eval_steps 250 \
        --logging_steps 5 \
        --warmup_steps 1000 \
        --train_splits train-clean-100 \
        --test_splits dev-clean \
        --per_device_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-4 \
        --weight_decay 5e-3 \
        --save_total_limit 2 \
        --huggingface_model_name "facebook/wav2vec2-base" \
        --pretrained_backbone "pretrained_huggingface" \
        --freeze_feature_extractor \
        --group_by_length

### Train with our own Pretrained Backbone (with model.safetensor weights) ###
CUDA_VISIBLE_DEVICES=0 python finetune_wav2vec2.py \
        --experiment_name "finetune_my_backbone" \
        --working_directory "work_dir" \
        --path_to_dataset_root "/mnt/datadrive/data/LibriSpeech/" \
        --num_train_epochs 50 \
        --save_steps 250 \
        --eval_steps 250 \
        --logging_steps 5 \
        --warmup_steps 1000 \
        --train_splits train-clean-100 \
        --test_splits dev-clean \
        --per_device_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-4 \
        --weight_decay 5e-3 \
        --save_total_limit 2 \
        --pretrained_backbone "pretrained" \
        --path_to_pretrained_backbone "work_dir/Pretraing_Wav2Vec2Base/checkpoint_100000/model.safetensors" \
        --freeze_feature_extractor \
        --group_by_length

### Train from Scratch ###
CUDA_VISIBLE_DEVICES=0 python finetune_wav2vec2.py \
        --experiment_name "finetune_from_scratch" \
        --working_directory "work_dir" \
        --path_to_dataset_root "/mnt/datadrive/data/LibriSpeech/" \
        --num_train_epochs 50 \
        --save_steps 250 \
        --eval_steps 250 \
        --logging_steps 5 \
        --warmup_steps 1000 \
        --train_splits train-clean-100 \
        --test_splits dev-clean \
        --per_device_batch_size 32 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-4 \
        --weight_decay 5e-3 \
        --save_total_limit 2 \
        --pretrained_backbone "random" \
        --group_by_length