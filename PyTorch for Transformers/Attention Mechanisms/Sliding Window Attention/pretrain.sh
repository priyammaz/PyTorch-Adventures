accelerate launch pretrain_roberta.py \
    --experiment_name "RoBERTa_Pretraining_localattention" \
    --working_directory "work_dir" \
    --hf_model_name "FacebookAI/roberta-base" \
    --path_to_prepped_data "/mnt/datadrive/data/prepped_data/roberta_data_long" \
    --context_length 2048 \
    --masking_probability 0.15 \
    --num_workers 24 \
    --hidden_dropout_p 0.1 \
    --attention_dropout_p 0.1 \
    --num_transformer_blocks 12 \
    --num_attention_heads 12 \
    --embedding_dimension 768 \
    --mlp_ratio 4 \
    --max_grad_norm 1.0 \
    --layer_norm_eps 1e-5 \
    --initializer_range 0.02 \
    --per_gpu_batch_size 40 \
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



