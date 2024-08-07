python prepare_data.py \
    --test_split_pct 0.005 \
    --context_length 512 \
    --path_to_data_store "/mnt/datadrive/data/prepped_data/roberta_data" \
    --huggingface_cache_dir "/mnt/datadrive/data/huggingface_cache" \
    --dataset_split_seed 42 \
    --num_workers 24 \
    --hf_model_name "FacebookAI/roberta-base"