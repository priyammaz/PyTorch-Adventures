python evaluate_squad_score.py \
    --path_to_model_weights "work_dir/finetune_qa_hf_roberta_backbone/model.safetensors" \
    --path_to_store "work_dir/finetune_qa_hf_roberta_backbone/squad_score.json" \
    --cache_dir "/mnt/datadrive/data/huggingface_cache" \
    --huggingface_model