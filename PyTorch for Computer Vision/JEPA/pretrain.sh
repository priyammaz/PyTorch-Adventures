accelerate launch pretrain.py \
    --experiment_name       "ijepa_vitb" \
    --wandb_run_name        "ijepa_vitb_pretraining" \
    --path_to_data          "/mnt/datadrive/data/ImageNet" \
    --working_directory     "work_dir" \
    --epochs                300 \
    --warmup_epochs         15 \
    --per_gpu_batch_size    512 \
    --num_workers           16 \
    --log_wandb
 