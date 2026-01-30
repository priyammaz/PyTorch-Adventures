python stage1_vae_trainer.py \
  --experiment_name "VAETrainer" \
  --wandb_run_name "vae_cc" \
  --working_directory "work_dir/vae_cc" \
  --training_config "configs/stage1_vae_train.yaml" \
  --model_config "configs/ldm.yaml" \
  --dataset conceptual_captions \
  --path_to_dataset "/mnt/datadrive/data/ConceptualCaptions/hf_train" \
  --path_to_save_gens "src/conceptual_captions"
