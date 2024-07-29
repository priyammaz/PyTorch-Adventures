accelerate launch train.py --experiment_name "CELEBA_DIFFUSION" \
                           --path_to_data "<PATH_TO_YOUR_DATA>" \
                           --working_directory "<PATH_TO_YOUR_WORKING_DIRECTORY>" \
                           --generated_directory "<PATH_TO_YOUR_GENERATION_DIRECTORY>" \
                           --num_diffusion_timesteps 1000 \
                           --plot_freq_interval 100 \
                           --num_generations 3 \
                           --num_training_steps 250000 \
                           --warmup_steps 2000 \
                           --evaluation_interval 500 \
                           --batch_size 64 \
                           --gradient_accumulation_steps 2 \
                           --learning_rate 1e-4 \
                           --loss_fn mse \
                           --img_size 128 \
                           --starting_channels 128 \
                           --num_workers 24 \
                           --num_keep_checkpoints 3


