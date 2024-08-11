#!/bin/bash

accelerate launch train.py --experiment_name="ResNet50_Reproduce" \
                           --path_to_data="<PATH_TO_IMAGENET_DATA>" \
                           --working_directory="<PATH_TO_WORK_DIR>" \
                           --epochs=90 \
                           --save_checkpoint_interval=10 \
                           --batch_size=128 \
                           --gradient_accumulation_steps=1 \
                           --learning_rate=0.1 \
                           --weight_decay=1e-4 \
                           --momentum=0.9 \
                           --step_lr_decay=0.1 \
                           --lr_step_size=30 \
                           --lr_warmup_start_factor=0.1 \
                           --max_grad_norm=1.0 \
                           --img_size=224 \
                           --num_workers=24
