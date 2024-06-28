#!/bin/bash

accelerate launch train.py --experiment_name="resnet_test" \
                           --path_to_data="/mnt/datadrive/data/ImageNet" \
                           --working_directory="work_dir" \
                           --num_classes=1000 \
                           --epochs=90 \
                           --batch_size=512 \
                           --gradient_accumulation_steps=1 \
                           --learning_rate=5e-4 \
                           --weight_decay=1e-4 \
                           --img_size=224 \
                           --num_workers=16
