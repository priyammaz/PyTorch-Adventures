# Huggingface Accelerate 

To train larger models efficiently, we need to be able to utilize multiple GPUs at a time. I used to use regular PyTorch DDP, but these days I basically just use Huggingface Accelerate just due to the convenience. We will be looking at an example of training ResNet50 on the ImageNet dataset, following as close as I can the [training recipie](https://github.com/pytorch/vision/tree/main/references/classification#resnet) provided by PyTorch. What we will learn here is:

- Controlling our scripts through the command line using [Argparse](https://docs.python.org/3/library/argparse.html)
- Using Huggingface Accelerate to train on multiple GPUs
- Incorporating a learning rate scheduler
- Using gradient accumulation to simulate larger batch sizes
- Gathering metrics across GPUs
- Checkpointing/Resuming model training
- Training logs stored in [Weights and Biases](https://wandb.ai/site)
  
To run this script, first downlod the ImageNet dataset. To do this you can just follow the [instructions from Google](https://cloud.google.com/tpu/docs/imagenet-setup) to set up the training and validation data! Next create a folder for where you want to store your checkpoints. Also verify that you have Accelerate installed and go through the prompts when you enter ```accelerate config``` to let it know information about the system and what resources you want to use. Pass the path to your imagenet data and working directory to the command below and you are good to go!

```
accelerate launch train.py --experiment_name="ResNet50" \
                           --path_to_data="<PATH_TO_IMAGENET_ROOT>" \
                           --working_directory="<PATH_TO_WORK_DIR>" \
                           --num_classes=1000 \
                           --epochs=90 \
                           --save_checkpoint_interval=2 \
                           --batch_size=512 \
                           --gradient_accumulation_steps=1 \
                           --learning_rate=0.1 \
                           --weight_decay=1e-4 \
                           --momentum=0.9 \
                           --step_lr_decay=0.1 \
                           --lr_step_size=30 \
                           --warmup_epochs=5 \
                           --lr_warmup_start_factor=0.1 \
                           --max_grad_norm=1.0 \
                           --img_size=224 \
                           --num_workers=18
```

Here are the [results](https://api.wandb.ai/links/exploratorydataadventure/q8on5kzo) from the training run stored in Weights and Biases! This model reached about a 75% Top-1 Accuracy on Imagenet, which is just shy of the 76% that PyTorch reports on their website. Theres probably something small missing thats causing that discrepancy, but this is enough to learn from!
