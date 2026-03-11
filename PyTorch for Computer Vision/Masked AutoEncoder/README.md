# Masked AutoEncoders are Scalable Vision Learners

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/mae.png?raw=true" alt="drawing" width="700"/>


## Model Structure
Just as Masked Language Modeling underpins architectures like BERT, Masked Image Modeling serves as a powerful method for large-scale image pretraining. This implementation focuses on reproducing the Masked Autoencoder (MAE) on the ImageNet dataset.

A key difference between the MAE architecture and models like BERT lies in their Encoder/Decoder structure. In BERT, text sequences are randomly masked, and the input includes the masked tokens replaced with a specific mask token. In contrast, the MAE approach masks 75% of the image patches, and only the remaining 25% of visible patches are passed to the encoder. This design significantly reduces the computational load on the encoder.The decoder then processes the full sequence of image patches, consisting of both the encoded visible patches and the masked tokens, to reconstruct the original image. The decoder is lightweight, utilizing a smaller embedding dimension and fewer transformer blocks compared to the encoder. This architectural design reduces the overall computation required for the reconstruction task while enabling efficient learning of high-quality visual representations.

The main benefit is, once pretrained, the encoder is just a normal Vision Transformer. We can pass in the full images (rather than mask) and finetune for downstream tasks like Classification and Segmentation, things we will implement here today!

## PreTraining MAE

The first stage is to pretrain our Masked AutoEncoder on ImageNet. I will be trying to follow the [Original MAE Repo](https://github.com/facebookresearch/mae) as closely as I can, but with Huggingface 🤗 [Accelerate](https://huggingface.co/docs/accelerate/en/index) as our distributed training setup. 


This can be done by the following:

```sh
accelerate launch pretrain_mae.py \
    --experiment_name "MAEPretraining" \
    --wandb_run_name "pretrain_mae" \
    --path_to_data "<PATH_TO_IMAGENET>" \
    --working_directory "work_dir" \
    --img_size 224 \
    --input_channels 3 \
    --epochs 800 \
    --warmup_epochs 40 \
    --save_checkpoint_interval 10 \
    --per_gpu_batch_size 2048 \
    --gradient_accumulation_steps 4 \
    --base_learning_rate 1.5e-4 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --img_size 224 \
    --encoder_embed_dim 768 \
    --encoder_depth 12 \
    --encoder_num_heads 12 \
    --encoder_mlp_ratio 4 \
    --decoder_embed_dim 512 \
    --decoder_depth 8 \
    --decoder_num_heads 16 \
    --decoder_mlp_ratio 4 \
    --num_workers 32 \
    --custom_weight_init \
    --log_wandb
```

### PreTraining Results

The results for the pretraining can be seen [here](https://api.wandb.ai/links/exploratorydataadventure/9l3zwqz5)    


### Reconstructions

Now that we have pretrained our MAE, what do the reconstructions look like?

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/PyTorch%20for%20Computer%20Vision/Masked%20AutoEncoder/src/mae_reconstruction.png?raw=true" alt="drawing" width="800"/>


## Downstream Tasks

This model on its own is pretty useless, so lets finetune it to do stuff we care about! We will be doing two cases here:

1) Classification on Imagenet to compare to our ViT model trained from scratch
2) Semantic Segmentation with UperNet Head


### Classification

I train for 100 epochs here (with 5 epochs of warmup) to match the MAE finetuning in the original repo, but I dont have the additional complications of layer-wise weight decay and other things that could further improve training. Basically everything here though was left the same as my original [ViT training script](https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20Computer%20Vision/Vision%20Transformer). 

You can run this with the following:

```sh
accelerate launch finetune_classifier.py \
    --experiment_name "MAE_Imagenet_Finetuning" \
    --wandb_run_name "finetune_mae" \
    --path_to_data "<PATH_TO_IMAGENET>" \
    --path_to_pretrained_mae_weights <PATH_TO_CHECKPOINT_FOLDER> \
    --working_directory "work_dir" \
    --num_classes 1000 \
    --epochs 100 \
    --warmup_epochs 5 \
    --save_checkpoint_interval 10 \
    --per_gpu_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.002 \
    --weight_decay 0.1 \
    --random_aug_magnitude 9 \
    --mixup_alpha 0.2 \
    --cutmix_alpha 1.0 \
    --label_smoothing 0.1 \
    --max_grad_norm 1.0 \
    --img_size 224 \
    --num_workers 32 \
    
```

where PATH_TO_CHECKPOINT_FOLDER is the pretrained checkpoint from your MAE pretraining so we can load that backbone. 

Results for this can be see [here](https://api.wandb.ai/links/exploratorydataadventure/omnnka0c)

 Training Time | ViT (Scratch)    | MAE (Pretrained 800 Epochs) |
| -------- | ------- | ------- |
| 100 Epochs  | 62% Top-1    | 81% Top-1 |
| 300 Epochs | 79% Top-1     | N/A|

The main takeaway here is that, we were able to beat our ViT Trained from scratch on ImageNet for 300 epochs, with our MAE trained only for 100 Epochs! Better results and less training time, so nothing to complain about here!

The Original MAE repo [reports](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md) a 83.6% accuracy with their MAE-Base, but they also have way more gpus to train on larger batch sizes, so this is close enough!

### Segmentation

Image classification is fine, but I wanted to do something more interesting, Image Segmentation. [This](https://huggingface.co/docs/transformers/en/model_doc/upernet) caught my eye, where I saw the UperNet model was being used as a decoder on encoders like [Swin](https://arxiv.org/abs/2103.14030) and [ConvNeXt](https://arxiv.org/abs/2201.03545) (different vision backbones). So I thought why not apply it to our MAE? Segmentation labeling is painfully slow, so the question is, if we pretrain on a ton of images and finetune on some segmentation masks can we actually get decent segmentations? I applied this method to one of my papers [Self-Supervised Digital Elevation Modeling](https://arxiv.org/abs/2309.03367) and it worked! Though at that time I was using [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Lets go ahead and implement this method!

There are two parts to this:

1) We need to define the UperNet Head that includes a Pyramid Pooling Module and a Feature Pyramid Network. A lot of this code is very close to a really great package [PyTorch-Segmentation](https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py)
   
<img src="https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/refs/heads/main/src/visuals/upernet_head.png" alt="drawing" width="600"/>

2) We need to identify which encodings we want to take from our MAE. In the original UperNet, a Resnet was used, so they grabbed the output of the 4 resnet blocks. In our case, we will grab 4 outputs from our 12 transformer layers. To ensure a heirarchy of features, we will grab layers 3, 5, 7 and 11, also matching the MMSegmentation implementation. 

```sh
accelerate launch finetune_segmentation.py \
    --path_to_data <PATH_TO_ADE20K_DATASET> \
    --path_to_backbone_checkpoint <PATH_TO_PRETRAINED_MAE_BACKBONE>
```

The training code is nearly identical to my [UNET]() implementation, just updated for the MAE+UperNet Model. We will also be training on the ADE20K dataset. 

#### Results

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/PyTorch%20for%20Computer%20Vision/Masked%20AutoEncoder/src/seg1.png?raw=true" alt="drawing" width="600"/>

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/PyTorch%20for%20Computer%20Vision/Masked%20AutoEncoder/src/seg2.png?raw=true" alt="drawing" width="600"/>

I didn't do any robust testing here for DICE scores or IOUs, this was just to see if we get something reasonable and it looks like we do!