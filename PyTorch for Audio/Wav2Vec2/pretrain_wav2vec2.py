import os
import shutil
import numpy as np
import argparse
import torch
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import get_scheduler, set_seed

from dataset import LibriSpeechDataset, Wav2Vec2CollateFunctionForPreTraining
from model import Wav2Vec2ForPreTraining
from utils import Wav2Vec2Config

def parse_args():
    ### PARSE COMMAND LINE ARGS ###
    parser = argparse.ArgumentParser(description="Wav2Vec2 Pretraining Arguments on Librispeech")
    parser.add_argument(
        "--experiment_name", 
        required=True, 
        type=str
    )

    parser.add_argument(
        "--working_directory", 
        required=True, 
        type=str
    )

    #########################
    ### DATASET ARGUMENTS ###
    #########################

    parser.add_argument(
        "--path_to_data_root", 
        help="Path to data root directory",
        required=True,
        type=str
    )

    parser.add_argument(
        "--train_splits", 
        help="Select Librispeech Training Splits (960 Hours of data Split into 100, 360 and 500 hour sections)",
        default=["train-clean-100", "train-clean-360", "train-other-500"],
        choices=("train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "test-clean"),
        nargs='+',
        type=str
    )

    parser.add_argument(
        "--test_splits", 
        help="Select Librispeech Testing Splits (Select from limited Validation + Testing Datasets)",
        default=["dev-clean", "test-clean"],
        choices=("train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "test-clean"),
        nargs="+",
        type=str
    )

    parser.add_argument(
        "--minimum_audio_duration",
        help="Filter out any audio sample less than these many seconds", 
        default=2.0, 
        type=float
    )

    parser.add_argument(
        "--maximum_audio_duration",
        help="Filter out any audio samples greater than these many seconds",
        default=20.0, 
        type=float
    )

    parser.add_argument(
        "--sampling_rate",
        help="Sampling rate to sample all audio to before passing to model",
        default=16000,
        type=int
    )

    parser.add_argument(
        "--audio_input_channels",
        help="Number of input channels from audio",
        default=1, 
        type=int
    )

    parser.add_argument(
        "--masking_probability",
        help="The probability for each token to be the start of a span mask",
        default=0.065, 
        type=float
    )

    parser.add_argument(
        "--masking_span_length",
        help="Number of consecutive tokens to mask in every span",
        default=10, 
        type=int
    )

    parser.add_argument(
        "--minimum_spans",
        help="Minimum number of span masks to have in every sequence",
        default=2, 
        type=int
    )

    parser.add_argument(
        "--num_negatives",
        help="For every masked token, how many negatives do we want to sample for contrastive loss?",
        default=100, 
        type=int
    )

    parser.add_argument(
        "--num_workers",
        help="Number of workers for dataloading",
        default=16, 
        type=int
    )

    #######################
    ### MODEL ARGUMENTS ###
    #######################

    ### WAV2VEC2 FEATURE ENCODER CONVOlUTION CONFIG ###
    parser.add_argument(
        "--conv_dim",
        help="Sequence of channels dims in encoder convolutions",
        default=(512, 512, 512, 512, 512, 512, 512),
        nargs="+",
        type=int
    )

    parser.add_argument(
        "--conv_kernel",
        help="Kernel size for each convolution in encoder convolutions",
        default=(10, 3, 3, 3, 3, 2, 2),
        nargs="+",
        type=int
    )

    parser.add_argument(
        "--conv_stride",
        help="Strides for each convolution in encoder convolutions",
        default=(5, 2, 2, 2, 2, 2, 2),
        nargs="+",
        type=int
    )

    parser.add_argument(
        "--disable_conv_bias",
        help="Do you want to have bias on the convolutional encoder",
        action=argparse.BooleanOptionalAction,
        default=False
    )

    parser.add_argument(
        "--feature_proj_dropout_p",
        help="Dropout on feature projection from convolution to transformer embedding dims",
        default=0.0,
        type=float
    )

    ### CONVOLUTIONAL POSITIONAL EMBEDDINGS ###
    parser.add_argument(
        "--conv_positional_emb_drop_p",
        help="Positional Embedding Dropout probability",
        default=0.0,
        type=float
    )

    parser.add_argument(
        "--conv_positional_emb_groups",
        help="Number of groups in convolution for positional encodings",
        default=16,
        type=int
    )

    parser.add_argument(
        "--conv_positional_emb_kernel_size",
        help="Kernel size for convolutional positional encodings",
        default=128,
        type=int
    )

    ### TRANSFORMER CONFIG ###
    parser.add_argument(
        "--num_transformer_layers",
        help="Number of transformer blocks in model",
        default=12,
        type=int
    )

    parser.add_argument(
        "--num_attention_heads",
        help="Number of heads of attention",
        default=12,
        type=int
    )

    parser.add_argument(
        "--embedding_dimension",
        help="Transformer embedding dimension",
        default=768,
        type=int
    )

    parser.add_argument(
        "--mlp_ratio",
        help="Hidden layer expansion factor for feed forward layers",
        default=4,
        type=int
    )

    parser.add_argument(
        "--mlp_dropout_p",
        help="Dropout probability in feedforward layers",
        default=0.0,
        type=float
    )

    parser.add_argument(
        "--attention_dropout_p",
        help="Dropout probability on attention matrix",
        default=0.0,
        type=float
    )

    parser.add_argument(
        "--transformer_encoder_dropout_p",
        help="Post transformer block dropout probability",
        default=0.0,
        type=float
    )

    parser.add_argument(
        "--layer_dropout",
        help="Entire transformer layerblock dropout (https://paperswithcode.com/method/layerdrop). \
            though im not sure how to implement this with DDP, as it throws unused parameters error",
        default=0.0,
        type=float
    )

    parser.add_argument(
        "--initializer_range",
        help="Standard deviation of linear layers initialized as normal distribution",
        default=0.02,
        type=float
    )

    ### GUMBEL SOFTMAX CONFIG ###
    parser.add_argument(
        "--num_codevector_groups",
        help="Number of codebooks in our quantizer",
        default=2,
        type=int
    )

    parser.add_argument(
        "--num_codevectors_per_group",
        help="Number of codevectors per group",
        default=320,
        type=int
    )

    parser.add_argument(
        "--codevector_dim",
        help="Dimension of codevectors in vector quantization",
        default=256,
        type=int
    )

    parser.add_argument(
        "--pre_quantizer_dropout_p",
        help="Dropout before quantization of tokens",
        default=0.0,
        type=float
    )

    parser.add_argument(
        "--max_gumbel_temperature",
        type=float,
        default=2.0,
        help="Maximum temperature for gumbel softmax.",
    )

    parser.add_argument(
        "--min_gumbel_temperature",
        type=float,
        default=0.5,
        help="Minimum temperature for gumbel softmax.",
    )

    parser.add_argument(
        "--gumbel_temperature_decay", 
        type=float, 
        default=0.999995, 
        help="Decay of gumbel temperature during training."
    )

    ##########################
    ### LOSS CONFIGURATION ###
    ##########################

    parser.add_argument(
        "--contrastive_logits_temperature",
        help="Temperature to scale cosine similarity before softmax",
        default=0.1, 
        type=float
    )

    parser.add_argument(
        "--diversity_loss_weight",
        help="Weight to scale diversity loss",
        default=0.1, 
        type=float
    )

    ##############################
    ### TRAINING CONFIGURATION ###
    ##############################

    parser.add_argument(
        "--per_gpu_batch_size",
        help="Overall batch size per gpu during training",
        default=64, 
        type=int
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="Splits per_gpu_batch_size by gradient_accumulation_steps",
        default=8, 
        type=int
    )

    parser.add_argument(
        "--num_training_steps", 
        help="Number of training steps to take",
        default=200000,
        type=int
    )

    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=32000, 
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="polynomial",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--logging_steps", 
        help="Number of iterations for every log of metrics to wandb",
        default=1,
        type=int
    )

    parser.add_argument(
        "--evaluation_interval", 
        help="Number of iterations for every evaluation and plotting",
        default=1000, 
        type=int
    )

    parser.add_argument(
        "--checkpoint_interval",
        help="Number of iterations for checkpointing",
        default=1000,
        type=int
    )

    parser.add_argument(
        "--learning_rate", 
        help="Max learning rate for all Learning Rate Schedulers", 
        default=0.001, 
        type=float
    )

    parser.add_argument(
        "--bias_weight_decay", 
        help="Apply weight decay to bias",
        default=False, 
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "--norm_weight_decay",
        help="Apply weight decay to normalization weight and bias",
        default=False,
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "--weight_decay",
        help="Weight decay constant for AdamW optimizer", 
        default=0.01, 
        type=float
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )

    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.98,
        help="Beta2 for AdamW optimizer",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-6,
        help="Epsilon for AdamW optimizer",
    )

    parser.add_argument(
        "--num_keep_checkpoints",
        help="Number of Checkpoints to Keep, if None, all checkpoints will be saved",
        default=None, 
        type=int
    )

    parser.add_argument(
        "--seed", 
        help="Set seed in model for reproducible training",
        default=None,
        type=int
    )

    parser.add_argument(
        "--resume_from_checkpoint", 
        help="Checkpoint folder for model to resume training from, inside the experiment folder", 
        default=None, 
        type=str
    )

    #############################
    ### LOGGING CONFIGURATION ###
    #############################
    
    parser.add_argument(
        "--log_wandb", 
        help="Flag to enable logging to wandb",
        default=False, 
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    return args


########################
### HELPER FUNCTIONS ###
########################

def multiply_gradients(params, constant):
    """
    All of our losses are summed up, not averaged. The number of tokens 
    that are masked is randomly sampled and will be different on every GPU
    if we trained on multi-gpu. The easiest way to handle this (as done by 
    the huggingface example) is to compute the derivative for the sum of all the
    losses (ignoring the non-masked tokens) and then gather the number of masked
    tokens at the end and scale the gradients before updating weights!
    """

    for param in params:
        if param.grad is not None:
            param.grad.data.mul_(constant)

def compute_gradient_norms(params, scale=1):
    """
    Compute the gradient norms as we are training. This is a good way to keep an eye on model
    health, if the gradients are extremely large or basically 0 we know there is a problem. As 
    recommended by the pretraining example from Huggingface we want this to be roughly between 
    0.5 and 2 during training!

    Also, if we are using mixed precision training, our gradients are automatically scaled by 
    huggingface accelerate. By default it is 1, but if we have an overflow, accelerate will 
    scale the gradients, so we pass in that scale as well here just so we are reporting what the 
    optimizer is seeing. 
    """
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm

def compute_batch_duration(attention_mask, sampling_rate):
    """
    This function is just to keep track of our total hours of audio
    within every batch! The attention mask in this case is at the 
    audio level, NOT ENCODED AUDIO. It is 1 for everywhere there was a
    valid audio sample and 0 where there is paddings. So we can just
    sum up the attention mask for every sample in the batch to get 
    the length, and then divide by the sampling rate to get the seconds
    of audio
    """
    total_duration_seconds = torch.sum(attention_mask.sum(axis=-1) / sampling_rate)
    total_duration_hours = total_duration_seconds / 3600
    return total_duration_hours    

####################################
### PREP EVERYTHING FOR TRAINING ###
####################################

### Parse Arguments ###
args = parse_args()

### Set Seed ###
if args.seed is not None:
    set_seed(args.seed)

### Instantiate Accelerate ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb" if args.log_wandb else None)
if args.log_wandb:
    accelerator.init_trackers(args.experiment_name)

### Prepare Config File ###
config = Wav2Vec2Config(

    conv_dim=tuple(args.conv_dim), 
    conv_stride=tuple(args.conv_stride),
    conv_kernel=tuple(args.conv_kernel),
    conv_bias=not args.disable_conv_bias, 
    feature_projection_dropout_p=args.feature_proj_dropout_p,
    conv_positional_emb_drop_p=args.conv_positional_emb_drop_p, 
    conv_positional_emb_groups=args.conv_positional_emb_groups,
    conv_positional_emb_kernel_size=args.conv_positional_emb_kernel_size,
    num_transformer_layers=args.num_transformer_layers,
    num_attention_heads=args.num_attention_heads,
    mlp_ratio=args.mlp_ratio,
    mlp_dropout_p=args.mlp_dropout_p, 
    attention_dropout_p=args.attention_dropout_p,
    transformer_encoder_dropout=args.transformer_encoder_dropout_p,
    layer_dropout=args.layer_dropout if accelerator.num_processes == 1 else 0.0, # Not really sure how to do layerdrop on multiple GPUs
    initializer_range=args.initializer_range,
    num_codevector_groups=args.num_codevector_groups,
    num_codevectors_per_group=args.num_codevectors_per_group,
    codevector_dim=args.codevector_dim, 
    pre_quantizer_dropout=args.pre_quantizer_dropout_p, 
    masking_probability=args.masking_probability, 
    masking_span_length=args.masking_span_length, 
    minimum_spans=args.minimum_spans, 
    contrastive_logits_temperature=args.contrastive_logits_temperature, 
    diversity_loss_weight=args.diversity_loss_weight,
    num_negatives=args.num_negatives

)


# initialize random model
model = Wav2Vec2ForPreTraining(config)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of Parameters:", params)

### Prepare Dataset ###
train_set = LibriSpeechDataset(path_to_data_root=args.path_to_data_root, 
                               include_splits=args.train_splits,
                               max_audio_duration=args.maximum_audio_duration, 
                               min_audio_duration=args.minimum_audio_duration,
                               sampling_rate=args.sampling_rate,
                               return_transcripts=False)

test_set = LibriSpeechDataset(path_to_data_root=args.path_to_data_root, 
                              include_splits=args.test_splits,
                              max_audio_duration=args.maximum_audio_duration, 
                              min_audio_duration=args.minimum_audio_duration,
                              sampling_rate=args.sampling_rate,
                              return_transcripts=False)

data_collator = Wav2Vec2CollateFunctionForPreTraining(config)
minibatch_size = args.per_gpu_batch_size // args.gradient_accumulation_steps
train_dataloader = DataLoader(train_set, 
                              batch_size=minibatch_size, 
                              shuffle=False, 
                              num_workers=8, 
                              collate_fn=data_collator)

eval_dataloader = DataLoader(test_set, 
                             batch_size=minibatch_size, 
                             shuffle=False, 
                             num_workers=8, 
                             collate_fn=data_collator)

### PREPARE OPTIMIZER ###
if (not args.bias_weight_decay) or (not args.norm_weight_decay):
    accelerator.print("Disabling Weight Decay on Some Parameters")
    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():

        if param.requires_grad:
            
            ### Dont have Weight decay on any bias parameter (including norm) ###
            if "bias" in name and not args.bias_weight_decay:
                no_weight_decay_params.append(param)

            ### Dont have Weight Decay on any Norm scales params (weights) ###
            elif "groupnorm" in name and not args.norm_weight_decay:
                no_weight_decay_params.append(param)

            else:
                weight_decay_params.append(param)

    optimizer_group = [
        {"params": weight_decay_params, "weight_decay": args.weight_decay},
        {"params": no_weight_decay_params, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_group, 
                                  lr=args.learning_rate,
                                  betas=[args.adam_beta1, args.adam_beta2],
                                  eps=args.adam_epsilon)

else:
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=args.learning_rate,
                                betas=[args.adam_beta1, args.adam_beta2],
                                eps=args.adam_epsilon)

### DEFINE SCHEDULER ###
scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps*accelerator.num_processes,
    num_training_steps=args.num_training_steps*accelerator.num_processes,
)

### PREPARE EVERYTHING ###
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
accelerator.register_for_checkpointing(scheduler)

########################################################
############## TRAINING SCRIPT #########################
########################################################

### RESUME FROM CHECKPOINT ###
if args.resume_from_checkpoint is not None:

    ### Grab path to checkpoint ###
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    
    ### Load checkpoint on main process first, recommended here: (https://huggingface.co/docs/accelerate/en/concept_guides/deferring_execution) ###
    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)
    
    ### Start completed steps from checkpoint index ###
    completed_steps = int(args.resume_from_checkpoint.split("_")[-1])
    accelerator.print(f"Resuming from Iteration: {completed_steps}")
else:
    completed_steps = 0

train = True
progress_bar = tqdm(range(completed_steps, args.num_training_steps), disable=not accelerator.is_local_main_process)

while train:

    ### Keep Track of Accumulated Mini-Steps ###
    accumulate_steps = 0

    ### Keep track of hours of audio per accumulation ###
    accumulated_hours_per_batch = 0

    ### Keep Track of Percent of Tokens Masked Out ###
    accumulated_percent_masked = 0
    
    for batch in train_dataloader:        

        ### Compute the Number of Masked Tokens to Scale Gradients ###
        num_losses = batch["mask_time_indices"].sum()
        
        ### Add Up Number of Hours of Audio in each MiniBatch ###
        hours_of_audio = compute_batch_duration(batch["attention_mask"], args.sampling_rate)
        accumulated_hours_per_batch += hours_of_audio

        ### Compute Percent Masked (attention mask is repeated so we just need the first one) ###
        percent_masked = num_losses / batch["sub_attention_mask"].sum()
        accumulated_percent_masked += percent_masked / args.gradient_accumulation_steps

        ### Make sure everything is on GPU ###
        batch = {k:v.to(accelerator.device) for (k,v) in batch.items()}

        ### Pass Batch through Model ###
        outputs = model(**batch)

        ### Scale Loss by Gradient Accumulation Steps ###
        loss = outputs.loss / args.gradient_accumulation_steps

        ### Compute Gradients ###
        accelerator.backward(loss)

        ### Scale Gradients by Num GPUs and num_losses ###
        if accelerator.state.num_processes > 1:
            num_losses = accelerator.gather_for_metrics(num_losses).sum()
            gradient_multiplier = accelerator.state.num_processes / num_losses
            multiply_gradients(model.module.parameters(), gradient_multiplier)
        else:
            multiply_gradients(model.parameters(), 1 / num_losses)

        accumulate_steps += 1

        if accumulate_steps % args.gradient_accumulation_steps == 0:

            ### Compute Gradient Norm to keep an eye on training health ###
            if hasattr(accelerator, "scaler") and accelerator.scaler is not None:
                scale = accelerator.scaler._scale.item()
            else:
                scale = 1

            if accelerator.state.num_processes > 1:
                grad_norm = compute_gradient_norms(model.module.parameters(), scale)
            else:
                grad_norm = compute_gradient_norms(model.parameters(), scale)

            ### Update Model ###
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            ### Update Gumbel Temperature ###
            gumbel_temperature = max(args.max_gumbel_temperature * args.gumbel_temperature_decay**completed_steps,
                                     args.min_gumbel_temperature)
            
            if hasattr(model, "module"):
                model.module.set_gumbel_temperature(gumbel_temperature)
            else:
                model.set_gumbel_temperature(gumbel_temperature)

            ### Log Results!! ###
            if completed_steps % args.logging_steps == 0:
                
                loss = outputs.loss.detach()
                contrastive_loss = outputs.contrastive_loss.detach()
                diversity_loss = outputs.diversity_loss.detach()
                perplexity = outputs.codevector_perplexity.detach()
                hours_per_batch = accumulated_hours_per_batch.detach()
                percent_masked = accumulated_percent_masked.detach()

                if accelerator.state.num_processes > 1:

                    loss = torch.sum(accelerator.gather_for_metrics(loss)) / num_losses
                    contrastive_loss = torch.sum(accelerator.gather_for_metrics(contrastive_loss)) / num_losses
                    diversity_loss = torch.sum(accelerator.gather_for_metrics(diversity_loss)) / num_losses
                    perplexity = torch.sum(accelerator.gather_for_metrics(perplexity)) / num_losses
                    hours_per_batch = torch.sum(accelerator.gather_for_metrics(hours_per_batch))
                    percent_masked = torch.mean(accelerator.gather_for_metrics(percent_masked))
                
                else:
                    loss = loss / num_losses
                    contrastive_loss = contrastive_loss / num_losses
                    diversity_loss = diversity_loss / num_losses
                    perplexity = perplexity / num_losses
                    hours_per_batch = hours_per_batch
                    percent_masked = percent_masked

                log = {"train_loss": loss,
                       "train_contrast_loss": contrastive_loss,
                       "train_div_loss": diversity_loss,
                       "pct_masked": percent_masked,
                       "batch_hours": hours_per_batch, 
                       "perplexity": perplexity,
                       "lr": scheduler.get_last_lr()[0],
                       "temp": gumbel_temperature,
                       "grad_norm": grad_norm}

                logging_string = ""
                for k, v in log.items():
                    logging_string += f"|{k[6:] if 'train_' in k else k}: {round(v.item() if torch.is_tensor(v) else v, 3)}"
                
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                
                if args.log_wandb:
                    accelerator.log(log, step=completed_steps)

            ### Evaluation Loop ###
            if completed_steps % args.evaluation_interval == 0:
                if accelerator.is_main_process:
                    progress_bar.write("Evaluating Model!!")
                
                model.eval()

                ### Dictionary to Store Results ###
                log = {"val_loss": 0, 
                       "val_contrast_loss": 0, 
                       "val_div_loss": 0}
                all_num_losses = 0

                ### Iterate Data ###
                for batch in tqdm(eval_dataloader, disable=not accelerator.is_main_process):
                    
                    ### Compute Number of Losses ###
                    num_losses = batch["mask_time_indices"].sum()

                    ### Move everything to GPU ###
                    batch = {k:v.to(accelerator.device) for (k,v) in batch.items()}

                    ### Pass through model ###
                    with torch.inference_mode():
                        output = model(**batch)

                    ### Grab Everything for Logging and Saving ###

                    loss = output.loss
                    contrastive_loss = output.contrastive_loss
                    diversity_loss = outputs.diversity_loss

                    if accelerator.num_processes > 1:
                        loss = torch.sum(accelerator.gather_for_metrics(loss))
                        contrastive_loss = torch.sum(accelerator.gather_for_metrics(contrastive_loss))
                        diversity_loss = torch.sum(accelerator.gather_for_metrics(diversity_loss))
                        num_losses = torch.sum(accelerator.gather_for_metrics(num_losses))

                    log["val_loss"] += loss
                    log["val_contrast_loss"] += contrastive_loss
                    log["val_div_loss"] += diversity_loss 
                    all_num_losses += num_losses
                
                ### Divide Log by Num Losses ###
                log = {k: v / all_num_losses for (k,v) in log.items()}

                ## Print to Console ###
                logging_string = ""
                for k, v in log.items():
                    logging_string += f"|{k[4:]}: {round(v.item() if torch.is_tensor(v) else v, 3)}"

                ### Print out Log ###
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                
                if args.log_wandb:
                    accelerator.log(log, step=completed_steps)

                ### Return Back to Training Mode ###
                model.train()

            ### Checkpoint Model (Only need main process for this) ###
            if (completed_steps % args.checkpoint_interval == 0):
                
                ### Save Checkpoint ### 
                path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")

                if accelerator.is_main_process:
                    progress_bar.write(f"Saving Checkpoint to {path_to_checkpoint}")

                ### Make sure that all processes have caught up before saving checkpoint! ###
                accelerator.wait_for_everyone()

                ### Save checkpoint using only the main process ###
                if accelerator.is_main_process:
                    accelerator.save_state(output_dir=path_to_checkpoint)

                ### Delete Old Checkpoints ###
                if args.num_keep_checkpoints is not None:
                    if accelerator.is_main_process:
                        all_checkpoints = os.listdir(path_to_experiment)
                        all_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split(".")[0].split("_")[-1]))
                        
                        if len(all_checkpoints) > args.num_keep_checkpoints:
                            checkpoints_to_delete = all_checkpoints[:-args.num_keep_checkpoints]

                            for checkpoint_to_delete in checkpoints_to_delete:
                                path_to_checkpoint_to_delete = os.path.join(path_to_experiment, checkpoint_to_delete)
                                if os.path.isdir(path_to_checkpoint_to_delete):
                                    shutil.rmtree(path_to_checkpoint_to_delete)
                                    
                ### Let all processes hang out while files are being deleted with the main process ###
                accelerator.wait_for_everyone()
                
            if completed_steps >= args.num_training_steps:
                train = False
                if accelerator.is_main_process:
                    progress_bar.write("Completed Training!!")
                break

            ### Iterate Progress Bar and Completed Steps ###
            completed_steps += 1
            progress_bar.update(1)

            ### After Accumulation, Zero our temporary storage variables ###
            accumulate_steps = accumulated_hours_per_batch = accumulated_percent_masked = 0

accelerator.end_training()