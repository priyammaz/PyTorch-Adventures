import os
import shutil
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from datasets import load_from_disk
from transformers import RobertaTokenizerFast, set_seed, get_scheduler
from accelerate import Accelerator

from utils import RobertaConfig, RobertaMaskedLMCollateFunction
from model import RobertaForMaskedLM

def parse_args():
    ### PARSE COMMAND LINE ARGS ###
    parser = argparse.ArgumentParser(description="RoBERTa Pretraining Arguments on Wikipedia + BookCorpus")
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
    
    ##########################
    ### HUGGINGFACE CONFIG ###
    ##########################

    parser.add_argument(
        "--hf_model_name",
        help="Huggingface model name we want to use for the tokenizer",
        default="FacebookAI/roberta-base",
        type=str
    )

    #########################
    ### DATASET ARGUMENTS ###
    #########################

    parser.add_argument(
        "--path_to_prepped_data",
        required=True,
        help="Path to data prepared in `prepare_data.py`",
        type=str
    )

    parser.add_argument(
        "--context_length",
        help="Max sequence length we want the model to accept",
        default=512, 
        type=int
    )

    parser.add_argument(
        "--masking_probability",
        help="Probability of token to be selected to be masked",
        default=0.15, 
        type=float
    )

    parser.add_argument(
        "--num_workers",
        help="Number of workers for dataloading",
        default=24, 
        type=int
    )

    #######################
    ### MODEL ARGUMENTS ###
    #######################

    ### WAV2VEC2 FEATURE ENCODER CONVOlUTION CONFIG ###

    parser.add_argument(
        "--hidden_dropout_p",
        help="Dropout probability on all linear layers",
        default=0.1,
        type=float
    )

    ### CONVOLUTIONAL POSITIONAL EMBEDDINGS ###
    parser.add_argument(
        "--attention_dropout_p",
        help="Dropout probability on attention matrix",
        default=0.1,
        type=float
    )

    ### TRANSFORMER CONFIG ###
    parser.add_argument(
        "--num_transformer_blocks",
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
        "--layer_norm_eps",
        help="error added to layer norm to avoid divide by zero",
        default=1e-5, 
        type=float
    )

    parser.add_argument(
        "--initializer_range",
        help="Standard deviation of linear layers initialized as normal distribution",
        default=0.02,
        type=float
    )

    ##############################
    ### TRAINING CONFIGURATION ###
    ##############################

    parser.add_argument(
        "--per_gpu_batch_size",
        help="Overall batch size per gpu during training",
        default=128, 
        type=int
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="Splits per_gpu_batch_size by gradient_accumulation_steps",
        default=4, 
        type=int
    )

    parser.add_argument(
        "--num_training_steps", 
        help="Number of training steps to take",
        default=250000,
        type=int
    )

    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=20000, 
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
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
        default=2500, 
        type=int
    )

    parser.add_argument(
        "--checkpoint_interval",
        help="Number of iterations for checkpointing",
        default=2500,
        type=int
    )

    parser.add_argument(
        "--learning_rate", 
        help="Max learning rate for all Learning Rate Schedulers", 
        default=6e-4, 
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

### Define Tokenizer ###
tokenizer = RobertaTokenizerFast.from_pretrained(args.hf_model_name)

### Prepare Config File ###
config = RobertaConfig(

    vocab_size = tokenizer.vocab_size, 
    start_token = tokenizer.bos_token_id,
    end_token = tokenizer.eos_token_id, 
    pad_token = tokenizer.pad_token_id, 
    mask_token = tokenizer.mask_token_id,
    embedding_dimension = args.embedding_dimension,
    num_transformer_blocks = args.num_transformer_blocks,
    num_attention_heads = args.num_attention_heads,
    mlp_ratio = args.mlp_ratio,
    layer_norm_eps = args.layer_norm_eps, 
    hidden_dropout_p = args.hidden_dropout_p,
    attention_dropout_p = args.attention_dropout_p, 
    context_length = args.context_length,
    masking_prob = args.masking_probability,
    hf_model_name = args.hf_model_name

)

### Load Dataset ###
tokenized_data = load_from_disk(args.path_to_prepped_data)

### Load Model ###
model = RobertaForMaskedLM(config)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of Parameters:", params)

## Define DataLoader ###
collate_fn = RobertaMaskedLMCollateFunction(config)
mini_batchsize = args.per_gpu_batch_size // args.gradient_accumulation_steps

train_dataloader = DataLoader(tokenized_data["train"], 
                              batch_size=mini_batchsize, 
                              collate_fn=collate_fn, 
                              shuffle=True)

eval_dataloader = DataLoader(tokenized_data["test"], 
                             batch_size=mini_batchsize, 
                             collate_fn=collate_fn, 
                             shuffle=False)

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
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps,
)

### PREPARE EVERYTHING ###
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
accelerator.register_for_checkpointing(scheduler)

### Define Accuracy ###
accuracy_func = Accuracy(task="multiclass", num_classes=tokenizer.vocab_size, ignore_index=-100).to(accelerator.device)

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
    
    ### Accumulated Loss ###
    accumulate_loss = 0
    
    ### Keep Track of Accuracy ###
    accuracy = 0
    
    for batch in train_dataloader:        

        ### Make sure everything is on GPU ###
        batch = {k:v.to(accelerator.device) for (k,v) in batch.items()}

        ### Pass Batch through Model ###
        logits, preds, loss = model(**batch)

        ### Scale Loss by Gradient Accumulation Steps ###
        loss = loss / args.gradient_accumulation_steps
        accumulate_loss += loss

        ### Compute Gradients ###
        accelerator.backward(loss)

        ### Compute TopK Accuracy ###
        labels = batch["labels"].flatten()
        acc = accuracy_func(preds, labels)
        accuracy += acc / args.gradient_accumulation_steps

        ### Iterate Accumulation ###
        accumulate_steps += 1

        if accumulate_steps % args.gradient_accumulation_steps == 0:

            ### Update Model ###
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            ### Update Scheduler ###
            scheduler.step()

            ### Log Results!! ###
            if completed_steps % args.logging_steps == 0:
                
                accumulate_loss = accumulate_loss.detach() * args.gradient_accumulation_steps
                accuracy = accuracy.detach()
            
                if accelerator.state.num_processes > 1:

                    accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))
                    accuracy = torch.mean(accelerator.gather_for_metrics(accuracy))

                log = {"train_loss": accumulate_loss,
                       "train_acc": accuracy,
                       "learning_rate": scheduler.get_last_lr()[0]}

                logging_string = f"[{completed_steps}/{args.num_training_steps}] Training Loss: {accumulate_loss} | Training Acc: {accuracy}"
                
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
                       "val_acc": 0}

                ### Iterate Data ###
                num_losses = 0
                for batch in tqdm(eval_dataloader, disable=not accelerator.is_main_process):
                    
                    ### Move everything to GPU ###
                    batch = {k:v.to(accelerator.device) for (k,v) in batch.items()}

                    ### Pass through model ###
                    with torch.inference_mode():
                        logits, preds, loss = model(**batch)

                    ### Grab Loss ###
                    loss = loss.detach()
                    if accelerator.num_processes > 1:
                        loss = torch.mean(accelerator.gather_for_metrics(loss))
                    
                    ### Compute Accuracy ###
                    labels = batch["labels"].flatten()
                    accuracy = accuracy_func(preds, labels)

                    ### Add to our Logs ###
                    log["val_loss"] += loss
                    log["val_acc"] += accuracy
                    num_losses += 1
                
                ### Divide Log by Num Losses ###
                log["val_loss"] = log["val_loss"] / num_losses
                log["val_acc"] = log["val_acc"] / num_losses

                ## Print to Console ###
                logging_string = f"[{completed_steps}/{args.num_training_steps}] Validation Loss: {log["val_loss"]} | Validation Acc: {log["val_acc"]}"
                
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

            ### Reset Topk Accuracy and Loss Accumulate For Next Accumulation ###
            accuracy = 0
            accumulate_loss = 0

accelerator.end_training()