import os
import argparse
import math
import torch
from accelerate import Accelerator

from dataset import LibriSpeechDataset, Wav2Vec2CollateFunctionForPreTraining
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from transformers import (
    get_scheduler,
    set_seed,
)
from model import Wav2Vec2ForPreTraining

from utils import compute_span_mask, sample_negative_indices, \
    compute_encoded_lengths, compute_sub_attention_mask, \
    Wav2Vec2Config



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--saving_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=200000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
 
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=32000, help="Number of steps for the warmup in the lr scheduler."
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
        "--gumbel_temperature_decay", type=float, default=0.999995, help="Decay of gumbel temperature during training."
    )
    parser.add_argument(
        "--max_duration_in_seconds",
        type=float,
        default=5.0,
        help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--min_duration_in_seconds",
        type=float,
        default=3.0,
        help="Filter out audio files that are shorter than `min_duration_in_seconds` seconds",
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
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )

    parser.add_argument(
        "--mask_time_prob",
        type=float,
        default=None,
        help=(
            "Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked in the"
            " contrastive task. If omitted, will pull value from model config."
        ),
    )
    parser.add_argument(
        "--mask_time_length",
        type=int,
        default=None,
        help=(
            "Length of each vector mask span to mask along the time axis in the contrastive task."
            " If omitted, will pull value from model config."
        ),
    )
    args = parser.parse_args()

    return args


### HELPER FUNCTIONS ###
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
    """Compute grad norm given a gradient scale."""
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


# See all possible arguments in src/transformers/args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.
args = parse_args()

set_seed(0)

### Instantiate Accelerate ###
accelerator = Accelerator()

# 3. Load model
config = Wav2Vec2Config()

# initialize random model
model = Wav2Vec2ForPreTraining(config)

### Prepare Dataset ###
train_set = LibriSpeechDataset(path_to_data_root="/mnt/datadrive/data/LibriSpeech/", 
                               include_splits=["dev-clean", "test-clean"],
                               max_audio_duration=20.0, 
                               min_audio_duration=2.0,
                               sampling_rate=16000,
                               return_transcripts=False)

test_set = LibriSpeechDataset(path_to_data_root="/mnt/datadrive/data/LibriSpeech/", 
                              include_splits=["dev-clean", "test-clean"],
                              max_audio_duration=20.0, 
                              min_audio_duration=2.0,
                              sampling_rate=16000,
                              return_transcripts=False)


data_collator = Wav2Vec2CollateFunctionForPreTraining(config)

train_dataloader = DataLoader(train_set, 
                         batch_size=args.per_device_train_batch_size, 
                         shuffle=True, 
                         num_workers=16, 
                         collate_fn=data_collator)

eval_dataloader = DataLoader(test_set, 
                        batch_size=args.per_device_eval_batch_size , 
                        shuffle=False, 
                        num_workers=16, 
                        collate_fn=data_collator)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=[args.adam_beta1, args.adam_beta2],
    eps=args.adam_epsilon,
)

# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps*accelerator.num_processes,
    num_training_steps=args.max_train_steps*accelerator.num_processes,
)


# Only show the progress bar once on each machine.
progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0

train = True
while train:

    accumulate_steps = 0
    accumulated_hours_per_batch = 0
    accumulated_percent_masked = 0
    
    for batch in train_dataloader:
        # compute num of losses
        num_losses = batch["mask_time_indices"].sum()
        percent_masked = num_losses / batch["sub_attention_mask"].sum()

        ### Make sure everything is on GPU ###
        batch = {k:v.to(accelerator.device) for (k,v) in batch.items()}

        # forward
        outputs = model(**batch)

        # divide loss by gradient accumulation steps since gradients
        # are accumulated for multiple backward passes in PyTorch
        loss = outputs.loss / args.gradient_accumulation_steps
        accelerator.backward(loss)

        # make sure that `num_losses` is summed for distributed training
        # and average gradients over losses of all devices
        if accelerator.state.num_processes > 1:
            num_losses = accelerator.gather_for_metrics(num_losses).sum()
            gradient_multiplier = accelerator.state.num_processes / num_losses
            multiply_gradients(model.module.parameters(), gradient_multiplier)
        else:
            multiply_gradients(model.parameters(), 1 / num_losses)

        accumulate_steps += 1

        # update step
        if accumulate_steps % args.gradient_accumulation_steps == 0:
            # compute grad norm for monitoring
            scale = (
                accelerator.scaler._scale.item()
                if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                else 1
            )
            if accelerator.state.num_processes > 1:
                grad_norm = compute_gradient_norms(model.module.parameters(), scale)
            else:
                grad_norm = compute_gradient_norms(model.parameters(), scale)

            # update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # update gumbel temperature
            gumbel_temperature = max(
                args.max_gumbel_temperature * args.gumbel_temperature_decay**completed_steps,
                args.min_gumbel_temperature,
            )
            if hasattr(model, "module"):
                model.module.set_gumbel_temperature(gumbel_temperature)
            else:
                model.set_gumbel_temperature(gumbel_temperature)

            progress_bar.update(1)
            completed_steps += 1

            # 6. Log all results
            if completed_steps % args.logging_steps == 0:
                loss.detach()
                outputs.contrastive_loss.detach()
                outputs.diversity_loss.detach()

                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather_for_metrics(loss).sum()
                    outputs.contrastive_loss = accelerator.gather_for_metrics(outputs.contrastive_loss).sum()
                    outputs.diversity_loss = accelerator.gather_for_metrics(outputs.diversity_loss).sum()
                    percent_masked = accelerator.gather_for_metrics(percent_masked).sum()

                train_logs = {
                    "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                    "constrast_loss": outputs.contrastive_loss / num_losses,
                    "div_loss": outputs.diversity_loss / num_losses,
                    "%_mask_idx": percent_masked / accelerator.num_processes,
                    "ppl": outputs.codevector_perplexity,
                    "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                    "temp": torch.tensor(gumbel_temperature),
                    "grad_norm": torch.tensor(grad_norm),
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)
        
        completed_steps += 1

