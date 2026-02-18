import os
import argparse
import numpy as np
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from models.gpt2 import GPT2, GPT2Config
from miniddp.accelerate import Accelerator
from tqdm import tqdm
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT2 with MyTorch")

    ### Experiment Config ###
    parser.add_argument("--project_name", type=str, default="GPT2Trainer")
    parser.add_argument("--run_name", type=str)

    ### Checkpointing Config ###
    parser.add_argument("--working_directory", type=str, default="work_dir")
    parser.add_argument("--checkpoint_iterations", type=int, default=10000)
    parser.add_argument("--always_save_checkpoint", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str)

    ### Model Config ###
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--model_size", type=str, choices=("small", "base"), default="base")
    parser.add_argument("--dropout_p", type=float, default=0.)
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--fused", action="store_true")

    ### Training Config ###
    parser.add_argument("--path_to_data", type=str, required=True)
    parser.add_argument("--train_iterations", type=int, default=150000)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--eval_iterations", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--mixed_precision", action="store_true")

    ### Logging Config ###
    parser.add_argument("--log_iter", type=int, default=100)
    parser.add_argument("--log_wandb", action="store_true")

    args = parser.parse_args()

    return args

args = parse_args()

### Load Model Variant ###
if args.model_size == "small":
    embed_dim = 384
    num_heads = 6
    num_blocks = 6
    mlp_ratio = 4
elif args.model_size == "base":
    embed_dim = 768
    num_heads = 12
    num_blocks = 12
    mlp_ratio = 4

### Check Dataset ###
if not os.path.exists(args.path_to_data):
    raise ValueError("Verify your path to data, it should contain "
                     "a train.bin, val.bin and optionally a tokenizer.pkl")

required_files = ["train.bin", "val.bin"]
optional_files = ["tokenizer.pkl"]

for fname in required_files:
    fpath = os.path.join(args.path_to_data, fname)
    if not os.path.exists(fpath):
        raise ValueError(f"Missing required file: {fname} in {args.path_to_data}")

# Optionally check tokenizer
tokenizer_path = os.path.join(args.path_to_data, "tokenizer.pkl")
tokenizer_exists = os.path.exists(tokenizer_path)
if not tokenizer_exists:
    print("No tokenizer.pkl found, assuming default gpt2 tokenizer!")
    vocab_size = 50257
else:
    with open(tokenizer_path, "rb") as f:
        tokenizer_meta = pickle.load(f)

    vocab_size = tokenizer_meta["vocab_size"]

### DataLoader ###
def get_batch(train=True):

    if train:
        data = np.memmap(os.path.join(args.path_to_data, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(args.path_to_data, 'val.bin'), dtype=np.uint16, mode='r')

    start_idx = np.random.randint(low=0, high=len(data) - args.context_length - 1, size=(args.batch_size//args.gradient_accumulation_steps))
    x = np.stack([data[i:i+args.context_length] for i in start_idx])
    y = np.stack([data[i+1:i+args.context_length+1] for i in start_idx])
    
    return mytorch.Tensor(x), mytorch.Tensor(y)

### Create Checkpoint Directory ###
path_to_experiment = os.path.join(args.working_directory, args.project_name)
if args.run_name is not None:
    path_to_experiment = os.path.join(path_to_experiment, args.run_name)
os.makedirs(path_to_experiment, exist_ok=True)

### Init DDP ###
### If we arent training in DDP or Mixed Precision Accelerator doesnt ###
### really do anything, just a wrapper that helps with logging essentially! ###
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                          mixed_precision=args.mixed_precision,
                          log_wand=args.log_wandb)

### Config to store model information (so we can load precise setup for inference!) ###
training_config = {
                    "vocab_size": vocab_size,
                    "context_length": args.context_length,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "num_blocks": num_blocks,
                    "mlp_ratio": mlp_ratio,
                    "use_bias": args.use_bias,
                    "dropout": args.dropout_p,
                    "batch_size": args.batch_size,
                    "grad_accumulation_steps": args.gradient_accumulation_steps,
                    "max_lr": args.max_lr,
                    "weight_decay": args.weight_decay,
                    "adam_beta1": args.beta1,
                    "adam_beta2": args.beta2,
                    "mixed_precision": args.mixed_precision,
                    "path_to_experiment": path_to_experiment,
                    "path_to_tokenizer": tokenizer_path if tokenizer_exists else None
                }

if accelerator.is_main_process:
    with open(os.path.join(path_to_experiment, "model_meta.pkl"), "wb") as f:
            pickle.dump(training_config, f)

if args.log_wandb:
    accelerator.init_tracker(project_name=args.project_name, 
                             run_name=args.run_name,
                             config=training_config)

### Load Model ###
gpt2config = GPT2Config(
    vocab_size=vocab_size, 
    max_seq_len=args.context_length,
    embed_dim=embed_dim, 
    num_heads=num_heads, 
    num_blocks=num_blocks, 
    attn_dropout_p=args.dropout_p, # Only works for non-fused attn 
    mlp_dropout_p=args.dropout_p, 
    mlp_ratio=mlp_ratio, 
    use_bias=args.use_bias, 
    use_fused_ops=args.fused
)
model = GPT2(gpt2config)

total_params = 0
for param in model.parameters():
    if param.requires_grad:
        total_params += np.prod(param.shape)
accelerator.print("Total Trainable Parameters:", total_params)

### Load Optimizer ###
optimizer = optim.AdamW(model.parameters(), 
                        lr=args.max_lr, 
                        weight_decay=args.weight_decay,
                        beta1=args.beta1, 
                        beta2=args.beta2)

### Load Scheduler ###
scheduler = mytorch.lr_scheduler.CosineLRScheduler(
    optimizer=optimizer, max_lr=args.max_lr, 
    min_lr=args.min_lr, total_steps=args.train_iterations,
    warmup_steps=args.warmup_steps
)

### Prepare Everything ###
model, optimizer = accelerator.prepare(model, optimizer)

### Resume from Checkpoint ###
if args.resume_from_checkpoint is not None:

    ### Grab path to checkpoint ###

    ### This is if we pass in a full path to checkpoint ###
    if os.path.exists(args.resume_from_checkpoint):
        path_to_checkpoint = args.resume_from_checkpoint
    
    ### Otherwise we can pass the folder name in our experiment directory ###
    else:
        path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    
    ### Load our State (model and optimizer) ###
    accelerator.load_state(path_to_checkpoint)
    
    ### Start completed steps from checkpoint index ###
    completed_steps = int(args.resume_from_checkpoint.split("_")[-1])
    accelerator.print(f"Resuming from Iteration: {completed_steps}")

    ### Advance our scheduler to the correct step ###
    scheduler.step_count = completed_steps

else:
    completed_steps = 0

### Load Loss Function ###
loss_fn = nn.CrossEntropyLoss(fused=args.fused)

### Train Model ###
pbar = tqdm(range(args.train_iterations), 
            disable=not accelerator.is_main_process(),
            initial=completed_steps)

for iter in range((args.train_iterations - completed_steps) * args.gradient_accumulation_steps):

    # Sample a batch
    inputs, targets = get_batch(train=True)
    inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)

    # Forward pass
    logits = model(inputs)
    loss = loss_fn(logits, targets)

    # Backward
    accelerator.backward(loss)
    
    # Clip gradients (and get the grads to check on training health)
    grad_norm = accelerator.clip_grad_norm_(args.max_grad_norm)

    # Step optimizer
    optimizer.step()
    optimizer.zero_grad()

    ### Accelerator tracks when accumulation is done, the flag is just sync_grad ###
    if accelerator.sync_grad:
        
        ### One full accumulation complete ###
        completed_steps += 1
        pbar.update(1)

        ### Update Scheduler ###
        scheduler.step()

        ### Gather metrics across GPUs
        if completed_steps % args.log_iter == 0:

            ### Gather (no-op if we are on single GPU) ###
            loss = accelerator.gather_for_metrics(loss)

            ### Grab our stored grad_norm for checking on model health ###
            log_statement = f"Iter {completed_steps}, Loss: {loss:.4f}, LR: {scheduler.get_last_lr():.2e}"
            if accelerator.grad_norm is not None:
                log_statement += f" Grad Norm: {accelerator.grad_norm:.3f}"

            ### Print to Console ###
            accelerator.print(log_statement)

            ### Log with Wandb if enabled ###
            if args.log_wandb:  
                logging_dict = {"loss": loss_val, "lr": scheduler.get_last_lr()}
                if accelerator.grad_norm is not None:
                    logging_dict["grad_norm"] = accelerator.grad_norm 
                accelerator.log(logging_dict, step=completed_steps)

        if completed_steps % args.checkpoint_iterations == 0 and args.always_save_checkpoint:
            accelerator.save_state(os.path.join(path_to_experiment, f"checkpoint_{completed_steps}"))

        if completed_steps % args.eval_interval == 0:
            
            accelerator.print("Evaluating!")
            model.eval()

            val_losses = []

            for _ in range(args.eval_iterations):

                inputs, targets = get_batch(train=False)
                inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)

                # Forward pass
                with mytorch.no_grad():
                    logits = model(inputs)

                ### Compute/Gather Loss ###
                loss = loss_fn(logits, targets)
                loss_val = accelerator.gather_for_metrics(loss)
                val_losses.append(loss_val)

            ### Log Loss ###
            val_losses = np.mean(val_losses)
            accelerator.print("Validation Loss:", val_losses)
            if args.log_wandb:
                logging_dict = {"val_loss": val_losses}
                accelerator.log(logging_dict, step=completed_steps)

            ### Set back into Training Mode ###
            model.train()

### Save final checkpoint once done ! ###
accelerator.save_state(os.path.join(path_to_experiment, f"final_checkpoint"))
accelerator.end_training()