import os
import numpy as np
import torch
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_from_disk
from tqdm import tqdm

from model import Transformer, TransformerConfig
from data import TranslationCollator
from tokenizer import FrenchTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

###########################
### Training Parameters ###
###########################

### Model Config ###
encoder_depth = 6
decoder_depth = 6
mlp_ratio = 4
attention_dropout_p = 0.1
hidden_dropout_p = 0.1
embedding_dimension = 512
num_attention_heads = 8
max_src_len = 512
max_tgt_len = 512
learn_pos_embed = False

### Tokenizer Config ###
tgt_tokenizer =  FrenchTokenizer("trained_tokenizer/french_wp.json")
src_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

### Dataloader Config ###
path_to_data = "/mnt/datadrive/data/machine_translation/english2french/tokenized_english2french_corpus"
batch_size = 128
gradient_accumulation_steps = 2
num_workers = 16

### Training Config ###
learning_rate = 1e-4
training_steps = 150000 
warmup_steps = 2000
scheduler_type = "cosine"
evaluation_steps = 2500
bias_norm_weight_decay = False
weight_decay = 0.001
betas = (0.9, 0.98)
adam_eps = 1e-6


### Logging Config ###
working_directory = "work_dir"
experiment_name = "Seq2Seq_Neural_Machine_Translation"
logging_interval = 1

### Resume from checkpoint ###
resume_from_checkpoint = None

#######################
### Training Script ###
#######################

### Prepare Accelerator ###
path_to_experiment = os.path.join(working_directory, experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb")
accelerator.init_trackers(experiment_name)

### Load Model Config ###
config = TransformerConfig(
    embedding_dimension=embedding_dimension, 
    num_attention_heads=num_attention_heads, 
    attention_dropout_p=attention_dropout_p, 
    hidden_dropout_p=hidden_dropout_p, 
    mlp_ratio=mlp_ratio, 
    encoder_depth=encoder_depth, 
    decoder_depth=decoder_depth, 
    src_vocab_size=src_tokenizer.vocab_size, 
    tgt_vocab_size=tgt_tokenizer.vocab_size, 
    max_src_len=max_src_len, 
    max_tgt_len=max_tgt_len, 
    learn_pos_embed=learn_pos_embed
)

### Prepare DataLoaders ###
dataset = load_from_disk(path_to_data)
accelerator.print(dataset)

collate_fn = TranslationCollator(src_tokenizer, tgt_tokenizer)
minbatch_size = batch_size // gradient_accumulation_steps

trainloader = DataLoader(dataset["train"], 
                         batch_size=minbatch_size, 
                         num_workers=num_workers, 
                         collate_fn=collate_fn, 
                         shuffle=True)

testloader = DataLoader(dataset["test"], 
                        batch_size=minbatch_size, 
                        collate_fn=collate_fn, 
                        shuffle=False)

### Prepare Model ###
model = Transformer(config)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of Parameters:", params)

### PREPARE OPTIMIZER ###
if not bias_norm_weight_decay:
    accelerator.print("Disabling Weight Decay on Some Parameters")
    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():

        if param.requires_grad:
            
            ### Dont have Weight decay on any bias parameter (including norm) ###
            if "bias" in name:
                no_weight_decay_params.append(param)

            ### Dont have Weight Decay on any Norm scales params (weights) ###
            elif "layernorm" in name:
                no_weight_decay_params.append(param)

            else:
                weight_decay_params.append(param)

    optimizer_group = [
        {"params": weight_decay_params, "weight_decay": weight_decay},
        {"params": no_weight_decay_params, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_group, 
                                  lr=learning_rate,
                                  betas=betas,
                                  eps=adam_eps)

else:
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=learning_rate,
                                  betas=betas,
                                  eps=adam_eps)
    
### Define Scheduler ###
scheduler = get_scheduler(
    name=scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=training_steps,
)

### Define Loss Function ###
loss_fn = torch.nn.CrossEntropyLoss()

### Define a Sample Sentence for Testing ###
src_ids = torch.tensor(src_tokenizer("I want to learn something new")["input_ids"]).unsqueeze(0)

### Prepare Everything ###
model, optimizer, trainloader, testloader = accelerator.prepare(
    model, optimizer, trainloader, testloader
)
accelerator.register_for_checkpointing(scheduler)

### RESUME FROM CHECKPOINT ###
if resume_from_checkpoint is not None:

    ### Grab path to checkpoint ###
    path_to_checkpoint = os.path.join(path_to_experiment, resume_from_checkpoint)
    
    ### Load checkpoint on main process first, recommended here: (https://huggingface.co/docs/accelerate/en/concept_guides/deferring_execution) ###
    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)
    
    ### Start completed steps from checkpoint index ###
    completed_steps = int(resume_from_checkpoint.split("_")[-1])
    accelerator.print(f"Resuming from Iteration: {completed_steps}")
else:
    completed_steps = 0


train = True
progress_bar = tqdm(range(completed_steps, training_steps), disable= not accelerator.is_local_main_process)

while train:

    ### Keep Track of Accumulated Mini-Steps ###
    accumulate_steps = 0
    
    ### Accumulated Loss ###
    accumulate_loss = 0
    
    ### Keep Track of Accuracy ###
    accuracy = 0

    for batch in trainloader:
        
        src_input_ids = batch["src_input_ids"].to(accelerator.device)
        src_pad_mask = batch["src_pad_mask"].to(accelerator.device)
        tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
        tgt_pad_mask = batch["tgt_pad_mask"].to(accelerator.device)
        tgt_outputs = batch["tgt_outputs"].to(accelerator.device)

        output = model(src_input_ids, 
                       tgt_input_ids, 
                       src_pad_mask, 
                       tgt_pad_mask)
        
        ### Flatten for Loss ###
        output = output.flatten(0,1)
        tgt_outputs = tgt_outputs.flatten()
        
        ### Compute Loss ###
        loss = loss_fn(output, tgt_outputs)

        ### Scale Loss and Accumulate ###
        loss = loss / gradient_accumulation_steps
        accumulate_loss += loss

        ### Compute Gradients ###
        accelerator.backward(loss)

        ### Compute Accuracy (ignoring -100 padding labels) ###
        output = output.argmax(axis=-1)
        mask = (tgt_outputs != -100)
        output = output[mask]
        tgt_outputs = tgt_outputs[mask]
        acc = (output == tgt_outputs).sum() / len(output)  
        accuracy += acc / gradient_accumulation_steps 
        
        ### Iterate Accumulation ###
        accumulate_steps += 1

        if accumulate_steps % gradient_accumulation_steps == 0:

            ### Clip and Update Model ###
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            ### Log Results ###
            if completed_steps % logging_interval == 0:

                accumulate_loss = accumulate_loss.detach()
                accuracy = accuracy.detach()

                if accelerator.num_processes > 1:
                     accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))
                     accuracy = torch.mean(accelerator.gather_for_metrics(accuracy))

                log = {"train_loss": accumulate_loss,
                       "training_acc": accuracy,
                       "learning_rate": scheduler.get_last_lr()[0]}
                
                accelerator.log(log, step=completed_steps)

                logging_string = f"[{completed_steps}/{training_steps}] Training Loss: {accumulate_loss} | Training Acc: {accuracy}"

                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                

            if completed_steps % evaluation_steps == 0:

                model.eval()
                
                print("Evaluating!")

                test_losses = []
                test_accs = []

                for batch in tqdm(testloader, disable=not accelerator.is_main_process):

                    src_input_ids = batch["src_input_ids"].to(accelerator.device)
                    src_pad_mask = batch["src_pad_mask"].to(accelerator.device)
                    tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
                    tgt_pad_mask = batch["tgt_pad_mask"].to(accelerator.device)
                    tgt_outputs = batch["tgt_outputs"].to(accelerator.device)

                    with torch.inference_mode():
                        output = model(src_input_ids, 
                                    tgt_input_ids, 
                                    src_pad_mask, 
                                    tgt_pad_mask)
                    
                    ### Flatten for Loss ###
                    output = output.flatten(0,1)
                    tgt_outputs = tgt_outputs.flatten()
                    
                    ### Compute Loss ###
                    loss = loss_fn(output, tgt_outputs)

                    ### Compute Accuracy (make sure to ignore -100 targets) ###
                    output = output.argmax(axis=-1)
                    mask = (tgt_outputs != -100)
                    output = output[mask]
                    tgt_outputs = tgt_outputs[mask]
                    accuracy = (output == tgt_outputs).sum() / len(output)   

                    ### Store Results ###
                    loss = loss.detach()
                    accuracy = accuracy.detach()

                    if accelerator.num_processes > 1:
                        loss = torch.mean(accelerator.gather_for_metrics(loss))
                        accuracy = torch.mean(accelerator.gather_for_metrics(accuracy))
            
                    ### Store Metrics ###
                    test_losses.append(loss.item())
                    test_accs.append(accuracy.item())

                test_loss = np.mean(test_losses)
                test_acc = np.mean(test_accs)

                log = {"test_loss": test_loss,
                        "test_acc": test_acc}   
                
                logging_string = f"Testing Loss: {test_loss} | Testing Acc: {test_acc}"
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                
                ### Log and Save Model ###
                accelerator.log(log, step=completed_steps)
                accelerator.save_state(os.path.join(path_to_experiment, f"checkpoint_{completed_steps}"))

                ### Testing Sentence ###
                if accelerator.is_main_process:
                    src_ids = src_ids.to(accelerator.device)
                    unrwapped = accelerator.unwrap_model(model)
                    translated = unrwapped.inference(src_ids, 
                                                    tgt_start_id=tgt_tokenizer.special_tokens_dict["[BOS]"],
                                                    tgt_end_id=tgt_tokenizer.special_tokens_dict["[EOS]"])
                    
                    translated = tgt_tokenizer.decode(translated, skip_special_tokens=False)

                    if accelerator.is_main_process:
                        progress_bar.write(f"Translation: {translated}")

                model.train()

            if completed_steps >= training_steps:
                train = False
                accelerator.save_state(os.path.join(path_to_experiment, f"final_checkpoint"))
                break
                
            ### Iterate Completed Steps ###
            completed_steps += 1
            progress_bar.update(1)

            ### Reset Accumulated Variables ###
            accumulate_loss = 0
            accuracy = 0

accelerator.end_training()