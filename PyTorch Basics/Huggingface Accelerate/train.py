import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.models import resnet50
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import Accuracy
from utils import LocalLogger

import warnings 
warnings.filterwarnings("ignore")

### Parse Training Arguments ###
parser = argparse.ArgumentParser(description="Arguments for Image Classification Training")
parser.add_argument("--experiment_name", 
                    help="Name of Experiment being Launched", 
                    required=True, 
                    type=str)
parser.add_argument("--path_to_data", 
                    help="Path to ImageNet root folder which should contain \train and \validation folders", 
                    required=True, 
                    type=str)
parser.add_argument("--working_directory", 
                    help="Working Directory where checkpoints and logs are stored, inside a \
                    folder labeled by the experiment name", 
                    required=True, 
                    type=str)
parser.add_argument("--epochs",
                    help="Number of Epochs to Train",
                    default=90, 
                    type=int)
parser.add_argument("--save_checkpoint_interval", 
                    help="After how many epochs to save model checkpoints",
                    default=10,
                    type=int)
parser.add_argument("--num_classes", 
                    help="How many classes is our network predicting?",
                    default=1000,
                    type=int)
parser.add_argument("--batch_size", 
                    help="Effective batch size. If split_batches is false, batch size is \
                         multiplied by number of GPUs utilized ", 
                    default=64, 
                    type=int)
parser.add_argument("--gradient_accumulation_steps", 
                    help="Number of Gradient Accumulation Steps for Training", 
                    default=1, 
                    type=int)
parser.add_argument("--learning_rate", 
                    help="Starting Learning Rate for StepLR", 
                    default=0.1,
                    type=float)
parser.add_argument("--weight_decay", 
                    help="Weight decay for optimizer", 
                    default=1e-4, 
                    type=float)
parser.add_argument("--momentum",
                    help="Momentum parameter for SGD optimizer",
                    default=0.9, 
                    type=float)
parser.add_argument("--step_lr_decay",
                    help="Decay for Step LR", 
                    default=0.1, 
                    type=float)
parser.add_argument("--lr_step_size",
                    help="Number of epochs for every step", 
                    default=30, 
                    type=int)
parser.add_argument("--lr_warmup_start_factor",
                    help="Learning rate start factor (i.e if learning rate is 0.1 and start factor is 0.01, then lr warm-up from 0.1*0.01 to 0.1)",
                    default=0.1, 
                    type=float)
parser.add_argument("--bias_weight_decay",
                    help="Apply weight decay to bias",
                    default=False, 
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--norm_weight_decay",
                    help="Apply weight decay to normalization weight and bias",
                    default=False, 
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--max_grad_norm", 
                    help="Maximum norm for gradient clipping", 
                    default=1.0, 
                    type=float)
parser.add_argument("--img_size", 
                    help="Width and Height of Images passed to model", 
                    default=224, 
                    type=int)
parser.add_argument("--num_workers", 
                    help="Number of workers for DataLoader", 
                    default=32, 
                    type=int)
parser.add_argument("--resume_from_checkpoint", 
                    help="Checkpoint folder for model to resume training from, inside the experiment folder", 
                    default=None, 
                    type=str)
args = parser.parse_args()

### Init Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          log_with="wandb")

### Init Logger ###
local_logger = LocalLogger(path_to_experiment)

### Weights and Biases Logger ###
experiment_config = {"epochs": args.epochs,
                     "effective_batch_size": args.batch_size*accelerator.num_processes, 
                     "learning_rate": args.learning_rate}
accelerator.init_trackers(args.experiment_name, config=experiment_config)

### Define Accuracy Metric ###
accuracy_fn = Accuracy(task="multiclass", num_classes=args.num_classes).to(accelerator.device)

### Load Model ###
model = resnet50()
if args.num_classes != 1000:
    ### Replace prediction head with nuber of classes ###
    model.fc = nn.Linear(512, args.num_classes)

### Set Transforms for Training and Testing ###
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(args.img_size,args.img_size)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((256,256)), 
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
)

### Load Dataset ###
path_to_train_data = os.path.join(args.path_to_data, "train")
path_to_valid_data = os.path.join(args.path_to_data, "validation")
trainset = datasets.ImageFolder(path_to_train_data, transform=train_transforms)
testset = datasets.ImageFolder(path_to_valid_data, transform=test_transform)

mini_batchsize = args.batch_size // args.gradient_accumulation_steps 
trainloader = DataLoader(trainset, batch_size=mini_batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True)
testloader = DataLoader(testset, batch_size=mini_batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True)

### Define Loss Function ###
loss_fn = nn.CrossEntropyLoss()

### Define Optimizer (And seperate out weight decay and no weight decay parameters) ###
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
            elif "bn" in name and not args.norm_weight_decay:
                no_weight_decay_params.append(param)

            else:
                weight_decay_params.append(param)

    optimizer_group = [
        {"params": weight_decay_params, "weight_decay": args.weight_decay},
        {"params": no_weight_decay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.SGD(optimizer_group, lr=args.learning_rate, momentum=args.momentum)

else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

### Define Scheduler ###
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.step_lr_decay)

### Prepare Everything ###
model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
    model, optimizer, trainloader, testloader, scheduler
)
accelerator.register_for_checkpointing(scheduler)

### Check if we are resuming from checkpoint ###
if args.resume_from_checkpoint is not None:
    accelerator.print(f"Resuming from Checkpoint: {args.resume_from_checkpoint}")
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    accelerator.load_state(path_to_checkpoint)
    starting_checkpoint = int(args.resume_from_checkpoint.split("_")[-1])
else:
    starting_checkpoint = 0

all_train_losses, all_test_losses = [], []
all_train_accs, all_test_accs = [], []

for epoch in range(starting_checkpoint, args.epochs):
    
    accelerator.print(f"Training Epoch {epoch}")

    ### Storage for Everything ###
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    accumulated_loss = 0 
    accumulated_accuracy = 0

    ### Training Progress Bar ###
    progress_bar = tqdm(range(len(trainloader)//args.gradient_accumulation_steps), 
                        disable=not accelerator.is_local_main_process)

    model.train()
    for images, targets in trainloader:

        ### Move Data to Correct GPU ###
        images, targets = images.to(accelerator.device), targets.to(accelerator.device)
        
        with accelerator.accumulate(model):
            
            ### Pass Through Model ###
            pred = model(images)

            ### Compute and Store Loss ##
            loss = loss_fn(pred, targets)
            accumulated_loss += loss / args.gradient_accumulation_steps

            ### Compute and Store Accuracy ###
            predicted = pred.argmax(axis=1)
            accuracy = accuracy_fn(predicted, targets)
            accumulated_accuracy += accuracy / args.gradient_accumulation_steps

            ### Compute Gradients ###
            accelerator.backward(loss)

            ### Clip Gradients ###
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            ### Update Model ###
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


        ### Only when GPUs are being synchronized (When all the grad accumulation is done) store metrics ###
        if accelerator.sync_gradients:
            
            ### Gather Metrics Across GPUs ###
            loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
            accuracy_gathered = accelerator.gather_for_metrics(accumulated_accuracy)

            ### Store Current Iteration Error ###
            train_loss.append(torch.mean(loss_gathered).item())
            train_acc.append(torch.mean(accuracy_gathered).item())

            ### Reset Accumulated for next Accumulation ###
            accumulated_loss, accumulated_accuracy = 0, 0

            ### Iterate Progress Bar ###
            progress_bar.update(1)


    model.eval()
    for images, targets in tqdm(testloader, disable=not accelerator.is_local_main_process):
        images, targets = images.to(accelerator.device), targets.to(accelerator.device)
        with torch.no_grad():
            pred = model(images)

        ### Compute Loss ###
        loss = loss_fn(pred, targets)

        ### Computed Accuracy ###
        predicted = pred.argmax(axis=1)
        accuracy = accuracy_fn(predicted, targets)

        ### Gather across GPUs ###
        loss_gathered = accelerator.gather_for_metrics(loss)
        accuracy_gathered = accelerator.gather_for_metrics(accuracy)

        ### Store Current Iteration Error ###
        test_loss.append(torch.mean(loss_gathered).item())
        test_acc.append(torch.mean(accuracy_gathered).item())
    
    epoch_train_loss = np.mean(train_loss)
    epoch_test_loss = np.mean(test_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_test_acc = np.mean(test_acc)

    all_train_losses.append(epoch_train_loss)
    all_test_losses.append(epoch_test_loss)
    all_train_accs.append(epoch_train_acc)
    all_test_accs.append(epoch_test_acc)

    accelerator.print(f"Training Accuracy: ", epoch_train_acc, "Training Loss:", epoch_train_loss)
    accelerator.print(f"Testing Accuracy: ", epoch_test_acc, "Testing Loss:", epoch_test_loss)

    ### Log with Local Logger ###
    if accelerator.is_main_process:
        local_logger.log(epoch=epoch, 
                         train_loss=epoch_train_loss,
                         test_loss=epoch_test_loss, 
                         train_acc=epoch_train_acc,
                         test_acc=epoch_test_acc)
        
    ### Log with Weights and Biases ###
    accelerator.log({"learning_rate": scheduler.get_last_lr()[0],
                     "training_loss": epoch_train_loss,
                     "testing_loss": epoch_test_loss, 
                     "training_acc": epoch_train_acc, 
                     "testing_acc": epoch_test_acc}, step=epoch)
    
    ### Iterate Learning Rate Scheduler ###
    scheduler.step()

    ### Checkpoint Model ###
    if epoch % args.save_checkpoint_interval == 0:
        path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{epoch}")
        accelerator.save_state(output_dir=path_to_checkpoint)

### End Training for Trackers to Exit ###
accelerator.end_training()