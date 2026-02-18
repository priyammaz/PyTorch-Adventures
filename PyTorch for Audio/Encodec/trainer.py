import os
from pathlib import Path
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import accelerate
from accelerate import Accelerator
import argparse 
import random
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from dataset import AudioDataset
from loss import generator_loss, discriminator_loss
from utils import load_audios, save_audios
from balancer import Balancer

torch.backends.cudnn.benchmark = True

class EncodecTrainer:

    def __init__(self, 
                 generator, 
                 discriminator, 
                 training_config):
        
        ######################
        ### TRAINING SETUP ###
        ######################

        ### EXPERIMENT SETUP ###
        self.experiment_name = training_config.get("experiment_name", "EnCodecTrainer")
        self.working_directory = training_config.get("working_directory", "work_dir")

        ### DATA SOURCE SETUP ###
        self.path_to_audio_dir = training_config.get("path_to_audio_dir", None)
        self.path_to_train_manifest = training_config.get("path_to_train_manifest", None)
        self.path_to_val_manifest = training_config.get("path_to_test_manifest", None)
        self.test_pct = training_config.get("test_pct", None)

        ### AUDIO PROCESSING SETUP ###
        self.sampling_rate = training_config.get("sampling_rate", 24000)
        self.segment_length = training_config.get("segment_length", 24000)
        self.num_mels = training_config.get("num_mels", 80)

        ### TRAINING SETUP ###
        self.total_iterations = training_config.get("total_iterations", 500000)
        self.warmup_iterations = training_config.get("warmup_iterations", 5000)
        self.checkpoint_iterations = training_config.get("checkpoint_iterations", 25000)
        self.eval_iterations = training_config.get("eval_iterations", 2500)
        self.generation_iterations = training_config.get("generation_iterations", 2500)
        self.per_gpu_batch_size = training_config.get("per_gpu_batch_size", 16)
        self.num_workers = training_config.get("num_workers", 16)
        self.pin_memory = training_config.get("pin_memory", False)
        self.max_grad_norm = training_config.get("max_grad_norm", None)

        ### LOGGING SETUP ###
        self.console_out_iters = training_config.get("console_out_iters", 5)
        self.log_wandb = training_config.get("log_wandb", False)
        self.wandb_log_iters = training_config.get("wandb_log_iters", 5)
        
        ### OPTIMIZER SETUP ###
        self.learning_rate = training_config.get("learning_rate", 1e-4)
        self.disc_learning_rate = training_config.get("disc_learning_rate", 1e-4)
        self.disc_update_prob = training_config.get("disc_update_prob", 0.666)
        self.beta1 = training_config.get("beta1", 0.5)
        self.beta2 = training_config.get("beta2", 0.9)

        ### LOSS SETUP ###
        self.time_loss_lambda = training_config.get("time_loss_lambda", 0.1)
        self.frequency_loss_lambda = training_config.get("frequency_loss_lambda", 1.0)
        self.feature_loss_lambda = training_config.get("feature_loss_lambda", 3.0)
        self.generator_loss_lambda = training_config.get("generator_loss_lambda", 3.0)
        self.quantizer_loss_lambda = training_config.get("quantizer_loss_lambda", 1.0)
        self.use_balancer = training_config.get("use_balancer", True)

        ### INFERENCE SETUP ###
        self.num_samples_for_reconstruction = training_config.get("num_samples_for_reconstruction", 10)

        ### Seed ###
        self.seed = training_config.get("seed", None)

        ### Prepare Accelerator ###
        self.path_to_experiment = self._get_path_to_experiment()
        os.makedirs(self.path_to_experiment, exist_ok=True)
        self.accelerator = Accelerator(project_dir=self.path_to_experiment, 
                                       log_with="wandb" if self.log_wandb else None)
        if self.log_wandb: ### ADD CONFIG
            self.accelerator.init_trackers(self.experiment_name)

        ### Store ###
        self.generator = generator
        self.discriminator = discriminator
        self.training_config = training_config

        if self.discriminator is None:
            self.accelerator.print("TRAINING WITHOUT DISCRIMINATOR!!!")

    def _get_path_to_experiment(self):
        return os.path.join(self.working_directory, self.experiment_name)

    def _find_audio_files(self):
        """extract audio files with durations longer than segment_length / sampling_rate"""
        extensions = {".wav", ".mp3",".flac"}
        min_duration = self.segment_length / self.sampling_rate

        audio_files = []
        iterator = Path(self.path_to_audio_dir).rglob("")

        for path in iterator:
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            try:
                duration = librosa.get_duration(path=str(path))
                if duration >= min_duration:
                    audio_files.append(str(path))
            except:
                pass
        
        self.accelerator.print(f"Found {len(audio_files)} Audio Files")
        
        return audio_files
    
    def _load_paths_from_text(self, path):
        with open(path, 'r') as f:
            audio_paths = [line.strip() for line in f if line.strip()]
        return audio_paths
    
    def _write_list_to_text(self, list_, path):
        with open(path, "w") as f:
            for l in list_:
                f.write(l+"\n")

    def train_test_split(self, list_, seed=None, test_pct=0.05):
        if seed is not None:
            random.seed(seed)
        random.shuffle(list_)
        num_test = int(self.test_pct * len(list_))

        train_split = list_[:-num_test]
        test_split = list_[-num_test:]

        return train_split, test_split
    
    def _get_datasets(self):

        if (self.path_to_audio_dir is None) == (self.path_to_train_manifest is None):
            raise ValueError(
                "Exactly one of `path_to_audio_dir` or `path_to_manifest` must be provided."
            )
        
        ### If we pass in a directory that contains audio ###
        if self.path_to_audio_dir is not None:
            assert self.test_pct is not None, "If no testing split is provided you must provide test_pct"
            audio_files = self._find_audio_files()
            self.train_files, self.test_files = self.train_test_split(audio_files, self.seed, self.test_pct)

        ### If we pass in a txt file with audio file paths ###
        else:
            audio_files = self._load_paths_from_text(self.path_to_train_manifest)
            if self.path_to_val_manifest is not None:
                self.train_files = audio_files
                self.test_files = self._load_paths_from_text(self.path_to_val_manifest)
            else:
                self.train_files, self.test_files = self.train_test_split(audio_files, self.seed, self.test_pct)

        ### Cache the dataset in our experiment directory so we select the same train/test split during inference time ###
        self._write_list_to_text(self.train_files, os.path.join(self.path_to_experiment, "train.txt"))
        self._write_list_to_text(self.test_files, os.path.join(self.path_to_experiment, "test.txt"))

        ### Create our Datasets ###
        trainset = AudioDataset(audio_paths=self.train_files, 
                                segment_length=self.segment_length, 
                                sample_rate=self.sampling_rate)
        testset = AudioDataset(audio_paths=self.test_files, 
                               segment_length=self.segment_length, 
                               sample_rate=self.sampling_rate)
        
        return trainset, testset

    def count_params(self, model):

        if model is not None:
            total = 0
            for param in model.parameters():
                total += param.numel()
            for param in model.buffers():
                total += param.numel()
            return total
        else:
            return 0
        
    def _print_run_info(self, starting_iteration=0):
        self.accelerator.print("=" * 60)
        self.accelerator.print(f"{'TRAINING CONFIGURATION':^60}")
        self.accelerator.print("=" * 60)
        self.accelerator.print(f"{'Training Samples':<25}: {len(self.train_files):>15,}")
        self.accelerator.print(f"{'Testing Samples':<25}: {len(self.test_files):>15,}")
        self.accelerator.print(f"{'Sampling Rate':<25}: {self.sampling_rate:>15,}")
        self.accelerator.print(f"{'Segment Size':<25}: {self.segment_length:>15,}")
        self.accelerator.print(f"{'Effective Batch Size':<25}: {self.per_gpu_batch_size * self.accelerator.num_processes:>15,}")
        self.accelerator.print(f"{'Num GPUs':<25}: {self.accelerator.num_processes:>15,}")
        self.accelerator.print("-" * 60)
        self.accelerator.print(f"{'Model Parameters':<25}: {self.count_params(self.generator):>15,}")
        self.accelerator.print(f"{'Discriminator Parameters':<25}: {self.count_params(self.discriminator):>15,}")
        self.accelerator.print("-" * 60)
        resume_status = "Yes" if starting_iteration != 0 else "No"
        self.accelerator.print(f"{'Resume From Checkpoint':<25}: {resume_status:>15}")
        self.accelerator.print(f"{'Starting From Iteration':<25}: {starting_iteration:>15}")
        self.accelerator.print("=" * 60)

    def train(self, resume=False):

        ### Load Datasets ###
        if not resume:
            trainset, testset = self._get_datasets()
        
        ### Load our cached files for the same train/test split ###
        else:

            ### Replace the class variables for train/test manifests to the cached files ###
            self.path_to_train_manifest = os.path.join(self.path_to_experiment, "train.txt")
            self.path_to_val_manifest = os.path.join(self.path_to_audio_dir, "test.txt")
            trainset, testset = self._get_datasets()
        
        ### Load Dataloader ###
        trainloader = DataLoader(trainset, batch_size=self.per_gpu_batch_size, 
                                 num_workers=self.num_workers, shuffle=True, 
                                 pin_memory=self.pin_memory)
        testloader = DataLoader(testset, batch_size=self.per_gpu_batch_size, 
                                num_workers=self.num_workers, shuffle=False, 
                                pin_memory=self.pin_memory)
        
        ### Load Cached Audios for Inference ###
        if not resume:
            sampled_paths = random.sample(testset.audio_paths, k=self.num_samples_for_reconstruction)
            self._write_list_to_text(sampled_paths, os.path.join(self.path_to_experiment, "samples.txt"))
        else:
            sampled_paths = self._load_paths_from_text(os.path.join(self.path_to_experiment, "samples.txt"))
        cached_audios = load_audios(sampled_paths, self.sampling_rate)

        ### Load Optimizers ###
        optimizer = torch.optim.Adam([p for p in self.generator.parameters() if p.requires_grad], lr=self.learning_rate)
        
        if self.discriminator is not None:
            disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_learning_rate)

        ### Load Scheduler ###
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.warmup_iterations * self.accelerator.num_processes, 
            num_training_steps=self.total_iterations * self.accelerator.num_processes
        )
        
        if self.discriminator is not None:
            disc_scheduler = get_cosine_schedule_with_warmup(
                disc_optimizer, 
                num_warmup_steps=self.warmup_iterations * self.accelerator.num_processes, 
                num_training_steps=self.total_iterations * self.accelerator.num_processes
            )

        ### Prepare Everything ###
        (
            self.generator, 
            optimizer, 
            scheduler, 
            trainloader, 
            testloader
        ) = self.accelerator.prepare(
            self.generator, 
            optimizer, 
            scheduler, 
            trainloader, 
            testloader
        )

        if self.discriminator is not None:
            (
                self.discriminator, 
                disc_optimizer, 
                disc_scheduler 
            ) = self.accelerator.prepare(
                self.discriminator, 
                disc_optimizer, 
                disc_scheduler
            )

        ### Initialize Balancer ###
        ### Use key names that match our loss function output later ###
        weights = {
            "time_loss": self.time_loss_lambda, 
            "frequency_loss": self.frequency_loss_lambda, 
            "generator_loss": self.generator_loss_lambda,
            "feature_loss": self.feature_loss_lambda
        }

        ### If no discriminator, then all losses involved should have 0 weight ###
        if self.discriminator is None:
            weights["generator_loss"] = 0.0
            weights["feature_loss"] = 0.0

        if self.use_balancer:
            balancer = Balancer(weights=weights, accelerator=self.accelerator)
        
        ### Resume from Checkpoint ###
        if resume:
            last_ckpt = max(
                (d for d in os.listdir(self.path_to_experiment) if d.startswith("checkpoint_")), 
                key=lambda x: int(x.split("_")[1]),
            )
            last_ckpt = os.path.join(self.path_to_experiment, last_ckpt)
            self.accelerator.load_state(last_ckpt)

            if self.use_balancer:
                path_to_balancer_state = os.path.join(last_ckpt, "balancer_state.bin")
                balancer_state = torch.load(path_to_balancer_state)
                balancer.load_state(balancer_state)
            
            completed_steps = int(last_ckpt.split("_")[-1])

        else:
            completed_steps = 0

        self._print_run_info(completed_steps)

        ### Init log ###
        log = {"accum_test_time_loss": [], "accum_test_freq_loss": []}

        train = True
        while train:

            for waveforms in trainloader:
                
                ### Move to GPU ###
                waveforms = waveforms.to(self.accelerator.device)

                ######################
                ### Generator Step ###
                ######################
                optimizer.zero_grad(set_to_none=True)
                
                ### Get output ###
                output = self.generator(waveforms)

                ### pass real and fake into disc ###
                if self.discriminator is not None:
                    logits_real, logits_fake, fmap_real, fmap_fake = self.discriminator(waveforms, output["decoded"])
                else:
                    logits_real = logits_fake = fmap_real = fmap_fake = None

                ### Compute Generator Loss ###
                losses = generator_loss(
                    fmap_real, fmap_fake, logits_fake, 
                    waveforms, output["decoded"], 
                    sample_rate=self.sampling_rate, 
                    num_mels=self.num_mels
                )

                ### Standard backward pass without the balancer ###
                if not self.use_balancer:
                    total_gen_loss = weights["time_loss"] * losses["time_loss"] + \
                                     weights["frequency_loss"] * losses["frequency_loss"] + \
                                     weights["generator_loss"] * losses["generator_loss"] + \
                                     weights["feature_loss"] * losses["feature_loss"] + \
                                     self.quantizer_loss_lambda * output["quantizer_loss"]
                    
                    ### Update model ###
                    self.accelerator.backward(total_gen_loss)
                    
                ### Otherwise use the autobalancer ###
                else:

                    ### Retain graph as we need to do a second backward ###
                    balancer.backward(losses, output["decoded"], retain_graph=True)

                    ### Compute separately grads w.r.t commitment loss ###
                    ### as this only effects the encoder portion of the model ###
                    ### https://github.com/facebookresearch/encodec/issues/20
                    self.accelerator.backward(output["quantizer_loss"])
                
                ### Graient Clipping ###
                if self.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
                
                ### Update the model with the compute grads ###
                optimizer.step()

                ### Update Generator Scheduler ###
                scheduler.step()

                ### Accumulate Loss ###
                if self.accelerator.num_processes > 1:
                    train_time_loss = self.accelerator.gather_for_metrics(losses["time_loss"]).mean().item()
                    train_freq_loss = self.accelerator.gather_for_metrics(losses["frequency_loss"]).mean().item()
                    train_gen_loss = self.accelerator.gather_for_metrics(losses["generator_loss"]).mean().item()
                    train_feat_loss = self.accelerator.gather_for_metrics(losses["feature_loss"]).mean().item()
                    train_commit_loss = self.accelerator.gather_for_metrics(output["quantizer_loss"]).mean().item()
                else:
                    train_time_loss = losses["time_loss"].item()
                    train_freq_loss = losses["frequency_loss"].item()
                    train_gen_loss = losses["generator_loss"].item()
                    train_feat_loss = losses["feature_loss"].item()
                    train_commit_loss = output["quantizer_loss"].item()

                if self.discriminator is not None:
                    ###########################
                    ### Dimscriminator Step ###
                    ###########################

                    disc_optimizer.zero_grad()

                    ### Random sample value between 0 and 1 ###
                    ### Ensure all GPUs have the same value ###
                    rand = torch.rand(size=(), device=self.accelerator.device)
                    rand = accelerate.utils.broadcast(rand, from_process=0)

                    if rand < self.disc_update_prob:
            
                        ### Pass Through Discriminator ###
                        logits_real, logits_fake, _, _ = self.discriminator(waveforms, output["decoded"].detach())

                        ### Compute Discriminator Loss ###
                        disc_loss = discriminator_loss(logits_real, logits_fake)

                        ### Update Disc ###
                        self.accelerator.backward(disc_loss)

                        ### Grad Clipping ###
                        if self.max_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)

                        ### Update Discriminator ###
                        disc_optimizer.step()
                    
                    else:
                        disc_loss = "skipped"
                    
                    ### Update Disc Scheduler anyway even if we skipped ###
                    disc_scheduler.step()

                ### Completed 1 Step of Training ###
                completed_steps += 1
                
                ### Logging ###
                progress = completed_steps / self.total_iterations * 100
                if completed_steps % self.console_out_iters == 0:
                    if self.discriminator is not None:
                        disc_loss_str = f"{disc_loss:.3f}" if isinstance(disc_loss, (int, float, torch.Tensor)) else str(disc_loss)
                        disc_lr_str = f"{disc_scheduler.get_last_lr()[0]:.2e}"
                        train_gen_loss_str = f"{train_gen_loss:.3f}"
                        train_feat_loss_str = f"{train_feat_loss:.3f}"
                    else:
                        disc_loss_str = disc_lr_str = train_gen_loss_str = train_feat_loss_str = "N/A"

                    self.accelerator.print(
                        f"Step {completed_steps:>6} ({progress:5.1f}%) | "
                        f"Time: {train_time_loss:.3f} | "
                        f"Freq: {train_freq_loss:.3f} | "
                        f"Gen: {train_gen_loss_str} | "
                        f"Feat: {train_feat_loss_str} | "
                        f"Disc: {disc_loss_str} | "
                        f"Quant: {train_commit_loss:.2e} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"D_LR: {disc_lr_str}"
                    )

                if (completed_steps % self.wandb_log_iters == 0) and self.log_wandb:

                    log_dict = {
                        "time_loss": train_time_loss,
                        "freq_loss": train_freq_loss, 
                        "gen_loss": train_gen_loss, 
                        "feat_loss": train_feat_loss, 
                        "quant": train_commit_loss,
                        "lr": scheduler.get_last_lr()[0]
                    }

                    if self.discriminator is not None:
                        log_dict["disc_lr"] = disc_scheduler.get_last_lr()[0]
                        if not isinstance(disc_loss, str):
                            log_dict["disc_loss"] = disc_loss

                    self.accelerator.log(log_dict, step=completed_steps)

                ### Evaluation ###
                if completed_steps % self.eval_iterations == 0:
                    
                    ### Throw into evaluation mode ###
                    self.generator.eval()

                    if self.discriminator is not None:
                        self.discriminator.eval()

                    for waveforms in tqdm(testloader, disable=not self.accelerator.is_main_process):
                        
                        with torch.no_grad():

                            ### Pass through generator ###
                            output = self.generator(waveforms)

                            ### pass real and fake into disc ###
                            if self.discriminator is not None:
                                logits_real, logits_fake, fmask_real, fmask_fake = self.discriminator(waveforms, output["decoded"])
                            else:
                                logits_real = logits_fake = fmask_real = fmask_fake = None

                        ### Compute Generator Loss ###
                        losses = generator_loss(
                            fmap_real, fmap_fake, logits_fake, 
                            waveforms, output["decoded"], 
                            sample_rate=self.sampling_rate, 
                            num_mels=self.num_mels 
                        )

                        ### Accumulate Loss ###
                        if self.accelerator.num_processes > 1:
                            test_time_loss = self.accelerator.gather_for_metrics(losses["time_loss"]).mean().item()
                            test_freq_loss = self.accelerator.gather_for_metrics(losses["frequency_loss"]).mean().item()
                        else:
                            test_time_loss = losses["time_loss"].item()
                            test_freq_loss = losses["frequency_loss"].item()
                        
                        log["accum_test_time_loss"].append(test_time_loss)
                        log["accum_test_freq_loss"].append(test_freq_loss)

                    ### Average Loss over Testing Split ###
                    test_time_loss = np.mean(log["accum_test_time_loss"])
                    test_freq_loss = np.mean(log["accum_test_freq_loss"])

                    self.accelerator.print(
                        f"VALIDATION LOSS: "
                        f"Time: {test_time_loss:.3f} | "
                        f"Freq: {test_freq_loss:.3f}"
                    )

                    if self.log_wandb:
                        self.accelerator.log(
                                {"test_time_loss": train_time_loss, "test_freq_loss": train_freq_loss},
                                step=completed_steps
                            )
                    
                    ### Reset log ###
                    log = {"accum_test_time_loss": [], "accum_test_freq_loss": []}

                    ### Set back to training mode ###
                    self.generator.train()

                    if self.discriminator is not None:
                        self.discriminator.train()

                ### Inference our Cached Files ###
                if completed_steps % self.generation_iterations == 0:

                    if self.accelerator.is_main_process:
                        unwrapped_model = self.accelerator.unwrap_model(self.generator)
                        gens = [unwrapped_model.passthrough(c.to(self.accelerator.device)).detach().cpu() for c in cached_audios]
                        
                        ### Save ###
                        path_to_save_dir = os.path.join(self.path_to_experiment, f"gens_iter_{completed_steps}")
                        os.makedirs(path_to_save_dir, exist_ok=True)
                        path_to_saves = [os.path.join(path_to_save_dir, file) for file in [f"gen_{i}.wav" for i in range(len(gens))]]
                        save_audios(gens, path_to_saves, self.sampling_rate)

                    self.accelerator.wait_for_everyone()

                ### Checkpoint Model ###
                if completed_steps % self.checkpoint_iterations == 0:
                    output_dir = os.path.join(self.path_to_experiment, f"checkpoint_{completed_steps}")
                    self.accelerator.save_state(output_dir, safe_serialization=False) # disable to use .bin due to weird lstm weights and safetensors

                    if self.use_balancer:
                        output_balancer = os.path.join(self.path_to_experiment, f"checkpoint_{completed_steps}", "balancer_state.bin")
                        balancer_state = balancer.state_dict()
                        torch.save(balancer_state, output_balancer)

                ### End Training ###
                if completed_steps >= self.total_iterations:
                    train = False
                    self.accelerator.print("COMPLETED TRAINING!!")
                    break 
        
        output_dir = os.path.join(self.path_to_experiment, f"final_checkpoint")
        self.accelerator.save_state(output_dir, safe_serialization=False)

