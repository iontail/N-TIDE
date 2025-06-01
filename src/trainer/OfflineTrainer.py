import os
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.cuda.amp import autocast

from src.utils.bias_metric import *
from src.model.get_model import get_model

class OfflineKDTrainer:
    def __init__(self, 
                 model, 
                 model_type,
                 train_loader,
                 val_loader,
                 optimizer,
                 scheduler,
                 device,
                 args):
        self.args = args
        self.device = device
        
        self.model = model 
        self.model_type = model_type  # "teacher" or "student"

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_epochs = self.args.num_epochs

        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, run_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.gender_criterion = nn.CrossEntropyLoss(label_smoothing=self.args.gender_smoothing)
        self.race_criterion = nn.CrossEntropyLoss(label_smoothing=self.args.race_smoothing)

        if self.model_type == 'student': 
            self.clip_pretrained, _ = get_model(self.args, self.device)
            self.clip_pretrained.load_state_dict(torch.load("~~~~")['model'])
            self.clip_pretrained = self.clip_pretrained.to(device)
            self.clip_pretrained.eval()

    def compute_losses(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        gender_labels, race_labels = labels[:, 1], labels[:, 2]  # [Age, Gender, Race]

        with autocast(device_type='cuda', dtype=torch.bfloat16 if self.args.bf16 else torch.float32):
            output = self.model(images)

            # Classification loss
            cls_g_loss = self.gender_criterion(output['gender_logits'], gender_labels)
            cls_r_loss = self.race_criterion(output['race_logits'], race_labels)
            
            losses = {
                "cls_gender_loss": cls_g_loss,
                "cls_race_loss": cls_r_loss
            }

            if self.model_type == 'teacher':
                # Alignment loss: neutral embeddings <-> bias-included text embeddings (MSE)
                align_loss = F.mse_loss(output['f_neutral'], output['f_biased'].detach())
                losses["feature_loss"] = align_loss
                losses["total_loss"] = (1 - self.args.c_lambda) * (cls_g_loss + cls_r_loss) + self.args.c_lambda * align_loss

            elif self.model_type == 'student': 
                # Knowledge distillation loss: CLIP's features <-> CV model's features (cosine similarity)
                with torch.no_grad():
                    clip_output = self.clip_pretrained(images)
                cosine_sim = F.cosine_similarity(output["features"], clip_output["f_neutral"].detach(), dim=-1)
                kd_loss = 1 - cosine_sim.mean()
                losses["feature_loss"] = kd_loss
                losses["total_loss"] = (1 - self.args.m_lambda) * (cls_g_loss + cls_r_loss) + self.args.m_lambda * kd_loss
            
        logits = {
            "model_gender": output["gender_logits"],
            "model_race": output["race_logits"],
        }

        return losses, logits
    
    def compute_accuracy(self, logits, labels):
        gender_labels = labels[:, 1].to(self.device)
        race_labels = labels[:, 2].to(self.device)

        gender_preds = logits['model_gender'].argmax(dim=1)
        race_preds = logits['model_race'].argmax(dim=1)

        gender_correct = (gender_preds == gender_labels).sum().item()
        race_correct = (race_preds == race_labels).sum().item()

        return gender_correct, race_correct

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        gender_correct, race_correct, total = 0, 0, 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)):
            losses, logits = self.compute_losses(batch)
            loss = losses['total_loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # Accuracy
            _, labels = batch
            g_corr, r_corr = self.compute_accuracy(logits, labels)
            gender_correct += g_corr
            race_correct += r_corr
            total += labels.size(0)

            if self.args.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "step": epoch * len(self.train_loader) + batch_idx,
                    "train/batch/total_loss": losses['total_loss'].item(),
                    "train/batch/cls_gender_loss": losses['cls_gender_loss'].item(), 
                    "train/batch/cls_race_loss": losses['cls_race_loss'].item(), 
                    "train/batch/feature_loss": losses['feature_loss'].item(),                    
                })

        self.scheduler.step()

        train_loss /= len(self.train_loader)
        train_gender_acc = gender_correct / total
        train_race_acc = race_correct / total

        train_results = {
            'train_loss': train_loss,
            'train_gender_acc': train_gender_acc,
            'train_race_acc': train_race_acc
        }

        eval_results = self.evaluate()
        return train_results, eval_results

    def evaluate(self):
        self.model.eval()
        eval_loss = 0.0
        gender_correct, race_correct, total = 0, 0, 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluation", leave=False):
                losses, logits = self.compute_losses(batch)
                eval_loss += losses['total_loss'].item()

                _, labels = batch
                g_corr, r_corr = self.compute_accuracy(logits, labels)
                gender_correct += g_corr
                race_correct += r_corr
                total += labels.size(0)

        eval_loss /= len(self.val_loader)
        gender_acc = gender_correct / total
        race_acc = race_correct / total

        eval_results = {
            'eval_loss': eval_loss,
            'eval_gender_acc': gender_acc,
            'eval_race_acc': race_acc
        }

        return eval_results

    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"N-TIDE_{self.model_type}_E{epoch}.pt"))

    def train(self):
        for epoch in range(self.num_epochs):
            train_results, eval_results = self.train_epoch(epoch)

            if self.args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/train_loss": train_results['train_loss'],
                    "epoch/train_gender_acc": train_results['train_gender_acc'],
                    "epoch/train_race_acc": train_results['train_race_acc'],
                    'epoch/eval_loss': eval_results['eval_loss'],
                    'epoch/eval_gender_acc': eval_results['eval_gender_acc'],
                    'epoch/eval_race_acc': eval_results['eval_race_acc']
                })

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)
        self.save_checkpoint(self.num_epochs)
