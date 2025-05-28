import os
import time
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.amp import autocast

from src.utils.bias_metric import *

class OfflineKDTrainer:
    def __init__(self, 
                 model, 
                 model_type,
                 train_loader,
                 val_loader,
                 optimizer,
                 device,
                 args,
                 ):
        self.args = args
        
        self.model = model 
        self.model_type = model_type # CLIP or CV model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.device = device

        self.num_epochs = self.args.num_epochs

        self.checkpoint_dir = self.args.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def compute_losses(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        gender_labels, race_labels = labels[:, 1], labels[:, 2] # labels: [Batch_size, 3] ->  [Age, Gender, Race]

        with autocast('cuda', dtype=torch.bfloat16 if self.args.bf16 else torch.float32):
            g_logits, r_logits, features = self.model(images)

            # Classification losses
            cls_g_loss = F.cross_entropy(g_logits, gender_labels)
            cls_r_loss = F.cross_entropy(r_logits, race_labels)

            if self.model_type == 'teacher':
                # Alignment loss: neutral embeddings <-> bias-included text embeddings
                # clip.neutral_embedding: shape [1, D]
                # clip.text_embeddings: shape [C, D]
                C = self.clip.text_embeddings.size(0)
                mse_loss = F.mse_loss(self.model.neutral_embedding.expand(C, -1), self.model.text_embeddings.detach())
                lambda_ = self.args.c_lambda

            elif self.model_type == 'student': 
                # pretrained CLIP load and extract featuers 
                clip_features = None

                # Knowledge distillation loss: CLIP's features <-> CV model's features 
                mse_loss = F.mse_loss(features, clip_features)
                lambda_ = self.args.m_lambda

            # Modle's Total loss
            total_loss = (1 - lambda_) * (cls_g_loss + cls_r_loss) + lambda_ * mse_loss
            

        losses = {
            "cls_gender_loss": cls_g_loss,
            "cls_race_loss": cls_r_loss,
            "mse_loss": mse_loss,
            "total_loss": total_loss
        }

        logits = {
            "model_gender": g_logits,
            "model_race": r_logits
        }

        return losses, logits

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
            losses, logits = self.compute_losses(batch)
            loss = losses['total_loss']

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss += loss.item()

            if self.args.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "train/batch_loss": loss.item(),

                    "train/cls_gender_loss": losses['cls_gender_loss'].item(),
                    "train/cls_race_loss": losses['cls_race_loss'].item(),
                    "train/mse_loss": losses['mse_loss'].item(),
                    
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_idx
                })

        train_loss /= len(self.train_loader)
        eval_losses = self.evaluate()

        return train_loss, eval_losses

    def evaluate(self):
        self.model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluation"):
                losses, logits = self.compute_losses(batch)
                eval_loss += losses['total_loss'].item()
                
        # Accuracy, Bias metric 등도 추가해야 함.
        eval_loss /= len(self.val_loader)

        eval_losses = { 
            'eval_loss': eval_loss, 
            'cls_gender_loss': losses['cls_gender_loss'].item(),
            'cls_race_loss': losses['cls_race_loss'].item(),
            'mse_loss': losses['mse_loss'].item()
        }

        return eval_losses


    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"N-TIDE_{self.model_type}_E{epoch}.pt"))

    def train(self):
        for epoch in range(self.num_epochs):
            start_time = time.time()
            train_loss, eval_losses = self.train_epoch(epoch)

            if self.args.use_wandb:
                wandb.log({
                    "epoch/time": (time.time() - start_time),
                    "epoch/total_loss": train_loss,
                    'epcoh/eval_loss': eval_losses['total_loss'],
                })

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)

        self.save_checkpoint(epoch + 1)
