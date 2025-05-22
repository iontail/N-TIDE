import os
import time
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.amp import autocast

from src.utils.bias_metric import *

class Trainer:
    def __init__(self, 
                 clip, 
                 model,
                 train_loader,
                 val_loader,
                 c_optimizer,
                 m_optimizer,
                 device,
                 args,
                 ):
        self.args = args

        self.clip = clip
        self.model = model 

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.c_optimizer = c_optimizer
        self.m_optimizer = m_optimizer
        self.device = device

        self.num_epochs = self.args.num_epochs

        self.checkpoint_dir = self.args.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def compute_losses(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        gender_labels, race_labels = labels[:, 1], labels[:, 2] # labels: [Batch_size, 3] ->  [Age, Gender, Race]

        with autocast('cuda', dtype=torch.bfloat16 if self.args.bf16 else torch.float32):
            # Forward pass
            gender_logits, race_logits = self.clip(images)

            # Classification losses
            gender_loss = F.cross_entropy(gender_logits, gender_labels)
            race_loss = F.cross_entropy(race_logits, race_labels)

            # Alignment loss: neutral embeddings <-> bias-included text embeddings
            # clip.neutral_embedding: shape [1, D]
            # clip.text_embeddings: shape [C, D]
            C = self.clip.text_embeddings.size(0)
            cosine_sim = F.cosine_similarity(self.clip.neutral_embedding.expand(C, -1), self.clip.text_embeddings.detach(), dim=-1)
            align_loss = -cosine_sim.mean()

            # CLIP's fine-tuning Total loss
            clip_loss = (1 - self.args.c_lambda) * (gender_loss + race_loss) + self.args.c_lambda * align_loss

        clip_losses = {
            "gender_loss": gender_loss,
            "race_loss": race_loss,
            "align_loss": align_loss,
            "total_loss": clip_loss
        }

        clip_logits = {
            "gender": gender_logits,
            "race": race_logits
        }

        # CLIP, CV Model 훈련 동시에 하면서, Knowledge Distillation 하려면,
        # CV Model's logits, losses 계산하는 부분 추가해야 함. 
        # 일단, CLIP's fine-tuning만 작성함. 
        return clip_losses, clip_logits

    def train_epoch(self, epoch):
        self.clip.train()
        train_loss = 0.0
        eval_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
            clip_losses, clip_logits = self.compute_losses(batch)
            clip_loss = clip_losses['total_loss']

            self.c_optimizer.zero_grad()
            clip_loss.backward() 
            torch.nn.utils.clip_grad_norm_(self.clip.parameters(), max_norm=1.0)
            self.c_optimizer.step()

            train_loss += clip_loss.item()

            if self.args.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "train/batch_loss": clip_loss.item(),
                    "train/gender_loss": clip_losses['gender_loss'].item(),
                    "train/race_loss": clip_losses['race_loss'].item(),
                    "train/align_loss": clip_losses['align_loss'].item(),
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_idx
                })

        train_loss /= len(self.train_loader)
        eval_loss = self.evaluate()

        return train_loss, eval_loss

    def evaluate(self):
        self.clip.eval()
        eval_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluation"):
                clip_losses, clip_logits = self.compute_losses(batch)
                clip_loss = clip_losses['total_loss']
                eval_loss += clip_loss.item()

        # Bias metric 추가해야 함.
        eval_loss /= len(self.val_loader)
        return eval_loss


    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "clip": self.clip.state_dict(),
            "c_optimizer": self.c_optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"CLIP-{self.args.clip_backbone}.pt"))

    def train(self):
        for epoch in range(self.num_epochs):
            start_time = time.time()
            train_loss, eval_loss = self.train_epoch(epoch)
            if self.args.use_wandb:
                wandb.log({
                    "epoch/time": (time.time() - start_time),
                    "epoch/avg_loss": train_loss,
                    'eval/avg_loss': eval_loss
                })
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)

        self.save_checkpoint(epoch + 1)
