import os
import time
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.amp import autocast

from src.utils.bias_metric import *

class OnlineKDTrainer:
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
            clip_g_logits, clip_r_logits, clip_features = self.clip(images)
            model_g_logits, model_r_logits, model_features = self.model(images)

            # Classification losses
            clip_g_loss = F.cross_entropy(clip_g_logits, gender_labels)
            clip_r_loss = F.cross_entropy(clip_r_logits, race_labels)

            model_g_loss = F.cross_entropy(model_g_logits, gender_labels)
            model_r_loss = F.cross_entropy(model_r_logits, race_labels)

            # Alignment loss: neutral embeddings <-> bias-included text embeddings
            # clip.neutral_embedding: shape [1, D]
            # clip.text_embeddings: shape [C, D]
            C = self.clip.text_embeddings.size(0)
            cosine_sim = F.cosine_similarity(self.clip.neutral_embedding.expand(C, -1), self.clip.text_embeddings.detach(), dim=-1)
            align_loss = -cosine_sim.mean()

            # Perceptual loss: CLIP's features <-> Models' features
            # CLIP's neutral features -> Knowledge distillation -> Models's features

            # Teacher 역할을 하는 CLIP이 계속 바뀌고 있으므로, Student인 CV Model이 안정적으로 학습하기 어려울 수 있음.
            # -> Student's feature는 CLIP's feature의 EMA 된 것을 따라가도록 하면 안정되지 않을까??? 
            pcp_loss = F.mse_loss(model_features, clip_features.detach()) # 일단 MSE로 함. 

            # CLIP's fine-tuning Total loss
            clip_loss = (1 - self.args.c_lambda) * (clip_g_loss + clip_r_loss) + self.args.c_lambda * align_loss

            # Modle's fine-tuning and distillation Total los
            model_loss = (1 - self.args.m_lambda) * (model_g_loss + model_r_loss) + self.args.m_lambda * pcp_loss

        losses = {
            "clip_gender_loss": clip_g_loss,
            "clip_race_loss": clip_r_loss,
            "clip_align_loss": align_loss,
            "clip_loss": clip_loss,
            "model_gender_loss": model_g_loss,
            "model_race_loss": model_r_loss,
            "model_pcp_loss": pcp_loss,
            "model_loss": model_loss
        }

        logits = {
            "clip_gender": clip_g_logits,
            "clip_race": clip_r_logits,
            "model_gender": model_g_logits,
            "model_race": model_r_logits
        }

        return losses, logits

    def train_epoch(self, epoch):
        self.clip.train()
        self.model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
            losses, logits = self.compute_losses(batch)
            loss = self.args.alpha * losses['clip_loss'] + (1-self.args.alpha) * losses['model_loss'] 

            self.m_optimizer.zero_grad()
            self.c_optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.clip.parameters(), max_norm=1.0)

            self.m_optimizer.step()
            self.c_optimizer.step()

            train_loss += loss.item()

            if self.args.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/clip_loss": losses['clip_loss'].item(),
                    "train/model_loss": losses['model_loss'] .item(),

                    "train/clip_gender_loss": losses['clip_gender_loss'].item(),
                    "train/clip_race_loss": losses['clip_race_loss'].item(),
                    "train/clip_align_loss": losses['clip_align_loss'].item(),

                    "train/model_gender_loss": losses['model_gender_loss'].item(), 
                    "train/model_race_loss": losses['model_race_loss'].item(),
                    "train/model_pcp_loss": losses['pcp_loss'].item(),

                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_idx
                })

        train_loss /= len(self.train_loader)
        eval_losses = self.evaluate()

        return train_loss, eval_losses

    def evaluate(self):
        self.clip.eval()
        self.model.eval()
        clip_eval_loss = 0.0
        model_eval_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluation"):
                losses, logits = self.compute_losses(batch)
                clip_cls_loss = losses['clip_gender_loss'] + losses['clip_race_loss']
                model_cls_loss = losses['model_gender_loss'] + losses['model_race_loss']
                clip_eval_loss += clip_cls_loss.item()
                model_eval_loss += model_cls_loss.item()

        # Accuracy, Bias metric 등도 추가해야 함.
        clip_eval_loss /= len(self.val_loader)
        model_eval_loss /= len(self.val_loader)

        eval_losses = { 
            'clip_cls_loss': clip_eval_loss,
            'model_eval_loss': model_eval_loss
        }

        return eval_losses


    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "clip": self.clip.state_dict(),
            "model": self.model.state_dict(),
            "c_optimizer": self.c_optimizer.state_dict(),
            "m_optimizer": self.m_optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"N-TIDE_E{epoch}.pt"))

    def train(self):
        for epoch in range(self.num_epochs):
            start_time = time.time()
            train_loss, eval_losses = self.train_epoch(epoch)

            if self.args.use_wandb:
                wandb.log({
                    "epoch/time": (time.time() - start_time),
                    "epoch/total_loss": train_loss,
                    'eval/clip_cls_loss': eval_losses['clip_cls_loss'],
                    'eval/model_cls_loss': eval_losses['model_cls_loss'],
                })

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)

        self.save_checkpoint(epoch + 1)
