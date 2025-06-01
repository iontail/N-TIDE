import os
import time
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.cuda.amp import autocast

from src.utils.bias_metric import *

class OnlineKDTrainer:
    def __init__(self, 
                 clip, 
                 model,
                 train_loader,
                 val_loader,
                 c_optimizer,
                 c_scheduler,
                 m_optimizer,
                 m_scheduler,
                 device,
                 args):
        self.args = args
        self.device = device

        self.clip = clip
        self.model = model 

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.c_optimizer = c_optimizer
        self.c_scheduler = c_scheduler
        self.m_optimizer = m_optimizer
        self.m_scheduler = m_scheduler

        self.num_epochs = self.args.num_epochs
        self.checkpoint_dir = self.args.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.gender_criterion = nn.CrossEntropyLoss(label_smoothing=self.args.gender_smoothing)
        self.race_criterion   = nn.CrossEntropyLoss(label_smoothing=self.args.race_smoothing)

    def compute_losses(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        gender_labels, race_labels = labels[:, 1], labels[:, 2]

        with autocast(device_type='cuda', dtype=torch.bfloat16 if self.args.bf16 else torch.float32):
            clip_output = self.clip(images)
            model_output = self.model(images)

            # Classification loss
            clip_g_loss = self.gender_criterion(clip_output["gender_logits"], gender_labels)
            clip_r_loss = self.race_criterion(clip_output["race_logits"], race_labels)

            model_g_loss = self.gender_criterion(model_output["gender_logits"], gender_labels)
            model_r_loss = self.race_criterion(model_output["race_logits"], race_labels)

            # Alignment loss: neutral embeddings <-> bias-included text embeddings (MSE)
            align_loss = F.mse_loss(clip_output['f_neutral'], clip_output['f_biased'].detach())

            # Similarity loss: CLIP's features <-> Models' features (cosine similarity)
            # CLIP's neutral features -> Knowledge distillation -> Models's features
            # Teacher 역할을 하는 CLIP이 계속 바뀌고 있으므로, Student인 CV Model이 안정적으로 학습하기 어려울 수 있음.
            # -> Student's feature는 CLIP's feature의 EMA 된 것을 따라가도록 하면 안정되지 않을까??? 
            cosine_sim = F.cosine_similarity(model_output["features"], clip_output["f_neutral"].detach(), dim=-1)
            kd_loss = 1 - cosine_sim.mean()

            # CLIP's fine-tuning Total loss
            clip_loss = (1 - self.args.c_lambda) * (clip_g_loss + clip_r_loss) + self.args.c_lambda * align_loss
            # Modle's fine-tuning and distillation Total los
            model_loss = (1 - self.args.m_lambda) * (model_g_loss + model_r_loss) + self.args.m_lambda * kd_loss

        losses = {
            "clip_gender_loss": clip_g_loss,
            "clip_race_loss": clip_r_loss,
            "clip_align_loss": align_loss,
            "clip_loss": clip_loss,
            "model_gender_loss": model_g_loss,
            "model_race_loss": model_r_loss,
            "model_kd_loss": kd_loss,
            "model_loss": model_loss
        }

        logits = {
            "clip_gender": clip_output["gender_logits"],
            "clip_race": clip_output["race_logits"],
            "model_gender": model_output["gender_logits"],
            "model_race": model_output["race_logits"]
        }

        return losses, logits

    def train_epoch(self, epoch):
        self.clip.train()
        self.model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)):
            losses, _ = self.compute_losses(batch)
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
                    "train/model_kd_loss": losses['kd_loss'].item(),

                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_idx
                })
                
        self.c_scheduler.step()
        self.m_scheduler.step()

        train_loss /= len(self.train_loader)
        eval_losses = self.evaluate()

        return train_loss, eval_losses

    def evaluate(self):
        self.clip.eval()
        self.model.eval()
        clip_eval_loss = 0.0
        model_eval_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluation", leave=False):
                losses, _ = self.compute_losses(batch)
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
        self.save_checkpoint(self.num_epochs)
