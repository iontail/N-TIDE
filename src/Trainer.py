import os
import time
import wandb
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss, CrossEntropyLoss
from torch.amp import autocast

from utils.bias_metric import *

class Trainer:
    def __init__(self, 
                 clip, 
                 model,
                 train_loader,
                 val_loader,
                 c_optimizer,
                 r_optimizer,
                 device,
                 args,
                 epochs=100,
                 eval_steps=500,
                 checkpoint_dir="./ckpt",
                 use_wandb=True,
                 project_name="MB_SLM",
                 run_name=None
                 ):
        self.args = args

        self.clip = clip
        self.model = model 

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.c_optimizer = c_optimizer
        self.r_optimizer = r_optimizer
        self.device = device

        self.epochs = epochs
        self.eval_steps = eval_steps

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.use_wandb = use_wandb

        self.ce_loss = CrossEntropyLoss()
        self.l1_loss = L1Loss()

    def compute_losses(self, batch):
        img, labels = batch
        img = img.to(self.device)
        labels = labels.to(self.device)

        # 설정 파일(ags)에서 지정한 인덱스로 주요 타겟 레이블 선택
        target = labels[:, self.args.dataset_target_label_idx]
        with autocast('cuda', dtype=torch.bfloat16 if self.args.bf16 else torch.float32):
            logits, features = self.model(img)

            # SLM 통과하는 부분 추가해줘야 할듯
            ce_loss = self.ce_loss(logits, target)

            r1_loss = torch.tensor(0.0).to(img.device) # R1 loss는 현재 사용하지 않음
            model_loss = ce_loss + self.args.r1_lambda * r1_loss

        losses = {
            "ce_loss": ce_loss,
            'r1_loss': r1_loss,
            "loss": model_loss
        }

        return model_loss, losses, logits


    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        eval_loss = 0.0

        epoch_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", unit="epoch", position=0, leave=True)
        for batch_idx, batch in enumerate(epoch_bar):
            self.model.train()

            model_loss, losses, outputs = self.compute_losses(batch)
            model_loss.backward() 

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += model_loss.item()

            if self.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "train/batch_loss": model_loss.item(),
                    "train/r1_loss": losses['r1_loss'].item(),
                    "train/ce_loss": losses['ce_loss'].item(),
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_idx
                })

            if batch_idx + 1 == len(self.train_loader):
                eval_loss = self.evaluate()
            epoch_bar.set_postfix(train_loss=f"{total_loss/((batch_idx +1) * self.args.batch_size):.6f}", eval_loss=f"{eval_loss/len(self.val_loader):.6f}")

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        log_images = []
        with torch.no_grad():
            cnt = 0
            for batch in self.val_loader:
                model_loss, losses, outputs = self.compute_losses(batch)
                total_loss += model_loss.item()
                cnt += 1

        if self.use_wandb:
            wandb.log({
                "eval/avg_loss": total_loss / cnt, # Use average loss
                "eval/r1_loss": losses['r1_loss'].item(), # Note: losses from the *last* batch
                "eval/ce_loss": losses['ce_loss'].item(),
            })

        if self.use_wandb and log_images:
            wandb.log({
                "eval/images": [wandb.Image(img) for img in log_images]
            })

        return total_loss

    def save_checkpoint(self, epoch):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"{self.args.model_name}.pt"))

    def train(self):
        for epoch in range(self.epochs):
            start_time = time.time()
            avg_loss = self.train_epoch(epoch)
            if self.use_wandb:
                wandb.log({
                    "epoch/avg_loss": avg_loss,
                    "epoch/time": time.time() - start_time
                })
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)

        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"{self.args.model_name}.pt"))