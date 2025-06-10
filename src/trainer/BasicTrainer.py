import os
import wandb
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn 
import torch.nn.functional as F

from src.utils.bias_metrics import compute_bias_metrics

class BasicTrainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, scheduler, device, args, run_name):
        self.args = args
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.num_epochs = self.args.num_epochs  

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.run_name = run_name
        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, run_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.target_attr = 'gender' if self.args.bias_attribute == 'race' else 'race'
        if self.target_attr == 'gender': 
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.gender_smoothing)
        elif self.target_attr == 'race': 
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.race_smoothing)

    def compute_losses(self, images, labels):
        # Forward
        outputs = self.model(images)

        # Classification Loss 
        loss = self.criterion(outputs['logits'], labels)
        return loss, outputs

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        correct, total = 0, 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            target_labels = labels[:, 1] if self.target_attr == 'gender' else labels[:, 2] # Label: [Age, Gender, Race]

            # Compute Loss
            loss, outputs = self.compute_losses(images, target_labels)

            # Backward 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # Compute Accuracy
            correct += (outputs['logits'].argmax(dim=1) == target_labels).sum().item()
            total += target_labels.size(0)
            
            # Wandb Logging 
            if self.args.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "step": epoch * len(self.train_loader) + batch_idx,
                    f"train/batch/{self.target_attr}_loss": loss.item(),
                })

        self.scheduler.step()

        # Logging
        train_log = {}
        train_log["train_loss"] = train_loss / len(self.train_loader)
        train_log["train_acc"] = correct / total

        # Validation
        eval_log = self.evaluate()
        return train_log, eval_log

    def evaluate(self):
        self.model.eval()
        eval_loss = 0.0
        correct, total = 0, 0

        bias_data = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluation", leave=False):
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                target_labels = labels[:, 1] if self.target_attr == 'gender' else labels[:, 2] # Label: [Age, Gender, Race]
                group_labels = labels[:, 2] if self.target_attr == 'gender' else labels[:, 1]  # Label: [Age, Gender, Race]

                # Compute Loss
                loss, outputs = self.compute_losses(images, target_labels)
                eval_loss += loss.item()

                # Compute Accuracy 
                correct += (outputs['logits'].argmax(dim=1) == target_labels).sum().item()
                total += target_labels.size(0)

                # Collect bias-related data
                bias_data["logits"].append(outputs['logits'].detach().cpu())
                bias_data["features"].append(outputs['features'].detach().cpu())
                bias_data["target_labels"].append(target_labels.detach().cpu())
                bias_data["group_labels"].append(group_labels.detach().cpu())

        for k in bias_data:
            bias_data[k] = torch.cat(bias_data[k], dim=0)

        # Compute bias metrics
        bias_metrics = compute_bias_metrics(
            logits=bias_data["logits"],
            labels=bias_data["target_labels"],
            group_labels=bias_data["group_labels"],
            features=bias_data["features"]
        )
        
        # Logging 
        eval_log = {}
        eval_log["eval_loss"] = eval_loss / len(self.val_loader)
        eval_log["eval_acc"] = correct / total
        
        for k, v in bias_metrics.items():
            eval_log[f"eval_{self.args.bias_attribute}_bias/{k}"] = v

        return eval_log

    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f"Base_ResNet50_E{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path  

    def train(self):
        # 수정 필요 - 삭제
        if self.args.use_wandb:
            artifact = wandb.Artifact(name=self.run_name, type="model")

        for epoch in range(self.num_epochs):
            train_log, eval_log = self.train_epoch(epoch)

            if self.args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/train_loss": train_log['train_loss'],
                    f"epoch/train_{self.target_attr}_acc": train_log['train_acc'],

                    "epoch/eval_loss": eval_log['eval_loss'],
                    f"epoch/eval_{self.target_attr}_acc": eval_log['eval_acc'],

                    **{f"epoch/{k}": v for k, v in eval_log.items() 
                    if k.startswith(f"eval_{self.args.bias_attribute}_bias/")}
                })

            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.num_epochs:
                checkpoint_path = self.save_checkpoint(epoch + 1)
        
        # 수정 필요 - 삭제
                if self.args.use_wandb:
                    artifact.add_file(checkpoint_path)
        if self.args.use_wandb:
            wandb.log_artifact(artifact)