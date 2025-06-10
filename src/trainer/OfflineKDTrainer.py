import os
import wandb
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn 
import torch.nn.functional as F

from src.model.get_models import get_models
from src.utils.bias_metrics import compute_bias_metrics


class OfflineKDTrainer:
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

        # For training student, load teacher model 
        if self.args.experiment_type == 'offline_student': 
            self.teacher, _ = get_models(self.args, self.device)
            self.teacher.load_state_dict(torch.load(args.teacher_ckpt_path)['model'])
            self.teacher = self.teacher.to(device)
            self.teacher.eval()

    def compute_losses(self, images, labels):
        # Forward
        outputs = self.model(images)

        # Classification Loss 
        cls_loss = self.criterion(outputs['logits'], labels)
        
        # Logging 
        losses = {} 
        losses["cls_loss"] = cls_loss

        # Teacher Loss: Cls Loss + Align Loss (MSE)
        if self.args.experiment_type == 'offline_teacher':
            align_loss = F.mse_loss(outputs['features'], outputs['f_null'].detach())
            losses["feature_loss"] = align_loss
            losses["total_loss"] = cls_loss + self.args.lambda_t * align_loss

        # Student Loss: Cls Loss + KD Loss (Cosine Similarity)
        elif self.args.experiment_type == 'offline_student': 
            with torch.no_grad():
                teacher_outputs = self.teacher(images)

            cosine_sim = F.cosine_similarity(outputs["features"], teacher_outputs["features"].detach(), dim=-1)
            kd_loss = 1 - cosine_sim.mean()
            losses["feature_loss"] = kd_loss
            losses["total_loss"] = cls_loss + self.args.lambda_s * kd_loss
            
        return losses, outputs

    def train_epoch(self, epoch):
        self.model.train()
        train_loss, feat_loss = 0.0, 0.0
        correct, total = 0, 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            target_labels = labels[:, 1] if self.target_attr == 'gender' else labels[:, 2] # Label: [Age, Gender, Race]
            
            # Compute Losses
            losses, outputs = self.compute_losses(images, target_labels)
            loss = losses['total_loss']

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += losses['total_loss'].item()
            feat_loss += losses['feature_loss'].item()

            # Compute Accuracy
            correct += (outputs['logits'].argmax(dim=1) == target_labels).sum().item()
            total += target_labels.size(0)

            # Wandb Logging
            if self.args.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "step": epoch * len(self.train_loader) + batch_idx,
                    "train/batch/total_loss": losses['total_loss'].item(),
                    "train/batch/feature_loss": losses['feature_loss'].item(),
                    f"train/batch/{self.target_attr}_loss": losses['cls_loss'].item(),                     
                })

        self.scheduler.step()

        # Logging 
        train_log = {}
        train_log["train_loss"] = train_loss / len(self.train_loader)
        train_log["train_feature_loss"] = feat_loss / len(self.train_loader)
        train_log["train_acc"] = correct / total

        # Validation
        eval_log = self.evaluate()
        return train_log, eval_log

    def evaluate(self):
        self.model.eval()
        eval_loss, feat_loss = 0.0, 0.0
        correct, total = 0, 0

        bias_data = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluation", leave=False):
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                target_labels = labels[:, 1] if self.target_attr == 'gender' else labels[:, 2] # Label: [Age, Gender, Race]
                group_labels = labels[:, 2] if self.target_attr == 'gender' else labels[:, 1]  # Label: [Age, Gender, Race]

                # Compute Losses 
                losses, outputs = self.compute_losses(images, target_labels)
                eval_loss += losses['total_loss'].item()
                feat_loss += losses['feature_loss'].item()

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
        eval_log["eval_feature_loss"] = feat_loss / len(self.val_loader)
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
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"N-TIDE_{self.args.experiment_type}_E{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path  

    def train(self):
        if self.args.use_wandb: # 수정 필요 - 삭제
            artifact = wandb.Artifact(name=self.run_name, type="model")

        for epoch in range(self.num_epochs):
            train_log, eval_log = self.train_epoch(epoch)
            
            if self.args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/train_loss": train_log['train_loss'],
                    'epoch/train_feature_loss': train_log['train_feature_loss'],
                    f"epoch/train_{self.target_attr}_acc": train_log['train_acc'],

                    "epoch/eval_loss": eval_log['eval_loss'],
                    "epoch/eval_feature_loss": eval_log['eval_feature_loss'],
                    f"epoch/eval_{self.target_attr}_acc": eval_log['eval_acc'],

                    **{f"epoch/{k}": v for k, v in eval_log.items() 
                    if k.startswith(f"eval_{self.args.bias_attribute}_bias/")}
                })

            if (epoch + 1) % 3 == 0 or (epoch + 1) == self.num_epochs:
                checkpoint_path = self.save_checkpoint(epoch + 1)

        # 수정 필요 - 삭제
                if self.args.use_wandb:
                    artifact.add_file(checkpoint_path)
        if self.args.use_wandb:
            wandb.log_artifact(artifact)
                