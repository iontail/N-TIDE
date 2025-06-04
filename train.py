import os
import sys 
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
import wandb
import time

from arguments import get_arguments
from src.dataset.get_dataset import get_dataset
from src.model.get_models import get_models
from src.trainer.BasicTrainer import BasicTrainer
from src.trainer.OfflineTrainer import OfflineKDTrainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(args.seed)

    train_dataset, val_dataset, data_collator = get_dataset(args)
    num_workers = min(4, os.cpu_count()) 

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=data_collator, drop_last=True, pin_memory=True,
        num_workers=num_workers, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=data_collator, pin_memory=True,
        num_workers=num_workers, persistent_workers=True
    )

    clip, model = get_models(args, device)
    clip, model = clip.to(device), model.to(device)

    # Basline Train: Fine-tuning ResNet50 (pretrained on ImageNet)
    if args.train_mode == 'baseline':
        run_name = f"Baseline_{time.strftime('%m%d_%H%M')}"
        if args.use_wandb:
            wandb.init(project='Intro-to-DL-N-TIDE', name=run_name, config=args)

        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.m_learning_rate, weight_decay=args.m_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs, eta_min=args.m_eta_min
        )
        trainer = BasicTrainer(
            model=model, 
            train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler, 
            device=device, args=args, run_name=run_name
        )
        
    # Offline KD Train: Fine-tuning CLIP (Teacher model)
    elif args.train_mode == 'offline_teacher':
        run_name = f"N-TIDE_Teacher_{time.strftime('%m%d_%H%M')}"
        if args.use_wandb:
            wandb.init(project='Intro-to-DL-N-TIDE', name=run_name, config=args)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, clip.parameters()), 
            lr=args.c_learning_rate, weight_decay=args.c_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs, eta_min=args.c_eta_min
        )
        trainer = OfflineKDTrainer(
            model=clip, model_type='teacher',
            train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler,
            device=device, args=args, run_name=run_name
        )

    # Offline KD Train: Fine-tuning ResNet50 (Student model)
    elif args.train_mode == 'offline_student':
        run_name = f"N-TIDE_Student_{time.strftime('%m%d_%H%M')}"
        if args.use_wandb:
            wandb.init(project='Intro-to-DL-N-TIDE', name=run_name, config=args)

        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.m_learning_rate, weight_decay=args.m_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs, eta_min=args.m_eta_min
        )
        trainer = OfflineKDTrainer(
            model=model, model_type='student',
            train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler,
            device=device, args=args, run_name=run_name
        )

    trainer.train()

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    
    main(args)