import os
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
import wandb
import time

from arguments import get_arguments
from src.dataset.get_dataset import get_dataset
from src.model.get_model import get_model
from src.trainer_offline import OfflineKDTrainer
from src.trainer_online import OnlineKDTrainer

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
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,  
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )

    clip, model = get_model(args, device)
    clip, model = clip.to(device), model.to(device)

    if args.bf16:
        clip, model = clip.to(torch.bfloat16), model.to(torch.bfloat16)

    c_optimizer = torch.optim.Adam(
        clip.parameters(),
        lr=args.c_learning_rate,
        weight_decay=args.c_weight_decay
    )
    c_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        c_optimizer,
        T_max=args.num_epochs,
        eta_min=args.c_eta_min
    )
    
    m_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.m_learning_rate,
        weight_decay=args.m_weight_decay
    )
    m_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        c_optimizer,
        T_max=args.num_epochs,
        eta_min=args.m_eta_min
    )


    if args.use_wandb:
        run_name = f"N-TIDE_run_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project='N-TIDE', name=run_name, config=args) 

    if args.distill_mode == 'offline':
        if args.finetune_model == 'teacher': 
            trainer = OfflineKDTrainer(
                model=clip,
                model_type='teacher',
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=c_optimizer,
                scheduler=c_scheduler,
                device=device,
                args=args
            )  
        elif args.finetune_model == 'student': 
            trainer = OfflineKDTrainer(
                model=model,
                model_type='student',
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=m_optimizer,
                scheduler=m_scheduler,
                device=device,
                args=args
            )

    elif args.distill_mode == 'online':
        trainer = OnlineKDTrainer(
            clip=clip,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            c_optimizer=c_optimizer,
            c_scheduler=c_scheduler,
            m_optimizer=m_optimizer,
            m_scheduler=m_scheduler,
            device=device,
            args=args
        )
    
    trainer.train()

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    
    main(args)