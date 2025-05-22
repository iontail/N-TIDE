import os
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
import wandb
import time

from arguments import get_arguments
from src.datasets.get_dataset import get_dataset
from src.model.get_model import get_model
from src.trainer import Trainer 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")

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

    clip, model = get_model(args)
    clip, model = clip.to(device), model.to(device)

    if args.bf16:
        clip, model = clip.to(torch.bfloat16), model.to(torch.bfloat16)

    c_optimizer = torch.optim.Adam(
        clip.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    r_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    run_name = f"N-TIDE_run_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project='N-TIDE', name=run_name, config=args) 

    trainer = Trainer(
        clip=clip,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        c_optimizer=c_optimizer,
        r_optimizer=r_optimizer,
        device=device,
        args=args,
        epochs=args.epochs,
        eval_steps=200, # Or pull from cfg if defined
        checkpoint_dir="./ckpt", 
        use_wandb=True,
        project_name="N-TIDE", # Match wandb.init
        run_name=run_name      # Match wandb.init
    )

    trainer.train()

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    
    main(args)