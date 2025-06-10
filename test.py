import os
import sys
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
import wandb
from tqdm import tqdm
from collections import defaultdict

from arguments import get_arguments
from src.dataset.get_dataset import get_dataset
from src.model.get_models import get_models
from src.utils.bias_metrics import compute_bias_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(args.seed)

    test_dataset, _, data_collator = get_dataset(args)
    num_workers = min(4, os.cpu_count()) 

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=data_collator, pin_memory=True,
        num_workers=num_workers, persistent_workers=True
    )

    # Baseline or Student model (ResNet50)
    if args.experiment_type == 'baseline' or args.experiment_type == 'offline_student':
        _, model = get_models(args, device) 
    
    # Teacher model (CLIP)
    elif args.experiment_type == 'offline_teacher':
        model, _ = get_models(args, device)

    model.load_state_dict(torch.load(args.infer_ckpt_path)['model']) 
    model = model.to(device)
    model.eval()

    bias_data = defaultdict(list)
    correct, total = 0, 0

    target_attr = 'gender' if args.bias_attribute == 'race' else 'race'
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Inference'):
            images, labels = images.to(device), labels.to(device)

            target_labels = labels[:, 1] if target_attr == 'gender' else labels[:, 2] # Label: [Age, Gender, Race]
            group_labels = labels[:, 2] if target_attr == 'gender' else labels[:, 1]  # Label: [Age, Gender, Race]

            # Forward 
            outputs = model(images)

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
    infer_log = {}
    infer_log['infer_acc'] = correct / total

    for k, v in bias_metrics.items():
        infer_log[f"{args.bias_attribute}_bias/{k}"] = v

    # Wandb Logging
    if args.use_wandb:
        run_name = "N-TIDE (ours)"
        wandb.init(project='Intro-to-DL-N-TIDE', name=run_name, config=args)

        wandb.log({
            # Accuracy
            f"Inference/{target_attr}_acc": infer_log['infer_acc'],
            # Bias metrics
            **{f"Inference/{k}": v for k, v in infer_log.items() 
            if k.startswith(f"{args.bias_attribute}_bias/")},
        })

if __name__ == "__main__":
    args = get_arguments()
    main(args)
