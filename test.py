import os
import sys
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from arguments import get_arguments
from src.dataset.get_dataset import get_dataset
from src.model.get_model import get_model

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

    _, model = get_models(args, device)
    model.load_state_dict(torch.load(" ")['model']) # 지정 필요.
    model = model.to(device)
    model.eval()

    bias_data = defaultdict(list)
    gender_correct, race_correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Inference'):
            images, labels = images.to(device), labels.to(device)
            gender_labels, race_labels = labels[:, 1], labels[:, 2]

            # Forward 
            outputs = model(images)

            # Compute Accuracy 
            gender_preds = outputs['gender_logits'].argmax(dim=1)
            race_preds = outputs['race_logits'].argmax(dim=1)
            g_correct = (gender_preds == gender_labels).sum().item()
            r_correct = (race_preds == race_labels).sum().item()

            gender_correct += g_correct
            race_correct += r_correct
            total += labels.size(0)

            # Collect bias-related data
            bias_data["gender_logits"].append(outputs['gender_logits'].detach().cpu())
            bias_data["race_logits"].append(outputs['race_logits'].detach().cpu())
            bias_data["features"].append(outputs['features'].detach().cpu())
            bias_data["gender_labels"].append(gender_labels.detach().cpu())
            bias_data["race_labels"].append(race_labels.detach().cpu())
        
    # Compute bias metrics
    # 1) Label: Gender / Group: Race
    gender_race_results = compute_bias_metrics(
        logits=bias_data["gender_logits"],
        labels=bias_data["gender_labels"],
        group_labels=bias_data["race_labels"],
        features=bias_data["features"]
    )

        # 2) Label: Race / Group: Gender
    race_gender_results = compute_bias_metrics(
        logits=bias_data["race_logits"],
        labels=bias_data["race_labels"],
        group_labels=bias_data["gender_labels"],
        features=bias_data["features"]
    )
    
    # Logging
    infer_log = {}
    infer_log['gender_acc'] = gender_correct / total
    infer_log['race_acc'] = race_correct / total

    for k, v in gender_race_results.items():
        infer_log[f"gender_race/{k}"] = v
    for k, v in race_gender_results.items():
        infer_log[f"race_gender/{k}"] = v

    # Wandb Logging
    if args.use_wandb:
        run_id = " " # 지정 필요.
        wandb.init(project='Intro-to-DL-N-TIDE', id=run_id, resume='must') 

        wandb.log({
            # Accuracy
            'infer/gender_acc': infer_log['gender_acc'],
            'infer/race_acc': infer_log['race_acc'],

            # Bias metrics
            **{f"infer/{k}": v for k, v in infer_log.items() if k.startswith("gender_race/")},
            **{f"infer/{k}": v for k, v in infer_log.items() if k.startswith("race_gender/")}
        })


if __name__ == "__main__":
    args = get_arguments()
    main(args)