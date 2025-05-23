import os
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
from tqdm import tqdm

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

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=data_collator,
    )

    clip, model = get_model(args, device)
    clip, model = clip.to(device), model.to(device)
    # clip.load_state_dict()
    # model.load_state_dict()

    clip.eval()
    model.eval()
    clip_preds = []
    model_preds = []
     
    with torch.no_grad():
        for images in tqdm(test_loader, desc='Inference'):
            images = images.to(device)

            # CLIP
            clip_g_logits, clip_r_logits, _ = clip(images)
            clip_preds.append(torch.cat([clip_g_logits, clip_r_logits], dim=1).cpu().numpy())

            # CV Model
            model_g_logits, model_r_logits, _ = model(images)
            model_preds.append(torch.cat([model_g_logits, model_r_logits], dim=1).cpu().numpy())

    clip_preds = np.concatenate(clip_preds, axis=0)
    model_preds = np.concatenate(model_preds, axis=0)

if __name__ == "__main__":
    args = get_arguments()
    main(args)