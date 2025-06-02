import os
import sys
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


# ===============
#    수정 필요   
# ===============
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
        for images, labels in tqdm(test_loader, desc='Inference'):
            images, labels = images.to(self.device), labels.to(self.device)
            gender_labels, race_labels = labels[:, 1], labels[:, 2]

            # CLIP
            # CV Model

    clip_preds = np.concatenate(clip_preds, axis=0)
    model_preds = np.concatenate(model_preds, axis=0)

if __name__ == "__main__":
    args = get_arguments()
    main(args)