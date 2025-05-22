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

    clip, model = get_model()
    # clip.load_state_dict()
    # model.load_state_dict()

    clip, model = clip.to(device), model.to(device)
    clip.eval()
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

if __name__ == "__main__":
    args = get_arguments()
    main(args)