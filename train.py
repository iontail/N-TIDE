import os
import sys
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np

import torch
import torch.nn as nn

from arguments import get_argument
from src.datasets.get_dataset import get_dataset
from src.model.get_model import get_model
from src.Trainer import Trainer 

from PIL import Image
import time
import wandb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")

    set_seed(args.seed)
