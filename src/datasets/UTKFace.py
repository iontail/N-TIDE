import torch
from torch.utils.data import Dataset

class UTKFace_Dataset(Dataset):
    """
    Hugging Face UTKFace-Cropped Dataset.
        - "jpg.chip.jpg": PIL.Image
        - "__key__": string formatted as "[age]_[gender]_[race]_..."
    Returns (image, label) where label = [age, gender, race]
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["jpg.chip.jpg"]
        key =  sample["__key__"].split("/")[-1]
        age, gender, race = map(int, key.split('_')[:3])

        label = torch.tensor([age, gender, race], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.dataset)
