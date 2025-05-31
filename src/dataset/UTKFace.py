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

    def __len__(self):
        return len(self.dataset)

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

if __name__ == "__main__":
    from argparse import Namespace
    from datasets import load_dataset
    import matplotlib.pyplot as plt 

    args = Namespace(dataset_path="py97/UTKFace-Cropped")

    dataset = load_dataset(args.dataset_path, split='train')
    dataset = UTKFace_Dataset(dataset)

    image, label = dataset[0]
    plt.imshow(image)
    plt.title(f"Age: {label[0]}, Gender: {label[1]}, Race: {label[2]}")
    plt.show()

