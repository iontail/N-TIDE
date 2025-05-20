import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Define transforms directly here or pass them during instantiation
augment_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

augment_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


class UTKFace_Dataset(torch.utils.data.Dataset):
    """
    UTKFace Dataset.
    Assumes filenames are in the format: [age]_[gender]_[race]_[date&time].jpg
    Uses 'age' as the label.
    """
    def __init__(self, root_dir, train_mode=True, test_mode =False, transform=None, test_split_ratio=0.2, random_state=42):
        super(UTKFace_Dataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.train_mode = train_mode
        self.test_mode = test_mode

        self.filepaths = []
        self.labels = []

        image_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))

        for filepath in image_paths:
            filename = os.path.basename(filepath)
            try:
                # Extract age from filename (e.g., "61_1_2_20170109150557335.jpg.chip.jpg")
                # Handle potential variations like ".chip.jpg" suffix if present
                split_name = filename.split('_')
                age = int(split_name[0])
                gender = int(split_name[1])
                race = int(split_name[2])
                
                self.filepaths.append(filepath)
                self.labels.append([age, gender, race])

            except (IndexError, ValueError):
                print(f"Warning: Could not parse age from filename: {filename}. Skipping file.")
                continue

        # Split data into train and validation sets
        if not self.test_mode:
            train_files, val_files, train_labels, val_labels = train_test_split(
                self.filepaths, self.labels, test_size=test_split_ratio, random_state=random_state
            )

            if self.train_mode:
                self.filepaths = train_files
                self.labels = train_labels

            else: # validation mode
                self.filepaths = val_files
                self.labels = val_labels
        else:
            pass

    def __getitem__(self, index):
        img_path = self.filepaths[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filepaths)