import numpy as np
import torch

class FairFaceDataset(torch.utils.data.Dataset):
    """
    Hugging Face FairFace Dataset.
        - Crop and align faces with padding = 0.25 in the main experiments
        - and padding = 1.25 for the bias measument experiment for commercial APIs.
    """
    
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        age, gender, race = sample["age"], sample["gender"], sample["race"]

        image = self.transform(image)
        label = torch.tensor([age, gender, race], dtype=torch.long)
        return image, label




  
    
if __name__ == "__main__":
    from datasets import load_dataset, Dataset, Image
    from sklearn.model_selection import train_test_split
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    def class_distribution(df, name):
        print(f"\n-- {name} set, Race and gender distribution:")
        race_dist = df["race"].value_counts(normalize=True).sort_index()

        for race, race_ratio in race_dist.items():
            print(f"Race {race}: {race_ratio:.4f}")
            sub_df = df[df["race"] == race]
            gender_dist = sub_df["gender"].value_counts(normalize=True).sort_index()

            for gender, gender_ratio in gender_dist.items():
                print(f"  Gender {gender}: {gender_ratio:.4f}")

    full_train_data = load_dataset("HuggingFaceM4/FairFace", "0.25", split='train')
    test_data = load_dataset("HuggingFaceM4/FairFace", "0.25", split='validation')

    df = full_train_data.to_pandas()
    test_df = test_data.to_pandas()

    df = df[df["race"].isin([0, 1, 2, 3])]
    test_df = test_df[test_df["race"].isin([0, 1, 2, 3])]

    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["race"], random_state=42)

    class_distribution(train_df, "Train")
    class_distribution(val_df, "Validation")
    class_distribution(test_df, "Test")

    print(f"\nTrain set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    train_data = Dataset.from_pandas(train_df.reset_index(drop=True))
    train_data = train_data.cast_column("image", Image())

    transform = transforms.Compose([transforms.Resize((224, 224))])
    train_data = FairFaceDataset(train_data, transform)

    image, label = train_data[0]
    plt.imshow(image)
    plt.title(f"Age: {label[0]}, Gender: {label[1]}, Race: {label[2]}")
    plt.show()


