import numpy as np
import torch
import io
from PIL import Image

class UTKFaceDataset(torch.utils.data.Dataset):
    """
    Hugging Face UTKFace-Cropped Dataset.
        - "jpg.chip.jpg": PIL.Image
        - "__key__": string formatted as "[age]_[gender]_[race]_..."
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["jpg.chip.jpg"]["bytes"]
        image = Image.open(io.BytesIO(image))
        key =  sample["__key__"].split("/")[-1]
        age, gender, race = map(int, key.split('_')[:3])

        image = self.transform(image)
        label = torch.tensor([age, gender, race], dtype=torch.long)
        return image, label


if __name__ == "__main__":
    from datasets import load_dataset, Dataset
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

    full_train_data = load_dataset("py97/UTKFace-Cropped", split='train')
    full_train_data = full_train_data.filter(
        lambda x: x["__key__"] not in ["UTKFace/55_0_0_20170116232725357jpg", # Image is None 
                                "UTKFace/39_1_20170116174525125",  # Lable is invalid
                                "UTKFace/61_1_20170109150557335",  # Lable is invalid
                                "UTKFace/61_1_20170109142408075",] # Lable is invalid
    )

    df = full_train_data.to_pandas()
    df[["age", "gender", "race"]] = df["__key__"].str.extract(r'(\d+)_(\d+)_(\d+)')
    df[["age", "gender", "race"]] = df[["age", "gender", "race"]].astype(int)

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['race'], random_state=42)
    val_df, test_df  = train_test_split(temp_df, test_size=0.2, stratify=temp_df['race'], random_state=42)

    class_distribution(train_df, "Train")
    class_distribution(val_df, "Validation")
    class_distribution(test_df, "Test")

    print(f"\nTrain set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    train_data = Dataset.from_pandas(train_df.reset_index(drop=True))

    transform = transforms.Compose([transforms.Resize((224, 224))])
    train_data = UTKFaceDataset(train_data, transform)

    image, label = train_data[0]
    plt.imshow(image)
    plt.title(f"Age: {label[0]}, Gender: {label[1]}, Race: {label[2]}")
    plt.show()

