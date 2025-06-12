from torch.utils.data.dataloader import default_collate 

from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, Image

from src.dataset.get_transforms import get_transforms
from src.dataset.UTKFace import UTKFaceDataset
from src.dataset.FairFace import FairFaceDataset

def get_dataset(args):
    train_transforms, test_transforms = get_transforms(args)

    # UTKFace Dataset
    if args.dataset_name == "UTKFace":
        train_ratio, valid_ratio, test_ratio = args.utkface_split_ratio
        test_size = test_ratio / (valid_ratio + test_ratio)

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

        train_df, temp_df = train_test_split(df, test_size=1-train_ratio, stratify=df['race'], random_state=args.seed)
        val_df, test_df  = train_test_split(temp_df, test_size=test_size, stratify=temp_df['race'], random_state=args.seed)

        train_data = Dataset.from_pandas(train_df.reset_index(drop=True))
        valid_data = Dataset.from_pandas(val_df.reset_index(drop=True))
        test_data = Dataset.from_pandas(test_df.reset_index(drop=True))
        
        if args.is_train:
            train_dataset = UTKFaceDataset(train_data, transform=train_transforms)
            valid_dataset = UTKFaceDataset(valid_data, transform=test_transforms)
            data_collator = default_collate
            return train_dataset, valid_dataset, data_collator
        else:
            test_dataset = UTKFaceDataset(test_data, transform=test_transforms)
            data_collator = default_collate
            return test_dataset, None, data_collator

    # FairFace Dataset
    elif args.dataset_name == "FairFace":
        train_ratio, valid_ratio = args.fairface_split_ratio

        full_train_data = load_dataset("HuggingFaceM4/FairFace", "0.25", split='train')
        test_data = load_dataset("HuggingFaceM4/FairFace", "0.25", split='validation')

        df = full_train_data.to_pandas()
        test_df = test_data.to_pandas()

        if not args.is_fairface_race_7: # 4-class
            df = df[df["race"].isin([0, 1, 2, 3])]
            test_df = test_df[test_df["race"].isin([0, 1, 2, 3])]

        train_df, val_df = train_test_split(df, test_size=valid_ratio, stratify=df["race"], random_state=args.seed)

        train_data = Dataset.from_pandas(train_df.reset_index(drop=True)).cast_column("image", Image())
        valid_data = Dataset.from_pandas(val_df.reset_index(drop=True)).cast_column("image", Image())
        test_data  = Dataset.from_pandas(test_df.reset_index(drop=True)).cast_column("image", Image())

        if args.is_train:
            train_dataset = FairFaceDataset(train_data, transform=train_transforms)
            valid_dataset = FairFaceDataset(valid_data, transform=test_transforms)
            return train_dataset, valid_dataset, default_collate
        else:
            test_dataset = FairFaceDataset(test_data, transform=test_transforms)
            return test_dataset, None, default_collate
        