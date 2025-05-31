import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate 
from datasets import load_dataset
from src.dataset.UTKFace import UTKFace_Dataset

def get_dataset(args):
    """
    Args:
        dataset_path (str): Hugging Face UTKFace-Cropped dataset
        dataset_split_ratio (list): [train, valid, test]
        random_state (int)

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    dataset = load_dataset(args.dataset_path, split='train')
    dataset = dataset.filter(
        lambda x: x["__key__"] not in ["UTKFace/55_0_0_20170116232725357jpg", # Image is None 
                                   "UTKFace/39_1_20170116174525125",  # Lable is invalid
                                   "UTKFace/61_1_20170109150557335",  # Lable is invalid
                                   "UTKFace/61_1_20170109142408075",] # Lable is invalid
    )
    
    assert sum(args.dataset_split_ratio) == 1.0, "Split ratios must sum to 1.0"
    train_ratio, valid_ratio, test_ratio = args.dataset_split_ratio
    total_size = len(dataset)
    train_end = int(train_ratio * total_size)
    valid_end = train_end + int(valid_ratio * total_size)

    dataset_sh = dataset.shuffle(seed=args.seed)
    train_data = dataset_sh.select(range(0, train_end))
    valid_data = dataset_sh.select(range(train_end, valid_end))
    test_data  = dataset_sh.select(range(valid_end, total_size))

    # Define transforms directly here or pass them during instantiation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    if args.is_train:
        train_dataset = UTKFace_Dataset(train_data, transform=train_transforms)
        valid_dataset = UTKFace_Dataset(valid_data, transform=test_transforms)
        data_collator = default_collate
        return train_dataset, valid_dataset, data_collator
    else:
        test_dataset = UTKFace_Dataset(test_data, transform=test_transforms)
        data_collator = default_collate
        return test_dataset, None, data_collator

