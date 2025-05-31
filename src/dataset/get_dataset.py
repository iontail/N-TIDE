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

    if args.train_mode == 'baseline' or args.train_mode == 'offline_student': 
        # For CV model pretrained on ImageNet
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
    elif args.train_mode == 'offline_teacher':
        # For CLIP model (Open AI)
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    if args.train_transform_type == 'strong':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.5),
        ])
    elif args.train_transform_type == 'weak':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
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

