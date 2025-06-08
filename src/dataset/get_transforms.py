import torchvision.transforms as transforms

def get_transforms(args):
    # Data Augmentation
    # For CV model pretrained on ImageNet
    if args.experiment_type == 'baseline':
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]

    # For CLIP model (Open AI)
    elif args.experiment_type == 'offline_teacher' or args.experiment_type == 'offline_student': 
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transforms, test_transforms