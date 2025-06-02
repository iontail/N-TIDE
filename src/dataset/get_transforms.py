import torchvision.transforms as transforms

def get_transforms(args):
    if args.train_mode == 'baseline' or args.train_mode == 'offline_student': 
        # For CV model pretrained on ImageNet
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]
    elif args.train_mode == 'offline_teacher':
        # For CLIP model (Open AI)
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    if args.train_transform_type == 'strong':
        # train_transforms = transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandAugment(num_ops=2, magnitude=9),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std),
        #     transforms.RandomErasing(p=0.5),
        # ])
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),

            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),

            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),

            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.3),
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

    return train_transforms, test_transforms