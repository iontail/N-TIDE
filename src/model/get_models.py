from src.model.architectures import CLIP_Model, ResNet_Model

def get_models(args, device):
    target_attr = 'gender' if args.bias_attribute == 'race' else 'race'

    # Bias attribute: Gender
    if target_attr == 'race': 
        num_classes = 2

    # Bias attribute: Race 
    elif target_attr == 'gender': 
        if args.dataset_name == "UTKFace":
            num_classes = len(args.utkface_race_class)

        elif args.dataset_name == "FairFace" and args.is_fairface_race_7:
            num_classes = len(args.fairface_race_class)

        elif args.dataset_name == "FairFace" and not args.is_fairface_race_7:
            num_classes = 4

    clip = CLIP_Model(num_classes, args, device)
    resnet = ResNet_Model(num_classes, args)
    return clip, resnet
