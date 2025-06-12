from src.model.debiasing_models import CLIP_Model, ResNet_Model

def get_models(args, device):
    target_attr = 'gender' if args.bias_attribute == 'race' else 'race'

    # Bias attribute: Race
    if target_attr == 'gender': 
        num_classes = 2

    # Bias attribute: Gender 
    elif target_attr == 'race': 
        if args.dataset_name == "UTKFace":
            num_classes = len(args.utkface_race_class)

        elif args.dataset_name == "FairFace":
            if args.is_fairface_race_7:
                num_classes = len(args.fairface_race_class)
            else:
                num_classes = 4  # ['East Asian', 'Indian', 'Black', 'White']

    clip = CLIP_Model(num_classes, args, device)
    resnet = ResNet_Model(num_classes, args)
    return clip, resnet
