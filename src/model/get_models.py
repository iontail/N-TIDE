from src.model.debiasing_models import CLIP_Model, ResNet_Model

def get_models(args, device):
    if args.dataset_name == "UTKFace":
        num_classes = [2, len(args.utkface_race_class)] # [Gender, Race]
    elif args.dataset_name == "FairFace":
        num_classes = [2, len(args.fairface_race_class)] # [Gender, Race]

    clip = CLIP_Model(num_classes, args, device)
    resnet = ResNet_Model(num_classes, args)
    return clip, resnet