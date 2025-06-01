from src.model.debiasing_models import CLIP_Model, CV_Model

def get_models(args, device):
    if args.dataset_name == "UTKFace":
        num_classes = [2, 5] # [Gender, Race]
        clip = CLIP_Model(num_classes, args, device)
        resnet = CV_Model(num_classes, args)

    elif args.dataset_name == "FairFace":
        num_classes = [2, 7] # [Gender, Race]
        clip = CLIP_Model(num_classes, args, device)
        resnet = CV_Model(num_classes, args)

    return clip, resnet