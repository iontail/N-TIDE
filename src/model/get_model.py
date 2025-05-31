from src.model.debiasing_models import CLIP_Model, CV_Model

def get_model(args, device):
    clip = CLIP_Model(args, device)
    resnet = CV_Model(args)

    return clip, resnet