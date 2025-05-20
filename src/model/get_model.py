from src.model.base_models import CLIP_Model, Base_Model

def get_model(args, device):
    clip = CLIP_Model(args, device)
    resnet = Base_Model(args)

    return clip, resnet