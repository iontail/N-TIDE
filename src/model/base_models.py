import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from torchvision.models import resnet50



class CLIP_Model(nn.Module):
    def __init__(self, args, device):
        super(CLIP_Model, self).__init__()
        self.args = args
        self.device = device

        self.clip, _ = clip.load(args.clip_backbone, device=self.device) 
        for param in self.clip.parameters(): # CLIP freeze
            param.requires_grad = False
        
        i_encoder_dim = self.clip.visual.output_dim
        t_encoder_dim = self.clip.text_projection.shape[1]
        self.neutral_embedding = nn.Parameter(torch.randn(1, t_encoder_dim))

        self.classifier =  nn.Linear(i_encoder_dim + t_encoder_dim, args.feature_dim)

        self.gender_classifier = nn.Linear(args.feature_dim, len(args.gender_classes))
        self.race_classifier = nn.Linear(args.feature_dim, len(args.race_classes))

        # "A photo of a Asian man.", "A photo of a White woman." Etc. Dataset Classes 
        text_prompts = [
            f"{args.clip_text_prompt} {race} {gender}."
            for race in args.race_classes
            for gender in args.gender_classes
        ] 
        with torch.no_grad():
            tokens = clip.tokenize(text_prompts).to(self.device)
            self.register_buffer("text_embeddings", self.clip.encode_text(tokens))  # [C, D]

    def forward(self, x):
        with torch.no_grad():
            image_embedding = self.clip.encode_image(x)
        
        B = x.size(0)
        fused_embedding = torch.cat([image_embedding, self.neutral_embedding.expand(B, -1)], dim=1)
        fused_embedding = self.classifier(fused_embedding)

        gender_logits = self.gender_classifier(fused_embedding)
        race_logits = self.race_classifier(fused_embedding)

        return (gender_logits, race_logits, fused_embedding) if self.args.return_features else (gender_logits, race_logits, None)
    


class Base_Model(nn.Module):
    def __init__(self, args):
        super(Base_Model, self).__init__()
        self.args = args
        
        self.model = resnet50(weights='IMAGENET1K_V2')
        resnet_dim = self.model.fc.in_features

        self.model.fc = nn.Linear(resnet_dim, args.feature_dim)
        self.gender_classifier = nn.Linear(args.feature_dim, len(args.gender_classes))
        self.race_classifier = nn.Linear(args.feature_dim, len(args.race_classes))

    def forward(self, x):
        x = self.model(x)
        features = torch.flatten(x, 1)

        gender_logits = self.gender_classifier(x)
        race_logits = self.race_classifier(x)

        return (gender_logits, race_logits, features) if self.args.return_features else (gender_logits, race_logits, None)
    


if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace(clip_backbone="RN50", clip_text_prompt='A photo of a', feature_dim = 512, 
                     gender_classes=['man', 'woman'], race_classes=['White', 'Black', 'Asian', 'Indian', 'Others'], return_features=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 64
    sample_image = torch.randn(batch_size, 3, 224, 224).to(device)

    clip_model = CLIP_Model(args, device).to(device)
    gender_logits, race_logtis, features = clip_model(sample_image)
    print("CLIP Model, Feature shape:", features.shape)
    print("CLIP Model, Gender output shape:", gender_logits.shape)
    print("CLIP Model, Race output shape:", race_logtis.shape)

    base_model = Base_Model(args).to(device)
    gender_logits, race_logtis, features = base_model(sample_image)
    print("Base Model, Feature shape:", features.shape)
    print("Base Model, Gender output shape:", gender_logits.shape)
    print("Base Model, Gender output shape:", race_logtis.shape)


