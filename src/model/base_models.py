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

        self.classifier = nn.Sequential(
            nn.Linear(i_encoder_dim + t_encoder_dim, args.clip_clf_dim),
            nn.ReLU(),
            nn.Linear(args.clip_clf_dim, args.clip_clf_dim)
        )

        self.gender_classifier = nn.Linear(args.clip_clf_dim, len(args.gender_classes))
        self.race_classifier = nn.Linear(args.clip_clf_dim, len(args.race_classes))

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

        return gender_logits, race_logits
    
class Base_Model(nn.Module):
    def __init__(self, args):
        super(Base_Model, self).__init__()
        self.args = args
        
        self.model = resnet50(weights='IMAGENET1K_V2')
        feature_dim = self.model.fc.in_features

        self.model.fc = nn.Identity()
        self.gender_classifier = nn.Linear(feature_dim, len(args.gender_classes))
        self.race_classifier = nn.Linear(feature_dim, len(args.race_classes))

    def forward(self, x):
        # 나중에 중간 레이어 출력 필요할 떄 대비해서 이렇게 코딩
        features = {}
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        features['layer0'] = x

        for i in range(1, 5):
            x = getattr(self.model, f'layer{i}')(x)
            features[f'layer{i}'] = x

        x = self.model.avgpool(x)  # global average pooling
        x = torch.flatten(x, 1)  # flatten the features

        gender_logits = self.gender_classifier(x)
        race_logits = self.race_classifier(x)

        return (gender_logits, race_logits, features) if self.args.return_features else (gender_logits, race_logits, None)
    

if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace(clip_backbone="RN50", clip_clf_dim = 256, clip_text_prompt='A photo of a',
                     gender_classes=['man', 'woman'], race_classes=['White', 'Black', 'Asian', 'Indian', 'Others'], return_features=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_image = torch.randn(1, 3, 224, 224).to(device)

    clip_model = CLIP_Model(args, device).to(device)
    gender_logits, race_logtis = clip_model(sample_image)
    print("CLIP Model, Gender output shape:", gender_logits.shape)
    print("CLIP Model, Race output shape:", race_logtis.shape)

    base_model = Base_Model(args).to(device)
    gender_logits, race_logtis, _ = base_model(sample_image)
    print("Base Model, Gender output shape:", gender_logits.shape)
    print("Base Model, Gender output shape:", race_logtis.shape)


