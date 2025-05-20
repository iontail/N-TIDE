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
        self.neutral_emebdding = nn.Parameter(torch.randn(1, t_encoder_dim))

        self.classifier = nn.Sequential(
            nn.Linear(i_encoder_dim + t_encoder_dim, args.clip_clf_dim),
            nn.ReLU(),
            nn.Linear(args.clip_clf_dim, args.num_classes)
        )

    def forward(self, image, text=None, is_neutral=False):
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
        
        if is_neutral:
            batch_size = image.size(0)
            text_features = self.neutral_embedding.expand(batch_size, -1)
        else:
            assert text is not None, "text must be not None if is_neutral=False"
            text_tokens = clip.tokenize(text).to(self.device)
            with torch.no_grad():
                text_features = self.clip.encode_text(text_tokens)

        features = torch.cat([image_features, text_features], dim=1)
        outputs = self.classifier(features)

        return outputs, image_features, text_features

class Base_Model(nn.Module):
    def __init__(self, args):
        super(Base_Model, self).__init__()
        self.args = args
        self.model = resnet50(weights='IMAGENET1K_V2')
        last_feature_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(last_feature_dim, args.num_classes)

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
        x = self.model.fc(x) 

        return x, features if self.args.return_features else (x, None)
    

if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace(clip_backbone="RN50", clip_clf_dim = 256, num_classes = 3, return_features = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_image = torch.randn(1, 3, 224, 224).to(device)
    sample_text = ["A photo of a man."]

    clip_model = CLIP_Model(args, device).to(device)
    outputs, image_features, text_features = clip_model(sample_image, sample_text, is_neutral=False)
    print("CLIP Model, Image features shape:", image_features.shape)
    print("CLIP Model, Text features shape:", text_features.shape)
    print("CLIP Model, Output shape:", outputs.shape)

    print("="*50)

    base_model = Base_Model(args).to(device)
    outputs, features = base_model(sample_image)
    print("Base Model, Output shape:", outputs.shape)


