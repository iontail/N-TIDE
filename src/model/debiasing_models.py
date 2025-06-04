import torch
import torch.nn as nn

import clip
from torchvision import models

class CLIP_Model(nn.Module):
    """
    Args:
        num_classes (tuple): A tuple of integers (num_gender_classes, num_race_classes).
        args:
            - clip_backbone (str): Name of the CLIP backbone to use (e.g., 'RN50').
            - clip_text_prompt (str): Prompt used to construct the null text embedding.
            - feature_dim (int): Hidden feature dimension used in fusion and classifiers.

    Returns:
        A model that takes an image tensor of shape (B, 3, 224, 224) and returns a dictionary with:
            - 'f_null' (Tensor): Fused feature from image and null text embedding, shape (B, feature_dim).
            - 'f_neutral' (Tensor): Fused feature from image and neutral text prompt, shape (B, feature_dim).
            - 'gender_logits' (Tensor): Gender classification logits, shape (B, num_gender_classes).
            - 'race_logits' (Tensor): Race classification logits, shape (B, num_race_classes).
    """
    def __init__(self, num_classes, args, device):
        super(CLIP_Model, self).__init__()
        self.args = args
        self.device = device
        gender_classes, race_classes = num_classes

        # CLIP Freeze
        self.clip, _ = clip.load(args.clip_backbone, device=self.device) 
        for param in self.clip.parameters():
            param.requires_grad = False

        # Fuse MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.clip.visual.output_dim + self.clip.text_projection.shape[1], args.feature_dim),
            nn.ReLU()
        )
        
        # Classification head
        self.gender_head = nn.Linear(args.feature_dim, gender_classes)
        self.race_head = nn.Linear(args.feature_dim, race_classes)
        
        # Null-Text prompt: ""
        with torch.no_grad():
            tokens = clip.tokenize([args.clip_null_text]).to(self.device)
            self.register_buffer("null_encoded", self.clip.encode_text(tokens))  
            
        
        # Neutral-Text prompt: "A photo of a [Neutral vector]"
        with torch.no_grad():
            tokens = clip.tokenize(["A photo of a neutral"]).to(device)
            self.register_buffer("neutral_embed", self.clip.token_embedding(tokens))     
            
        # [Neutral vector]"
        with torch.no_grad():
            tokens = clip.tokenize(["A photo of a person"]).to(device)
            token_embeds = self.clip.token_embedding(tokens)
            init_vector = token_embeds[0, 1:6].mean(dim=0, keepdim=True)
        self.neutral_vector = nn.Parameter(init_vector.clone())
            
            
    def forward(self, x):
        B = x.size(0)
        
        with torch.no_grad():
            # Image Encode            
            image_encoded = self.clip.encode_image(x) 
            
            # Null-text Encode 
            null_encoded = self.null_encoded.expand(B, -1)
            fused_null = torch.cat([image_encoded, null_encoded], dim=1)
            # Null-text Fuse
            fused_null = self.fusion_mlp(fused_null)
            
        # A photo of a Neutral -> A photo of a [Neutral vector]
        neutral_embed = self.neutral_embed.expand(B, -1, -1).clone() # (B, 77, D)
        neutral_embed[:, 5, :] = self.neutral_vector.expand(B, -1)    # "Neutral" index = 5
        
        # Neutral-text Encode
        neutral_encoded = neutral_embed + self.clip.positional_embedding 
        neutral_encoded = neutral_encoded.permute(1, 0, 2)
        neutral_encoded = self.clip.transformer(neutral_encoded)
        neutral_encoded = neutral_encoded.permute(1, 0, 2) 
        neutral_encoded = self.clip.ln_final(neutral_encoded)
        
        neutral_encoded = neutral_encoded[:, 6, :] # EOS token          
        neutral_encoded = torch.matmul(neutral_encoded, self.clip.text_projection)

        # Neutral-text Fuse
        fused_neutral = torch.cat([image_encoded, neutral_encoded], dim=1)
        fused_neutral = self.fusion_mlp(fused_neutral) 
        
        # Classification Head
        gender_logits = self.gender_head(fused_neutral)
        race_logits = self.race_head(fused_neutral)

        return {
            'f_null': fused_null,
            'f_neutral': fused_neutral,
            'gender_logits': gender_logits,
            'race_logits': race_logits,
        }
    

class CV_Model(nn.Module):
    """
    Args:
        num_classes (tuple): A tuple of integers (num_gender_classes, num_race_classes).
        args: 
            - feature_dim (int): Hidden feature dimension used in projection and classification.

    Returns:
        A model that takes an image tensor of shape (B, 3, 224, 224) and returns a dictionary with:
            - 'features' (Tensor): Extracted visual features, shape (B, feature_dim).
            - 'gender_logits' (Tensor): Gender classification logits, shape (B, num_gender_classes).
            - 'race_logits' (Tensor): Race classification logits, shape (B, num_race_classes).
    """

    def __init__(self, num_classes, args):
        super(CV_Model, self).__init__()
        self.args = args
        gender_classes, race_classes = num_classes

        # ResNet 
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, args.feature_dim),
            nn.ReLU()
        )
        
        # Classification Head 
        self.gender_head = nn.Linear(args.feature_dim, gender_classes)
        self.race_head = nn.Linear(args.feature_dim, race_classes)

    def forward(self, x):
        # Image Encode
        features = self.model(x)
        
        # Classification Head
        gender_logits = self.gender_head(features)
        race_logits = self.race_head(features)

        return {
            'features': features,
            'gender_logits': gender_logits,
            'race_logits': race_logits,
        }
    

if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace(clip_backbone="RN50", clip_null_text='', feature_dim = 512)
    device = torch.device("cpu")
    num_classes = [2, 7]
    batch_size = 64

    sample_image = torch.randn(batch_size, 3, 224, 224)

    clip_model = CLIP_Model(num_classes, args, device)
    outputs = clip_model(sample_image)
    print("-- CLIP Model:")
    print("Fused Neutral-text Feature shape:", outputs['f_neutral'].shape)
    print("Fused Null-text Feature shape:", outputs['f_null'].shape)
    print("Gender logits shape:", outputs['gender_logits'].shape)
    print("Race logits shape:", outputs['race_logits'].shape)

    cv_model = CV_Model(num_classes, args)
    outputs = cv_model(sample_image)
    print("\n-- CV Model:")
    print("Features shape:", outputs['features'].shape)
    print("Gender logits shape:", outputs['gender_logits'].shape)
    print("Race logits shape:", outputs['race_logits'].shape)


