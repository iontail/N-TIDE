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

        self.clip, _ = clip.load(args.clip_backbone, device=self.device) 
        for param in self.clip.parameters(): # CLIP freeze
            param.requires_grad = False
        
        img_out_dim = self.clip.visual.output_dim
        txt_in_dim = self.clip.token_embedding.weight.shape[1]
        txt_out_dim = self.clip.text_projection.shape[1]

        self.fusion_mlp =  nn.Sequential(
            nn.Linear(img_out_dim + txt_out_dim, args.feature_dim),
            nn.ReLU()
        )
        self.gender_head = nn.Linear(args.feature_dim, gender_classes)
        self.race_head = nn.Linear(args.feature_dim, race_classes)
        
        # Neutral-Text prompt: "A photo of [Neutral vector]"
        self.neutral_token = nn.Parameter(torch.randn(1, txt_in_dim)) 
        with torch.no_grad():
            tokenized = clip.tokenize(["A photo of a person"]).to(device)
            prefix_ids = tokenized[0][:5]  # [BOS, a, photo, of, a]
            self.register_buffer("prompt_embed", self.clip.token_embedding(prefix_ids)  ) 

        # Null-Text prompt: ""
        with torch.no_grad():
            tokens = clip.tokenize([args.clip_text_prompt]).to(self.device)
            self.register_buffer("null_text_embed", self.clip.encode_text(tokens))  

    def forward(self, x):
        B = x.size(0)
        with torch.no_grad():
            image_embedding = self.clip.encode_image(x) 
            null_embeddding = self.null_text_embed.expand(B, -1) 
            fused_null = torch.cat([image_embedding, null_embeddding], dim=1)
            fused_null = self.fusion_mlp(fused_null)
        
        prompt = self.prompt_embed.unsqueeze(0).expand(B, -1, -1) 
        neutral = self.neutral_token.expand(B, -1).unsqueeze(1)  
        neutral_prompt = torch.cat([prompt, neutral], dim=1)  # [BOS, a, photo, of, a, [Neutral]]

        with torch.no_grad():
            pad_len = 77 - neutral_prompt.size(1)
            pad_embed = self.clip.token_embedding(torch.zeros(pad_len, dtype=torch.long, device=x.device))  
            pad_embed = pad_embed.unsqueeze(0).expand(B, -1, -1)  
            neutral_prompt = torch.cat([neutral_prompt, pad_embed], dim=1)  # [B, 77, D]

        pos_embed = self.clip.positional_embedding[:neutral_prompt.size(1), :].unsqueeze(0)
        neutral_prompt = neutral_prompt + pos_embed
        neutral_prompt = neutral_prompt.permute(1, 0, 2) 

        neutral_prompt = self.clip.transformer(neutral_prompt)   
        neutral_prompt = neutral_prompt.permute(1, 0, 2) 
        neutral_prompt = self.clip.ln_final(neutral_prompt)          

        neutral_embed = neutral_prompt[:, 0, :]             
        neutral_embed = torch.matmul(neutral_embed, self.clip.text_projection)

        fused_neutral = torch.cat([image_embedding, neutral_embed], dim=1)
        fused_neutral = self.fusion_mlp(fused_neutral) 

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

        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, args.feature_dim),
            nn.ReLU()
        )
        self.gender_head = nn.Linear(args.feature_dim, gender_classes)
        self.race_head = nn.Linear(args.feature_dim, race_classes)

    def forward(self, x):
        features = self.model(x)
        gender_logits = self.gender_head(features)
        race_logits = self.race_head(features)

        return {
            'features': features,
            'gender_logits': gender_logits,
            'race_logits': race_logits,
        }
    

if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace(clip_backbone="RN50", clip_text_prompt='', feature_dim = 512)
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


