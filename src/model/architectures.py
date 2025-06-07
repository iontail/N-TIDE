import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from torchvision import models

class CLIP_Model(nn.Module):
    def __init__(self, num_classes, args, device):
        super().__init__()
        self.args = args    
        self.device = device
        gender_classes, race_classes = num_classes

        # CLIP (Freeze)
        self.model, _ = clip.load(args.clip_backbone, device=self.device) 
        self.model = self.model.float()

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Null-Text prompt: ""
        with torch.no_grad():
            tokens = clip.tokenize([args.clip_null_text]).to(self.device)
            self.register_buffer("null_encoded", self.model.encode_text(tokens))  
             
        # Neutral-Text prompt: "A photo of a [Neutral vector]"
        with torch.no_grad():
            tokens = clip.tokenize(["A photo of a neutral"]).to(device)
            self.register_buffer("neutral_token_embed", self.model.token_embedding(tokens)) 

        # Initialize [Neutral vector]
        self.neutral_vector = nn.Parameter(torch.empty(1, self.model.token_embedding.embedding_dim))
        nn.init.normal_(self.neutral_vector, mean=0.0, std=0.02)
        # with torch.no_grad():
        #     tokens = clip.tokenize(["A photo of a person"]).to(device)
        #     token_embed = self.model.token_embedding(tokens)
        # self.neutral_vector = nn.Parameter(token_embed[0, 5].clone())
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.model.visual.output_dim + self.model.text_projection.shape[1], args.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(args.feature_dim * 2, args.feature_dim),
            nn.ReLU(),
            nn.Linear(args.feature_dim, args.feature_dim)
        )

        # Classification Head
        self.gender_head = nn.Linear(args.feature_dim, gender_classes)
        self.race_head = nn.Linear(args.feature_dim, race_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()

    def _encode_image(self, x):
        return self.model.encode_image(x)

    def _encode_neutral_text(self, x):
        x = x + self.model.positional_embedding 
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2) 
        x = self.model.ln_final(x)
        
        x = x[:, 6, :] # EOS token, index = 6          
        x = torch.matmul(x, self.model.text_projection)
        return x

    def forward(self, x):
        B = x.size(0)          

        with torch.no_grad():
            # Image Encode
            image_enc = self._encode_image(x)
            image_enc = F.normalize(image_enc, dim=-1)

            # Null-text Encode and Fuse
            null_enc = self.null_encoded.expand(B, -1)
            null_enc = F.normalize(null_enc, dim=-1)

            fused_null = torch.cat([image_enc, null_enc], dim=1)
            fused_null = self.fusion_mlp(fused_null)
   
        # A photo of a neutral -> A photo of a [Neutral vector]
        neutral_embed = self.neutral_token_embed.expand(B, -1, -1).clone() 
        neutral_embed[:, 5, :] = self.neutral_vector.expand(B, -1)   

        # Neutral-text Encode and Fuse
        neutral_enc = self._encode_neutral_text(neutral_embed)
        neutral_enc = F.normalize(neutral_enc, dim=-1)

        fused_neutral = torch.cat([image_enc, neutral_enc], dim=1)
        fused_neutral = self.fusion_mlp(fused_neutral)

        # Classification Head
        gender_logits = self.gender_head(fused_neutral)
        race_logits = self.race_head(fused_neutral)

        return {
            'f_null': fused_null,
            'features': fused_neutral,
            'gender_logits': gender_logits,
            'race_logits': race_logits,
        }
    


class ResNet_Model(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        self.args = args
        gender_classes, race_classes = num_classes

        # ResNet 
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = nn.Linear(self.model.fc.in_features, args.feature_dim)

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
