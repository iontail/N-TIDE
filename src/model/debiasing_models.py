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

        # CLIP (Freeze)
        self.model, _ = clip.load(args.clip_backbone, device=self.device) 
        self.model = self.model.float()

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        with torch.no_grad():
            # General-Text prompt: "A photo of a person"
            tokens = clip.tokenize(["A photo of a person"]).to(self.device)
            self.register_buffer("text_encoded", self.model.encode_text(tokens))

            # Null-Text prompt: ""
            tokens = clip.tokenize([""]).to(self.device)
            self.register_buffer("null_encoded", self.model.encode_text(tokens))

            # Neutral-Text prompt: "A photo of a neutral" -> "A photo of a [Neutral vector]"
            tokens = clip.tokenize(["A photo of a neutral"]).to(self.device)
            self.register_buffer("neutral_token_embed", self.model.token_embedding(tokens))

        # Initialize [Neutral vector]
        if args.neutral_init == 'random':
            self.neutral_vector = nn.Parameter(
                torch.empty(1, self.model.token_embedding.embedding_dim)
            )
            nn.init.normal_(self.neutral_vector, mean=0.0, std=0.02)

        elif args.neutral_init == 'person':
            with torch.no_grad():
                tokens = clip.tokenize(["A photo of a person"]).to(device)
                token_embed = self.model.token_embedding(tokens)
                init_vector = token_embed[0, 5].clone()
            self.neutral_vector = nn.Parameter(init_vector.unsqueeze(0))
            
        # Fusion MLP (Image, Text features -> element-wise additing)
        in_features = self.model.visual.output_dim 
        self.fusion = nn.Sequential(
            nn.Linear(in_features, args.feature_dim),
            nn.ReLU(),
            nn.Linear(args.feature_dim, args.feature_dim)
        )

        # Classification Head
        self.head = nn.Linear(args.feature_dim, num_classes)

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

            fused_null = image_enc + null_enc   
            fused_null = self.fusion(fused_null)

        # A photo of a neutral -> A photo of a [Neutral vector]
        neutral_embed = self.neutral_token_embed.expand(B, -1, -1).clone() 
        neutral_embed[:, 5, :] = self.neutral_vector.expand(B, -1) # "neutral" token, index =5 

        # Neutral-text Encode and Fuse
        neutral_enc = self._encode_neutral_text(neutral_embed)
        neutral_enc = F.normalize(neutral_enc, dim=-1)

        fused_neutral = image_enc + neutral_enc
        fused_neutral = self.fusion(fused_neutral)

        # Classification Head
        logits = self.head(fused_neutral)

        return {
            'f_null': fused_null,
            'features': fused_neutral,
            'logits': logits
        }
    


class ResNet_Model(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        self.args = args

        # ResNet 
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        # Projection
        self.proj = nn.Linear(in_features, args.feature_dim)

        # Classification Head 
        self.head = nn.Linear(args.feature_dim, num_classes)

    def forward(self, x):
        # Image Encode
        features = self.model(x)
        features = self.proj(features)
        
        # Classification Head
        logits = self.head(features)

        return {
            'features': features,
            'logits': logits
        }
