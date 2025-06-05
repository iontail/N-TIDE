import torch
import torch.nn as nn

import clip
from torchvision import models

class Residual_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = output + self.skip(x)
        return self.relu(output)



class CV_Model(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        self.args = args
        gender_classes, race_classes = num_classes

        # ResNet 
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = Residual_MLP(
            input_dim = self.model.fc.in_features,
            hidden_dim = args.feature_dim,
            output_dim = args.feature_dim
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



class CLIP_Model(nn.Module):
    def __init__(self, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        gender_classes, race_classes = num_classes

        # CLIP (Freeze)
        self.model, _ = clip.load(args.clip_backbone, device=self.device) 
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
            self.register_buffer("neutral_embed", self.model.token_embedding(tokens))     
            
        # Initialize [Neutral vector]
        with torch.no_grad():
            tokens = clip.tokenize(["A photo of a person"]).to(device)
            token_embeds = self.model.token_embedding(tokens)
            init_vector = token_embeds[0, 1:6].mean(dim=0, keepdim=True)
        self.neutral_vector = nn.Parameter(init_vector.clone())

        # Fusion MLP
        self.fusion_mlp = Residual_MLP(
            # input_dim = self.model.visual.output_dim + self.model.text_projection.shape[1],
            input_dim = self.model.visual.output_dim,
            hidden_dim = args.feature_dim,
            output_dim = args.feature_dim,
        )

        # Classification Head
        self.gender_head = nn.Linear(args.feature_dim, gender_classes)
        self.race_head = nn.Linear(args.feature_dim, race_classes)


    def forward(self, x):
        B = x.size(0)
        with torch.no_grad():
            image_encoded = self.model.encode_image(x)
            # image_encoded = image_encoded / image_encoded.norm(dim=-1, keepdim=True)

        fused_neutral = self.fusion_mlp(image_encoded)
        gender_logits = self.gender_head(fused_neutral)
        race_logits = self.race_head(fused_neutral)

        fused_null = torch.zeros_like(fused_neutral)

        return {
            'f_null': fused_null,
            'f_neutral': fused_neutral,
            'gender_logits': gender_logits,
            'race_logits': race_logits,
        }

    # def forward(self, x):
    #     B = x.size(0)
        
    #     with torch.no_grad():
    #         # Image Encode            
    #         image_encoded = self.model.encode_image(x) 
            
    #         # Null-text Encode 
    #         null_encoded = self.null_encoded.expand(B, -1)
    #         fused_null = torch.cat([image_encoded, null_encoded], dim=1)
    #         # Null-text Fuse
    #         fused_null = self.fusion_mlp(fused_null)
            
    #     # A photo of a Neutral -> A photo of a [Neutral vector]
    #     neutral_embed = self.neutral_embed.expand(B, -1, -1).clone() # (B, 77, D)
    #     neutral_embed[:, 5, :] = self.neutral_vector.expand(B, -1)    # "Neutral" index = 5
        
    #     # Neutral-text Encode
    #     neutral_encoded = neutral_embed + self.model.positional_embedding 
    #     neutral_encoded = neutral_encoded.permute(1, 0, 2)
    #     neutral_encoded = self.model.transformer(neutral_encoded)
    #     neutral_encoded = neutral_encoded.permute(1, 0, 2) 
    #     neutral_encoded = self.model.ln_final(neutral_encoded)
        
    #     neutral_encoded = neutral_encoded[:, 6, :] # EOS token          
    #     neutral_encoded = torch.matmul(neutral_encoded, self.model.text_projection)

    #     # Neutral-text Fuse
    #     fused_neutral = torch.cat([image_encoded, neutral_encoded], dim=1)
    #     fused_neutral = self.fusion_mlp(fused_neutral) 

    #     # Classification Head
    #     gender_logits = self.gender_head(fused_neutral)
    #     race_logits = self.race_head(fused_neutral)

    #     return {
    #         'f_null': fused_null,
    #         'f_neutral': fused_neutral,
    #         'gender_logits': gender_logits,
    #         'race_logits': race_logits,
    #     }
    
    

if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace(clip_backbone="RN50", clip_null_text='', feature_dim = 512)
    device = torch.device("cpu")
    num_classes = [2, 7]
    batch_size = 64

    sample_image = torch.randn(batch_size, 3, 224, 224)

    cv_model = CV_Model(num_classes, args)
    outputs = cv_model(sample_image)
    print("-- CV Model:")
    print("Features shape:", outputs['features'].shape)
    print("Gender logits shape:", outputs['gender_logits'].shape)
    print("Race logits shape:", outputs['race_logits'].shape)

    clip_model = CLIP_Model(num_classes, args, device)
    outputs = clip_model(sample_image)
    print("\n-- CLIP Model:")
    print("Fused Neutral-text Feature shape:", outputs['f_neutral'].shape)
    print("Fused Null-text Feature shape:", outputs['f_null'].shape)
    print("Gender logits shape:", outputs['gender_logits'].shape)
    print("Race logits shape:", outputs['race_logits'].shape)


