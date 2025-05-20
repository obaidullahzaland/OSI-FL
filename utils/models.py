from torchvision import models, transforms
import torch.nn as nn
import torch


def initialize_model(backbone, num_classes, pretrained=True):
    if backbone == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'vit_b_16':
        model = models.vit_b_16(pretrained=pretrained)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'vit_b_32':
        model = models.vit_b_32(pretrained=pretrained)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Invalid backbone model name")
    return model


# class BackboneModel(nn.Module):
#     def __init__(self, backbone_model, num_classes, pretrained=True):
#         """
#         Initializes the ServerTune model with a flexible backbone and configurable number of classes.
#         Args:
#             backbone_model: A string specifying the backbone model (e.g., 'resnet18', 'resnet50').
#             num_classes: An integer specifying the number of output classes.
#             pretrained: A boolean indicating whether to load a pre-trained model.
#         """
#         super(BackboneModel, self).__init__()
#         # Initialize the backbone model dynamically
#         if backbone_model == 'resnet18':
#             self.encoder = models.resnet18(pretrained=pretrained)
#             feature_dim = self.encoder.fc.in_features
#             self.encoder.fc = nn.Identity()  # Remove the classification head
#         elif backbone_model == 'resnet50':
#             self.encoder = models.resnet50(pretrained=pretrained)
#             feature_dim = self.encoder.fc.in_features
#             self.encoder.fc = nn.Identity()
#         elif backbone_model == 'vit_b_16':
#             self.encoder = models.vit_b_16(pretrained=pretrained)
#             feature_dim = self.encoder.heads.head.in_features
#             self.encoder.heads.head = nn.Identity()
#         # Add other backbone options here as needed
#         else:
#             raise ValueError("Unsupported backbone model")
        
#         # Final projection for classification
#         self.final_proj = nn.Sequential(
#             nn.Linear(feature_dim, num_classes)
#         )
    
#     def forward(self, x, get_fea=False, input_image=True):
#         """
#         Forward pass for the ServerTune model.
#         Args:
#             x: Input tensor.
#             get_fea: If True, return the extracted features.
#             input_image: If True, the input is an image and will pass through the encoder.
#         """
#         if input_image:
#             with torch.no_grad():  # Freeze encoder during forward pass
#                 x = self.encoder(x)
        
#         if get_fea:  # Return extracted features
#             return x.view(x.shape[0], -1)
        
#         # Pass features to the final projection
#         out = self.final_proj(x.view(x.shape[0], -1))
#         return out
    

class BackboneModel(nn.Module):
    def __init__(self, backbone_model, num_classes, pretrained=True, freeze_encoder=True):
        super(BackboneModel, self).__init__()
        # Initialize the backbone dynamically
        if backbone_model == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif backbone_model == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif backbone_model == 'vit_b_16':
            self.encoder = models.vit_b_16(pretrained=pretrained)
            feature_dim = self.encoder.heads.head.in_features
            self.encoder.heads.head = nn.Identity()
        else:
            raise ValueError("Unsupported backbone model")

        # Freeze encoder layers if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Final projection for classification
        self.final_proj = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x, get_fea=False):
        x = self.encoder(x)  # Always compute gradients if training
        if get_fea:
            return x.view(x.shape[0], -1)
        out = self.final_proj(x.view(x.shape[0], -1))
        return out
    
class ServerTune(nn.Module):
    def __init__(self, classes=60):
        super(ServerTune, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.final_proj = nn.Sequential(
            nn.Linear(512, classes)
        )
    
    def forward(self, x, get_fea=False, input_image=True):
        if input_image:
            with torch.no_grad():  # Freeze encoder during forward pass
                x = self.encoder(x)
        
        if get_fea:  # Return extracted features
            return x.view(x.shape[0], -1)
        
        # Pass features to the final projection
        out = self.final_proj(x.view(x.shape[0], -1))
        return out
    
