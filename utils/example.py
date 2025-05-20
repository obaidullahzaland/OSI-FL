import torch
import torchvision.models as models

# Example model
model = models.resnet18(pretrained=True)

# Calculate total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')