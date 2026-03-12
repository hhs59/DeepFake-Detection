import torch
import torch.nn as nn
from torchvision import models


class DeepfakeModel(nn.Module):

    def __init__(self, num_classes=2, freeze_backbone=True):

        super().__init__()

        # Load pretrained EfficientNet
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Get feature size
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

    def forward(self, x):

        return self.backbone(x)


    def unfreeze_backbone(self):

        for param in self.backbone.features.parameters():
            param.requires_grad = True