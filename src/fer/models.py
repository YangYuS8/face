from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class FERResNet(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_feats = net.fc.in_features
            net.fc = nn.Linear(in_feats, num_classes)
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_feats = net.fc.in_features
            net.fc = nn.Linear(in_feats, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(num_classes: int, backbone: str = "resnet18", pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    return FERResNet(num_classes=num_classes, backbone=backbone, pretrained=pretrained, dropout=dropout)
