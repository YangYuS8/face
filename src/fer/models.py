from typing import Tuple, Optional

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
    # Native torchvision backbones
    if backbone in {"resnet18", "resnet50"}:
        return FERResNet(num_classes=num_classes, backbone=backbone, pretrained=pretrained, dropout=dropout)

    # Try timm backbones
    try:
        import timm  # type: ignore
    except Exception as e:
        raise ValueError(f"Backbone '{backbone}' requires timm. Please install timm to use this model. ({e})")

    # Many timm models accept num_classes directly
    try:
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        return model
    except Exception as e:
        # Fallback: create with features head, then attach classifier
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        in_feats = model.num_features if hasattr(model, 'num_features') else None
        if in_feats is None:
            raise
        classifier = nn.Linear(in_feats, num_classes)
        model.reset_classifier(num_classes) if hasattr(model, 'reset_classifier') else None
        # Attach a generic classifier head
        model.fc = classifier  # type: ignore[attr-defined]
        return model
