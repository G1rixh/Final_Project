"""
Three models, all producing 14 raw logits (sigmoid applied at loss/inference).

  * resnet18  : ImageNet-pretrained, final FC swapped to 14 outputs.
  * vgg19     : ImageNet-pretrained, classifier head swapped to 14 outputs.
  * customcnn : from-scratch conv net (expected to trail the pretrained ones --
                that gap is itself a reportable finding).
"""
import torch
import torch.nn as nn
from torchvision import models

import config as C


def build_resnet18(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, C.NUM_LABELS)
    return m


def build_vgg19(pretrained: bool = True) -> nn.Module:
    weights = models.VGG19_Weights.DEFAULT if pretrained else None
    m = models.vgg19(weights=weights)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, C.NUM_LABELS)
    return m


class CustomCNN(nn.Module):
    """Compact 4-block CNN -> global average pool -> classifier."""
    def __init__(self, num_labels: int = C.NUM_LABELS):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_labels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def build_custom_cnn(**_) -> nn.Module:
    return CustomCNN()


MODEL_REGISTRY = {
    "resnet18": build_resnet18,
    "vgg19": build_vgg19,
    "customcnn": build_custom_cnn,
}


def get_model(name: str, pretrained: bool = True) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](pretrained=pretrained)


def gradcam_target_layer(model: nn.Module, name: str):
    """Return the conv layer Grad-CAM should hook for each architecture."""
    if name == "resnet18":
        return model.layer4[-1]
    if name == "vgg19":
        return model.features[-1]
    if name == "customcnn":
        return model.features[-1][-3]  # last conv in final block
    raise ValueError(name)
