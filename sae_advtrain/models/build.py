"""Model factory. Wraps every backbone in a `NormalizedModel` so that PGD
attacks are always crafted in raw [0, 1] pixel space (standard practice --
see Madry et al. / RobustBench conventions) while the network still sees
normalized inputs internally.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .preact_resnet import PreActResNet18, PreActResNet34
from .wideresnet import WideResNet28_10, WideResNet34_10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2471, 0.2435, 0.2616)

_ARCHS = {
    "preact_resnet18": PreActResNet18,
    "preact_resnet34": PreActResNet34,
    "wideresnet28_10": WideResNet28_10,
    "wideresnet34_10": WideResNet34_10,
}


class NormalizedModel(nn.Module):
    """backbone((x - mean) / std). `x` is expected in [0, 1]."""

    def __init__(self, backbone: nn.Module, mean=CIFAR_MEAN, std=CIFAR_STD):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def build_model(arch: str, num_classes: int = 10) -> NormalizedModel:
    if arch not in _ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Available: {list(_ARCHS)}")
    backbone = _ARCHS[arch](num_classes=num_classes)
    return NormalizedModel(backbone)


def resolve_feature_layer(model: NormalizedModel, feature_layer: str) -> Tuple[str, int]:
    """Resolve 'auto' to the architecture's recommended layer and return
    (dotted_module_path, channel_width) where dotted_module_path is relative
    to `model` (i.e. prefixed with 'backbone.')."""
    backbone = model.backbone
    if feature_layer == "auto":
        feature_layer = backbone.default_feature_layer
    if feature_layer not in backbone.layer_widths:
        raise ValueError(
            f"'{feature_layer}' is not a hookable layer of {type(backbone).__name__}. "
            f"Available: {list(backbone.layer_widths)}"
        )
    width = backbone.layer_widths[feature_layer]
    return f"backbone.{feature_layer}", width
