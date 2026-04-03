"""
TransFuse classification-oriented adaptation.

Important:
    TransFuse is originally proposed for segmentation, combining transformer and
    CNN streams. This implementation adapts the same design principle to
    classification by fusing:
        - a CNN stream
        - a transformer stream
    followed by a classifier head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .common import MLPHead


class TransFuseClassifier(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        # CNN stream
        self.cnn = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)

        # Transformer stream
        self.transformer = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)

        # Align both features to same dimension
        self.cnn_proj = nn.Linear(self.cnn.num_features, 768)
        self.tr_proj = nn.Identity()

        self.head = MLPHead(768, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_proj(self.cnn(x))
        tr_feat = self.tr_proj(self.transformer(x))
        fused = 0.5 * cnn_feat + 0.5 * tr_feat
        return self.head(fused)
