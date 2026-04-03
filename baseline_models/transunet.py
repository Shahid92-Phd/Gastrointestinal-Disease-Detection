"""
TransUNet classification-oriented adaptation.

Important:
    TransUNet is originally a segmentation architecture. This implementation is
    a clean classification baseline inspired by a CNN+Transformer encoder-decoder
    design, adapted for image-level GI disease classification.

Design:
    - CNN stem for local texture extraction
    - ViT-style transformer encoder for global modeling
    - Global average pooling
    - Classification head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .common import MLPHead


class TransUNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        # CNN-like local stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Transformer encoder branch
        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0
        )

        # Project stem features to image space-like representation
        self.local_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.local_proj = nn.Linear(64, self.encoder.num_features)

        self.head = MLPHead(self.encoder.num_features, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.stem(x)
        local = self.local_pool(local).flatten(1)
        local = self.local_proj(local)

        global_feat = self.encoder(x)
        fused = 0.5 * local + 0.5 * global_feat
        return self.head(fused)
