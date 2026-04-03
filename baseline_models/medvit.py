"""
MedViT classification-oriented baseline.

Important:
    Because "MedViT" is not consistently exposed as a standard pretrained model
    across common PyTorch libraries, this implementation provides a practical
    medical-imaging-oriented approximation:
        - convolutional stem for local inductive bias
        - transformer encoder for global context
        - lightweight classifier head

This is intended as a clean experimental baseline for GI image classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .common import MLPHead


class MedViTClassifier(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.local_proj = nn.Linear(64, 768)

        self.transformer = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        self.head = MLPHead(768, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.conv_stem(x)
        local = self.global_pool(local).flatten(1)
        local = self.local_proj(local)

        global_feat = self.transformer(x)
        fused = 0.5 * local + 0.5 * global_feat
        return self.head(fused)
