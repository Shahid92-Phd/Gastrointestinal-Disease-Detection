"""
Common helpers for baseline models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class TimmClassifier(nn.Module):
    """
    Thin wrapper around timm.create_model for classification baselines.
    """
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPHead(nn.Module):
    """
    Small reusable classifier head.
    """
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
