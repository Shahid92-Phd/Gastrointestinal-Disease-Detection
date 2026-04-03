"""
Swin-UNETR classification adaptation.

Important:
    Swin-UNETR is originally a segmentation architecture from MONAI.
    This adaptation uses the encoder output as an image-level representation
    for GI disease classification.

Requirements:
    pip install monai
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLPHead


class SwinUNETRClassifier(nn.Module):
    def __init__(self, num_classes: int = 8, img_size: int = 224, in_channels: int = 3, feature_size: int = 48, dropout: float = 0.2):
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:
            raise ImportError(
                "MONAI is required for SwinUNETRClassifier. Install it with: pip install monai"
            ) from exc

        # Create SwinUNETR backbone
        self.backbone = SwinUNETR(
            img_size=(img_size, img_size),
            in_channels=in_channels,
            out_channels=2,   # dummy segmentation channels; not used directly
            feature_size=feature_size,
            spatial_dims=2
        )

        # We will use encoder-like hidden features via a feature projection path
        # Since MONAI SwinUNETR does not expose a direct classification head,
        # we pool the deepest output from the backbone output logits map.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = MLPHead(2, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)    # [B, 2, H, W]
        pooled = self.pool(feat_map).flatten(1)
        return self.head(pooled)
