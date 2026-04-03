"""
Fusion module for combining ViT and MaxViT features.
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Adaptive gated fusion for combining two feature vectors."""

    def __init__(self, input_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_vit: torch.Tensor, feat_maxvit: torch.Tensor):
        combined = torch.cat([feat_vit, feat_maxvit], dim=1)
        alpha = self.gate(combined)
        fused = alpha * feat_vit + (1.0 - alpha) * feat_maxvit
        return fused, alpha
