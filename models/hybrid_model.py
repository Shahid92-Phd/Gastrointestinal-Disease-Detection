"""
Main hybrid model implementation.
IEEE-style modular structure:
- ViT feature extractor
- MaxViT feature extractor
- Projection layers
- Gated fusion
- Classification head
"""

import torch
import torch.nn as nn
import timm

from models.fusion import GatedFusion


class HybridTransformerModel(nn.Module):
    """Hybrid ViT-L/32 + MaxViT-L model with adaptive gated fusion."""

    def __init__(self, config):
        super().__init__()

        self.vit = timm.create_model(
            config.VIT_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.maxvit = timm.create_model(
            config.MAXVIT_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.vit_dim = self.vit.num_features
        self.maxvit_dim = self.maxvit.num_features

        self.vit_proj = nn.Linear(self.vit_dim, config.FUSION_DIM)
        self.maxvit_proj = nn.Linear(self.maxvit_dim, config.FUSION_DIM)

        self.fusion = GatedFusion(
            input_dim=config.FUSION_DIM,
            dropout=config.DROPOUT_FUSION
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.FUSION_DIM),
            nn.Dropout(config.DROPOUT_HEAD),
            nn.Linear(config.FUSION_DIM, config.NUM_CLASSES)
        )

    def forward(self, x: torch.Tensor):
        feat_vit = self.vit(x)
        feat_maxvit = self.maxvit(x)

        feat_vit = self.vit_proj(feat_vit)
        feat_maxvit = self.maxvit_proj(feat_maxvit)

        fused_feat, gate_score = self.fusion(feat_vit, feat_maxvit)
        logits = self.classifier(fused_feat)
        return logits, gate_score
