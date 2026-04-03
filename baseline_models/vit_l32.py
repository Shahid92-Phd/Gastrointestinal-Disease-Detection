"""
ViT-L/32 baseline using timm.
"""

from .common import TimmClassifier


class ViTL32Classifier(TimmClassifier):
    def __init__(self, num_classes: int = 8, pretrained: bool = True, model_name: str = "vit_large_patch32_224"):
        super().__init__(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
