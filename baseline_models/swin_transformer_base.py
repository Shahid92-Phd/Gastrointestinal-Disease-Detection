"""
Swin Transformer-Base baseline using timm.
"""

from .common import TimmClassifier


class SwinTransformerBaseClassifier(TimmClassifier):
    def __init__(self, num_classes: int = 8, pretrained: bool = True, model_name: str = "swin_base_patch4_window7_224"):
        super().__init__(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
