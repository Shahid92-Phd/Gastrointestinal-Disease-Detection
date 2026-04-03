"""
DeiT-Base baseline using timm.
"""

from .common import TimmClassifier


class DeiTBaseClassifier(TimmClassifier):
    def __init__(self, num_classes: int = 8, pretrained: bool = True, model_name: str = "deit_base_patch16_224"):
        super().__init__(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
