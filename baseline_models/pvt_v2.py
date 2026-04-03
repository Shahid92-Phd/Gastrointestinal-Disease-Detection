"""
Pyramid Vision Transformer (PVT-v2) baseline using timm.

Default choice: pvt_v2_b2
You may switch to pvt_v2_b3 or pvt_v2_b4 depending on GPU budget.
"""

from .common import TimmClassifier


class PyramidVisionTransformerClassifier(TimmClassifier):
    def __init__(self, num_classes: int = 8, pretrained: bool = True, model_name: str = "pvt_v2_b2"):
        super().__init__(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
