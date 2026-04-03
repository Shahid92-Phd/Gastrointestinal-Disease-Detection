"""
Factory function for baseline model construction.
"""

from .deit_base import DeiTBaseClassifier
from .vit_b16 import ViTB16Classifier
from .vit_l32 import ViTL32Classifier
from .transunet import TransUNetClassifier
from .transfuse import TransFuseClassifier
from .medvit import MedViTClassifier
from .pvt_v2 import PyramidVisionTransformerClassifier
from .swin_transformer_base import SwinTransformerBaseClassifier
from .swin_unetr import SwinUNETRClassifier
from .maxvit_large import MaxViTLargeClassifier


def build_baseline_model(model_name: str, num_classes: int = 8, pretrained: bool = True):
    name = model_name.lower()

    if name == "deit-base":
        return DeiTBaseClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "vit-b/16":
        return ViTB16Classifier(num_classes=num_classes, pretrained=pretrained)
    if name == "vit-l/32":
        return ViTL32Classifier(num_classes=num_classes, pretrained=pretrained)
    if name == "transunet":
        return TransUNetClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "transfuse":
        return TransFuseClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "medvit":
        return MedViTClassifier(num_classes=num_classes, pretrained=pretrained)
    if name in ("pyramid vision transformer", "pvt", "pvt-v2"):
        return PyramidVisionTransformerClassifier(num_classes=num_classes, pretrained=pretrained)
    if name in ("swin transformer-base", "swin-base", "swin"):
        return SwinTransformerBaseClassifier(num_classes=num_classes, pretrained=pretrained)
    if name == "swin-unetr":
        return SwinUNETRClassifier(num_classes=num_classes)
    if name in ("maxvit-large", "maxvit"):
        return MaxViTLargeClassifier(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(
        "Unsupported model_name. Choose from: "
        "DeiT-Base, ViT-B/16, ViT-L/32, TransUNet, TransFuse, MedViT, "
        "PVT-v2, Swin Transformer-Base, Swin-UNETR, MaxViT-Large"
    )
