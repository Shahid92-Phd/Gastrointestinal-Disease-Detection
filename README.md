# Additional Python Files for the Kvasir Hybrid Transformer Repository

This package contains:

1. `xai_analysis_full.py`
   - Full XAI implementation for:
     - SmoothGrad
     - Integrated Gradients
     - Occlusion Sensitivity
     - LIME
     - Grad-CAM
     - Grad-CAM++
     - Score-CAM
   - Generates a paper-style grid:
     `Original | SmoothGrad | Integrated Gradients | Occlusion | LIME | GradCAM | GradCAM++ | ScoreCAM`

2. `baseline_models/`
   - Baseline `.py` implementations for:
     - DeiT-Base
     - ViT-B/16
     - ViT-L/32
     - TransUNet (classification adaptation)
     - TransFuse (classification adaptation)
     - MedViT (lightweight classification-oriented approximation)
     - Pyramid Vision Transformer (PVT-v2)
     - Swin Transformer-Base
     - Swin-UNETR (classification adaptation)
     - MaxViT-Large

## Notes
- Models backed by `timm` use well-known model identifiers and pretrained weights when available.
- `SwinUNETR` uses `MONAI` if installed.
- `TransUNet`, `TransFuse`, and `MedViT` are classification-oriented baseline adaptations intended for fair comparison in our GI Detection and classification paper.

## Installation
```bash
pip install torch torchvision timm numpy matplotlib scikit-learn pillow opencv-python captum lime scikit-image grad-cam
# Optional for Swin-UNETR:
pip install monai
```
