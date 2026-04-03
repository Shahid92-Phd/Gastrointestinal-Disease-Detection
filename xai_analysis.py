"""
Full XAI implementation for the proposed Hybrid ViT-L/32 + MaxViT-L model.

Methods:
    1. SmoothGrad
    2. Integrated Gradients
    3. Occlusion Sensitivity
    4. LIME
    5. Grad-CAM
    6. Grad-CAM++
    7. Score-CAM

Output:
    A paper-style comparison grid with the following columns:
    Original | SmoothGrad | Integrated Gradients | Occlusion |
    LIME | GradCAM | GradCAM++ | ScoreCAM

Usage:
    python xai_analysis_full.py --image_path /path/to/img.png --checkpoint /path/to/best_model.pth

Notes:
    - This script expects the repository structure used in the earlier hybrid model package.
    - It imports `HybridTransformerModel` from `models/hybrid_model.py`.
"""

from __future__ import annotations

import os
import argparse
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from captum.attr import Saliency, IntegratedGradients, Occlusion, NoiseTunnel
from lime import lime_image
from skimage.segmentation import slic

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Project imports
from configs.config import Config
from models.hybrid_model import HybridTransformerModel
from utils.checkpoints import load_checkpoint


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def denormalize_imagenet(t: torch.Tensor) -> torch.Tensor:
    """Convert normalized tensor back to displayable [0,1] RGB."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
    x = t * std + mean
    return torch.clamp(x, 0.0, 1.0)


def tensor_to_rgb_np01(x: torch.Tensor) -> np.ndarray:
    """Convert CHW torch tensor [0,1] to HWC numpy [0,1]."""
    return x.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def heatmap_from_attr(attr: torch.Tensor) -> np.ndarray:
    """Aggregate attribution across channels and normalize to [0,1]."""
    if attr.dim() == 4:
        attr = attr[0]
    hm = attr.abs().mean(dim=0).detach().cpu().numpy()
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    return hm


def overlay_heatmap(rgb01: np.ndarray, heat01: np.ndarray) -> np.ndarray:
    """Overlay a JET heatmap on an RGB image."""
    import cv2

    heat = cv2.applyColorMap((heat01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    out = 0.55 * rgb01 + 0.45 * heat
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def find_last_conv2d(module: nn.Module) -> nn.Module:
    """Find the last Conv2d layer for CAM-based explanations."""
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found. CAM methods require a convolutional target layer.")
    return last


# ---------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------
class LogitsOnlyWrapper(nn.Module):
    """Wrap a model that returns (logits, aux) so XAI methods receive logits only."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.base_model(x)
        return logits


# ---------------------------------------------------------------------
# XAI engine
# ---------------------------------------------------------------------
class XAIEngine:
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str]):
        self.model = model.to(device).eval()
        self.device = device
        self.class_names = class_names
        self.model_logits = LogitsOnlyWrapper(self.model).to(device).eval()
        self.cam_target_layers = [find_last_conv2d(self.model.maxvit)]

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model_logits(x)
        return F.softmax(logits, dim=1)

    def smoothgrad(self, x: torch.Tensor, target: int) -> np.ndarray:
        sal = Saliency(self.model_logits)
        nt = NoiseTunnel(sal)
        attr = nt.attribute(x, nt_type="smoothgrad", stdevs=0.15, n_samples=20, target=target)
        hm = heatmap_from_attr(attr)
        rgb = tensor_to_rgb_np01(denormalize_imagenet(x[0]))
        return overlay_heatmap(rgb, hm)

    def integrated_gradients(self, x: torch.Tensor, target: int) -> np.ndarray:
        ig = IntegratedGradients(self.model_logits)
        baseline = torch.zeros_like(x)
        attr = ig.attribute(x, baselines=baseline, target=target, n_steps=50)
        hm = heatmap_from_attr(attr)
        rgb = tensor_to_rgb_np01(denormalize_imagenet(x[0]))
        return overlay_heatmap(rgb, hm)

    def occlusion(self, x: torch.Tensor, target: int) -> np.ndarray:
        occ = Occlusion(self.model_logits)
        attr = occ.attribute(
            x,
            target=target,
            strides=(3, 16, 16),
            sliding_window_shapes=(3, 32, 32),
            baselines=0
        )
        hm = heatmap_from_attr(attr)
        rgb = tensor_to_rgb_np01(denormalize_imagenet(x[0]))
        return overlay_heatmap(rgb, hm)

    def lime(self, x: torch.Tensor, target: int) -> np.ndarray:
        rgb01 = tensor_to_rgb_np01(denormalize_imagenet(x[0]))

        def batch_predict(images):
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            batch_tensors = []
            for im in images:
                im = np.clip(im, 0, 1).astype(np.float32)
                im = (im - mean) / std
                t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
                batch_tensors.append(t)
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            with torch.no_grad():
                probs = self.predict_proba(batch).cpu().numpy()
            return probs

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            rgb01.astype(np.double),
            classifier_fn=batch_predict,
            top_labels=len(self.class_names),
            hide_color=0,
            num_samples=1000,
            segmentation_fn=lambda im: slic(im, n_segments=150, compactness=10, sigma=1),
        )

        _, mask = explanation.get_image_and_mask(
            label=target,
            positive_only=True,
            num_features=12,
            hide_rest=False
        )
        mask = (mask > 0).astype(np.float32)
        mask = mask - mask.min()
        mask = mask / (mask.max() + 1e-8)
        return overlay_heatmap(rgb01, mask)

    def cam(self, x: torch.Tensor, target: int, method: str) -> np.ndarray:
        rgb01 = tensor_to_rgb_np01(denormalize_imagenet(x[0]))
        if method == "gradcam":
            cam = GradCAM(model=self.model_logits, target_layers=self.cam_target_layers)
        elif method == "gradcam++":
            cam = GradCAMPlusPlus(model=self.model_logits, target_layers=self.cam_target_layers)
        elif method == "scorecam":
            cam = ScoreCAM(model=self.model_logits, target_layers=self.cam_target_layers)
        else:
            raise ValueError(f"Unsupported CAM method: {method}")

        targets = [ClassifierOutputTarget(target)]
        grayscale_cam = cam(input_tensor=x, targets=targets)[0]
        vis = show_cam_on_image(rgb01, grayscale_cam, use_rgb=True)
        return vis

    def create_grid(self, x: torch.Tensor, target: int, save_path: str, title: Optional[str] = None) -> str:
        x = x.to(self.device)

        original = (tensor_to_rgb_np01(denormalize_imagenet(x[0])) * 255).astype(np.uint8)
        smooth = self.smoothgrad(x, target)
        ig = self.integrated_gradients(x, target)
        occ = self.occlusion(x, target)
        lime_img = self.lime(x, target)
        gc = self.cam(x, target, "gradcam")
        gcpp = self.cam(x, target, "gradcam++")
        score = self.cam(x, target, "scorecam")

        images = [original, smooth, ig, occ, lime_img, gc, gcpp, score]
        titles = [
            "Original image",
            "SmoothGrad",
            "Integrated Gradients",
            "Occlusion Siliency Map",
            "LIME",
            "GradCAM",
            "GradCAM++",
            "ScoreCAM",
        ]

        plt.figure(figsize=(24, 4))
        for i, (im, name) in enumerate(zip(images, titles), start=1):
            ax = plt.subplot(1, 8, i)
            ax.imshow(im)
            ax.set_title(name, fontsize=10)
            ax.axis("off")

        if title is None:
            title = f"Class-wise XAI Analysis: {self.class_names[target]}"
        plt.suptitle(title, fontsize=14, y=1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--output", type=str, default="xai_grid.png", help="Output PNG path")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint if args.checkpoint is not None else config.CHECKPOINT_PATH

    model = HybridTransformerModel(config).to(device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    transform = build_transform(config.IMAGE_SIZE)
    image = Image.open(args.image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(x)
        pred = torch.argmax(logits, dim=1).item()

    engine = XAIEngine(model=model, device=device, class_names=config.CLASS_NAMES)
    save_path = engine.create_grid(
        x=x,
        target=pred,
        save_path=args.output,
        title=f"XAI Results for Predicted Class: {config.CLASS_NAMES[pred]}"
    )
    print(f"Saved XAI grid to: {save_path}")


if __name__ == "__main__":
    main()
