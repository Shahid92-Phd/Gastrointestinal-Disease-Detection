# Gastrointestinal-Disease-Detection
# Hybrid ViT-L/32 + MaxViT-L with Gated Fusion for Gastrointestinal Disease Classification

This repository provides the implementation of a hybrid transformer framework for multi-class gastrointestinal disease classification using endoscopic images from the Kvasir dataset.

## Features
- Dual-branch architecture: ViT-L/32 + MaxViT-L
- Adaptive gated fusion
- Training, evaluation, inference, and XAI support
- IEEE-style modular implementation for readability and reproducibility

## Dataset
Kvasir dataset with the following 8 classes:
- Dyed-lifted-polyps
- Dyed-resection-margins
- Esophagitis
- Normal-cecum
- Normal-pylorus
- Normal-z-line
- Polyps
- Ulcerative-colitis

## Installation
```bash
pip install -r requirements.txt
Training
python train.py
Evaluation
python evaluate.py
XAI Analysis
python xai_analysis.py

---

# 2. requirements.txt

```txt
torch
torchvision
timm
numpy
pandas
matplotlib
scikit-learn
opencv-python
Pillow
tqdm
captum
