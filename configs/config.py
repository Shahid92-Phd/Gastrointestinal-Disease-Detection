"""
Configuration file for the hybrid transformer model.
All hyperparameters and paths are defined here for reproducibility.
"""

import os


class Config:
    DATA_DIR = "./kvasir-dataset"
    CLASS_NAMES = [
        "Dyed-lifted-polyps",
        "Dyed-resection-margins",
        "Esophagitis",
        "Normal-cecum",
        "Normal-pylorus",
        "Normal-z-line",
        "Polyps",
        "Ulcerative-colitis",
    ]
    CLASS_ABBR = ["DLP", "DRM", "EP", "NC", "NP", "NZL", "PP", "UC"]
    NUM_CLASSES = 8
    IMAGE_SIZE = 224

    TRAIN_RATIO = 0.60
    VAL_RATIO = 0.20
    TEST_RATIO = 0.20

    BATCH_SIZE = 16
    EPOCHS = 100
    LR_HEAD = 3e-4
    LR_BACKBONE = 1e-4
    WEIGHT_DECAY = 1e-2
    LABEL_SMOOTHING = 0.1
    NUM_WORKERS = 4
    SEED = 42

    VIT_MODEL_NAME = "vit_large_patch32_224"
    MAXVIT_MODEL_NAME = "maxvit_large_tf_224"
    FUSION_DIM = 1024
    DROPOUT_FUSION = 0.3
    DROPOUT_HEAD = 0.2

    OUTPUT_DIR = "./outputs"
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
