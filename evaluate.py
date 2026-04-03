"""Evaluation script for test set performance."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import Config
from datasets.kvasir_dataset import KvasirDataset, collect_image_paths, create_splits
from models.hybrid_model import HybridTransformerModel
from utils.metrics import compute_metrics
from utils.checkpoints import load_checkpoint


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, labels = collect_image_paths(config.DATA_DIR, config.CLASS_NAMES)
    x_train, x_val, x_test, y_train, y_val, y_test = create_splits(
        image_paths, labels, seed=config.SEED
    )

    test_dataset = KvasirDataset(x_test, y_test, image_size=config.IMAGE_SIZE, train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )

    model = HybridTransformerModel(config).to(device)
    model = load_checkpoint(model, config.CHECKPOINT_PATH, device)
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs, _ = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))

    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
