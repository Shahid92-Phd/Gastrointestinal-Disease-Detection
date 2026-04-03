"""
Training script for the hybrid transformer model.
Step-by-step and well-commented for GitHub and IEEE-style reproducibility.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import Config
from datasets.kvasir_dataset import KvasirDataset, collect_image_paths, create_splits
from models.hybrid_model import HybridTransformerModel
from utils.seed import set_seed
from utils.checkpoints import save_checkpoint
from utils.plots import plot_training_history


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """Evaluate the model on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    return running_loss / len(loader), val_acc


def main():
    config = Config()
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, labels = collect_image_paths(config.DATA_DIR, config.CLASS_NAMES)
    x_train, x_val, x_test, y_train, y_val, y_test = create_splits(
        image_paths, labels, seed=config.SEED
    )

    train_dataset = KvasirDataset(x_train, y_train, image_size=config.IMAGE_SIZE, train=True)
    val_dataset = KvasirDataset(x_val, y_val, image_size=config.IMAGE_SIZE, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )

    model = HybridTransformerModel(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR_HEAD,
        weight_decay=config.WEIGHT_DECAY
    )

    best_val_acc = 0.0
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, config.CHECKPOINT_PATH)

    plot_training_history(train_losses, val_losses, val_accuracies, config.OUTPUT_DIR)
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
