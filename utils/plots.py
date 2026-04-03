"""Basic plotting utilities for training curves."""

import os
import matplotlib.pyplot as plt


def plot_training_history(train_losses, val_losses, val_accuracies, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=300)
    plt.close()
