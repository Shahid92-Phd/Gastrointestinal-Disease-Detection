"""Checkpoint save/load utilities."""

import torch


def save_checkpoint(model, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
