"""Single-image inference script."""

import argparse
import torch
from PIL import Image
from torchvision import transforms

from configs.config import Config
from models.hybrid_model import HybridTransformerModel
from utils.checkpoints import load_checkpoint


def predict_image(image_path: str):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridTransformerModel(config).to(device)
    model = load_checkpoint(model, config.CHECKPOINT_PATH, device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(image)
        pred = torch.argmax(outputs, dim=1).item()

    print("Predicted Class:", config.CLASS_NAMES[pred])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    predict_image(args.image_path)
