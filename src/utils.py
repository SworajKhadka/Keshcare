"""Small shared helpers: image preprocessing and device selection."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src import config


def get_device() -> str:
    """Picks GPU if available, otherwise CPU. Spaces' free CPU tier will
    always resolve to 'cpu' here, which is fine for inference."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_inference_transform() -> transforms.Compose:
    """Preprocessing pipeline applied to any image before it's fed to the
    model. Must match the transform used at training time (see
    training/train.py) or predictions will be unreliable."""
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def load_and_preprocess_image(image: Image.Image) -> torch.Tensor:
    """Converts a PIL image into a normalized tensor batch of shape
    (1, 3, IMAGE_SIZE, IMAGE_SIZE), ready for the model."""
    image = image.convert("RGB")
    transform = get_inference_transform()
    tensor = transform(image)
    return tensor.unsqueeze(0)  # add batch dimension


def pil_to_normalized_numpy(image: Image.Image) -> np.ndarray:
    """Returns a resized image as a float32 numpy array in [0, 1], which is
    the format pytorch-grad-cam expects for overlaying the heatmap."""
    image = image.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    arr = np.array(image).astype(np.float32) / 255.0
    return arr
