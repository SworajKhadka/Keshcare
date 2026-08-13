"""
End-to-end inference pipeline: image in -> prediction + confidence +
Grad-CAM heatmap out. This is the module app.py calls; it deliberately
knows nothing about Streamlit so it can also be unit-tested or reused in
a FastAPI backend later without changes.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src import config
from src.gradcam import generate_gradcam_overlay
from src.model import load_trained_model
from src.utils import get_device, load_and_preprocess_image, pil_to_normalized_numpy


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_index: int
    confidence: float
    class_probabilities: dict  # class_name -> probability
    gradcam_overlay: np.ndarray  # RGB uint8 image
    is_fine_tuned: bool  # False => model hasn't been trained on real data yet


_MODEL_CACHE = {}


def _get_cached_model():
    """Loads the model once and reuses it across requests — avoids
    re-loading the checkpoint from disk on every single prediction, which
    matters on a CPU-only free-tier deployment."""
    device = get_device()
    if "model" not in _MODEL_CACHE:
        model, class_names, is_fine_tuned = load_trained_model(device=device)
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["class_names"] = class_names
        _MODEL_CACHE["is_fine_tuned"] = is_fine_tuned
        _MODEL_CACHE["device"] = device
    return (
        _MODEL_CACHE["model"],
        _MODEL_CACHE["class_names"],
        _MODEL_CACHE["is_fine_tuned"],
        _MODEL_CACHE["device"],
    )


def predict_image(image: Image.Image) -> PredictionResult:
    """Runs the full pipeline on a single PIL image."""
    model, class_names, is_fine_tuned, device = _get_cached_model()

    input_tensor = load_and_preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item())

    class_probabilities = {
        class_names[i]: float(probabilities[i].item()) for i in range(len(class_names))
    }

    # Grad-CAM needs gradients, so run it separately from the no_grad block.
    rgb_float = pil_to_normalized_numpy(image)
    gradcam_overlay = generate_gradcam_overlay(
        model=model,
        input_tensor=input_tensor,
        rgb_image_float=rgb_float,
        target_class=predicted_index,
    )

    return PredictionResult(
        predicted_class=class_names[predicted_index],
        predicted_index=predicted_index,
        confidence=confidence,
        class_probabilities=class_probabilities,
        gradcam_overlay=gradcam_overlay,
        is_fine_tuned=is_fine_tuned,
    )


def get_class_names() -> List[str]:
    _, class_names, _, _ = _get_cached_model()
    return class_names
