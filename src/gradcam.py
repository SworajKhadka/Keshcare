"""
Grad-CAM explainability for the classifier.

Grad-CAM highlights *which pixels* in the input image most influenced the
model's prediction, rendered as a heatmap overlay. This is what separates
"I fine-tuned a CNN" from "I fine-tuned a CNN and can show it's looking at
the right region of the scalp/hairline" — worth keeping even though it adds
one extra dependency (pytorch-grad-cam).
"""

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_target_layer(model: torch.nn.Module):
    """EfficientNet-B0's last convolutional block — the standard choice of
    target layer for Grad-CAM on this architecture."""
    return model.features[-1]


def generate_gradcam_overlay(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_image_float: np.ndarray,
    target_class: int,
) -> np.ndarray:
    """Runs Grad-CAM and returns an RGB uint8 image (heatmap blended onto
    the original picture) ready to display in Streamlit.

    Args:
        model: the (eval-mode) classifier.
        input_tensor: preprocessed input, shape (1, 3, H, W).
        rgb_image_float: original image resized to the same H, W, as a
            float32 numpy array in [0, 1], shape (H, W, 3).
        target_class: index of the class to explain (usually the model's
            top prediction).
    """
    target_layers = [get_target_layer(model)]
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # grayscale_cam shape: (batch, H, W) -> take the first (only) image
        grayscale_cam = grayscale_cam[0, :]
        overlay = show_cam_on_image(rgb_image_float, grayscale_cam, use_rgb=True)
        return overlay
