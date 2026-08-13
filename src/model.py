"""
Model architecture and checkpoint loading for the hair/scalp classifier.

We use EfficientNet-B0 pretrained on ImageNet as the backbone (transfer
learning) and replace its final classification layer with one sized to our
own number of classes. This is a standard, well-tested approach for
small/medium image datasets — you get ImageNet's learned visual features
for free and only need to fine-tune the last layer(s) on your own data.
"""

import json
import os

import torch
import torch.nn as nn
from torchvision import models

from src import config


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Builds an EfficientNet-B0 with a custom classification head.

    Args:
        num_classes: number of output classes.
        pretrained: if True, loads ImageNet-pretrained weights for the
            backbone (recommended, this is what makes it "transfer learning").
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freezes every layer except the final classifier head.

    Useful for the first phase of fine-tuning on a small dataset: training
    only the new head first avoids destroying the pretrained features before
    they've had a chance to adapt.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreezes every layer — call this for a second fine-tuning phase
    with a lower learning rate once the head has stabilized."""
    for param in model.parameters():
        param.requires_grad = True


def load_class_names() -> list:
    """Loads class names produced by training, falling back to the default
    list in config.py if no training run has happened yet."""
    if os.path.exists(config.CLASS_NAMES_PATH):
        with open(config.CLASS_NAMES_PATH, "r") as f:
            return json.load(f)
    return config.DEFAULT_CLASS_NAMES


def load_trained_model(device: str = "cpu"):
    """Loads the classifier for inference.

    Returns a tuple (model, class_names, is_fine_tuned).

    is_fine_tuned is False when no checkpoint exists yet (i.e. you haven't
    run the Colab training notebook), in which case we still return a
    usable model — the ImageNet-pretrained backbone with a freshly
    initialized (untrained) head — so the app can run end-to-end in a
    clearly-labeled demo mode instead of crashing.
    """
    class_names = load_class_names()
    num_classes = len(class_names)

    if os.path.exists(config.CLASSIFIER_WEIGHTS_PATH):
        model = build_model(num_classes=num_classes, pretrained=False)
        checkpoint = torch.load(config.CLASSIFIER_WEIGHTS_PATH, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, class_names, True

    # No fine-tuned weights yet -> demo mode.
    model = build_model(num_classes=num_classes, pretrained=True)
    model.to(device)
    model.eval()
    return model, class_names, False
