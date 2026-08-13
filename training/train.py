"""
Fine-tunes the EfficientNet-B0 classifier on your hair/scalp image dataset.

This is designed to run in Google Colab with a free GPU (Runtime -> Change
runtime type -> GPU). It also runs on CPU (just slower) if you'd rather run
it locally in VS Code on a small dataset.

Expected data layout (standard torchvision ImageFolder format) — you build
this yourself after downloading a dataset (see SETUP_GUIDE.md for dataset
suggestions and links):

    data/
      train/
        Healthy/
          img001.jpg
          img002.jpg
          ...
        Early_Stage/
          ...
        Moderate/
          ...
        Advanced/
          ...
      val/
        Healthy/
          ...
        Early_Stage/
          ...
        ...

The folder names become your class names automatically — you are not
restricted to the 4 example categories above, use whatever classes your
chosen dataset provides.

Run:
    python training/train.py --data_dir data --epochs 15 --lr 0.001
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.model import build_model, freeze_backbone, unfreeze_all  # noqa: E402


def get_data_loaders(data_dir: str, batch_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, train_dataset.classes


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_correct += torch.sum(preds == labels).item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def train_model(data_dir: str, epochs: int, lr: float, batch_size: int, freeze_epochs: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size)
    print(f"Classes found: {class_names}")

    model = build_model(num_classes=len(class_names), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        start = time.time()

        # Train only the classifier head for the first `freeze_epochs`
        # epochs, then unfreeze the whole network for fine-tuning. This
        # two-phase approach is standard practice and gives noticeably
        # better results than unfreezing everything from epoch 1 on a
        # small dataset.
        if epoch < freeze_epochs:
            freeze_backbone(model)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr
            )
        else:
            unfreeze_all(model)
            optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        elapsed = time.time() - start
        print(
            f"Epoch {epoch + 1}/{epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save({"model_state_dict": best_weights}, config.CLASSIFIER_WEIGHTS_PATH)
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Saved weights to {config.CLASSIFIER_WEIGHTS_PATH}")
    print(f"Saved class names to {config.CLASS_NAMES_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=3,
        help="Number of initial epochs to train only the classifier head.",
    )
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        freeze_epochs=args.freeze_epochs,
    )
