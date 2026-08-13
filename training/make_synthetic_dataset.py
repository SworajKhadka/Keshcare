"""
Generates a tiny synthetic image dataset purely so you can smoke-test the
training pipeline (train.py) end-to-end in a couple of minutes BEFORE
spending time downloading a real dataset. The images are random noise
patterns, not real hair/scalp photos — they exist only to prove the code
runs without errors. Do not use the model trained on this data for real
predictions; replace this data with a real dataset (see SETUP_GUIDE.md)
before doing your actual training run.

Run:
    python training/make_synthetic_dataset.py
"""

import os

import numpy as np
from PIL import Image

CLASSES = ["Healthy", "Early_Stage", "Moderate", "Advanced"]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
IMAGES_PER_CLASS_TRAIN = 24
IMAGES_PER_CLASS_VAL = 8
IMG_SIZE = 224


def make_image(seed: int, density: float) -> Image.Image:
    """Creates a synthetic RGB image where `density` (0-1) controls how
    many bright 'dots' are drawn, loosely standing in for hair density so
    the four classes are at least visually distinguishable to a CNN."""
    rng = np.random.default_rng(seed)
    base = rng.integers(180, 220, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    n_dots = int(density * 3000)
    ys = rng.integers(0, IMG_SIZE, size=n_dots)
    xs = rng.integers(0, IMG_SIZE, size=n_dots)
    base[ys, xs] = rng.integers(20, 80, size=(n_dots, 3), dtype=np.uint8)

    return Image.fromarray(base, mode="RGB")


def build_split(split_dir: str, n_per_class: int, seed_offset: int):
    # Higher class index => lower "density" => less hair, mimicking more
    # advanced hair loss.
    densities = [0.85, 0.6, 0.35, 0.12]

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(n_per_class):
            seed = seed_offset + class_idx * 10_000 + i
            img = make_image(seed=seed, density=densities[class_idx])
            img.save(os.path.join(class_dir, f"{class_name.lower()}_{i:03d}.jpg"))


if __name__ == "__main__":
    build_split(os.path.join(OUT_DIR, "train"), IMAGES_PER_CLASS_TRAIN, seed_offset=0)
    build_split(os.path.join(OUT_DIR, "val"), IMAGES_PER_CLASS_VAL, seed_offset=999_999)
    print(f"Synthetic smoke-test dataset written under {OUT_DIR}/train and {OUT_DIR}/val")
    print("Reminder: this is fake data for testing the pipeline only.")
