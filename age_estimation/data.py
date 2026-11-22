"""Shared image preprocessing and loading for age regression."""

import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


INPUT_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FLIP_PROBABILITY = 0.5
BRIGHTNESS_JITTER = 0.15


def preprocessing_config():
    return {
        "color": "RGB",
        "exif_orientation": True,
        "size": INPUT_SIZE,
        "resize_interpolation": "bilinear",
        "mean": MEAN,
        "std": STD,
        "train_horizontal_flip_probability": FLIP_PROBABILITY,
        "train_brightness_jitter": BRIGHTNESS_JITTER,
    }


def oriented_rgb(image):
    return ImageOps.exif_transpose(image).convert("RGB")


def build_preprocess(*, training=False):
    steps = [oriented_rgb, transforms.Resize(INPUT_SIZE)]
    if training:
        steps.extend([
            transforms.RandomHorizontalFlip(p=FLIP_PROBABILITY),
            transforms.ColorJitter(brightness=BRIGHTNESS_JITTER),
        ])
    steps.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(steps)


class AgeDataset(Dataset):
    def __init__(self, image_root, manifest, *, training=False):
        self.image_root = Path(image_root)
        with Path(manifest).open(encoding="utf-8", newline="") as file:
            self.samples = [(Path(row["path"]), int(row["age"])) for row in csv.DictReader(file)]
        self.preprocess = build_preprocess(training=training)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, age = self.samples[index]
        with Image.open(self.image_root / path) as image:
            pixels = self.preprocess(image)
        return pixels, torch.tensor(age, dtype=torch.float32)
