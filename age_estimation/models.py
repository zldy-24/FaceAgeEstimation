"""Single-output age regressors and checkpoint loading."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from age_estimation.data import preprocessing_config


RESNET18_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
CHECKPOINT_VERSION = 1


class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = (3, 16, 32, 64, 128)
        for inputs, outputs in zip(channels, channels[1:]):
            layers.extend([
                nn.Conv2d(inputs, outputs, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, images):
        return self.fc(self.pool(self.features(images)).flatten(1))


def build_model(architecture, *, pretrained=False):
    if architecture == "cnn":
        return BasicCNN()
    if architecture == "resnet18":
        model = resnet18(weights=RESNET18_WEIGHTS if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    raise ValueError(f"Unknown model architecture: {architecture}")


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["version"] != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version in {path}")
    config = checkpoint["config"]
    if config["preprocessing"] != preprocessing_config():
        raise ValueError(f"Checkpoint preprocessing differs from the current project in {path}")
    model = build_model(config["architecture"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
