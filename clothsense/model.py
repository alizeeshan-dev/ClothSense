from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class FashionCNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        first_channels, second_channels = config.conv_channels
        self.features = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                first_channels,
                kernel_size=config.kernel_size,
                padding=config.padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(config.pool_size),
            nn.Conv2d(
                first_channels,
                second_channels,
                kernel_size=config.kernel_size,
                padding=config.padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(config.pool_size),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second_channels * 7 * 7, config.hidden_units),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_units, config.num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))

