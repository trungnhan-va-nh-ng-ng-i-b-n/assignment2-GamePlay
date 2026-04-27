from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        channels: int = 64,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, channels)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * 8 * 8, 64)

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(1 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(self.stem(x))

        policy = self.policy_head(features)
        policy_logits = self.policy_fc(policy.flatten(1))

        value = self.value_head(features)
        value = self.value_fc(value.flatten(1)).squeeze(1)

        return policy_logits, value
