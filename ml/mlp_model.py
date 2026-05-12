"""
OthelloMLP — Pure Multi-Layer Perceptron (NO CNN) for Othello.

Architecture:
    Flatten(3×8×8 = 192) → FC(512) → FC(256) → FC(128)
        ├── Policy Head: FC(64) → masked softmax → move probabilities
        └── Value Head:  FC(1)  → tanh           → board evaluation in [-1, +1]

Key difference from PolicyValueNet (CNN):
    - No convolutional layers → no spatial inductive bias
    - Board is treated as a flat vector of 192 independent features
    - Must learn spatial relationships (adjacency, corners, edges) from data alone
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class OthelloMLP(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_sizes: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.in_features = in_channels * 8 * 8
        self.in_channels = in_channels

        # Build FC backbone
        layers = []
        prev_size = self.in_features
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_size = h

        self.backbone = nn.Sequential(*layers)

        # Policy Head
        self.policy_head = nn.Linear(prev_size, 64)

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, C, 8, 8) float tensor — same input format as CNN version

        Returns:
            policy_logits: (batch, 64)
            value:         (batch,)
        """
        # Flatten: (batch, C, 8, 8) → (batch, C*64)
        x = x.view(x.size(0), -1)

        features = self.backbone(x)

        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(1)

        return policy_logits, value
