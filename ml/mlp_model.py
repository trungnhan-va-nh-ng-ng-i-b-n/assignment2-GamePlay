"""
ml/mlp_model.py — OthelloMLP architecture definition.

Architecture:
    Input (4×8×8 = 256) → 6 FC hidden layers → dual output heads
        ├── policy_head: FC(64) → logits over 64 squares
        └── value_head:  FC(64) → FC(1) → tanh → eval ∈ [-1, +1]

Input encoding (4 channels):
    ch0: own discs   (1.0 where current player has disc)
    ch1: opp discs   (1.0 where opponent has disc)
    ch2: empty cells (1.0 where cell is empty)
    ch3: turn plane  (1.0 if player==1 Black, 0.0 if player==-1 White)

Note: This file defines the architecture only.
      For the full agent (search + inference), see ml/mlp_agent.py.
"""

from __future__ import annotations

import torch
from torch import nn


class OthelloMLP(nn.Module):
    """
    6-layer MLP for Othello position evaluation.

    Input:  (B, 4, 8, 8) board tensor  →  flatten  →  256 features
    Output: (policy_logits, value)
              policy_logits: (B, 64) — unnormalised move scores
              value:         (B,)    — tanh position eval ∈ [-1, 1]
                                       +1 = current player wins, -1 = loses
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden: tuple[int, ...] = (1024, 512, 512, 256, 256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Build FC backbone: flatten → hidden layers with BN + ReLU + Dropout
        layers, prev = [], in_channels * 64
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h

        self.backbone    = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, 64)
        self.value_head  = nn.Sequential(
            nn.Linear(prev, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),   nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 4, 8, 8) float tensor

        Returns:
            policy_logits: (B, 64)
            value:         (B,)
        """
        f = self.backbone(x.view(x.size(0), -1))
        return self.policy_head(f), self.value_head(f).squeeze(1)
