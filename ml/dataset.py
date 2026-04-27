from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class OthelloNpzDataset(Dataset):
    REQUIRED_KEYS = ("states", "legal_masks", "policy_targets", "outcomes")

    def __init__(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        payload = np.load(file_path)

        missing = [key for key in self.REQUIRED_KEYS if key not in payload]
        if missing:
            raise ValueError(f"Dataset missing keys: {missing}")

        states = payload["states"].astype(np.float32)
        legal_masks = payload["legal_masks"].astype(np.float32)
        policy_targets = payload["policy_targets"].astype(np.int64)
        outcomes = payload["outcomes"].astype(np.float32)

        if len(states) == 0:
            raise ValueError("Dataset is empty")

        self.states = torch.from_numpy(states)
        self.legal_masks = torch.from_numpy(legal_masks)
        self.policy_targets = torch.from_numpy(policy_targets)
        self.outcomes = torch.from_numpy(outcomes)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int):
        return (
            self.states[index],
            self.legal_masks[index],
            self.policy_targets[index],
            self.outcomes[index],
        )


def split_dataset(dataset: Dataset, val_ratio: float, seed: int):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset
