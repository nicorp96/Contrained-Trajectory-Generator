"""Episode-level train/validation split and per-epoch repetition of training episodes."""

from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset, random_split


class RepeatDataset(Dataset):
    """Every item `repeats` times per epoch. Datasets that draw a random window per
    access therefore yield `repeats` different windows of each episode."""

    def __init__(self, dataset: Dataset, repeats: int):
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        self.dataset = dataset
        self.repeats = int(repeats)

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        if not -len(self) <= idx < len(self):
            raise IndexError(idx)
        return self.dataset[idx % len(self.dataset)]


def split_episodes(n: int, val_ratio: float, seed: int = 0) -> Tuple[list, list]:
    """Seeded episode indices (train, val); identical for training, stats and evaluation."""
    if not val_ratio:
        return list(range(n)), []
    val_size = max(1, int(round(val_ratio * n)))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed))).tolist()
    return perm[: n - val_size], perm[n - val_size:]


def make_splits(
    dataset: Dataset, val_ratio: float = 0.2, seed: int = 0, samples_per_episode: int = 1
) -> Tuple[Dataset, Optional[Dataset]]:
    """Train/val datasets. Repetition is applied after the split, so validation is unchanged."""
    train_idx, val_idx = split_episodes(len(dataset), val_ratio, seed)
    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx) if val_idx else None
    if samples_per_episode > 1:
        train = RepeatDataset(train, samples_per_episode)
    return train, val


def base_dataset(dataset: Dataset) -> Dataset:
    """The dataset under Subset / RepeatDataset wrappers."""
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset
