import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from common.get_class import get_class_dict


def get_ds_from_cfg(config):
    """
    Get the dataset from the configuration dictionary.
    The train/val split is seeded (config["split_seed"]) so it is identical
    across runs, resumes and evaluation. config["samples_per_episode"] > 1 lists
    every training episode that many times per epoch (each with its own random
    window); it is applied after the split, so validation is unchanged.
    :param config: Configuration dictionary
    :return: (train, val) datasets
    """
    ds_train = get_class_dict(config)
    ds_val = None
    if config.get("val", False):
        n = len(ds_train)
        val_size = max(1, int(round(config["val_ratio"] * n)))
        train_size = n - val_size
        generator = torch.Generator().manual_seed(int(config.get("split_seed", 0)))
        ds_train, ds_val = random_split(
            ds_train, [train_size, val_size], generator=generator
        )
    repeats = int(config.get("samples_per_episode", 1))
    if repeats > 1:
        ds_train = RepeatDataset(ds_train, repeats)
    return ds_train, ds_val


class RepeatDataset(Dataset):
    """Every item `repeats` times per epoch; datasets that draw a random window
    per access therefore yield `repeats` different windows of each episode."""

    def __init__(self, dataset, repeats):
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


def base_dataset(dataset):
    """The dataset under Subset / RepeatDataset wrappers."""
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def get_max_min(list_items):
    full_np = np.concatenate(list_items)
    max = full_np.max(axis=0)
    min = full_np.min(axis=0)
    return max, min


def get_mean_std(list_items):
    full_np = np.concatenate(list_items)
    mean = full_np.mean(axis=0)
    std = full_np.std(axis=0)
    return mean, std
