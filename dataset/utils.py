import numpy as np
import torch
from torch.utils.data import random_split
from common.get_class import get_class_dict


def get_ds_from_cfg(config):
    """
    Get the dataset from the configuration dictionary.
    The train/val split is seeded (config["split_seed"]) so it is identical
    across runs, resumes and evaluation.
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
    return ds_train, ds_val


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
