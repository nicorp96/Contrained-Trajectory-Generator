import numpy as np
from torch.utils.data import random_split
from common.get_class import get_class_dict


def get_ds_from_cfg(config):
    """
    Get the dataset from the configuration dictionary.
    :param config: Configuration dictionary
    :return: Dataset object
    """
    ds_train = get_class_dict(config)
    ds_val = None
    if config["val"]:
        train_size = int((1.0 - config["val_ratio"]) * len(ds_train))
        # TODO: Check if +1
        val_size = int(config["val_ratio"] * len(ds_train))
        ds_train, ds_val = random_split(ds_train, [train_size, val_size])
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
