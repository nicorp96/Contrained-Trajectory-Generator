"""Normalization statistics from the training episodes only (validation does not leak in)."""

from typing import Dict, Iterable, Sequence

import h5py
import numpy as np

from ctg.data.maniskill import DemoDataset, find_dataset
from ctg.data.splits import split_episodes

MIN_SPREAD = 1e-4  # constant channels must not explode under min-max / standard normalization


def field_stats(data: np.ndarray) -> Dict[str, list]:
    """mean / std / min / max per channel of data [N, d]; min/max widened to MIN_SPREAD."""
    lo, hi = data.min(axis=0), data.max(axis=0)
    mid, spread = 0.5 * (lo + hi), np.maximum(hi - lo, MIN_SPREAD)
    return {
        "mean": data.mean(axis=0).tolist(),
        "std": np.maximum(data.std(axis=0), MIN_SPREAD).tolist(),
        "min": (mid - 0.5 * spread).tolist(),
        "max": (mid + 0.5 * spread).tolist(),
    }


def compute_stats(dataset: DemoDataset, keys: Iterable[str], val_ratio: float, seed: int) -> Dict[str, dict]:
    """Statistics of `keys` over all steps of the training episodes of `dataset`."""
    train_idx, _ = split_episodes(len(dataset), val_ratio, seed)
    keys: Sequence[str] = list(dict.fromkeys(keys))
    chunks = {k: [] for k in keys}
    with h5py.File(dataset.path, "r") as h5:
        for i in train_idx:
            traj = h5[f"traj_{dataset.episodes[i]['episode_id']}"]
            n_steps = traj[dataset.action_key].shape[0]
            for key in keys:
                data = np.asarray(find_dataset(traj, key)[:n_steps], dtype=np.float64)
                chunks[key].append(data.reshape(n_steps, -1))
    return {k: field_stats(np.concatenate(v, axis=0)) for k, v in chunks.items()}
