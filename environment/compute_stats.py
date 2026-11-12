"""
Normalizer statistics (stats.json) for a ManiSkill demo dataset.

Statistics are computed on the *training* episodes only, using the same seeded
episode split as dataset.utils.get_ds_from_cfg, so validation data does not leak
into the normalization.

  python environment/compute_stats.py -c diffusion_traj_pad \
      --set dataset.path=<h5> --out <stats.json>

Keys: every non-camera key in the config's state/goal/obs shapes.
Output format per key: {"mean": [...], "std": [...], "min": [...], "max": [...]}.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import apply_overrides, load_config  # noqa: E402
from dataset.dataset_maniskill import BaseDataset, _resolve_path  # noqa: E402
from global_parameters import ConfigGlobalP  # noqa: E402

cfg_gp = ConfigGlobalP()
MIN_SPREAD = 1e-4  # avoids exploding normalization for constant channels


def training_episode_indices(n_episodes, ds_cfg):
    """Replicates torch.utils.data.random_split in dataset.utils.get_ds_from_cfg."""
    if not ds_cfg.get("val", False):
        return list(range(n_episodes))
    val_size = max(1, int(round(ds_cfg["val_ratio"] * n_episodes)))
    train_size = n_episodes - val_size
    generator = torch.Generator().manual_seed(int(ds_cfg.get("split_seed", 0)))
    perm = torch.randperm(n_episodes, generator=generator).tolist()
    return perm[:train_size]


def main(args):
    config = load_config(os.path.join(cfg_gp.ROOT_DIR, "configs", args.config + ".yaml"))
    apply_overrides(config, args.set)
    ds_cfg = config["dataset"]

    h5_path = _resolve_path(ds_cfg["path"])
    with open(h5_path.replace(".h5", ".json")) as f:
        episodes = json.load(f)["episodes"]
    if ds_cfg.get("success_only", True):
        episodes = [e for e in episodes if e.get("success", False)]
    train_idx = training_episode_indices(len(episodes), ds_cfg)

    keys = []
    for group in ("state_shapes", "goal_shapes", "obs_shapes"):
        for key in (ds_cfg.get(group) or {}):
            if "camera" not in key and key not in keys:
                keys.append(key)

    values = {k: [] for k in keys}
    with h5py.File(h5_path, "r") as h5:
        for i in tqdm(train_idx, desc="episodes"):
            traj = h5[f"traj_{episodes[i]['episode_id']}"]
            eps_len = traj["actions"].shape[0]
            for key in keys:
                data = np.asarray(BaseDataset.find_dataset(traj, key)[:eps_len], dtype=np.float64)
                values[key].append(data.reshape(eps_len, -1))

    stats = {}
    for key, chunks in values.items():
        data = np.concatenate(chunks, axis=0)
        lo, hi = data.min(axis=0), data.max(axis=0)
        mid = 0.5 * (lo + hi)
        spread = np.maximum(hi - lo, MIN_SPREAD)
        stats[key] = {
            "mean": data.mean(axis=0).tolist(),
            "std": np.maximum(data.std(axis=0), MIN_SPREAD).tolist(),
            "min": (mid - 0.5 * spread).tolist(),
            "max": (mid + 0.5 * spread).tolist(),
        }
        print(f"{key:>16}: dim={data.shape[1]} samples={data.shape[0]}")

    out = args.out or os.path.join(os.path.dirname(h5_path), "stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"saved {out} ({len(train_idx)} training episodes of {len(episodes)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", required=True, help="config name in configs/")
    parser.add_argument("--set", action="append", default=[], help="override, e.g. dataset.path=...")
    parser.add_argument("--out", default=None, help="defaults to stats.json next to the h5 file")
    main(parser.parse_args())
