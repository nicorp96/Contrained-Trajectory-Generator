import os, glob, gzip, pickle
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from models.utils.normalizer import Normalizer, DictNormalizer


# ----------------------------
# utils
# ----------------------------
def _load_pickle_gz(path: str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _to_torch(d):
    if isinstance(d, dict):
        return {k: _to_torch(v) for k, v in d.items()}
    arr = (
        torch.from_numpy(d.copy()) if isinstance(d, np.ndarray) else torch.as_tensor(d)
    )
    return arr


# ----------------------------
# Dataset
# ----------------------------
class SimpleTrajDataset(Dataset):
    """
    Reads gzipped-pickle shards written by make_synth_trajs.py and normalizes with stats.npz.

    Each __getitem__ returns:
      states_nm:       {"position": [N,3], "vel_ned": [N,3]} (normalized)
      observations_nm: {"start_position": [1,3], "start_velocity": [1,3]} (normalized)
      environment_nm:  {"goal": [1,3]} (normalized)
      extras:          {"t":[N], "T":(), "dt":(), "family": int} (unnormalized)
    Optionally return raw (unnormalized) copies if return_raw=True.
    """

    def __init__(
        self,
        config: Dict,
        val: bool = False,
    ):
        super().__init__()
        self.config = config
        self.root = config["root_dir"]
        self.split = "val" if val else "train"
        self.return_raw = config["return_raw"]
        self.cache_last_shard = config["cache_last_shard"]
        self.expect_constant_N = config["expect_constant_N"]

        self.state_shapes = config["state_shapes"]
        self.env_shapes = config["environment_shapes"]
        self.observation_shapes = config["observation_shapes"]
        self.dict_norm_states = DictNormalizer(self.state_shapes)
        self.dict_norm_obs = DictNormalizer(self.observation_shapes)
        self.dict_norm_env = DictNormalizer(self.env_shapes)
        # discover shards
        pattern = os.path.join(self.root, f"{self.split}.*.pkl.gz")
        self.shard_paths = sorted(glob.glob(pattern))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards matching {pattern}")

        # load stats
        stats_path = os.path.join(self.root, "stats.npz")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Missing stats file: {stats_path}")
        self.stats = np.load(stats_path)
        self.mean_pos = self.stats["mean_pos"].astype(np.float32)  # [3]
        self.std_pos = np.maximum(self.stats["std_pos"].astype(np.float32), 1e-8)
        self.mean_vel = self.stats["mean_vel"].astype(np.float32)  # [3]
        self.std_vel = np.maximum(self.stats["std_vel"].astype(np.float32), 1e-8)

        # index across shards (global idx -> (shard_idx, local_idx))
        self._index: List[Tuple[int, int]] = []
        self._shard_sizes: List[int] = []
        for si, spath in enumerate(self.shard_paths):
            recs = _load_pickle_gz(spath)  # list of dicts
            n = len(recs)
            self._shard_sizes.append(n)
            self._index.extend([(si, j) for j in range(n)])

        # simple cache for most recent shard
        self._cache_si: Optional[int] = None
        self._cache_records: Optional[List[dict]] = None

        # verify constant N if requested
        if self.expect_constant_N:
            N0 = None
            for si, spath in enumerate(
                self.shard_paths[:1]
            ):  # just peek first shard's first record
                recs = _load_pickle_gz(spath)
                if len(recs) == 0:
                    continue
                N0 = recs[0]["states"]["position"].shape[0]
                break
            self._N = N0

    def __len__(self):
        return len(self._index)

    # ------------- normalization helpers -------------
    def _norm_pos(self, x: np.ndarray) -> np.ndarray:
        # x: [..., 3]
        return ((x - self.mean_pos) / self.std_pos).astype(np.float32)

    def _norm_vel(self, v: np.ndarray) -> np.ndarray:
        # v: [..., 3]
        return ((v - self.mean_vel) / self.std_vel).astype(np.float32)

    def _get_record(self, global_idx: int) -> dict:
        si, li = self._index[global_idx]
        if (
            self.cache_last_shard
            and self._cache_si == si
            and self._cache_records is not None
        ):
            recs = self._cache_records
        else:
            recs = _load_pickle_gz(self.shard_paths[si])
            if self.cache_last_shard:
                self._cache_si = si
                self._cache_records = recs
        return recs[li]

    def __getitem__(self, idx: int):
        rec = self._get_record(idx)

        # raw numpy views
        states = rec["states"]  # dict: position [N,3], vel_ned [N,3]
        obs = rec["observations"]  # dict: start_position [1,3], start_velocity [1,3]
        env = rec["environment"]  # dict: goal [1,3]
        extras = rec["extras"]  # dict: t [N], T, dt, family

        # normalized copies
        states_nm = {
            "position": self._norm_pos(states["position"]),
            "vel_ned": self._norm_vel(states["vel_ned"]),
        }
        observations_nm = {
            "start_position": self._norm_pos(obs["start_position"]),
            "start_velocity": self._norm_vel(obs["start_velocity"]),
        }
        environment_nm = {
            "goal": self._norm_pos(env["goal"]),
        }

        if self.return_raw:
            item = {
                "states": _to_torch(states),
                "observations": _to_torch(obs),
                "environment": _to_torch(env),
                # "states_nm": _to_torch(states_nm),
                # "observations_nm": _to_torch(observations_nm),
                # "environment_nm": _to_torch(environment_nm),
                "extras": _to_torch(extras),
            }
        else:
            item = {
                "states": _to_torch(states_nm),  # expose normalized by default
                "observations": _to_torch(observations_nm),
                "environment": _to_torch(environment_nm),
                "extras": _to_torch(extras),  # keep extras unnormalized
            }
        return item

    def get_stats_from_file(self):
        stats = self.stats
        stats_dict = {
            "position": {"mean": stats["mean_pos"], "std": stats["std_pos"]},
            "vel_ned": {"mean": stats["mean_vel"], "std": stats["std_vel"]},
        }
        stats_dict_obs = {
            "start_position": {"mean": stats["mean_pos"], "std": stats["std_pos"]},
            "start_velocity": {"mean": stats["mean_vel"], "std": stats["std_vel"]},
        }
        stats_dict_env = {
            "goal": {"mean": stats["mean_pos"], "std": stats["std_pos"]},
        }
        self.dict_norm_states.set_stats_from_json(stats_dict)
        self.dict_norm_obs.set_stats_from_json(stats_dict_obs)
        self.dict_norm_env.set_stats_from_json(stats_dict_env)
        return [self.dict_norm_states, self.dict_norm_obs, self.dict_norm_env]


# ----------------------------
# Collate
# ----------------------------
def collate_traj_batch(batch: List[Dict]):
    """
    Stacks nested dicts into batched tensors.
    Assumes constant N across the batch (as produced by generator).
    Returns:
      states:       {"position":[B,N,3], "vel_ned":[B,N,3]}
      observations: {"start_position":[B,1,3], "start_velocity":[B,1,3]}
      environment:  {"goal":[B,1,3]}
      extras:       {"t":[B,N], "T":[B], "dt":[B], "family":[B]}
    """

    def stack_dicts(dicts: List[Dict]):
        out = {}
        keys = dicts[0].keys()
        for k in keys:
            vs = [d[k] for d in dicts]
            if isinstance(vs[0], dict):
                out[k] = stack_dicts(vs)
            else:
                out[k] = torch.stack(vs, dim=0)
        return out

    # unified field names regardless of return_raw
    if "states_nm" in batch[0]:  # return_raw=True path
        states = [b["states_nm"] for b in batch]
        obs = [b["observations_nm"] for b in batch]
        env = [b["environment_nm"] for b in batch]
        extras = [b["extras"] for b in batch]
        out = {
            "states": stack_dicts(states),
            "observations": stack_dicts(obs),
            "environment": stack_dicts(env),
            "extras": stack_dicts(extras),
        }
    else:
        states = [b["states"] for b in batch]
        obs = [b["observations"] for b in batch]
        env = [b["environment"] for b in batch]
        extras = [b["extras"] for b in batch]
        out = {
            "states": stack_dicts(states),
            "observations": stack_dicts(obs),
            "environment": stack_dicts(env),
            "extras": stack_dicts(extras),
        }
    # ensure types
    out["extras"]["family"] = out["extras"]["family"].long()
    return out


# ----------------------------
# Example DataLoader usage
# ----------------------------
def make_loader(
    root: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    return_raw: bool = False,
):
    ds = SimpleTrajDataset(root, split=split, return_raw=return_raw)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_traj_batch,
    )
    return dl


if __name__ == "__main__":
    # quick smoke test
    root = "/home/nrodriguez/Documents/DProjekt/research_t_opt/synth_trajs"
    train_loader = make_loader(root, "train", batch_size=8, shuffle=True, num_workers=0)
    val_loader = make_loader(root, "val", batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(train_loader))
    print(
        {
            k: (
                {kk: vv.shape for kk, vv in batch[k].items()}
                if isinstance(batch[k], dict)
                else type(batch[k])
            )
            for k in batch
        }
    )
    batch = next(iter(val_loader))
    print(
        {
            k: (
                {kk: vv.shape for kk, vv in batch[k].items()}
                if isinstance(batch[k], dict)
                else type(batch[k])
            )
            for k in batch
        }
    )
