import json
import os

import h5py
import numpy as np
import torch
from mani_skill.utils.io_utils import load_json
from torch.utils.data import Dataset

from common.get_class import get_class_dict
from common.resampling_algo import ResamplingBase, build_schema_from_layout
from global_parameters import ConfigGlobalP
from models.utils.normalizer import DictNormalizer

cfg_gp = ConfigGlobalP()


def _resolve_path(path):
    """Return `path` if it exists, otherwise resolve it relative to DATA_DIR."""
    if os.path.exists(path):
        return str(path)
    candidate = os.path.join(cfg_gp.DATA_DIR, path)
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"{path} does not exist (also tried {candidate})")
    return candidate


def history_window(dset, end: int, length: int, pad_value=None) -> np.ndarray:
    """
    Rows [end - length, end) of `dset`, left-padded to `length` rows.
    Padding repeats the first row (pad_value=None) or uses a constant, which
    mirrors how the policy initialises its history buffer at episode start.
    """
    start = max(end - length, 0)
    rows = dset[start:end] if end > 0 else dset[0:0]
    n_pad = length - rows.shape[0]
    if n_pad > 0:
        if pad_value is None:
            pad = np.repeat(dset[0:1], n_pad, axis=0)
        else:
            pad = np.full((n_pad,) + tuple(dset.shape[1:]), pad_value, dtype=dset.dtype)
        rows = np.concatenate([pad, rows], axis=0)
    return rows


class BaseDataset(Dataset):
    """
    ManiSkill demonstrations (trajectory.h5 + trajectory.json), one item per episode.

    Each item is (state_dict, goal_dict, obs_dict):
      state_dict: future trajectory starting at a random time step t   [L, d_k]
      goal_dict:  goal quantities read at the last step of the episode  [d_k]
      obs_dict:   observation history ending at t (actions end at t-1) [H, ...]
                  + "mask" [L] (True = real step, False = padding) and "seq_len_L"

    The h5 file is opened lazily per process (DataLoader workers) and only the
    required slices are read.
    """

    def __init__(self, config, dtype=torch.float32):
        self.config = config
        self.dtype = dtype
        self.h5_path = _resolve_path(config["path"])
        self.stats_path = (
            _resolve_path(config["path_stats"]) if "path_stats" in config else None
        )
        self._h5 = None

        self.json_data = load_json(self.h5_path.replace(".h5", ".json"))
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.control_mode = self.env_kwargs.get("control_mode", "")

        episodes = self.json_data["episodes"]
        if config.get("success_only", True):
            if any("success" not in eps for eps in episodes):
                raise ValueError(
                    "episodes in this dataset do not have the success attribute, "
                    "cannot load dataset with success_only=True"
                )
            episodes = [eps for eps in episodes if eps["success"]]
        self.episodes = episodes

        self.state_shapes = config["state_shapes"]
        self.goal_shapes = config["goal_shapes"]
        self.obs_shapes = config.get("obs_shapes") or {}
        self.sequence_len = int(config.get("sequence_len", 256))
        self.sequence_len_H = int(config.get("sequence_hist", 6))
        self.action_key = config.get("action_key", "actions")

        # Padding after the episode end: velocities and delta actions are zero
        # (robot at rest), absolute quantities hold their last value.
        default_zero = [k for k in self.state_shapes if "vel" in k]
        if "delta" in self.control_mode and self.action_key in self.state_shapes:
            default_zero.append(self.action_key)
        self.zero_pad_keys = set(config.get("zero_pad_keys", default_zero))

        self.dict_norm_states = DictNormalizer(self.state_shapes)
        self.dict_norm_goals = DictNormalizer(self.goal_shapes)
        self.dict_norm_contex = DictNormalizer(self.obs_shapes)

    def __len__(self):
        return len(self.episodes)

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def get_normalizers_from_file(self):
        with open(self.stats_path, "r") as f:
            json_stats_dict_r = json.load(f)
        self.dict_norm_states.set_stats_from_json(json_stats_dict_r)
        self.dict_norm_goals.set_stats_from_json(json_stats_dict_r)
        self.dict_norm_contex.set_stats_from_json(json_stats_dict_r)
        return self.dict_norm_states, self.dict_norm_goals, self.dict_norm_contex

    @staticmethod
    def find_dataset(traj: h5py.Group, key: str, image_type: str = "rgb"):
        """Top-level datasets (e.g. actions) win over obs/<group>/<key>."""
        if key in traj and isinstance(traj[key], h5py.Dataset):
            return traj[key]
        for group_name, group in traj["obs"].items():
            if group_name == "sensor_param" or not isinstance(group, h5py.Group):
                continue
            if key in group:
                node = group[key]
                return node[image_type] if isinstance(node, h5py.Group) else node
        raise KeyError(f"Key '{key}' not found in {traj.name}")

    def _episode(self, idx):
        eps = self.episodes[idx]
        traj = self.h5[f"traj_{eps['episode_id']}"]
        eps_len = traj["actions"].shape[0]
        # torch RNG is seeded per DataLoader worker
        high = max(eps_len - 2, 1)
        start_idx = int(torch.randint(0, high, (1,)).item())
        return traj, eps_len, start_idx

    def _goals(self, traj, eps_len):
        goal_dict = {}
        for key in self.goal_shapes:
            data = self.find_dataset(traj, key)
            goal_dict[key] = torch.tensor(data[eps_len - 1], dtype=self.dtype)
        return goal_dict

    def _history(self, traj, start_idx):
        obs_dict = {}
        H = self.sequence_len_H
        for key, spec in self.obs_shapes.items():
            if key == self.action_key:
                # actions already executed before t: a_{t-H} .. a_{t-1}
                dset = self.find_dataset(traj, key)
                window = history_window(dset, start_idx, H, pad_value=0)
                obs_dict[key] = torch.tensor(window, dtype=self.dtype)
            elif "camera" in key:
                dset = self.find_dataset(traj, key, spec.get("type", "rgb"))
                window = history_window(dset, start_idx + 1, H)  # [H, h, w, c]
                # keep uint8; ImageNormalizer converts to float in [0, 1]
                obs_dict[key] = torch.from_numpy(
                    np.ascontiguousarray(window.transpose(0, 3, 1, 2))
                )
            else:
                # observations o_{t-H+1} .. o_t (includes the current step)
                dset = self.find_dataset(traj, key)
                window = history_window(dset, start_idx + 1, H)
                obs_dict[key] = torch.tensor(window, dtype=self.dtype)
        return obs_dict


class TrajectoryPadDatasetHDF5(BaseDataset):
    """Future trajectory from t to the episode end, padded to `sequence_len`."""

    def pad_tail(self, value: torch.Tensor, key: str) -> torch.Tensor:
        L, D = value.shape
        if key in self.zero_pad_keys:
            out = torch.zeros((self.sequence_len, D), dtype=value.dtype)
        else:
            out = value[-1:].repeat(self.sequence_len, 1)
        out[:L] = value
        return out

    def __getitem__(self, idx):
        traj, eps_len, start_idx = self._episode(idx)
        # windows longer than sequence_len are truncated (sliding window)
        end_idx = min(eps_len, start_idx + self.sequence_len)
        seq_len_L = end_idx - start_idx

        state_dict = {}
        for key in self.state_shapes:
            data = self.find_dataset(traj, key)[start_idx:end_idx]
            state_dict[key] = self.pad_tail(torch.tensor(data, dtype=self.dtype), key)

        obs_dict = self._history(traj, start_idx)
        mask = torch.zeros(self.sequence_len, dtype=torch.bool)
        mask[:seq_len_L] = True
        obs_dict["mask"] = mask
        obs_dict["seq_len_L"] = seq_len_L
        return state_dict, self._goals(traj, eps_len), obs_dict


class TrajectoryKDatasetHDF5(BaseDataset):
    """Future trajectory from t to the episode end, resampled in time to `sequence_len` points."""

    def __init__(self, config, dtype=torch.float32):
        super().__init__(config, dtype)
        resampling_cfg = dict(config["resampling_algo"])
        resampling_cfg.setdefault("K", self.sequence_len)
        self.resampling_clss: ResamplingBase = get_class_dict(resampling_cfg)
        self.schema = build_schema_from_layout(
            self.state_shapes, interp_kind=self.resampling_clss.interp_kind
        )
        self.time_key = config.get("time_key", "time")

    def __getitem__(self, idx):
        traj, eps_len, start_idx = self._episode(idx)
        t = np.asarray(self.find_dataset(traj, self.time_key)[start_idx:eps_len]).reshape(-1)
        parts = [
            np.asarray(self.find_dataset(traj, key)[start_idx:eps_len]).reshape(
                eps_len - start_idx, -1
            )
            for key in self.state_shapes
        ]
        out = self.resampling_clss.resample(
            t, states=np.concatenate(parts, axis=1), schema=self.schema, dt=None
        )
        Y = torch.from_numpy(out["X"]).to(self.dtype)
        state_dict = {k: Y[:, list(spec.cols)] for k, spec in self.schema.items()}

        obs_dict = self._history(traj, start_idx)
        obs_dict["mask"] = torch.ones(Y.shape[0], dtype=torch.bool)
        obs_dict["seq_len_L"] = Y.shape[0]
        return state_dict, self._goals(traj, eps_len), obs_dict
