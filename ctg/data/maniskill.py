"""
ManiSkill demonstrations (trajectory.h5 + trajectory.json) as training samples.

One item per episode: a random start step t, the plan (future trajectory from t),
the goal (read at the last step) and the observation history ending at t.

  item["plan"]     {field: [L, d]}  future states/actions, padded after the episode end
  item["goal"]     {field: [d]}
  item["history"]  {field: [H, d]} and {camera: [H, C, h, w] uint8}; actions end at t-1
  item["mask"]     [L] bool, True for real (not padded) plan steps
  item["time"]     [L] seconds since t for every plan point

ResampledDemoDataset replaces the fixed-step plan by the rest of the episode
resampled to L points (see ctg.data.resampling); its time stamps are non-uniform.
"""

import json
import os
from typing import Mapping, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ctg.data.layout import Layout, make_layout


def history_window(dset, end: int, length: int, pad_value=None) -> np.ndarray:
    """Rows [end - length, end) of dset, left-padded to `length` rows by repeating
    the first row (pad_value=None) or with a constant - as the policy fills its
    history buffer at the start of an episode."""
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


def find_dataset(traj: h5py.Group, key: str, image_type: str = "rgb"):
    """Top-level datasets (e.g. actions) win over obs/<group>/<key> (agent, extra, sensor_data)."""
    if key in traj and isinstance(traj[key], h5py.Dataset):
        return traj[key]
    for group_name, group in traj["obs"].items():
        if group_name == "sensor_param" or not isinstance(group, h5py.Group):
            continue
        if key in group:
            node = group[key]
            return node[image_type] if isinstance(node, h5py.Group) else node
    raise KeyError(f"key {key!r} not found in {traj.name}")


class DemoDataset(Dataset):
    """Fixed-step plans of `plan_len` control steps.

    Dataset handles are looked up once per process and reused: every h5py path
    lookup leaks ~0.25 KB inside HDF5, which over ~1e8 samples filled 125 GB of RAM.
    """

    def __init__(
        self,
        path: str,
        plan: Mapping,
        goal: Mapping,
        history: Mapping,
        cameras: Sequence[str] = (),
        plan_len: int = 16,
        history_len: int = 2,
        action_key: str = "actions",
        control_dt: float = 0.05,
        success_only: bool = True,
        zero_pad: Optional[Sequence[str]] = None,
    ):
        self.path = str(path)
        self.plan_layout: Layout = make_layout(plan)
        self.goal_layout: Layout = make_layout(goal)
        self.history_layout: Layout = make_layout(history)
        self.cameras = tuple(cameras)
        self.plan_len = int(plan_len)
        self.history_len = int(history_len)
        self.action_key = action_key
        self.control_dt = float(control_dt)

        with open(self.path.replace(".h5", ".json")) as f:
            meta = json.load(f)
        self.env_info = meta["env_info"]
        episodes = meta["episodes"]
        if success_only:
            episodes = [e for e in episodes if e.get("success", False)]
        self.episodes = episodes

        # after the episode end the robot rests: velocities and delta actions are zero,
        # absolute quantities keep their last value
        if zero_pad is None:
            control_mode = self.env_info.get("env_kwargs", {}).get("control_mode", "")
            zero_pad = [k for k in self.plan_layout if "vel" in k]
            if "delta" in control_mode and action_key in self.plan_layout:
                zero_pad.append(action_key)
        self.zero_pad = set(zero_pad)

        self._h5 = None
        self._handles = {}

    def __len__(self):
        return len(self.episodes)

    def __getstate__(self):  # DataLoader workers open their own file and handles
        state = self.__dict__.copy()
        state["_h5"], state["_handles"] = None, {}
        return state

    # ------------------------------------------------------------------ h5 access
    @property
    def h5(self):
        if self._h5 is None:
            # no chunk cache: each window is read once, and every cached handle would
            # otherwise keep up to 1 MB of decompressed chunks
            self._h5 = h5py.File(self.path, "r", rdcc_nbytes=0)
        return self._h5

    def _dset(self, ep_id, key, image_type="rgb"):
        handle_key = (ep_id, key, image_type)
        if handle_key not in self._handles:
            self._handles[handle_key] = find_dataset(self.h5[f"traj_{ep_id}"], key, image_type)
        return self._handles[handle_key]

    def _length(self, ep_id) -> int:
        """Number of actions (= control steps) of the episode."""
        if ("len", ep_id) not in self._handles:
            self._handles[("len", ep_id)] = self._dset(ep_id, self.action_key).shape[0]
        return self._handles[("len", ep_id)]

    # ------------------------------------------------------------------ sample parts
    def _sample_start(self, n_steps: int) -> int:
        return int(torch.randint(0, max(n_steps - 2, 1), (1,)).item())  # torch RNG: seeded per worker

    def _goal(self, ep_id, n_steps):
        return {k: torch.tensor(self._dset(ep_id, k)[n_steps - 1], dtype=torch.float32) for k in self.goal_layout}

    def _history(self, ep_id, start):
        out, H = {}, self.history_len
        for key in self.history_layout:
            if key == self.action_key:  # actions already executed before t: a[t-H] .. a[t-1]
                window = history_window(self._dset(ep_id, key), start, H, pad_value=0)
            else:  # observations o[t-H+1] .. o[t]
                window = history_window(self._dset(ep_id, key), start + 1, H)
            out[key] = torch.tensor(window, dtype=torch.float32)
        for cam in self.cameras:
            window = history_window(self._dset(ep_id, cam), start + 1, H)  # [H, h, w, c] uint8
            out[cam] = torch.from_numpy(np.ascontiguousarray(window.transpose(0, 3, 1, 2)))
        return out

    def _pad(self, value: torch.Tensor, key: str) -> torch.Tensor:
        out = torch.zeros((self.plan_len, value.shape[1])) if key in self.zero_pad \
            else value[-1:].repeat(self.plan_len, 1)
        out[: value.shape[0]] = value
        return out

    def _plan(self, ep_id, start, n_steps):
        end = min(n_steps, start + self.plan_len)
        plan = {k: self._pad(torch.tensor(self._dset(ep_id, k)[start:end], dtype=torch.float32), k)
                for k in self.plan_layout}
        mask = torch.zeros(self.plan_len, dtype=torch.bool)
        mask[: end - start] = True
        time = torch.arange(self.plan_len, dtype=torch.float32) * self.control_dt
        return plan, mask, time

    def __getitem__(self, idx):
        ep_id = self.episodes[idx]["episode_id"]
        n_steps = self._length(ep_id)
        start = self._sample_start(n_steps)
        plan, mask, time = self._plan(ep_id, start, n_steps)
        return {"plan": plan, "goal": self._goal(ep_id, n_steps), "history": self._history(ep_id, start),
                "mask": mask, "time": time}


class ResampledDemoDataset(DemoDataset):
    """Plans cover the whole rest of the episode, resampled to plan_len points by `resampler`."""

    def __init__(self, resampler, **kwargs):
        super().__init__(**kwargs)
        if resampler.num_points != self.plan_len:
            raise ValueError(f"resampler.num_points ({resampler.num_points}) must equal plan_len ({self.plan_len})")
        self.resampler = resampler
        self.zero_pad = set()  # resampled plans end at the episode end: no padding

    def _plan(self, ep_id, start, n_steps):
        stop = n_steps  # states of steps start .. n_steps-1 (one action per step)
        t = np.arange(start, stop, dtype=np.float64) * self.control_dt
        if len(t) < 2:
            start, t = stop - 2, np.arange(stop - 2, stop, dtype=np.float64) * self.control_dt
        fields = {k: np.asarray(self._dset(ep_id, k)[start:stop], dtype=np.float64) for k in self.plan_layout}
        kinds = {k: spec.interp for k, spec in self.plan_layout.items()}
        t_new, resampled = self.resampler(t, fields, kinds)
        plan = {k: torch.tensor(v, dtype=torch.float32) for k, v in resampled.items()}
        mask = torch.ones(self.plan_len, dtype=torch.bool)
        return plan, mask, torch.tensor(t_new - t[0], dtype=torch.float32)


def resolve_path(path, data_dir) -> str:
    """Absolute path, or relative to data_dir."""
    return str(path) if os.path.isabs(str(path)) else os.path.join(str(data_dir), str(path))
