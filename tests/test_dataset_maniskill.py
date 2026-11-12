"""TrajectoryPadDatasetHDF5 on a small synthetic file with the ManiSkill layout."""

import json
import pickle

import h5py
import numpy as np
import pytest
import torch

from dataset.dataset_maniskill import BaseDataset, TrajectoryPadDatasetHDF5

LENGTHS = {0: 6, 1: 9}  # actions per episode; observations have one more step


@pytest.fixture
def h5_path(tmp_path):
    path = tmp_path / "demo.h5"
    with h5py.File(path, "w") as f:
        for ep, n in LENGTHS.items():
            g = f.create_group(f"traj_{ep}")
            g["actions"] = np.full((n, 2), ep, dtype=np.float32) + np.arange(n)[:, None]
            g["obs/agent/qpos"] = np.arange(n + 1, dtype=np.float32)[:, None].repeat(2, 1) + 100 * ep
            g["obs/extra/goal"] = np.full((n + 1, 3), ep, dtype=np.float32)
            g["obs/sensor_data/base_camera/rgb"] = np.full((n + 1, 4, 4, 3), ep, dtype=np.uint8)
    meta = {"env_info": {"env_id": "X", "env_kwargs": {"control_mode": "pd_joint_delta_pos"}},
            "episodes": [{"episode_id": ep, "success": True} for ep in LENGTHS]}
    (tmp_path / "demo.json").write_text(json.dumps(meta))
    return str(path)


@pytest.fixture
def ds(h5_path):
    return TrajectoryPadDatasetHDF5({
        "path": h5_path,
        "state_shapes": {"qpos": {"shape": 2, "method_norm": "minmax"},
                         "actions": {"shape": 2, "method_norm": "standard"}},
        "goal_shapes": {"goal": {"shape": 3, "method_norm": "minmax"}},
        "obs_shapes": {"qpos": {"shape": 2, "method_norm": "minmax"},
                       "actions": {"shape": 2, "method_norm": "standard"},
                       "base_camera": {"shape": "(3, 4, 4)", "type": "rgb", "method_norm": "none"}},
        "sequence_len": 4, "sequence_hist": 2,
    })


def test_item_shapes_and_padding(ds):
    torch.manual_seed(0)
    for idx in (0, 1):
        states, goals, obs = ds[idx]
        assert states["qpos"].shape == (4, 2) and states["actions"].shape == (4, 2)
        assert goals["goal"].shape == (3,)
        assert obs["qpos"].shape == (2, 2) and obs["actions"].shape == (2, 2)
        assert obs["base_camera"].shape == (2, 3, 4, 4) and obs["base_camera"].dtype == torch.uint8
        L = obs["seq_len_L"]
        assert obs["mask"].tolist() == [True] * L + [False] * (4 - L)
        assert torch.all(states["actions"][L:] == 0)  # delta actions are zero-padded
        assert torch.all(goals["goal"] == idx)


def test_dataset_handles_are_looked_up_once(ds, monkeypatch):
    calls = []
    original = BaseDataset.find_dataset

    def counting(traj, key, image_type="rgb"):
        calls.append(key)
        return original(traj, key, image_type)

    monkeypatch.setattr(BaseDataset, "find_dataset", staticmethod(counting))
    for _ in range(20):
        ds[0], ds[1]
    # actions, qpos, goal, base_camera per episode: one lookup each, however many samples
    assert sorted(calls) == sorted(["actions", "qpos", "goal", "base_camera"] * 2)


def test_file_opened_without_chunk_cache(ds):
    rdcc_nbytes = ds.h5.id.get_access_plist().get_cache()[2]
    assert rdcc_nbytes == 0


def test_pickling_drops_open_handles(ds):
    ds[0]
    assert ds._handles and ds._h5 is not None
    copy = pickle.loads(pickle.dumps(ds))  # what each DataLoader worker receives
    assert copy._handles == {} and copy._h5 is None
    states, _, _ = copy[1]
    assert states["qpos"].shape == (4, 2)


def test_episode_length_cached(ds):
    assert ds._length(1) == LENGTHS[1]
    assert ("len", 1) in ds._handles
