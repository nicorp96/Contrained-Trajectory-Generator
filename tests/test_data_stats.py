"""ctg/data/stats.py"""

import h5py
import numpy as np
import pytest

from ctg.data.maniskill import DemoDataset
from ctg.data.splits import split_episodes
from ctg.data.stats import MIN_SPREAD, compute_stats, field_stats


def test_field_stats_widens_constant_channels():
    data = np.array([[1.0, 5.0], [3.0, 5.0]])
    s = field_stats(data)
    assert s["mean"] == [2.0, 5.0] and s["min"][0] == 1.0 and s["max"][0] == 3.0
    assert s["max"][1] - s["min"][1] == pytest.approx(MIN_SPREAD) and s["std"][1] == MIN_SPREAD


def test_compute_stats_uses_only_training_episodes(demo_h5):
    ds = DemoDataset(demo_h5, plan={"actions": {"size": 8}}, goal={}, history={"qpos": {"size": 9}})
    stats = compute_stats(ds, ["actions", "qpos", "actions"], val_ratio=0.4, seed=0)
    assert set(stats) == {"actions", "qpos"} and len(stats["qpos"]["mean"]) == 9
    train, _ = split_episodes(len(ds), 0.4, 0)
    with h5py.File(demo_h5, "r") as f:  # all steps of the training episodes (n actions, n states)
        expected = np.concatenate([f[f"traj_{ds.episodes[i]['episode_id']}/obs/agent/qpos"][:-1] for i in train])
    np.testing.assert_allclose(stats["qpos"]["mean"], expected.mean(axis=0), rtol=1e-6)
