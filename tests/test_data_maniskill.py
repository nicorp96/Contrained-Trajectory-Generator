"""ctg/data/maniskill.py on the synthetic PegInsertion demonstrations (tests/conftest.py)."""

import pickle

import h5py
import numpy as np
import pytest
import torch

import ctg.data.maniskill as maniskill
from ctg.data.maniskill import (DemoDataset, ResampledDemoDataset, find_dataset, history_window,
                                resolve_path)
from ctg.data.resampling import ArcLengthResampler, TimeResampler

from conftest import DT, FAILED, LENGTHS

FIELDS = dict(
    plan={"actions": {"size": 8}, "qpos": {"size": 9}, "qvel": {"size": 9}},
    goal={"box_hole_pose": {"size": 7}},
    history={"qpos": {"size": 9}, "actions": {"size": 8}},
    cameras=["base_camera"],
    plan_len=8,
    history_len=3,
)


@pytest.fixture
def ds(demo_h5):
    return DemoDataset(demo_h5, **FIELDS)


def test_history_window_pads_at_the_start():
    data = np.arange(5)[:, None]
    np.testing.assert_array_equal(history_window(data, 1, 3)[:, 0], [0, 0, 0])
    np.testing.assert_array_equal(history_window(data, 4, 3)[:, 0], [1, 2, 3])
    np.testing.assert_array_equal(history_window(data, 0, 2, pad_value=0)[:, 0], [0, 0])
    np.testing.assert_array_equal(history_window(data, 1, 3, pad_value=-1)[:, 0], [-1, -1, 0])


def test_find_dataset_groups(demo_h5):
    with h5py.File(demo_h5, "r") as f:
        traj = f["traj_0"]
        assert find_dataset(traj, "actions").name.endswith("/actions")
        assert find_dataset(traj, "qpos").name.endswith("agent/qpos")
        assert find_dataset(traj, "base_camera").name.endswith("base_camera/rgb")
        with pytest.raises(KeyError, match="intrinsic_cv"):  # sensor_param is never searched
            find_dataset(traj, "intrinsic_cv")


def test_only_successful_episodes(demo_h5, ds):
    assert len(ds) == len(LENGTHS)
    assert len(DemoDataset(demo_h5, success_only=False, **FIELDS)) == len(LENGTHS) + 1
    assert FAILED not in [e["episode_id"] for e in ds.episodes]


def test_item_shapes_mask_and_time(ds):
    torch.manual_seed(0)
    for idx in range(len(ds)):
        item = ds[idx]
        assert {k: v.shape for k, v in item["plan"].items()} == {"actions": (8, 8), "qpos": (8, 9), "qvel": (8, 9)}
        assert item["goal"]["box_hole_pose"].shape == (7,)
        assert item["history"]["qpos"].shape == (3, 9) and item["history"]["actions"].shape == (3, 8)
        assert item["history"]["base_camera"].shape == (3, 3, 16, 16)
        assert item["history"]["base_camera"].dtype == torch.uint8
        assert item["mask"].dtype == torch.bool and item["mask"][0]
        torch.testing.assert_close(item["time"], torch.arange(8) * DT)


def test_window_contents_and_padding(ds, monkeypatch, demo_h5):
    n = LENGTHS[0]
    start = n - 3  # only 3 real plan steps left
    monkeypatch.setattr(DemoDataset, "_sample_start", lambda self, n_steps: start)
    item = ds[0]
    with h5py.File(demo_h5, "r") as f:
        g = f["traj_0"]
        actions, qpos, qvel = g["actions"][:], g["obs/agent/qpos"][:], g["obs/agent/qvel"][:]
        goal = g["obs/extra/box_hole_pose"][n - 1]
    assert item["mask"].tolist() == [True] * 3 + [False] * 5
    torch.testing.assert_close(item["plan"]["actions"][:3], torch.tensor(actions[start:]))
    assert torch.all(item["plan"]["actions"][3:] == 0)  # delta actions: zero after the end
    assert torch.all(item["plan"]["qvel"][3:] == 0)  # velocities: zero
    torch.testing.assert_close(item["plan"]["qpos"][3:], torch.tensor(qpos[n - 1]).expand(5, 9))  # positions hold
    torch.testing.assert_close(item["goal"]["box_hole_pose"], torch.tensor(goal))
    torch.testing.assert_close(item["history"]["qpos"], torch.tensor(qpos[start - 2:start + 1]))  # o[t-2] .. o[t]
    torch.testing.assert_close(item["history"]["actions"], torch.tensor(actions[start - 3:start]))  # a[t-3] .. a[t-1]


def test_history_at_the_episode_start(ds, monkeypatch):
    monkeypatch.setattr(DemoDataset, "_sample_start", lambda self, n_steps: 0)
    item = ds[0]
    assert torch.all(item["history"]["actions"] == 0)  # nothing executed yet
    assert torch.equal(item["history"]["qpos"][0], item["history"]["qpos"][2])  # first state repeated


def test_zero_pad_follows_the_control_mode(demo_h5):
    assert DemoDataset(demo_h5, **FIELDS).zero_pad == {"qvel", "actions"}
    assert DemoDataset(demo_h5, zero_pad=[], **FIELDS).zero_pad == set()


def test_dataset_handles_are_looked_up_once(ds, monkeypatch):
    calls = []
    original = maniskill.find_dataset

    def counting(traj, key, image_type="rgb"):
        calls.append(key)
        return original(traj, key, image_type)

    monkeypatch.setattr(maniskill, "find_dataset", counting)
    for _ in range(10):
        ds[0], ds[1]
    keys = ["actions", "qpos", "qvel", "box_hole_pose", "base_camera"]
    assert sorted(calls) == sorted(keys * 2)


def test_file_opened_without_chunk_cache(ds):
    assert ds.h5.id.get_access_plist().get_cache()[2] == 0


def test_pickling_drops_open_handles(ds):
    ds[0]
    assert ds._handles and ds._h5 is not None
    copy = pickle.loads(pickle.dumps(ds))  # what each DataLoader worker receives
    assert copy._handles == {} and copy._h5 is None
    assert copy[1]["plan"]["qpos"].shape == (8, 9)


def resampled(demo_h5, resampler, plan_len=8):
    return ResampledDemoDataset(resampler=resampler, path=demo_h5, **{**FIELDS, "plan_len": plan_len})


@pytest.mark.parametrize("resampler", [TimeResampler(8), ArcLengthResampler(8, "qpos")])
def test_resampled_plan_covers_the_rest_of_the_episode(demo_h5, resampler, monkeypatch):
    start = 5
    monkeypatch.setattr(DemoDataset, "_sample_start", lambda self, n_steps: start)
    item = resampled(demo_h5, resampler)[0]
    n = LENGTHS[0]
    with h5py.File(demo_h5, "r") as f:
        qpos = f["traj_0/obs/agent/qpos"][:]
    assert item["mask"].all() and item["time"][0] == 0
    assert item["time"][-1].item() == pytest.approx((n - 1 - start) * DT)
    assert torch.all(item["time"][1:] >= item["time"][:-1])
    torch.testing.assert_close(item["plan"]["qpos"][0], torch.tensor(qpos[start]))
    torch.testing.assert_close(item["plan"]["qpos"][-1], torch.tensor(qpos[n - 1]))


def test_resampled_plan_at_the_last_step(demo_h5, monkeypatch):
    monkeypatch.setattr(DemoDataset, "_sample_start", lambda self, n_steps: n_steps - 1)
    item = resampled(demo_h5, TimeResampler(8))[0]
    assert item["time"][-1].item() == pytest.approx(DT)


def test_resampler_must_match_plan_len(demo_h5):
    with pytest.raises(ValueError, match="plan_len"):
        resampled(demo_h5, TimeResampler(4))


def test_resolve_path(tmp_path):
    assert resolve_path("/abs/file.h5", tmp_path) == "/abs/file.h5"
    assert resolve_path("rel/file.h5", tmp_path) == str(tmp_path / "rel/file.h5")
