"""Observation history of the policy: must match the training history window."""

import torch

from models.policies.diff_policy import DiffPolicy, TensorHistory


def stub_policy(hist=4):
    """DiffPolicy without checkpoint loading: only the history logic is exercised."""
    policy = DiffPolicy.__new__(DiffPolicy)
    policy.config = {"dataset": {"sequence_hist": hist, "obs_shapes": {
        "qpos": {"shape": 2},
        "actions": {"shape": 1},
        "base_camera": {"shape": (3, 2, 2), "type": "rgb"},
    }}}
    policy.reset()
    return policy


def obs(t):
    return {
        "agent": {"qpos": torch.full((1, 2), float(t))},
        "sensor_data": {"base_camera": {"rgb": torch.full((1, 2, 2, 3), t, dtype=torch.uint8)}},
        "actions": torch.full((1, 1), float(t)),
    }


def test_first_observation_fills_the_buffer():
    h = TensorHistory(3)
    h.update({"x": torch.tensor([[1.0]])})
    assert h.get_stacked()["x"].flatten().tolist() == [1.0, 1.0, 1.0]


def test_history_keeps_last_steps_in_order():
    h = TensorHistory(3)
    for t in range(5):
        h.update({"x": torch.tensor([[float(t)]])})
    assert h.get_stacked()["x"].flatten().tolist() == [2.0, 3.0, 4.0]


def test_observe_once_per_step_gives_consecutive_frames():
    policy = stub_policy(hist=4)
    for t in range(6):  # reset + 5 executed steps of one action chunk
        policy.observe(obs(t))
    stacked = policy.contex_history.get_stacked()
    assert stacked["qpos"][0, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert stacked["actions"][0, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    # cameras: HWC -> CHW, uint8 kept
    assert stacked["base_camera"].shape == (1, 4, 3, 2, 2)
    assert stacked["base_camera"].dtype == torch.uint8
    assert stacked["base_camera"][0, :, 0, 0, 0].tolist() == [2, 3, 4, 5]


def test_reset_clears_the_history():
    policy = stub_policy(hist=2)
    policy.observe(obs(7))
    policy.reset()
    policy.observe(obs(1))
    assert policy.contex_history.get_stacked()["qpos"][0, :, 0].tolist() == [1.0, 1.0]
