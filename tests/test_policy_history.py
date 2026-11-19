"""ctg/policy/history.py"""

import torch

from ctg.policy.history import ObservationHistory


def obs(t):
    return {"qpos": torch.full((1, 3), float(t)), "base_camera": torch.full((1, 3, 2, 2), t, dtype=torch.uint8)}


def test_first_observation_fills_the_buffer():
    h = ObservationHistory(4)
    h.update(obs(7))
    s = h.stacked()
    assert s["qpos"].shape == (1, 4, 3) and s["base_camera"].shape == (1, 4, 3, 2, 2)
    assert torch.all(s["qpos"] == 7)


def test_history_keeps_the_last_steps_in_order():
    h = ObservationHistory(3)
    for t in range(6):
        h.update(obs(t))
    assert h.stacked()["qpos"][0, :, 0].tolist() == [3.0, 4.0, 5.0]


def test_values_are_copied():
    h = ObservationHistory(2)
    o = obs(1)
    h.update(o)
    o["qpos"].fill_(9.0)
    assert torch.all(h.stacked()["qpos"] == 1)


def test_unbatched_values_and_reset():
    h = ObservationHistory(2)
    h.update({"qpos": torch.ones(3)})
    assert h.stacked()["qpos"].shape == (1, 2, 3)
    h.reset()
    assert h.stacked() == {}
