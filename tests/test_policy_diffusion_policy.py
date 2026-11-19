"""ctg/policy/diffusion_policy.py with a checkpoint trained on the synthetic demonstrations."""

import h5py
import pytest
import torch

from ctg.policy.diffusion_policy import DiffusionPolicy, lookup

from conftest import GROUPS


def observation(demo_h5, step=0):
    """A ManiSkill-style observation (batch 1) from the synthetic demonstrations."""
    obs = {"agent": {}, "extra": {}, "sensor_data": {}}
    with h5py.File(demo_h5, "r") as f:
        g = f["traj_0"]
        for key, group in GROUPS.items():
            obs[group][key] = torch.tensor(g[f"obs/{group}/{key}"][step:step + 1])
        obs["sensor_data"]["base_camera"] = {"rgb": torch.tensor(g["obs/sensor_data/base_camera/rgb"][step:step + 1])}
    obs["actions"] = torch.zeros(1, 8)
    return obs


def test_lookup():
    obs = {"actions": 1, "agent": {"qpos": 2}, "sensor_param": {"x": 3}}
    assert lookup(obs, "actions") == 1 and lookup(obs, "qpos") == 2
    with pytest.raises(KeyError, match="'x'"):
        lookup(obs, "x")


def test_policy_from_checkpoint_plans_and_acts(e9_checkpoint, demo_h5):
    policy = DiffusionPolicy.from_checkpoint(e9_checkpoint, action_horizon=3)
    ckpt = torch.load(e9_checkpoint, map_location="cpu", weights_only=False)
    for k, v in ckpt["ema_model"].items():
        assert torch.equal(policy.model.state_dict()[k], v)
    obs = observation(demo_h5)
    plan = policy.plan(obs)
    assert {k: v.shape for k, v in plan.items()} == {"actions": (1, 16, 8), "qpos": (1, 16, 9), "qvel": (1, 16, 9)}
    assert policy.act(obs, observe=True).shape == (1, 3, 8)
    assert policy.history.stacked()["qpos"].shape == (1, 2, 9)
    policy.reset()
    assert policy.history.stacked() == {}


def test_overrides_sections_and_inpainting(e9_checkpoint, demo_h5):
    from ctg.config import compose_config

    physics = compose_config(["physics=panda"]).physics
    policy = DiffusionPolicy.from_checkpoint(
        e9_checkpoint, overrides=["diffusion.num_inference_steps=2", "diffusion.inpainting.start_fields=[qpos]"],
        sections={"physics": physics})
    assert policy.sampler.num_inference_steps == 2 and policy.start_dims == list(range(8, 17))
    obs = observation(demo_h5, step=4)
    plan = policy.plan(obs)
    torch.testing.assert_close(plan["qpos"][0, 0], obs["agent"]["qpos"][0])  # starts at the current state
    assert policy.action_horizon == 8  # evaluation.action_horizon


def test_image_policy_observes_cameras(demo_h5, demo_stats, tmp_path):
    from conftest import train_tiny

    train_tiny(demo_h5, demo_stats, tmp_path, "experiment=e8", "data.dataset.cameras=[base_camera]",
               "encoder.pretrained=false", "encoder.num_keypoints=4", "training.epochs=1",
               "diffusion.num_inference_steps=2")
    policy = DiffusionPolicy.from_checkpoint(tmp_path / "checkpoints" / "epoch_00000.pt")
    assert policy.act(observation(demo_h5)).shape == (1, 8, 8)
    assert policy.history.stacked()["base_camera"].dtype == torch.uint8
