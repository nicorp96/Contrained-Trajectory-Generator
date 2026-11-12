"""Extra cameras of PegInsertionSide-Extended (needs ManiSkill + a renderer)."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill")

import environment.extended_peg_insertion  # noqa: E402,F401
from environment.extended_peg_insertion import camera_config  # noqa: E402


def test_camera_config_defaults_and_pose():
    cfg = camera_config("side", eye=[1, 0, 0], target=[0, 0, 0])
    assert (cfg.uid, cfg.width, cfg.height, cfg.fov) == ("side", 128, 128, 1.0)
    np.testing.assert_allclose(np.asarray(cfg.pose.p).reshape(-1), [1, 0, 0])


@pytest.fixture(scope="module")
def env():
    e = gym.make(
        "PegInsertionSide-Extended", obs_mode="rgb", num_envs=1, sim_backend="physx_cpu",
        sensor_configs=dict(width=64, height=64),
        extra_cameras=[dict(uid="side", eye=[0.4, 0.15, 0.2], target=[0, 0.22, 0.1], width=48, height=48)],
    )
    yield e
    e.close()


def test_extra_camera_is_rendered(env):
    obs, _ = env.reset(seed=0)
    assert set(obs["sensor_data"]) == {"base_camera", "hand_camera", "side"}
    # global sensor_configs override every camera, including the extra one
    assert obs["sensor_data"]["base_camera"]["rgb"].shape[1:3] == (64, 64)
    assert obs["sensor_data"]["side"]["rgb"].shape[1:3] == (64, 64)


def test_without_extra_cameras_matches_base_task():
    e = gym.make("PegInsertionSide-Extended", obs_mode="rgb", num_envs=1, sim_backend="physx_cpu")
    obs, _ = e.reset(seed=0)
    e.close()
    assert set(obs["sensor_data"]) == {"base_camera", "hand_camera"}
    assert obs["sensor_data"]["base_camera"]["rgb"].shape[1:3] == (128, 128)
