"""environment/replay_cameras.py: camera setup in the dataset json and a real one-episode replay."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from environment.replay_cameras import ENV_ID, EXTRA_CAMERAS, camera_env_info, output_path, prepare

ENV_INFO = {
    "env_id": "PegInsertionSide-v1",
    "max_episode_steps": 100,
    "env_kwargs": {"obs_mode": "state_dict+rgb", "control_mode": "pd_joint_delta_pos",
                   "sensor_configs": {"shader_pack": "default"}, "sim_backend": "physx_cpu", "num_envs": 1},
}


def test_camera_env_info():
    info = camera_env_info(ENV_INFO, 256, ["top"])
    assert info["env_id"] == ENV_ID and info["max_episode_steps"] == 100
    kw = info["env_kwargs"]
    assert kw["sensor_configs"] == {"shader_pack": "default", "width": 256, "height": 256}
    assert kw["extra_cameras"] == [dict(EXTRA_CAMERAS["top"], width=256, height=256)]
    assert kw["control_mode"] == "pd_joint_delta_pos"  # actions stay in the same control mode
    assert ENV_INFO["env_kwargs"]["sensor_configs"] == {"shader_pack": "default"}  # input untouched


def test_camera_env_info_without_extra_cameras():
    assert camera_env_info(ENV_INFO, 128, [])["env_kwargs"]["extra_cameras"] == []


def test_prepare_links_states_and_writes_json(tmp_path):
    src = tmp_path / "src" / "demo.h5"
    src.parent.mkdir()
    src.write_bytes(b"h5")
    src.with_suffix(".json").write_text(json.dumps({"env_info": ENV_INFO, "episodes": [{"episode_id": 0}]}))

    out = tmp_path / "out"
    link = prepare(str(src), str(out), 256, ["top"])
    assert os.path.islink(link) and os.path.realpath(link) == str(src.resolve())
    meta = json.loads((out / "trajectory.json").read_text())
    assert meta["episodes"] == [{"episode_id": 0}]
    assert meta["env_info"]["env_id"] == ENV_ID
    assert output_path(link) == str(out / "trajectory.state_dict+rgb.pd_joint_delta_pos.physx_cpu.h5")

    assert prepare(str(src), str(out), 128, []) == link  # re-running replaces the link and json
    assert json.loads((out / "trajectory.json").read_text())["env_info"]["env_kwargs"]["extra_cameras"] == []


def test_replay_one_episode_keeps_actions(tmp_path):
    pytest.importorskip("mani_skill")
    h5py = pytest.importorskip("h5py")
    from environment.replay_cameras import main
    from global_parameters import ConfigGlobalP

    src = os.path.join(ConfigGlobalP().DATA_DIR, "mani_skill_data/demos/PegInsertionSide-v1/motionplanning/"
                       "trajectory.state_dict+rgb.pd_joint_delta_pos.physx_cpu.h5")
    if not os.path.exists(src):
        pytest.skip("converted demonstrations not available")
    out = main(SimpleNamespace(src=src, out_dir=str(tmp_path), resolution=64, cameras=["top"], num_procs=1, count=1))

    meta = json.loads((tmp_path / "trajectory.json").read_text())
    ep = meta["episodes"][0]["episode_id"]
    with h5py.File(src) as old, h5py.File(out) as new:
        a, b = old[f"traj_{ep}"], new[f"traj_{ep}"]
        np.testing.assert_array_equal(a["actions"][:], b["actions"][:])
        # the replay re-simulates each step from the saved state; contact steps differ slightly
        np.testing.assert_allclose(a["obs/agent/qpos"][:], b["obs/agent/qpos"][:], atol=5e-3)
        cams = b["obs/sensor_data"]
        assert set(cams) == {"base_camera", "hand_camera", "top_camera"}
        assert cams["top_camera/rgb"].shape[1:] == (64, 64, 3)
        assert cams["base_camera/rgb"].shape[1:] == (64, 64, 3)
        assert bool(b["success"][-1])
