"""Tests for the env and video helpers of run_evaluation.py."""

import pytest

pytest.importorskip("mani_skill")

from run_evaluation import camera_kwargs, make_env, rename_videos  # noqa: E402

ENV_ID = "PegInsertionSide-v1"
CONTROL_MODE = "pd_joint_delta_pos"


def test_rename_videos_marks_success_and_failure(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}.mp4").write_bytes(b"x")
    results = [
        {"episode": 0, "success": True},
        {"episode": 1, "success": False},
        {"episode": 2, "success": False},
    ]
    renamed = rename_videos(tmp_path, results)
    assert sorted(p.name for p in renamed) == [
        "episode_000_success.mp4",
        "episode_001_fail.mp4",
        "episode_002_fail.mp4",
    ]
    assert not any(tmp_path.glob("[0-9].mp4"))


def test_rename_videos_skips_missing_files(tmp_path):
    (tmp_path / "1.mp4").write_bytes(b"x")
    results = [{"episode": 0, "success": True}, {"episode": 1, "success": True}]
    assert [p.name for p in rename_videos(tmp_path, results)] == ["episode_001_success.mp4"]


def test_rename_videos_empty_results(tmp_path):
    assert rename_videos(tmp_path, []) == []


def test_make_env_without_video_has_no_render():
    env = make_env(ENV_ID, CONTROL_MODE, max_steps=5)
    try:
        assert env.render_mode is None
        assert "RecordEpisode" not in _wrapper_names(env)
    finally:
        env.close()


def test_make_env_records_one_video_per_episode(tmp_path):
    env = make_env(ENV_ID, CONTROL_MODE, max_steps=5, video_dir=tmp_path)
    try:
        assert "RecordEpisode" in _wrapper_names(env)
        for seed in range(2):
            env.reset(seed=seed)
            for _ in range(3):
                env.step(env.action_space.sample() * 0)
    finally:
        env.close()
    assert sorted(p.name for p in tmp_path.glob("*.mp4")) == ["0.mp4", "1.mp4"]


def test_camera_kwargs_picks_camera_settings_only():
    env_info = {"env_id": "X", "control_mode": "pd_joint_delta_pos", "sensor_configs": {"width": 256},
                "extra_cameras": [{"uid": "top_camera"}], "sim_backend": "physx_cpu"}
    assert camera_kwargs(env_info) == {"sensor_configs": {"width": 256}, "extra_cameras": [{"uid": "top_camera"}]}
    assert camera_kwargs({"env_id": "X"}) == {}


def test_make_env_uses_dataset_cameras():
    env = make_env(
        "PegInsertionSide-Extended", CONTROL_MODE, max_steps=5,
        sensor_configs={"width": 48, "height": 48},
        extra_cameras=[{"uid": "top_camera", "eye": [0.0, 0.12, 0.65], "target": [0.0, 0.22, 0.1]}],
    )
    try:
        obs, _ = env.reset(seed=0)
        cams = obs["sensor_data"]
        assert set(cams) == {"base_camera", "hand_camera", "top_camera"}
        assert all(cams[c]["rgb"].shape[1:3] == (48, 48) for c in cams)
    finally:
        env.close()


def _wrapper_names(env):
    names = []
    while env is not None:
        names.append(type(env).__name__)
        env = getattr(env, "env", None)
    return names
