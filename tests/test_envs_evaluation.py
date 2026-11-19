"""ctg/envs/evaluation.py: env, video and end-state helpers."""

import pytest

pytest.importorskip("mani_skill")

from ctg.envs.evaluation import camera_kwargs, dataset_env_info, insertion_state, make_env, rename_videos  # noqa: E402

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


def test_insertion_state_other_task_is_empty():
    assert insertion_state(None, {"success": True}) == {}


def test_insertion_state_at_reset():
    env = make_env(ENV_ID, CONTROL_MODE, max_steps=5)
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample() * 0)
        state = insertion_state(env, info)
    finally:
        env.close()
    x, y, z = state["peg_head_at_hole"]
    assert x < -state["peg_half_length"]  # peg still in front of the entrance
    assert abs(state["peg_height"]) < 0.005  # lying on the table
    clearance = state["hole_radius"] - state["peg_radius"]
    assert clearance == pytest.approx(0.003, abs=1e-6)  # task definition


def test_insertion_state_matches_success_check():
    env = make_env(ENV_ID, CONTROL_MODE, max_steps=5)
    try:
        env.reset(seed=1)
        base = env.unwrapped
        base.peg.set_pose(base.goal_pose)  # peg head exactly at the hole centre
        info = base.evaluate()
        state = insertion_state(env, info)
    finally:
        env.close()
    assert bool(info["success"][0])
    assert max(abs(v) for v in state["peg_head_at_hole"]) < 1e-5


def _wrapper_names(env):
    names = []
    while env is not None:
        names.append(type(env).__name__)
        env = getattr(env, "env", None)
    return names


def test_dataset_env_info(tmp_path):
    (tmp_path / "demo.json").write_text('{"env_info": {"env_id": "X-v1", "env_kwargs": {"control_mode": "c"}}}')
    assert dataset_env_info(str(tmp_path / "demo.h5")) == {"env_id": "X-v1", "control_mode": "c"}
    assert dataset_env_info(str(tmp_path / "missing.h5")) == {}


def test_evaluate_runs_closed_loop_with_constraint_metrics(e9_checkpoint):
    from ctg.envs.evaluation import evaluate
    from ctg.policy.diffusion_policy import DiffusionPolicy

    policy = DiffusionPolicy.from_checkpoint(e9_checkpoint, overrides=["diffusion.num_inference_steps=2"],
                                             action_horizon=4)
    env = make_env(ENV_ID, CONTROL_MODE, max_steps=6)
    try:
        summary = evaluate(policy, env, episodes=2, seed=0)
    finally:
        env.close()
    assert summary["episodes"] == 2 and 0.0 <= summary["success_rate"] <= 1.0
    assert summary["mean_steps"] == 6  # truncated after max_steps
    assert "joint_limits_violation_rate" in summary["constraints"]
    record = summary["per_episode"][0]
    assert record["seed"] == 0 and "peg_head_at_hole" in record
