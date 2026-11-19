"""Shared fixtures: a small synthetic demonstration file with the ManiSkill PegInsertion layout,
its statistics, and the real data directory (tests that need it are skipped without it)."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from ctg.data.stats import field_stats

DT = 0.05
LENGTHS = {0: 20, 1: 27, 2: 24, 3: 30, 4: 22}  # actions per episode (observations: one more)
FAILED = 5  # an extra unsuccessful episode
SIZES = {"qpos": 9, "qvel": 9, "tcp_pose": 7, "peg_pose": 7, "peg_half_size": 3, "box_hole_pose": 7}
GROUPS = {"qpos": "agent", "qvel": "agent", "tcp_pose": "extra", "peg_pose": "extra",
          "peg_half_size": "extra", "box_hole_pose": "extra"}
IMAGE = 16  # camera resolution


def episode(ep: int, n: int, camera: bool) -> dict:
    """Smooth joint motion inside the Panda limits with consistent velocities and actions."""
    t = np.arange(n + 1) * DT
    phase = np.linspace(0, 1, n + 1)[:, None]
    q0 = np.array([0.0, 0.2, 0.0, -2.0, 0.0, 2.0, 0.8, 0.02, 0.02])
    amp = np.array([0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3, 0.01, 0.01]) * (1 + 0.1 * ep)
    qpos = q0 + amp * np.sin(np.pi * phase)
    qvel = amp * np.pi * np.cos(np.pi * phase) / t[-1]
    data = {
        "actions": np.clip(np.diff(qpos, axis=0)[:, :8] / 0.1, -1, 1),
        "qpos": qpos, "qvel": qvel,
        "tcp_pose": np.tile([0.1 * ep, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], (n + 1, 1)) + 0.01 * phase,
        "peg_pose": np.tile([0.0, 0.1 * ep, 0.05, 1.0, 0.0, 0.0, 0.0], (n + 1, 1)),
        "peg_half_size": np.tile([0.1, 0.02, 0.02], (n + 1, 1)),
        "box_hole_pose": np.tile([0.0, 0.3, 0.1 + 0.01 * ep, 1.0, 0.0, 0.0, 0.0], (n + 1, 1)),
    }
    if camera:
        data["base_camera"] = np.full((n + 1, IMAGE, IMAGE, 3), 10 * ep, dtype=np.uint8)
    return data


def write_demos(directory: Path, camera: bool = True) -> Path:
    path = directory / "trajectory.h5"
    episodes = {**LENGTHS, FAILED: 10}
    with h5py.File(path, "w") as f:
        for ep, n in episodes.items():
            g = f.create_group(f"traj_{ep}")
            for key, value in episode(ep, n, camera).items():
                if key == "actions":
                    g[key] = value.astype(np.float32)
                elif key == "base_camera":
                    g[f"obs/sensor_data/{key}/rgb"] = value
                else:
                    g[f"obs/{GROUPS[key]}/{key}"] = value.astype(np.float32)
            g["obs/sensor_param/base_camera/intrinsic_cv"] = np.zeros((n + 1, 3, 3), dtype=np.float32)
    meta = {"env_info": {"env_id": "PegInsertionSide-v1", "max_episode_steps": 100,
                         "env_kwargs": {"control_mode": "pd_joint_delta_pos", "obs_mode": "state_dict+rgb"}},
            "episodes": [{"episode_id": ep, "success": ep != FAILED} for ep in episodes]}
    (directory / "trajectory.json").write_text(json.dumps(meta))
    return path


@pytest.fixture(scope="session")
def demo_h5(tmp_path_factory) -> str:
    return str(write_demos(tmp_path_factory.mktemp("demos")))


@pytest.fixture(scope="session")
def demo_stats(demo_h5, tmp_path_factory) -> str:
    """Statistics over all steps of all episodes (enough for tests)."""
    chunks = {k: [] for k in [*SIZES, "actions"]}
    with h5py.File(demo_h5, "r") as f:
        for g in f.values():
            chunks["actions"].append(g["actions"][:])
            for key in SIZES:
                chunks[key].append(g[f"obs/{GROUPS[key]}/{key}"][:])
    stats = {k: field_stats(np.concatenate(v).astype(np.float64)) for k, v in chunks.items()}
    path = tmp_path_factory.mktemp("stats") / "stats.json"
    path.write_text(json.dumps(stats))
    return str(path)


@pytest.fixture
def demos_dir() -> Path:
    """Real PegInsertionSide demonstrations below $CTG_DATA_DIR."""
    data_dir = os.environ.get("CTG_DATA_DIR")
    if not data_dir:
        pytest.skip("CTG_DATA_DIR not set")
    return Path(data_dir) / "mani_skill_data/demos/PegInsertionSide-v1/motionplanning"


def tiny_overrides(demo_h5: str, demo_stats: str, logs: Path, *extra: str) -> list:
    """Config overrides for a seconds-long CPU training run on the synthetic demonstrations."""
    return [f"data.dataset.path={demo_h5}", f"data.stats={demo_stats}", f"paths.logs={logs}",
            "data.samples_per_episode=2", "data.val_ratio=0.4",
            "training.epochs=2", "training.batch_size=4", "training.num_workers=0",
            "training.mixed_precision=no", "training.warmup_steps=1", "training.val_every=1",
            "training.save_every=1", "diffusion.num_inference_steps=4", *extra]


def train_tiny(demo_h5, demo_stats, run_dir: Path, *extra: str):
    from ctg.config import compose_config
    from ctg.training.trainer import Trainer

    cfg = compose_config(tiny_overrides(demo_h5, demo_stats, run_dir, *extra))
    trainer = Trainer(cfg, str(run_dir))
    trainer.train()
    return trainer


@pytest.fixture(scope="session")
def e9_checkpoint(demo_h5, demo_stats, tmp_path_factory) -> Path:
    """E9 (actions + joint state, Panda constraints) trained for two epochs."""
    run_dir = tmp_path_factory.mktemp("e9")
    train_tiny(demo_h5, demo_stats, run_dir, "experiment=e9", "physics.loss_weight=0.1")
    return run_dir / "checkpoints" / "epoch_00001.pt"
