"""
Closed-loop evaluation of a policy in ManiSkill: seeded episodes, success, the
peg-hole end state and physical-constraint metrics of the executed motion.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# env kwargs of the demonstrations that define what the cameras see
CAMERA_KWARGS = ("sensor_configs", "extra_cameras")


def dataset_env_info(dataset_path: str) -> Dict:
    """env_id and env kwargs the demonstrations were recorded with ({} if unknown)."""
    json_path = str(dataset_path).replace(".h5", ".json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        env_info = json.load(f)["env_info"]
    return {"env_id": env_info["env_id"], **env_info.get("env_kwargs", {})}


def camera_kwargs(env_info: Dict) -> Dict:
    """Camera settings recorded with the demonstrations (resolution, extra cameras)."""
    return {k: env_info[k] for k in CAMERA_KWARGS if k in env_info}


def make_env(env_id: str, control_mode: str, max_steps: int, video_dir: Optional[str] = None, **env_kwargs):
    """Evaluation env; with video_dir every episode is written as <index>.mp4."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers the ManiSkill tasks)
    from mani_skill.utils.wrappers import RecordEpisode

    import ctg.envs.peg_insertion  # noqa: F401  (registers PegInsertionSide-Extended)

    env = gym.make(env_id, obs_mode="state_dict+rgb", control_mode=control_mode,
                   render_mode="rgb_array" if video_dir else None, max_episode_steps=max_steps, **env_kwargs)
    if video_dir:
        env = RecordEpisode(env, output_dir=str(video_dir), save_trajectory=False, save_video=True,
                            max_steps_per_video=max_steps)
    return env


def rename_videos(video_dir: str, results: List[Dict]) -> List[Path]:
    """<index>.mp4 -> episode_<index>_<success|fail>.mp4; returns the new paths."""
    renamed = []
    for r in results:
        src = Path(video_dir) / f"{r['episode']}.mp4"
        if src.exists():
            dst = src.with_name(f"episode_{r['episode']:03d}_{'success' if r['success'] else 'fail'}.mp4")
            src.rename(dst)
            renamed.append(dst)
    return renamed


def insertion_state(env, info: Dict) -> Dict:
    """Peg head relative to the hole at the end of an episode, for miss analysis.
    Hole frame of PegInsertionSide: x along the insertion axis (entrance at -peg_half_length,
    success needs x >= -0.015), y sideways, z up; {} for other tasks."""
    if "peg_head_pos_at_hole" not in info:
        return {}
    base = env.unwrapped
    return {
        "peg_head_at_hole": torch.as_tensor(info["peg_head_pos_at_hole"]).reshape(-1)[:3].tolist(),
        "hole_radius": float(base.box_hole_radii[0]),
        "peg_radius": float(base.peg_half_sizes[0, 2]),
        "peg_half_length": float(base.peg_half_sizes[0, 0]),
        "peg_height": float(base.peg.pose.p[0, 2] - base.peg_half_sizes[0, 2]),  # 0 = lying on the table
    }


def to_bool(x) -> bool:
    return bool(torch.as_tensor(x).reshape(-1)[0].item())


def run_episode(policy, env, seed: int, action_dim: int) -> Dict:
    """One closed-loop episode; returns success, steps, end state and the executed motion."""
    obs, info = env.reset(seed=seed)
    policy.reset()
    obs["actions"] = torch.zeros((1, action_dim))
    policy.observe(obs)
    executed = {"qpos": [], "qvel": [], "actions": []}
    success, steps, done = False, 0, False
    while not done:
        actions = policy.act(obs, observe=False)  # the history already holds obs
        for i in range(actions.shape[1]):
            action = actions[:, i].detach().cpu().float()
            executed["qpos"].append(obs["agent"]["qpos"].float().cpu())
            executed["qvel"].append(obs["agent"]["qvel"].float().cpu())
            executed["actions"].append(action)
            obs, _, terminated, truncated, info = env.step(action)
            obs["actions"] = action
            policy.observe(obs)
            steps += 1
            success = success or to_bool(info.get("success", False))
            if success or to_bool(terminated) or to_bool(truncated):
                done = True
                break
    record = {"seed": seed, "success": success, "steps": steps, **insertion_state(env, info)}
    return record, {k: torch.stack(v, dim=1) for k, v in executed.items()}


def evaluate(policy, env, episodes: int, seed: int = 0) -> Dict:
    """Seeded closed-loop episodes; summary with success rate and executed-motion constraint metrics."""
    action_dim = env.action_space.shape[-1]
    results = []
    for ep in range(episodes):
        record, executed = run_episode(policy, env, seed + ep, action_dim)
        record["episode"] = ep
        if policy.constraints is not None:
            record["constraints"] = policy.constraints.metrics({k: v.to(policy.device) for k, v in executed.items()})
        results.append(record)
        print(f"episode {ep}: success={record['success']} steps={record['steps']}")
    summary = {"episodes": len(results),
               "success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
               "mean_steps": float(np.mean([r["steps"] for r in results])) if results else 0.0}
    if results and "constraints" in results[0]:
        summary["constraints"] = {k: float(np.mean([r["constraints"][k] for r in results]))
                                  for k in results[0]["constraints"]}
    summary["per_episode"] = results
    return summary
