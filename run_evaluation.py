"""
Closed-loop evaluation of a trained policy in ManiSkill.

  python run_evaluation.py --ckpt <path/to/checkpoint.pth> --episodes 100 --seed 0

Reports success rate over seeded episodes plus physical-constraint violations of
the executed joint trajectories; writes a JSON summary next to the checkpoint.
Inference-time ablations: --constraint-eta, --project-final, --guidance-scale.
--video-dir records one mp4 per episode (episode_XXX_success|fail.mp4).
"""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers environments)
import numpy as np
import torch
from mani_skill.utils.wrappers import RecordEpisode

import environment.extended_peg_insertion  # noqa: F401  (registers PegInsertionSide-Extended)
from global_parameters import ConfigGlobalP
from models.policies.diff_policy import DiffPolicyEncoder

cfg_global_p = ConfigGlobalP()


def dataset_env_info(config):
    """env_id / control_mode the demonstrations were recorded with."""
    path = config["dataset"]["path"]
    if not os.path.exists(path):
        path = os.path.join(cfg_global_p.DATA_DIR, path)
    json_path = path.replace(".h5", ".json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        env_info = json.load(f)["env_info"]
    return {"env_id": env_info["env_id"], **env_info.get("env_kwargs", {})}


# env kwargs of the demonstrations that define what the cameras see
CAMERA_KWARGS = ("sensor_configs", "extra_cameras")


def camera_kwargs(env_info):
    """Camera settings recorded with the demonstrations (resolution, extra cameras)."""
    return {k: env_info[k] for k in CAMERA_KWARGS if k in env_info}


def make_env(env_id, control_mode, max_steps, video_dir=None, **env_kwargs):
    """Evaluation env; with video_dir every episode is written as <index>.mp4.
    env_kwargs: e.g. camera_kwargs() of the dataset."""
    env = gym.make(
        env_id,
        obs_mode="state_dict+rgb",
        control_mode=control_mode,
        render_mode="rgb_array" if video_dir else None,
        max_episode_steps=max_steps,
        **env_kwargs,
    )
    if video_dir:
        env = RecordEpisode(
            env, output_dir=str(video_dir), save_trajectory=False, save_video=True,
            max_steps_per_video=max_steps,
        )
    return env


def rename_videos(video_dir, results):
    """<index>.mp4 -> episode_<index>_<success|fail>.mp4; returns the new paths."""
    renamed = []
    for r in results:
        src = Path(video_dir) / f"{r['episode']}.mp4"
        if not src.exists():
            continue
        dst = src.with_name(f"episode_{r['episode']:03d}_{'success' if r['success'] else 'fail'}.mp4")
        src.rename(dst)
        renamed.append(dst)
    return renamed


def to_bool(x):
    return bool(torch.as_tensor(x).reshape(-1)[0].item())


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute() and not ckpt.exists():
        ckpt = Path(cfg_global_p.LOGS_DIR) / ckpt

    guidance = {}
    if args.constraint_eta is not None:
        guidance["constraint_eta"] = args.constraint_eta
    if args.project_final:
        guidance["project_final"] = True
    if args.guidance_scale is not None:
        guidance["guidance_scale"] = args.guidance_scale

    policy = DiffPolicyEncoder(
        ckpt,
        device,
        only_actions=False,
        action_key="actions",
        action_horizon=args.action_horizon,
        guidance=guidance,
    )
    env_info = dataset_env_info(policy.config)
    env_id = args.env_id or env_info.get("env_id", "PegInsertionSide-v1")
    control_mode = args.control_mode or env_info.get("control_mode", "pd_joint_delta_pos")
    print(f"env_id={env_id} control_mode={control_mode}")

    cams = camera_kwargs(env_info)
    print(f"cameras: {cams or 'environment default'}")
    env = make_env(env_id, control_mode, args.max_steps, args.video_dir, **cams)
    act_dim = env.action_space.shape[-1]

    results = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            policy.reset()
            obs["actions"] = torch.zeros((1, act_dim))
            policy.observe(obs)
            executed = {"qpos": [], "qvel": [], "actions": []}
            success, steps, done = False, 0, False
            while not done:
                plan = policy(obs, observe=False)  # history already holds obs
                actions = plan["actions"][:, : policy.action_horizon]
                for i in range(actions.shape[1]):
                    action = actions[:, i].detach().cpu()
                    executed["qpos"].append(obs["agent"]["qpos"].float().cpu())
                    executed["qvel"].append(obs["agent"]["qvel"].float().cpu())
                    executed["actions"].append(action.float())
                    obs, _, terminated, truncated, info = env.step(action)
                    obs["actions"] = action
                    policy.observe(obs)
                    steps += 1
                    success = success or to_bool(info.get("success", False))
                    if success or to_bool(terminated) or to_bool(truncated):
                        done = True
                        break

            record = {"episode": ep, "seed": args.seed + ep, "success": success, "steps": steps}
            if policy.constraints is not None:
                traj = {
                    k: torch.stack(v, dim=1).to(policy.device)
                    for k, v in executed.items()
                    if k in policy.config["dataset"]["state_shapes"]
                }
                record["constraints"] = policy.constraints.metrics(traj)
            results.append(record)
            print(f"episode {ep}: success={success} steps={steps}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt: summarising finished episodes")
    finally:
        env.close()
    if args.video_dir:
        rename_videos(args.video_dir, results)

    if not results:
        return
    summary = {
        "checkpoint": str(ckpt),
        "env_id": env_id,
        "control_mode": control_mode,
        "guidance": guidance,
        "episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
    }
    if "constraints" in results[0]:
        keys = results[0]["constraints"].keys()
        summary["constraints"] = {
            k: float(np.mean([r["constraints"][k] for r in results])) for k in keys
        }
    summary["per_episode"] = results
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))

    out_path = Path(args.output) if args.output else ckpt.with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trajectory Generator")
    parser.add_argument("--ckpt", required=True, help="checkpoint (absolute or relative to LOGS_DIR)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--action-horizon", type=int, default=None,
        help="actions executed per plan; defaults to config evaluation.action_horizon (or 1)",
    )
    parser.add_argument("--env-id", default=None, help="defaults to the dataset's env_id")
    parser.add_argument("--control-mode", default=None, help="defaults to the dataset's control_mode")
    parser.add_argument("--constraint-eta", type=float, default=None)
    parser.add_argument("--project-final", action="store_true")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--video-dir", default=None, help="record one mp4 per episode here")
    parser.add_argument("--output", default=None)
    main(parser.parse_args())
