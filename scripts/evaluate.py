"""
Closed-loop evaluation of a trained checkpoint in ManiSkill.

  python scripts/evaluate.py <checkpoint.pt> --episodes 100 --video-dir videos/
  python scripts/evaluate.py <checkpoint.pt> diffusion.num_inference_steps=25 physics.guidance_eta=1
  python scripts/evaluate.py <checkpoint.pt> --physics panda    # constraints from configs/physics/

Positional key=value arguments override the checkpoint's config before the policy is
built (sampling, guidance, physics). Writes a JSON summary (success rate, constraint
metrics of the executed motion, per-episode end states).
"""

import argparse
import json
from pathlib import Path

import torch

from ctg.config import compose_config
from ctg.envs.evaluation import camera_kwargs, dataset_env_info, evaluate, make_env, rename_videos
from ctg.policy.diffusion_policy import DiffusionPolicy


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    overrides = list(args.overrides)
    # a physics option from configs/physics/ (e.g. stricter limits) replaces the checkpoint's
    sections = {"physics": compose_config([f"physics={args.physics}"]).physics} if args.physics else None
    policy = DiffusionPolicy.from_checkpoint(args.checkpoint, device, overrides, args.action_horizon, sections)

    env_info = dataset_env_info(policy.cfg.data.dataset.path)
    env_id = env_info.get("env_id", "PegInsertionSide-v1")
    control_mode = env_info.get("control_mode", "pd_joint_delta_pos")
    print(f"env {env_id} | control {control_mode} | cameras {camera_kwargs(env_info) or 'default'}")
    env = make_env(env_id, control_mode, args.max_steps, args.video_dir, **camera_kwargs(env_info))
    try:
        summary = evaluate(policy, env, args.episodes, args.seed)
    finally:
        env.close()
    if args.video_dir:
        rename_videos(args.video_dir, summary["per_episode"])

    summary = {"checkpoint": str(args.checkpoint), "env_id": env_id, "control_mode": control_mode,
               "overrides": overrides, "physics": args.physics, "action_horizon": policy.action_horizon, **summary}
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))
    out = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".eval.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("overrides", nargs="*", help="config overrides key=value")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--action-horizon", type=int, default=None, help="default: evaluation.action_horizon")
    parser.add_argument("--physics", default=None, help="physics option from configs/physics/ (replaces the checkpoint's)")
    parser.add_argument("--video-dir", default=None, help="record one mp4 per episode here")
    parser.add_argument("--output", default=None, help="default: <checkpoint>.eval.json")
    main(parser.parse_args())
