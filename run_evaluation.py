import argparse
import os
import gymnasium as gym
from dataclasses import dataclass
from pathlib import Path
import mani_skill.envs
import torch

from common.get_class import get_class_dict
from common.utils import load_config
from global_parameters import ConfigGlobalP
from models.policies.diff_policy import DiffPolicyEncoder


@dataclass
class EvaluationConfig:
    CKPT: Path = Path(
        "logs/diffusion_trj_vel_padding/20260320-150115/logs/checkpoint_2700.pth"
    )


cfg_global_p = ConfigGlobalP()
cfg_eval = EvaluationConfig()


def get_environment(id="PegInsertionSide-v1"):
    env = gym.make(
        id,
        obs_mode="state_dict+rgb+depth",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        reconfiguration_freq=1,
    )
    return env


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        checkpoint_path = os.path.join(cfg_global_p.LOGS_DIR, cfg_eval.CKPT)
        policy = DiffPolicyEncoder(
            checkpoint_path,
            device,
            only_actions=True,
            action_key="actions",
            action_horizon=1,
        )
        env = get_environment(policy.config.get("env_id", "PegInsertionSide-v1"))
        obs, info = env.reset()
        init_action = torch.zeros((1, 8))
        obs.update({"actions": init_action})
        while not info["success"]:
            actions = policy(obs)
            for step in range(policy.action_horizon):
                obs, rew, terminated, truncated, info = env.step(actions[:, step, :])
                obs.update({"actions": actions[:, step, :]})
                env.render_human()
            # if terminated or truncated:
            #     obs, info = env.reset()
            # env.render_human()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trajectory Generator")
    parser.add_argument(
        "-c", "--config", help="Name of config file", default="diffusion"
    )
    args = parser.parse_args()
    main(args)
