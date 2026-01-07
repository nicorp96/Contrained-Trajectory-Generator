import os
import numpy as np
import torch
from typing import Dict

from common.utils import split_state_tensor
from common.get_class import get_class_dict
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from dataset.utils import get_ds_from_cfg


class DiffPolicy:
    def __init__(self, config, device):
        self.config_policy = config
        self.device = device
        self.model = None
        self.checkpoint = None
        self.config = None
        self.diff_model = None
        self.noise_scheduler = None
        self.horizon = None
        self.state_normalizer, self.action_normalizer, self.env_normalizer = (
            None,
            None,
            None,
        )
        self.load_ckp_diff_model()
        self.setup_model()
        self.setup_scheduler()
        self.setup_normalizer()

    def load_ckp_diff_model(self):
        config_policy = self.config_policy
        checkpoint_path = config_policy["path_ckp"]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint["config"]
        self.horizon = self.config["dataset"]["horizon"]

    def setup_model(self):
        self.model = get_class_dict(self.config["model"])
        self.model.load_state_dict(self.checkpoint["state_dicts"]["model_state"])
        self.model = self.model.to(self.device)

    def setup_normalizer(self):
        dataset_dict = self.config["dataset"]
        train_dataset, validation_dataset = get_ds_from_cfg(dataset_dict)
        self.train_dataset = train_dataset
        # TODO: load normalizer from dict
        self.state_normalizer, self.action_normalizer, self.env_normalizer = (
            self.train_dataset.dataset.get_normalizer()
        )
        self.model.set_normalizer(
            self.state_normalizer, self.action_normalizer, self.env_normalizer
        )

    def setup_scheduler(self):
        scheduler_cfg = self.config["model"]["diffusion_model"]["noise_scheduler"]
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=scheduler_cfg["num_train_timesteps"],
            beta_start=scheduler_cfg["beta_start"],
            beta_end=scheduler_cfg["beta_end"],
            beta_schedule=scheduler_cfg["beta_schedule"],
            clip_sample=scheduler_cfg["clip_sample"],
            set_alpha_to_one=scheduler_cfg["set_alpha_to_one"],
            steps_offset=scheduler_cfg["steps_offset"],
            prediction_type=scheduler_cfg["prediction_type"],
        )
        self.model.set_scheduler(scheduler=self.noise_scheduler)

    def __call__(self, states, action):
        config = self.config
        model = self.model
        device = self.device
        horizon = self.horizon
        desired_pos_init = torch.from_numpy(action[0, :2]).to(device)
        desired_pos_full = torch.ones((1, horizon, 2)).to(device)
        desired_pos_full[0, 0, :] = desired_pos_init
        states = {
            "desired_pos": desired_pos_full,
            "current_pos": desired_pos_full,
        }
        # actions = {"vel": torch.zeros((1, horizon, 2), device=device)}
        act_torch = torch.from_numpy(action).unsqueeze(0).to(device)
        actions = {"vel": act_torch}
        environment = {
            "goal": torch.ones((1, 2), device=device),
        }
        states_gt_nm_dict = self.state_normalizer(states)
        action_gt_nm_dict = self.action_normalizer(actions)
        env_nm_dict = self.env_normalizer(environment)
        out_trj = model.conditional_sample(
            states_gt_nm_dict, action_gt_nm_dict, env_nm_dict, device=device
        )
        out_dict = split_state_tensor(
            out_trj["traj_pred"][:, :, 2:], config["dataset"]["state_shapes"]
        )
        out_dict = self.state_normalizer.unnormalize(out_dict)
        out_action_dict = split_state_tensor(
            out_trj["traj_pred"][:, :, :2], config["dataset"]["actions_shapes"]
        )
        out_action_dict = self.action_normalizer.unnormalize(out_action_dict)
        return out_action_dict, out_dict


if __name__ == "__main__":
    config = {
        "path_ckp": "/mnt/data_nrp/research_t_opt/logs/GCDiTDiff/20251216-111751/logs/checkpoint_1195.pth"
    }
    device = torch.device("cuda:0")
    policy = DiffPolicy(config, device)
    states = {
        "desired_pos": torch.ones((1, 16, 2)),
        "current_pos": torch.ones((1, 16, 2)),
    }
    actions = {"vel": torch.ones((1, 16, 2))}
    environment = {
        "goal": torch.ones((1, 2)),
    }
    out = policy(states, actions, environment)
    print(out.size())
