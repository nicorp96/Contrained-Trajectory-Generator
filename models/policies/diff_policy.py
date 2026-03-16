import os
import numpy as np
import torch
from typing import Dict
from einops import repeat, rearrange

from common.utils import split_state_tensor
from common.get_class import get_class_dict
from common.inpainting import apply_inpainting, make_inpainting_mask
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from dataset.utils import get_ds_from_cfg
from models.policies.base_policy import Policy


class DiffPolicy(Policy):
    def __init__(self, config, device, only_actions=True, **kwargs):
        super().__init__(config, device, only_actions, **kwargs)
        self.action_key = (
            kwargs["action_key"] if "action_key" in kwargs.keys() else "actions"
        )
        self.action_horizon = (
            kwargs["action_horizon"] if "action_horizon" in kwargs.keys() else 1
        )

    def get_dict_from_obs(self, obs, key):
        data = None
        for obs_key in obs.keys():
            if obs_key == "sensor_param":
                continue
            if key in obs[obs_key].keys():
                data = obs[obs_key][key]
                break
        # if key in obs:
        #     data = obs[key]
        if data is None:
            print(f"Key {key} not found in obs")
        return data
        #     data_zeros = torch.zeros((1, sequence_len, shape_dict[key]["shape"]))
        #     data_zeros[:, 0, :] = data
        #     out[key] = data_zeros
        # return out

    def get_states_from_obs(self, obs):
        out = {}
        shapes_dict = self.config["dataset"]["state_shapes"]
        for key in shapes_dict.keys():
            data = self.get_dict_from_obs(obs, key)
            if key == self.action_key:
                data = torch.zeros((1, shapes_dict[key]["shape"]))
            data_zeros = repeat(
                data, "B D -> B L D", L=self.sequence_len
            )  # torch.zeros((1, self.sequence_len, shapes_dict[key]["shape"]))
            # data_zeros[:, -1, :] = data
            out[key] = data_zeros
        return out

    def get_goal_from_obs(self, obs):
        out = {}
        shapes_dict = self.config["dataset"]["goal_shapes"]
        for key in shapes_dict.keys():
            data = self.get_dict_from_obs(obs, key)
            out[key] = data
        return out

    def get_context_from_obs(self, obs):
        out = {}
        shapes_dict = self.config["dataset"]["obs_shapes"]
        sequence_hist = self.config["dataset"]["sequence_hist"]
        for key in shapes_dict.keys():
            data = self.get_dict_from_obs(obs, key)
            if key == self.action_key:
                data = torch.zeros((1, shapes_dict[key]["shape"]))
            if "camera" in key:
                data = data["rgb"]
                data = rearrange(data, "B H W C -> B C H W")
                data_zeros = repeat(data, "B C H W -> B L C H W", L=sequence_hist)
                # data_zeros[:, -1, :, :, :] = data
            else:
                data_zeros = torch.zeros((1, sequence_hist, shapes_dict[key]["shape"]))
                # data_zeros[:, -1, :] = data
            out[key] = data_zeros
        return out

    def get_states_goal_extra_from_obs(self, obs):
        # config = self.config
        # states_dict, goals_dict, extra = {}, {}, {}
        states_dict = self.get_states_from_obs(obs)
        goals_dict = self.get_goal_from_obs(obs)
        extra = self.get_context_from_obs(obs)
        return states_dict, goals_dict, extra

    @torch.inference_mode()
    def conditional_sample(self, states_norm_dict, goals_norm, history_dict_nm, device):
        model = self.ema_model if self.ema_model is not None else self.model
        inpt = self.inpainting
        guidance_algo = self.guidance_algo
        scheduler = self.noise_scheduler
        trj = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)
        cond = torch.cat(
            [goals_norm[key] for key in goals_norm.keys()],
            dim=1,
        ).to(device)
        trajectory_history = torch.cat(
            [history_dict_nm[key].to(device) for key in history_dict_nm.keys()],
            dim=2,
        )
        mask_inp = make_inpainting_mask(
            trajectory=trj,
            start_indices=self.start_indices,
            goal_indices=self.goal_indices,
        )

        trajectory_n = torch.randn(size=trj.shape, dtype=trj.dtype, device=trj.device)

        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)

        if cond is None:
            raise ValueError(
                "cond is required for CFG (set guidance_scale=1.0 to disable)."
            )
        if inpt:
            trajectory_n = apply_inpainting(trajectory_n, mask_inp, trj, noise=False)
        for k in scheduler.timesteps:
            trajectory_n = guidance_algo(
                input_noised=trajectory_n,
                k=k,
                cond=cond,
                scheduler=scheduler,
                model=model,
                state_hist=trajectory_history,
            )
            if inpt:
                trajectory_n = apply_inpainting(
                    trajectory_n, mask_inp, trj, noise=False
                )
        return trajectory_n

    def __call__(self, obs):
        states_dict, goals_dict, extra = self.get_states_goal_extra_from_obs(obs)
        device = self.device
        # Normalize
        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        history_dict_nm = self.normalizer_obs(extra)
        # Sample trajectory
        out_trj = self.conditional_sample(
            states_norm_dict, goals_norm, history_dict_nm, device=device
        )
        output_dict = split_state_tensor(
            out_trj, self.config["dataset"]["state_shapes"]
        )
        output_unnormalized = self.normalizer_state.unnormalize(output_dict)
        if self.only_actions:
            output_unnormalized = output_unnormalized[self.action_key][
                :, : self.action_horizon, :
            ]
        return output_unnormalized


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
