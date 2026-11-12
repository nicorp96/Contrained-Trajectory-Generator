from collections import deque
import time

import torch
from einops import rearrange, repeat

from common.inpainting import apply_inpainting, make_inpainting_mask
from common.utils import split_trajectory
from models.policies.base_policy import Policy
from trainer.trainer_diffusion_traj import build_image_encoders


class TensorHistory:
    """Rolling observation history; the first observation fills the whole buffer."""

    def __init__(self, H):
        self.H = H
        self.history = {}

    def update(self, current_data: dict):
        for key, value in current_data.items():
            value = value.clone()
            if value.shape[0] == 1:  # drop batch dim of size 1
                value = value.squeeze(0)
            if key not in self.history:
                self.history[key] = deque([value.clone() for _ in range(self.H)], maxlen=self.H)
            else:
                self.history[key].append(value)

    def get_stacked(self):
        return {
            k: torch.stack(list(v), dim=0).unsqueeze(0) for k, v in self.history.items()
        }


class DiffPolicy(Policy):
    """
    Receding-horizon policy: at each call, sample a future trajectory starting at the
    current state and return its first `action_horizon` actions.

    obs must contain the ManiSkill observation dict plus obs["actions"]: the last
    executed action (zeros at the first step), matching the training history.
    """

    def __init__(self, config, device, only_actions=True, **kwargs):
        super().__init__(config, device, only_actions, **kwargs)
        self.action_key = kwargs.get("action_key", "actions")
        self.action_horizon = int(kwargs.get("action_horizon", 1))
        self.verbose = bool(kwargs.get("verbose", False))
        self.reset()

    def reset(self):
        self.contex_history = TensorHistory(self.config["dataset"]["sequence_hist"])

    @staticmethod
    def get_dict_from_obs(obs, key):
        if key in obs:
            return obs[key]
        for obs_key, group in obs.items():
            if obs_key == "sensor_param" or not isinstance(group, dict):
                continue
            if key in group:
                return group[key]
        raise KeyError(f"Key {key} not found in obs")

    def get_states_from_obs(self, obs):
        """Current state repeated over the horizon (shape template + inpainting values)."""
        out = {}
        for key in self.config["dataset"]["state_shapes"].keys():
            data = self.get_dict_from_obs(obs, key).float()
            out[key] = repeat(data, "B D -> B L D", L=self.sequence_len)
        return out

    def get_goal_from_obs(self, obs):
        return {
            key: self.get_dict_from_obs(obs, key).float()
            for key in self.config["dataset"]["goal_shapes"].keys()
        }

    def get_context_from_obs(self, obs):
        out = {}
        obs_shapes = self.config["dataset"].get("obs_shapes") or {}
        for key, spec in obs_shapes.items():
            data = self.get_dict_from_obs(obs, key)
            if "camera" in key:
                data = rearrange(data[spec.get("type", "rgb")], "B H W C -> B C H W")
            else:
                data = data.float()
            out[key] = data
        return out

    def encode_history(self, history_dict_nm):
        return history_dict_nm

    @torch.no_grad()
    def conditional_sample(self, states_norm_dict, goals_norm, history_dict_nm):
        # no_grad (not inference_mode) so constraint guidance can use autograd
        device = self.device
        model = self.ema_model if self.ema_model is not None else self.model
        scheduler = self.noise_scheduler
        trj = torch.cat([states_norm_dict[k] for k in states_norm_dict], dim=2).to(device)
        cond = torch.cat([goals_norm[k] for k in goals_norm], dim=1).to(device)
        state_hist = torch.cat(
            [history_dict_nm[k].to(device).float() for k in history_dict_nm], dim=2
        )
        mask_known = (
            make_inpainting_mask(
                trajectory=trj,
                start_indices=self.start_indices,
                goal_indices=self.goal_indices,
            )
            if self.inpainting
            else None
        )
        kwargs = {
            "state_hist": state_hist,
            "mask_known": mask_known,
            "constraints": self.constraints,
            "normalizer_state": self.normalizer_state,
            "state_shapes": self.config["dataset"]["state_shapes"],
        }

        scheduler.set_timesteps(self.num_inference_steps, device=device)
        x = torch.randn(size=trj.shape, dtype=trj.dtype, device=device)
        if mask_known is not None:
            x = apply_inpainting(x, mask_known, trj, noise=False)
        for k in scheduler.timesteps:
            x = self.guidance_algo(
                input_noised=x, k=k, cond=cond, scheduler=scheduler, model=model, **kwargs
            )
            if mask_known is not None:
                x = apply_inpainting(x, mask_known, trj, noise=False)
        x = self.guidance_algo.finalize(x, **kwargs)
        if mask_known is not None:
            x = apply_inpainting(x, mask_known, trj, noise=False)
        return x

    def __call__(self, obs):
        states_dict = self.get_states_from_obs(obs)
        goals_dict = self.get_goal_from_obs(obs)
        self.contex_history.update(self.get_context_from_obs(obs))

        states_norm_dict = self.normalizer_state(
            {k: v.to(self.device) for k, v in states_dict.items()}
        )
        goals_norm = self.normalizer_goal({k: v.to(self.device) for k, v in goals_dict.items()})
        ctx_hist = {k: v.to(self.device) for k, v in self.contex_history.get_stacked().items()}
        history_dict_nm = self.encode_history(self.normalizer_obs(ctx_hist))

        t0 = time.perf_counter()
        out_trj = self.conditional_sample(states_norm_dict, goals_norm, history_dict_nm)
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"conditional_sample: {(time.perf_counter() - t0) * 1000:.1f} ms")

        with torch.no_grad():
            output = self.normalizer_state.unnormalize(
                split_trajectory(out_trj, self.config["dataset"]["state_shapes"])
            )
        if self.only_actions:
            return output[self.action_key][:, : self.action_horizon, :]
        return output


class DiffPolicyEncoder(DiffPolicy):
    """DiffPolicy with image encoders (cameras in obs_shapes)."""

    def setup_model(self):
        super().setup_model()
        sd = self.checkpoint["state_dicts"]
        self.image_encoder = build_image_encoders(self.config, for_inference=True)
        # EMA encoder weights match the EMA diffusion model used for sampling
        enc_state = sd.get("ema_image_encoders_state") or sd.get("image_encoders_state")
        if enc_state is not None:
            self.image_encoder.load_state_dict(enc_state)
        self.image_encoder = self.image_encoder.to(self.device).eval()

    @torch.no_grad()
    def encode_history(self, history_dict_nm):
        return self.image_encoder(history_dict_nm)
