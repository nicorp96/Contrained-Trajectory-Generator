"""
Receding-horizon diffusion policy for closed-loop evaluation.

Per planning call it samples a plan from the current observation history and goal
and returns its first `action_horizon` actions. With several actions per plan,
call observe(obs) after every executed env step and plan with observe=False, so
the history holds consecutive steps as in training.
"""

from typing import Dict, Mapping, Optional, Sequence

import torch
from omegaconf import OmegaConf

from ctg.data.layout import DataSpec, concat
from ctg.diffusion.inpainting import known_mask
from ctg.policy.history import ObservationHistory
from ctg.training.trainer import build_components, channel_indices


def lookup(obs: Mapping, key: str):
    """obs[key], or obs[group][key] for the ManiSkill groups (agent, extra, sensor_data)."""
    if key in obs:
        return obs[key]
    for group_name, group in obs.items():
        if group_name != "sensor_param" and isinstance(group, Mapping) and key in group:
            return group[key]
    raise KeyError(f"key {key!r} not in observation")


class DiffusionPolicy:
    def __init__(self, cfg, device="cpu", action_horizon: Optional[int] = None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.spec = DataSpec.from_config(cfg.data.dataset)
        parts = build_components(cfg, self.spec, pretrained_encoders=False)
        self.normalizers = {k: v.to(self.device) for k, v in parts["normalizers"].items()}
        self.encoders = parts["encoders"].to(self.device).eval()
        self.model = parts["model"].to(self.device).eval()
        self.sampler, self.codec, self.constraints = parts["sampler"], parts["codec"], parts["constraints"]
        self.action_horizon = int(action_horizon or cfg.evaluation.action_horizon)
        self.start_dims = channel_indices(self.spec.plan_layout, cfg.diffusion.inpainting.start_fields)
        self.history = ObservationHistory(self.spec.history_len)

    @classmethod
    def from_checkpoint(cls, path, device="cpu", overrides: Sequence[str] = (), action_horizon: Optional[int] = None,
                        sections: Optional[Mapping] = None):
        """overrides: dotted config overrides applied before building, e.g. "diffusion.num_inference_steps=25";
        sections: whole config sections to replace, e.g. {"physics": <a configs/physics option>}."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(ckpt["config"])
        for name, section in (sections or {}).items():
            cfg[name] = section
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        policy = cls(cfg, device, action_horizon)
        policy.model.load_state_dict(ckpt["ema_model"])  # sampling uses the EMA weights
        policy.encoders.load_state_dict(ckpt["ema_encoders"] or ckpt["encoders"])
        for name, state in ckpt["normalizers"].items():
            policy.normalizers[name].load_state_dict(state)
        return policy

    # ------------------------------------------------------------------ interface
    def reset(self):
        self.history.reset()

    def observe(self, obs: Mapping):
        """Append one observation (once per executed env step). obs must also contain
        obs[action_key]: the last executed action (zeros at the first step)."""
        entry = {k: lookup(obs, k).float() for k in self.spec.history_layout}
        entry.update({cam: lookup(obs, cam)["rgb"] for cam in self.spec.cameras})
        self.history.update(entry)

    @torch.no_grad()
    def plan(self, obs: Mapping, observe: bool = True) -> Dict[str, torch.Tensor]:
        """Physical plan {field: [1, plan_len, d]} from the current observation."""
        if observe:
            self.observe(obs)
        d = self.device
        goal = concat(self.normalizers["goal"]({k: lookup(obs, k).float().to(d) for k in self.spec.goal_layout}),
                      self.spec.goal_layout)
        hist = self.encoders(self.normalizers["history"]({k: v.to(d) for k, v in self.history.stacked().items()}))
        history = torch.cat([hist[k].float() for k in list(self.spec.history_layout) + list(self.spec.cameras)], dim=-1)
        shape = (1, self.spec.plan_len, sum(s.width for s in self.spec.plan_layout.values()))
        known = known_mask(shape, self.start_dims, device=d)
        known_values = None
        if known is not None:  # the current state, repeated over the plan
            current = {k: lookup(obs, k).float().to(d).reshape(1, 1, -1).expand(1, shape[1], -1)
                       for k in self.spec.plan_layout}
            known_values = self.codec.encode(current)
        time = (torch.arange(shape[1], device=d, dtype=torch.float32) * self.spec.control_dt).unsqueeze(0)
        plan = self.sampler.sample(self.model, shape, goal, history, self.codec, known, known_values, time)
        return self.codec.decode(plan)

    def act(self, obs: Mapping, observe: bool = True) -> torch.Tensor:
        """The first action_horizon actions [1, action_horizon, action_dim]."""
        return self.plan(obs, observe)[self.spec.action_key][:, : self.action_horizon]
