from einops import repeat
from typing import Optional, Dict
import torch


class Constraint:
    def __init__(self, config: Optional[Dict]):
        self.config = config

    def __call__(
        self, value: Optional[torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        raise NotImplementedError()


class ObstacleConstraint(Constraint):
    def __init__(self, config: Optional[Dict]):
        super().__init__(config)
        self.obstacles_torch = []
        for keys in self.config.keys():
            radius_torch = torch.tensor([self.config[keys]["radius"]])
            center_torch = torch.tensor(self.config[keys]["center"])
            self.obstacles_torch.append(
                {"radius": radius_torch, "center": center_torch}
            )

    def __call__(
        self, value: Optional[torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        obstacles = self.obstacles_torch
        device = value.device
        B, H, D = value.shape
        residual_total = torch.zeros((B, H), device=device, dtype=value.dtype)
        energy_total = torch.zeros((), device=device, dtype=value.dtype)

        for obstacle_dict in obstacles:
            center = repeat(obstacle_dict["center"], "d -> B H d", B=B, H=H).to(device)
            radius = (
                repeat(obstacle_dict["radius"], "d -> B H", B=B, H=H)
                .squeeze(2)
                .to(device)
            )
            d = torch.norm(value - center, dim=-1) - radius
            # residual = penetration depth (0 if safe)
            r = torch.relu(-d)  # (B,H)

            # quadratic energy per step (smooth, differentiable)
            e = 0.5 * (r**2)  # (B,H)

            residual_total = residual_total + r
            energy_total = energy_total + e.mean()  # or e.sum() depending on scaling
        dict_out = {"residual_t": residual_total, "energy_t": energy_total}
        return dict_out
