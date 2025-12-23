from typing import Optional, Dict
import torch


class Constraint:
    def __init__(self, config: Optional[Dict]):
        self.config = config
        self.config.pop("target")

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

    def __call__(self, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        # value: [B, H, D]  (should require_grad=True when coming from x_var slices)
        B, H, D = value.shape
        device = value.device
        dtype = value.dtype

        residual_total = torch.zeros((B, H), device=device, dtype=dtype)
        energy_per_batch = torch.zeros(
            (B,), device=device, dtype=dtype
        )  # <-- [B], not [B,H]

        for obstacle_dict in self.obstacles_torch:
            # center = (
            #     obstacle_dict["center"].to(device=device, dtype=dtype).view(1, 1, D)
            # )  # [1,1,D]
            radius = (
                obstacle_dict["radius"].to(device=device, dtype=dtype).view(1, 1)
            )  # [1,1]
            center = torch.concat([obstacle_dict["center"], obstacle_dict["center"]], dim=-1).to(device=device, dtype=dtype).view(1, 1, D)
            d = torch.norm(value - center, dim=-1) - radius  # [B,H] signed distance
            r = torch.relu(-d)  # [B,H] penetration depth
            e = 0.5 * (r**2)  # [B,H] energy per step

            residual_total = residual_total + r
            energy_per_batch = energy_per_batch + e.mean(dim=1)  # [B] mean over time

        return {"residual_t": residual_total, "energy_t": energy_per_batch}


class ObstacleConstraintGrad(Constraint):
    def __init__(self, config: Optional[Dict]):
        super().__init__(config)
        self.obstacles_torch = []
        for keys in self.config.keys():
            radius_torch = torch.tensor([self.config[keys]["radius"]])
            center_torch = torch.tensor(self.config[keys]["center"])
            self.obstacles_torch.append(
                {"radius": radius_torch, "center": center_torch}
            )

    def __call__(self, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        # value: [B, H, D]  (should require_grad=True when coming from x_var slices)
        B, H, D = value.shape
        device = value.device
        dtype = value.dtype

        residual_total = torch.zeros((B, H, D), device=device, dtype=dtype)
        energy_grad_per_batch = torch.zeros(
            (B, H, D), device=device, dtype=dtype
        )  # <-- [B], not [B,H]

        for obstacle_dict in self.obstacles_torch:
            # center = (
            #     obstacle_dict["center"].to(device=device, dtype=dtype).view(1, 1, D)
            # )  # [1,1,D]
            center = (
                torch.concat([obstacle_dict["center"], obstacle_dict["center"]], dim=-1)
                .to(device=device, dtype=dtype)
                .view(1, 1, D)
            )
            radius = (
                obstacle_dict["radius"].to(device=device, dtype=dtype).view(1, 1)
            )  # [1,1]

            r_vec = value - center  # [B,H,D]
            x_norm = torch.linalg.norm(r_vec, dim=-1)  # [B,H]

            d = x_norm - radius  # [B,H]   (distance-to-surface)
            inv = 1.0 / (x_norm + 1e-6)  # [B,H]

            grad_e = (d * inv).unsqueeze(
                -1
            ) * r_vec  # [B,H,D]  == d * r_vec / (x_norm+eps)
            residual_total = residual_total + r_vec
            # energy_per_batch = energy_per_batch + e.mean(dim=1)  # [B] mean over time
            energy_grad_per_batch = energy_grad_per_batch + grad_e  # [B, H]
        return {"residual_t": residual_total, "energy_t": energy_grad_per_batch}

