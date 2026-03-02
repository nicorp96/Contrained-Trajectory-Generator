from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import repeat
import torch
import torch.nn.functional as F
from typing import Optional

from common.utils import split_state_tensor


class BaseGuidance:
    def __init__(self, config):
        self.config = config
        self.guidance_scale = config["guidance_scale"]

    def __call__(
        self,
        input_noised: torch.Tensor,
        k: torch.Tensor,
        cond: torch.Tensor,
        scheduler: Optional[DDIMScheduler],
        model: Optional[torch.nn.Module],
        **kwargs,
    ):
        raise NotImplementedError


class CFGGuidance(BaseGuidance):
    def __init__(self, config):
        super().__init__(config)

    def __call__(
        self,
        input_noised: torch.Tensor,
        k: torch.Tensor,
        cond: torch.Tensor,
        scheduler: Optional[DDIMScheduler],
        model: Optional[torch.nn.Module],
        **kwargs,
    ):
        device = input_noised.device
        guidance_scale = self.guidance_scale
        uncond = torch.zeros_like(cond)
        # batched uncond/cond forward
        traj_in = torch.cat([input_noised, input_noised], dim=0)
        cond_in = torch.cat([uncond, cond], dim=0)
        if "state_hist" in kwargs.keys():
            history_cache = kwargs["state_hist"]
            kwargs["state_hist"] = torch.cat([history_cache, history_cache], dim=0)
        k_in = k.to(device)
        if k_in.ndim > 0:
            assert k_in.shape[0] == input_noised.shape[0]
            k_in = torch.cat([k_in, k_in], dim=0)
        model_out = model(traj_in, k_in, cond_in, **kwargs)
        out_uncond, out_cond = model_out.chunk(2, dim=0)
        pred_type = scheduler.config.prediction_type
        guided = out_cond
        if guidance_scale == 1.0:
            guided = out_cond
        elif guidance_scale == 0.0:
            guided = out_uncond
        elif pred_type in ("epsilon", "v_prediction"):
            guided = out_uncond + guidance_scale * (out_cond - out_uncond)
        x_prev = scheduler.step(guided, k, input_noised).prev_sample
        return x_prev


def compute_energy_grad(trajectory: torch.Tensor, goal, k=10, eps=1e-6):
    energy_grad = torch.zeros_like(trajectory)
    diff_pos = trajectory[:, -k:, :3] - goal
    dist_norm = torch.norm(diff_pos, p=2, dim=-1).unsqueeze(-1)
    energy_grad[:, -k:, :3] = diff_pos / (dist_norm + eps)
    return energy_grad


@torch.enable_grad()
def compute_energy_vel(
    trajectory: torch.Tensor, max_vel, min_vel, k=20.0, eps=1e-6, **kwargs
):
    energy_grad = torch.zeros_like(trajectory)
    x_var = trajectory.detach().clone()  # .requires_grad_(True)
    assert "normalizer_state" in kwargs.keys()
    assert "state_shapes" in kwargs.keys()
    normalizer = kwargs["normalizer_state"]
    state_shapes = kwargs["state_shapes"]
    trajectory_norm_dict = split_state_tensor(x_var, state_shapes)
    trajectory_unorm_dict = normalizer.unnormalize(trajectory_norm_dict)
    traj_unorm = torch.cat(
        [trajectory_unorm_dict[key] for key in trajectory_unorm_dict.keys()],
        dim=2,
    ).to(x_var.device)
    traj_vel = traj_unorm[:, :, 3:6]
    max_speed = torch.linalg.norm(torch.tensor(max_vel)).to(x_var.device)
    min_speed = torch.linalg.norm(torch.tensor(min_vel)).to(x_var.device)
    speed = torch.linalg.norm(traj_vel, dim=-1)  # [B,N]

    hi = F.softplus(k * (speed - max_speed)) / k
    lo = F.softplus(k * (min_speed - speed)) / k
    dhi_ds = torch.sigmoid(k * (speed - max_speed))  # [B, N]
    dlo_ds = -torch.sigmoid(k * (min_speed - speed))  # [B, N]
    dJ_ds = 2.0 * hi * dhi_ds + 2.0 * lo * dlo_ds
    ds_dv = traj_vel / (speed[..., None] + eps)  # [B, N, 3]

    dJ_dv = dJ_ds[..., None] * ds_dv  # [B, N, 3]
    energy_grad[:, :, 3:6] = dJ_dv
    return energy_grad


def compute_grad_speed_mag(
    trajectory: torch.Tensor,
    s_max: float,
    s_min: float,
    k=50.0,
    eps=1e-6,
    **kwargs,
):
    with torch.enable_grad():
        x_var = trajectory.detach().requires_grad_(True)

        normalizer = kwargs["normalizer_state"]
        state_shapes = kwargs["state_shapes"]

        traj_norm_dict = split_state_tensor(x_var, state_shapes)
        traj_unorm_dict = normalizer.unnormalize(traj_norm_dict)
        traj_unorm = torch.cat(
            [traj_unorm_dict[key] for key in traj_unorm_dict.keys()], dim=2
        )

        v = traj_unorm[:, :, 3:6]  # [B,N,3]
        speed = torch.sqrt((v * v).sum(dim=-1) + eps)  # [B,N]

        s_max_t = torch.tensor(s_max, device=v.device, dtype=v.dtype)
        s_min_t = torch.tensor(s_min, device=v.device, dtype=v.dtype)

        hi = F.softplus(k * (speed - s_max_t)) / k
        lo = F.softplus(k * (s_min_t - speed)) / k

        J = (hi**2 + lo**2).sum()
        grad_x = torch.autograd.grad(
            J, traj_unorm, retain_graph=False, create_graph=False
        )[0]
        return grad_x


def compute_energy_grad_opt(trajectory: torch.Tensor, goal, k=10, eps=1e-6):
    energy_grad = torch.zeros_like(trajectory)
    pos = trajectory[:, -k:, :3].detach().requires_grad_(True)
    goal_pip = goal.detach().clone().requires_grad_(True)
    diff_pos = pos - goal_pip
    # dist_norm = torch.norm(diff_pos, p=2, dim=-1).unsqueeze(-1)
    with torch.enable_grad():
        dist_norm = diff_pos.pow(2)
        dist_norm.backward()

    energy_grad[:, -k:, :3] = diff_pos.detach()
    return energy_grad


def project_speed_magnitude(v, s_min, s_max, eps=1e-8):
    s = torch.sqrt((v * v).sum(dim=-1, keepdim=True) + eps)  # [B,T,1]
    scale = torch.ones_like(s)
    scale = torch.where(s > s_max, s_max / s, scale)
    if s_min > 0:
        scale = torch.where(s < s_min, s_min / s, scale)
    return v * scale


def project_speed_in_xnorm(
    x_norm, vel_slice=slice(3, 6), s_min=0.0, s_max=300.0, **kwargs
):
    normalizer = kwargs["normalizer_state"]
    state_shapes = kwargs["state_shapes"]
    # unnormalize
    traj_norm_dict = split_state_tensor(x_norm, state_shapes)
    traj_unorm_dict = normalizer.unnormalize(traj_norm_dict)
    traj_unorm = torch.cat([traj_unorm_dict[k] for k in traj_unorm_dict.keys()], dim=2)

    # project in physical space
    v = traj_unorm[:, :, vel_slice]
    traj_unorm[:, :, vel_slice] = project_speed_magnitude(v, s_min, s_max)

    # renormalize back (you need normalizer.normalize)
    traj_unorm_dict2 = split_state_tensor(traj_unorm, state_shapes)
    traj_norm_dict2 = normalizer(traj_unorm_dict2)
    x_norm_proj = torch.cat([traj_norm_dict2[k] for k in traj_norm_dict2.keys()], dim=2)

    return x_norm_proj


class EnergyGuidanceGrad(BaseGuidance):  # EnergyGuidanceGrad class CFGGuidance
    def __init__(self, config):
        super().__init__(config)
        self.eta = 0.0005  # 3e-07  # config["guidance_eta"]

    def __call__(
        self,
        input_noised: torch.Tensor,
        k: torch.Tensor,
        cond: torch.Tensor,
        scheduler: Optional[DDIMScheduler],
        model: Optional[torch.nn.Module],
        **kwargs,
    ) -> torch.Tensor:
        eta = self.eta
        assert "goal" in kwargs.keys()
        # uncond = torch.zeros_like(cond)
        # grad_E = compute_grad_speed_mag(
        #     input_noised,
        #     max_vel=[900.0, 700.0, 592.1],
        #     min_vel=[-304.1, -93.9, -1060.5],
        #     **kwargs,
        # )
        grad_E = compute_grad_speed_mag(
            input_noised,
            s_max=1100.0,
            s_min=1.0,
            **kwargs,
        )
        alpha_bar = scheduler.alphas_cumprod[k.long()]
        sigma_t = (
            torch.sqrt(1.0 - alpha_bar).view(1, 1, 1).to(input_noised.device)
        )  # [1,1,1]
        eps = model(input_noised, k, cond, **kwargs)
        # sigma_t = scheduler.sigmas[t]
        eps_guided = eps  # + sigma_t * eta * grad_E
        x_prev = (
            scheduler.step(eps_guided, k, input_noised).prev_sample
            - sigma_t * eta * grad_E
        )
        x_prev = project_speed_in_xnorm(
            x_prev, vel_slice=slice(3, 6), s_min=0.0, s_max=1100.0, **kwargs
        )
        return x_prev


# ProjectionGuidanceGrad
class ProjectionGuidanceGrad(BaseGuidance):
    def __init__(self, config):
        super().__init__(config)
        self.eta = 0.2  # config["guidance_eta"]

    def __call__(
        self,
        input_noised: torch.Tensor,
        k: torch.Tensor,
        cond: torch.Tensor,
        scheduler: Optional[DDIMScheduler],
        model: Optional[torch.nn.Module],
        **kwargs,
    ) -> torch.Tensor:
        eta = self.eta
        pos_idx = 3
        assert "goal" in kwargs.keys()
        # uncond = torch.zeros_like(cond)
        eps = model(input_noised, k, cond, **kwargs)
        x_prev = scheduler.step(eps, k, input_noised).prev_sample
        x_prev = project_speed_in_xnorm(
            x_prev, vel_slice=slice(3, 6), s_min=0.0, s_max=1100.0, **kwargs
        )
        return x_prev


class CFGGuidanceKSampling(BaseGuidance):
    def __init__(self, config):
        super().__init__(config)
        self.grad_guid = False  # config.get("grad_guid", True)
        self.eta = config["guidance_eta"]
        self.s_max = config.get("s_max", 1644.0)
        self.s_min = config.get("s_min", 1.0)

    def __call__(
        self,
        input_noised: torch.Tensor,
        k: torch.Tensor,
        cond: torch.Tensor,
        scheduler: Optional[DDIMScheduler],
        model: Optional[torch.nn.Module],
        **kwargs,
    ):
        eta = self.eta
        grad_guid = self.grad_guid
        device = input_noised.device
        guidance_scale = self.guidance_scale
        uncond = torch.zeros_like(cond)
        B, L, C = input_noised.shape
        # batched uncond/cond forward
        traj_in = torch.cat([input_noised, input_noised], dim=0)
        cond_in = torch.cat([uncond, cond], dim=0)
        if "state_hist" in kwargs.keys():
            history_cache = kwargs["state_hist"]
            kwargs["state_hist"] = torch.cat([history_cache, history_cache], dim=0)
        k_in = k.to(device)
        # if k_in.shape[0] == input_noised.shape[0]:
        if k_in.ndim > 0:
            assert k_in.shape[0] == input_noised.shape[0]
            k_in = torch.cat([k_in, k_in], dim=0)
        kwargs["dist"] = torch.cat([kwargs["dist"], kwargs["dist"]], dim=0)
        model_out = model(traj_in, k_in, cond_in, **kwargs)
        out_uncond, out_cond = model_out.chunk(2, dim=0)
        pred_type = scheduler.config.prediction_type
        guided = out_cond
        if guidance_scale == 1.0:
            guided = out_cond
        elif guidance_scale == 0.0:
            guided = out_uncond
        elif pred_type in ("epsilon", "v_prediction"):
            guided = out_uncond + guidance_scale * (out_cond - out_uncond)

        if k_in.ndim > 0:
            x_prev = []
            for i in range(B):
                x_prev_i = scheduler.step(
                    guided[i, :, :], k[i], input_noised[i, :, :]
                ).prev_sample
                x_prev.append(x_prev_i)
            x_prev = torch.stack(x_prev, dim=0)
        else:
            x_prev = scheduler.step(guided, k, input_noised).prev_sample
        if grad_guid:
            alpha_bar = scheduler.alphas_cumprod[k.long()]
            sigma_t = torch.sqrt(1.0 - alpha_bar).to(input_noised.device)
            sigma_t = repeat(sigma_t, "B -> B H", H=input_noised.shape[-1]).unsqueeze(1)
            grad_E = compute_grad_speed_mag(
                input_noised,
                s_max=self.s_max,
                s_min=self.s_min,
                **kwargs,
            )
            x_prev = x_prev - sigma_t * eta * grad_E
        return x_prev