from typing import Optional

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from common.utils import split_trajectory
from models.utils.robot_constraints import RobotConstraints


def to_physical(x_norm, normalizer_state, state_shapes):
    """Normalized [B, T, D] -> dict of unnormalized [B, T, d_k] (differentiable)."""
    return normalizer_state.unnormalize(split_trajectory(x_norm, state_shapes))


def to_normalized(x_phys: dict, normalizer_state):
    x_norm = normalizer_state(x_phys)
    return torch.cat([x_norm[k] for k in x_norm.keys()], dim=-1)


class BaseGuidance:
    def __init__(self, config):
        self.config = config
        self.guidance_scale = float(config.get("guidance_scale", 1.0))

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

    def finalize(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Hook applied once after the last denoising step."""
        return x


class CFGGuidance(BaseGuidance):
    """
    One DDIM step with
      1. classifier-free guidance on the goal (learned null token, see `cond_drop`),
      2. optional constraint guidance: the gradient of the constraint energy,
         evaluated on the denoised estimate x0_hat in physical units, shifts the
         noise prediction (eps <- eps + eta * sqrt(1 - a_bar) * grad),
      3. optional hard projection of the final sample onto joint/velocity/action limits.

    config:
      guidance_scale:        CFG weight (1.0 = conditional only)
      constraint_eta:        constraint guidance step size (0 disables it)
      constraint_grad_clip:  max per-sample L2 norm of the constraint gradient
      constraint_start:      apply constraint guidance only when k <= start * T_train
      project_final:         clamp the final sample with RobotConstraints.project
    """

    def __init__(self, config):
        super().__init__(config)
        self.eta = float(config.get("constraint_eta", 0.0))
        self.grad_clip = float(config.get("constraint_grad_clip", 1.0))
        self.start = float(config.get("constraint_start", 1.0))
        self.project_final = bool(config.get("project_final", False))

    def model_output(self, x_t, k, cond, scheduler, model, state_hist=None):
        model_kwargs = {}
        if self.guidance_scale == 1.0:
            if state_hist is not None:
                model_kwargs["state_hist"] = state_hist
            return model(x_t, k, cond, **model_kwargs)

        B = x_t.shape[0]
        traj_in = torch.cat([x_t, x_t], dim=0)
        cond_in = torch.cat([cond, cond], dim=0)
        cond_drop = torch.cat(
            [
                torch.ones(B, dtype=torch.bool, device=x_t.device),
                torch.zeros(B, dtype=torch.bool, device=x_t.device),
            ]
        )
        if state_hist is not None:
            model_kwargs["state_hist"] = torch.cat([state_hist, state_hist], dim=0)
        k_in = k if k.ndim == 0 else torch.cat([k, k], dim=0)
        out = model(traj_in, k_in, cond_in, cond_drop=cond_drop, **model_kwargs)
        out_uncond, out_cond = out.chunk(2, dim=0)
        # same linear rule for eps, v and x0 predictions
        return out_uncond + self.guidance_scale * (out_cond - out_uncond)

    def constraint_step(
        self,
        out,
        x_t,
        k,
        scheduler,
        constraints: RobotConstraints,
        normalizer_state,
        state_shapes,
        mask_known=None,
    ):
        pred_type = scheduler.config.prediction_type
        a_bar = scheduler.alphas_cumprod.to(x_t.device)[k.long()].float()
        a_bar = a_bar.view(-1, *([1] * (x_t.ndim - 1)))
        sqrt_ab, sqrt_1mab = a_bar.sqrt(), (1.0 - a_bar).sqrt()
        x_t32, out32 = x_t.float(), out.float()

        if pred_type == "epsilon":
            eps = out32
            x0 = (x_t32 - sqrt_1mab * eps) / sqrt_ab
        elif pred_type == "v_prediction":
            x0 = sqrt_ab * x_t32 - sqrt_1mab * out32
            eps = sqrt_1mab * x_t32 + sqrt_ab * out32
        elif pred_type == "sample":
            x0 = out32
            eps = (x_t32 - sqrt_ab * x0) / sqrt_1mab
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        with torch.enable_grad():
            x0_var = x0.detach().requires_grad_(True)
            phys = to_physical(x0_var, normalizer_state, state_shapes)
            energy = constraints.energy(phys)["total"].sum()
            (grad,) = torch.autograd.grad(energy, x0_var)

        if mask_known is not None:
            grad = grad * (~mask_known.bool()).to(grad.dtype)
        norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
        scale = (self.grad_clip / norm).clamp(max=1.0).view(-1, *([1] * (grad.ndim - 1)))
        grad = grad * scale

        eps = eps + self.eta * sqrt_1mab * grad
        x0 = (x_t32 - sqrt_1mab * eps) / sqrt_ab
        if pred_type == "epsilon":
            new_out = eps
        elif pred_type == "v_prediction":
            new_out = sqrt_ab * eps - sqrt_1mab * x0
        else:
            new_out = x0
        return new_out.to(out.dtype)

    def __call__(
        self,
        input_noised: torch.Tensor,
        k: torch.Tensor,
        cond: torch.Tensor,
        scheduler: Optional[DDIMScheduler],
        model: Optional[torch.nn.Module],
        **kwargs,
    ):
        k = k.to(input_noised.device)
        out = self.model_output(
            input_noised, k, cond, scheduler, model, kwargs.get("state_hist")
        )
        constraints = kwargs.get("constraints")
        n_train = scheduler.config.num_train_timesteps
        if (
            constraints is not None
            and self.eta > 0.0
            and float(k.max()) <= self.start * n_train
        ):
            out = self.constraint_step(
                out,
                input_noised,
                k,
                scheduler,
                constraints,
                kwargs["normalizer_state"],
                kwargs["state_shapes"],
                kwargs.get("mask_known"),
            )
        return scheduler.step(out, k, input_noised).prev_sample

    @torch.no_grad()
    def finalize(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        constraints = kwargs.get("constraints")
        if not self.project_final or constraints is None:
            return x
        normalizer_state = kwargs["normalizer_state"]
        phys = to_physical(x, normalizer_state, kwargs["state_shapes"])
        return to_normalized(constraints.project(phys), normalizer_state).to(x.dtype)


# Kept so existing configs/checkpoints that name this class keep working
class CFGGuidanceKSampling(CFGGuidance):
    pass
