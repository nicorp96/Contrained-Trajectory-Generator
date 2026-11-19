"""
Sampling plans from a trained denoiser, used by validation and by the policy.

Each denoising step can combine
  1. classifier-free guidance on the goal (the denoiser's learned null goal),
  2. physics guidance: the gradient of the constraint energy of the denoised
     estimate (in physical units) shifts the prediction toward feasible plans,
  3. inpainting of known plan entries;
and the final plan can be projected onto the joint / velocity / action limits.
"""

from typing import Optional

import torch

from ctg.data.normalization import PlanCodec
from ctg.diffusion.inpainting import apply_known
from ctg.diffusion.scheduler import predicted_x0


class DiffusionSampler:
    def __init__(self, scheduler, num_inference_steps: int = 100, guidance_scale: float = 1.0,
                 constraints=None, guidance_eta: float = 0.0, guidance_grad_clip: float = 1.0,
                 guidance_start: float = 1.0, project_final: bool = False):
        """guidance_eta: physics guidance step (0 = off); guidance_start: apply it only for
        diffusion steps k <= guidance_start * num_train_timesteps (late, low-noise steps)."""
        self.scheduler = scheduler
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.constraints = constraints
        self.guidance_eta = float(guidance_eta)
        self.guidance_grad_clip = float(guidance_grad_clip)
        self.guidance_start = float(guidance_start)
        self.project_final = bool(project_final)

    # ------------------------------------------------------------------ pieces
    def model_output(self, model, x, k, goal, history):
        """Denoiser output with classifier-free guidance on the goal."""
        if self.guidance_scale == 1.0:
            return model(x, k, goal, history)
        B = x.shape[0]
        drop = torch.cat([torch.ones(B, dtype=torch.bool, device=x.device), torch.zeros(B, dtype=torch.bool, device=x.device)])
        out = model(torch.cat([x, x]), k if k.ndim == 0 else torch.cat([k, k]), torch.cat([goal, goal]),
                    torch.cat([history, history]), goal_drop=drop)
        uncond, cond = out.chunk(2)
        return uncond + self.guidance_scale * (cond - uncond)

    def physics_step(self, out, x, k, codec: PlanCodec, known=None, time=None):
        """Shift the model output so that its clean estimate moves down the constraint energy:
        eps <- eps + eta * sqrt(1 - a_bar) * clip(grad_x0 E)."""
        scheduler = self.scheduler
        a_bar = scheduler.alphas_cumprod.to(x.device)[k.long()].float().view(-1, *([1] * (x.ndim - 1)))
        x0 = predicted_x0(scheduler, out.float(), x.float(), k.expand(x.shape[0]) if k.ndim == 0 else k)
        with torch.enable_grad():
            x0_var = x0.detach().requires_grad_(True)
            energy = self.constraints.energy(codec.decode(x0_var), time=time)["total"].sum()
            (grad,) = torch.autograd.grad(energy, x0_var)
        if known is not None:
            grad = grad * (~known).to(grad.dtype)
        norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
        grad = grad * (self.guidance_grad_clip / norm).clamp(max=1.0).view(-1, *([1] * (grad.ndim - 1)))
        eps = (x.float() - a_bar.sqrt() * x0) / (1 - a_bar).sqrt() + self.guidance_eta * (1 - a_bar).sqrt() * grad
        x0 = (x.float() - (1 - a_bar).sqrt() * eps) / a_bar.sqrt()
        kind = scheduler.config.prediction_type
        new = eps if kind == "epsilon" else (a_bar.sqrt() * eps - (1 - a_bar).sqrt() * x0 if kind == "v_prediction" else x0)
        return new.to(out.dtype)

    def _physics_active(self, k) -> bool:
        return (self.constraints is not None and self.guidance_eta > 0.0
                and float(k.max()) <= self.guidance_start * self.scheduler.config.num_train_timesteps)

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()  # not inference_mode: physics guidance needs autograd
    def sample(self, model, shape, goal, history, codec: PlanCodec, known: Optional[torch.Tensor] = None,
               known_values: Optional[torch.Tensor] = None, time: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Normalized plan [B, T, D]. known / known_values: inpainting mask and values (normalized)."""
        scheduler = self.scheduler
        scheduler.set_timesteps(self.num_inference_steps, device=goal.device)
        x = torch.randn(shape, device=goal.device, generator=generator)
        x = apply_known(x, known, known_values)
        for k in scheduler.timesteps:
            out = self.model_output(model, x, k, goal, history)
            if self._physics_active(k):
                out = self.physics_step(out, x, k, codec, known, time)
            x = apply_known(scheduler.step(out, k, x, generator=generator).prev_sample, known, known_values)
        if self.project_final and self.constraints is not None:
            x = apply_known(codec.encode(self.constraints.project(codec.decode(x))).to(x.dtype), known, known_values)
        return x
