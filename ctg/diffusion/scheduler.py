"""Noise schedulers (diffusers) shared by training and sampling."""

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

SAMPLERS = {"ddim": DDIMScheduler, "ddpm": DDPMScheduler}
# Below this alpha_bar an epsilon prediction no longer determines the sample: the
# clean-sample estimate (x - sqrt(1 - a) * eps) / sqrt(a) amplifies model errors by
# 1 / sqrt(a) (squaredcos_cap_v2: a = 2.4e-9 at t = 999, i.e. x20000).
MIN_START_ALPHA_BAR = 1e-4


def build_noise_scheduler(
    sampler: str = "ddpm",
    num_train_timesteps: int = 100,
    beta_schedule: str = "squaredcos_cap_v2",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    prediction_type: str = "epsilon",
    clip_sample: bool = False,
    timestep_spacing: str = "leading",
    set_alpha_to_one: bool = True,
    steps_offset: int = 0,
):
    """DDIM (deterministic) or DDPM (stochastic) scheduler. Both share the noise
    schedule, so a model trained with either can be sampled with the other.
    timestep_spacing: "leading" starts sampling at T - T/steps, "trailing" at T - 1."""
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; choose from {sorted(SAMPLERS)}")
    kwargs = dict(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule, beta_start=beta_start,
                  beta_end=beta_end, prediction_type=prediction_type, clip_sample=clip_sample,
                  timestep_spacing=timestep_spacing, steps_offset=steps_offset)
    if sampler == "ddim":
        kwargs["set_alpha_to_one"] = set_alpha_to_one
    scheduler = SAMPLERS[sampler](**kwargs)
    start_alpha_bar = float(scheduler.alphas_cumprod[-1])
    if timestep_spacing == "trailing" and prediction_type == "epsilon" and start_alpha_bar < MIN_START_ALPHA_BAR:
        raise ValueError(
            f"timestep_spacing='trailing' starts sampling at alpha_bar={start_alpha_bar:.1e}, where an epsilon "
            "prediction amplifies errors by 1/sqrt(alpha_bar); use 'leading' or prediction_type='v_prediction'"
        )
    return scheduler


def training_target(scheduler, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """What the denoiser learns to predict for the configured prediction type."""
    kind = scheduler.config.prediction_type
    if kind == "epsilon":
        return noise
    if kind == "sample":
        return x0
    if kind == "v_prediction":
        return scheduler.get_velocity(x0, noise, t)
    raise ValueError(f"unsupported prediction type {kind!r}")


def predicted_x0(scheduler, pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Clean-sample estimate from the model output (used by physics loss and guidance)."""
    a_bar = scheduler.alphas_cumprod.to(x_t.device)[t.long()].float().view(-1, *([1] * (x_t.ndim - 1)))
    kind = scheduler.config.prediction_type
    if kind == "epsilon":
        return (x_t - (1 - a_bar).sqrt() * pred) / a_bar.sqrt()
    if kind == "v_prediction":
        return a_bar.sqrt() * x_t - (1 - a_bar).sqrt() * pred
    if kind == "sample":
        return pred
    raise ValueError(f"unsupported prediction type {kind!r}")
