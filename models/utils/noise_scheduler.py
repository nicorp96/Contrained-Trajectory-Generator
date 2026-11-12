from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# scheduler arguments read from config["model"]["noise_scheduler"]
REQUIRED = (
    "num_train_timesteps", "beta_start", "beta_end", "beta_schedule",
    "clip_sample", "set_alpha_to_one", "steps_offset", "prediction_type",
)
# "leading" (diffusers default) starts sampling at t = T - T/steps, not at pure noise;
# "trailing" starts at t = T - 1 (Lin et al., 2024).
# sampler: "ddim" (deterministic) or "ddpm" (stochastic ancestral sampling). Both use
# the same noise schedule, so a model trained with either can be sampled with both.
OPTIONAL = {"timestep_spacing": "leading", "sampler": "ddim"}
SAMPLERS = {"ddim": DDIMScheduler, "ddpm": DDPMScheduler}
DDIM_ONLY = ("set_alpha_to_one",)
# Below this alpha_bar an epsilon prediction no longer determines the sample: DDIM's
# x0 = (x - sqrt(1 - a) * eps) / sqrt(a) amplifies model errors by 1 / sqrt(a)
# (squaredcos_cap_v2: a = 2.4e-9 at t = 999, i.e. x20000).
MIN_START_ALPHA_BAR = 1e-4


def build_noise_scheduler(cfg, **overrides):
    """DDIM or DDPM scheduler from a noise_scheduler config; `overrides` replace
    optional entries (inference-time only, e.g. timestep_spacing, sampler)."""
    unknown = set(overrides) - set(OPTIONAL)
    if unknown:
        raise ValueError(f"cannot override {sorted(unknown)}; allowed: {sorted(OPTIONAL)}")
    kwargs = {k: cfg[k] for k in REQUIRED}
    kwargs.update({k: cfg.get(k, default) for k, default in OPTIONAL.items()})
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    sampler = kwargs.pop("sampler")
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; choose from {sorted(SAMPLERS)}")
    if sampler != "ddim":
        kwargs = {k: v for k, v in kwargs.items() if k not in DDIM_ONLY}
    scheduler = SAMPLERS[sampler](**kwargs)
    start_alpha_bar = float(scheduler.alphas_cumprod[-1])
    if (kwargs["timestep_spacing"] == "trailing" and kwargs["prediction_type"] == "epsilon"
            and start_alpha_bar < MIN_START_ALPHA_BAR):
        raise ValueError(
            f"timestep_spacing='trailing' starts sampling at alpha_bar={start_alpha_bar:.1e}, where an "
            "epsilon prediction amplifies errors by 1/sqrt(alpha_bar); use 'leading' or train with "
            "prediction_type='v_prediction'"
        )
    return scheduler
