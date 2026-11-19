"""ctg/diffusion/scheduler.py"""

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from ctg.diffusion.scheduler import build_noise_scheduler, predicted_x0, training_target

DDIM = dict(sampler="ddim", num_train_timesteps=1000, beta_start=1e-5, beta_end=0.0085)  # configs/diffusion/ddim.yaml
V = dict(DDIM, prediction_type="v_prediction")


def test_ddim_matches_direct_construction():
    old = DDIMScheduler(num_train_timesteps=1000, beta_start=1e-5, beta_end=0.0085, beta_schedule="squaredcos_cap_v2",
                        clip_sample=False, set_alpha_to_one=True, steps_offset=0, prediction_type="epsilon")
    new = build_noise_scheduler(**DDIM)
    assert new.config.timestep_spacing == "leading"
    assert torch.equal(old.alphas_cumprod, new.alphas_cumprod)
    old.set_timesteps(25)
    new.set_timesteps(25)
    assert torch.equal(old.timesteps, new.timesteps)


def test_trailing_starts_at_pure_noise():
    s = build_noise_scheduler(**V, timestep_spacing="trailing")
    s.set_timesteps(25)
    assert s.timesteps[0] == 999 and s.timesteps[-1] == 39
    lead = build_noise_scheduler(**DDIM)
    lead.set_timesteps(25)
    assert lead.timesteps[0] == 960 and lead.timesteps[-1] == 0


def test_trailing_with_epsilon_at_zero_snr_rejected():
    """Measured: 0/100 successes, actions up to ~800 instead of [-1, 1]."""
    with pytest.raises(ValueError, match="amplifies errors"):
        build_noise_scheduler(**DDIM, timestep_spacing="trailing")


def test_trailing_with_epsilon_allowed_when_signal_remains():
    s = build_noise_scheduler(num_train_timesteps=100, beta_schedule="linear", beta_start=1e-4, beta_end=0.02,
                              timestep_spacing="trailing")
    assert float(s.alphas_cumprod[-1]) >= 1e-4


def test_epsilon_error_amplification_at_first_step():
    """Why the guard exists: a 1 % noise error at the first trailing step vs. leading."""
    ab = build_noise_scheduler(**DDIM).alphas_cumprod.double()
    amp = lambda t: 0.01 * (1 - ab[t]).sqrt() / ab[t].sqrt()  # error in the x0 estimate
    assert amp(999) > 100 and amp(960) < 0.2


def test_unknown_sampler_rejected():
    with pytest.raises(ValueError, match="unknown sampler"):
        build_noise_scheduler(sampler="euler")


def test_ddpm_shares_the_noise_schedule():
    ddim, ddpm = build_noise_scheduler(**DDIM), build_noise_scheduler(**{**DDIM, "sampler": "ddpm"})
    assert isinstance(ddpm, DDPMScheduler) and isinstance(ddim, DDIMScheduler)
    assert torch.equal(ddim.alphas_cumprod, ddpm.alphas_cumprod)


def oracle_run(s, steps, **step_kw):
    s.set_timesteps(steps)
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(2, 16, 5, generator=gen, dtype=torch.float64)
    x = torch.randn(2, 16, 5, generator=gen, dtype=torch.float64)
    ab = s.alphas_cumprod.double()
    for k in s.timesteps:
        eps = (x - ab[k].sqrt() * x0) / (1 - ab[k]).sqrt()
        x = s.step(eps, k, x, **({"generator": gen} if step_kw else {})).prev_sample
    return x, x0


@pytest.mark.parametrize("steps", [25, 100])
def test_ddim_oracle_noise_prediction_recovers_sample(steps):
    x, x0 = oracle_run(build_noise_scheduler(**DDIM), steps)
    torch.testing.assert_close(x, x0)


@pytest.mark.parametrize("steps", [25, 100])
def test_ddpm_oracle_noise_prediction_recovers_sample(steps):
    """DDPM adds noise at every step but none at the last one."""
    x, x0 = oracle_run(build_noise_scheduler(num_train_timesteps=100), steps, generator=True)
    torch.testing.assert_close(x, x0)


@pytest.mark.parametrize("kind", ["epsilon", "v_prediction", "sample"])
def test_training_target_and_predicted_x0_are_consistent(kind):
    s = build_noise_scheduler(num_train_timesteps=100, prediction_type=kind)
    x0, noise = torch.randn(4, 8, 3), torch.randn(4, 8, 3)
    t = torch.tensor([0, 10, 50, 99])
    x_t = s.add_noise(x0, noise, t)
    torch.testing.assert_close(predicted_x0(s, training_target(s, x0, noise, t), x_t, t), x0, atol=2e-4, rtol=1e-4)


def test_unsupported_prediction_type():
    s = build_noise_scheduler()
    s.register_to_config(prediction_type="flow")
    x = torch.zeros(1, 2, 1)
    with pytest.raises(ValueError, match="unsupported"):
        training_target(s, x, x, torch.tensor([0]))
    with pytest.raises(ValueError, match="unsupported"):
        predicted_x0(s, x, x, torch.tensor([0]))
