"""models/utils/noise_scheduler.py and the scheduler overrides of the policy."""

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from models.policies.base_policy import Policy
from models.utils.noise_scheduler import build_noise_scheduler

CFG = {  # as in configs/diffusion_traj_pad_dp_cam.yaml
    "num_train_timesteps": 1000, "beta_start": 1e-5, "beta_end": 0.0085,
    "beta_schedule": "squaredcos_cap_v2", "clip_sample": False, "set_alpha_to_one": True,
    "steps_offset": 0, "prediction_type": "epsilon",
}


def test_default_matches_previous_construction():
    old = DDIMScheduler(**CFG)  # what policy and trainer built before
    new = build_noise_scheduler(CFG)
    assert new.config.timestep_spacing == "leading"
    assert torch.equal(old.alphas_cumprod, new.alphas_cumprod)
    old.set_timesteps(25)
    new.set_timesteps(25)
    assert torch.equal(old.timesteps, new.timesteps)


V_CFG = {**CFG, "prediction_type": "v_prediction"}


def test_trailing_starts_at_pure_noise():
    s = build_noise_scheduler(V_CFG, timestep_spacing="trailing")
    s.set_timesteps(25)
    assert s.timesteps[0] == 999 and s.timesteps[-1] == 39
    lead = build_noise_scheduler(CFG)
    lead.set_timesteps(25)
    assert lead.timesteps[0] == 960 and lead.timesteps[-1] == 0


def test_config_value_and_override_precedence():
    assert build_noise_scheduler({**V_CFG, "timestep_spacing": "trailing"}).config.timestep_spacing == "trailing"
    s = build_noise_scheduler({**V_CFG, "timestep_spacing": "trailing"}, timestep_spacing="leading")
    assert s.config.timestep_spacing == "leading"
    # None means "not set on the command line": keep the config value
    s = build_noise_scheduler({**V_CFG, "timestep_spacing": "trailing"}, timestep_spacing=None)
    assert s.config.timestep_spacing == "trailing"


def test_trailing_with_epsilon_at_zero_snr_rejected():
    """Measured: 0/100 successes, actions up to ~800 instead of [-1, 1]."""
    with pytest.raises(ValueError, match="amplifies errors"):
        build_noise_scheduler(CFG, timestep_spacing="trailing")
    with pytest.raises(ValueError, match="amplifies errors"):
        build_noise_scheduler({**CFG, "timestep_spacing": "trailing"})


def test_trailing_with_epsilon_allowed_when_signal_remains():
    linear = {**CFG, "beta_schedule": "linear", "beta_start": 1e-4, "beta_end": 0.02}  # alpha_bar(T) = 4e-5
    s = build_noise_scheduler({**linear, "num_train_timesteps": 100}, timestep_spacing="trailing")
    assert float(s.alphas_cumprod[-1]) >= 1e-4


def test_epsilon_error_amplification_at_first_step():
    """Why the guard exists: a 1 % noise error at the first trailing step vs. leading."""
    s = build_noise_scheduler(CFG)
    ab = s.alphas_cumprod.double()
    err = 0.01
    amp = lambda t: err * (1 - ab[t]).sqrt() / ab[t].sqrt()  # error in the x0 estimate
    assert amp(999) > 100 and amp(960) < 0.2


def test_unknown_override_rejected():
    with pytest.raises(ValueError, match="beta_schedule"):
        build_noise_scheduler(CFG, beta_schedule="linear")


def test_missing_required_entry():
    with pytest.raises(KeyError):
        build_noise_scheduler({k: v for k, v in CFG.items() if k != "prediction_type"})


@pytest.mark.parametrize("spacing, steps", [("leading", 25), ("leading", 100)])
def test_oracle_noise_prediction_recovers_sample(spacing, steps):
    """With the exact noise as model output, deterministic DDIM must end at x0."""
    s = build_noise_scheduler(CFG, timestep_spacing=spacing)
    s.set_timesteps(steps)
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(2, 64, 26, generator=gen, dtype=torch.float64)
    x = torch.randn(2, 64, 26, generator=gen, dtype=torch.float64)
    ab = s.alphas_cumprod.double()
    for k in s.timesteps:
        eps = (x - ab[k].sqrt() * x0) / (1 - ab[k]).sqrt()
        x = s.step(eps, k, x).prev_sample
    torch.testing.assert_close(x, x0)


def test_ddpm_shares_the_training_noise_schedule():
    ddim, ddpm = build_noise_scheduler(CFG), build_noise_scheduler(CFG, sampler="ddpm")
    assert isinstance(ddpm, DDPMScheduler) and isinstance(ddim, DDIMScheduler)
    assert torch.equal(ddim.alphas_cumprod, ddpm.alphas_cumprod)
    assert build_noise_scheduler({**CFG, "sampler": "ddpm"}).config.prediction_type == "epsilon"


def test_unknown_sampler_rejected():
    with pytest.raises(ValueError, match="unknown sampler"):
        build_noise_scheduler(CFG, sampler="euler")


@pytest.mark.parametrize("steps", [25, 1000])
def test_ddpm_oracle_noise_prediction_recovers_sample(steps):
    """DDPM adds noise at every step but none at the last one: with the exact noise
    as model output the sample must still end at x0."""
    s = build_noise_scheduler(CFG, sampler="ddpm")
    s.set_timesteps(steps)
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(2, 64, 26, generator=gen, dtype=torch.float64)
    x = torch.randn(2, 64, 26, generator=gen, dtype=torch.float64)
    ab = s.alphas_cumprod.double()
    for k in s.timesteps:
        eps = (x - ab[k].sqrt() * x0) / (1 - ab[k]).sqrt()
        x = s.step(eps, k, x, generator=gen).prev_sample
    torch.testing.assert_close(x, x0)


def test_ddpm_is_stochastic():
    s = build_noise_scheduler(CFG, sampler="ddpm")
    s.set_timesteps(25)
    x, eps = torch.zeros(1, 4, 3), torch.zeros(1, 4, 3)
    a = s.step(eps, s.timesteps[0], x, generator=torch.Generator().manual_seed(0)).prev_sample
    b = s.step(eps, s.timesteps[0], x, generator=torch.Generator().manual_seed(1)).prev_sample
    assert not torch.equal(a, b)


@pytest.fixture
def light_policy(monkeypatch):
    """Policy without checkpoint I/O: only config handling runs."""
    config = {
        "model": {"noise_scheduler": dict(CFG), "num_inference_steps": 25,
                  "guidance": {"target": "models.utils.guidance_robot.CFGGuidance", "guidance_scale": 1.0}},
        "dataset": {"sequence_len": 64},
    }

    def load(self, path):
        self.config = config

    monkeypatch.setattr(Policy, "__load_ckp__", load)
    monkeypatch.setattr(Policy, "setup_model", lambda self: None)
    monkeypatch.setattr(Policy, "setup_normalizer", lambda self: None)
    return Policy


def test_policy_defaults_from_config(light_policy):
    p = light_policy("ckpt.pth", "cpu")
    assert p.num_inference_steps == 25
    assert p.noise_scheduler.config.timestep_spacing == "leading"


def test_policy_overrides(light_policy):
    p = light_policy("ckpt.pth", "cpu", num_inference_steps=100, scheduler={"timestep_spacing": "linspace"})
    assert p.num_inference_steps == 100
    assert p.noise_scheduler.config.timestep_spacing == "linspace"


def test_policy_ddpm_override(light_policy):
    p = light_policy("ckpt.pth", "cpu", scheduler={"sampler": "ddpm", "timestep_spacing": None})
    assert isinstance(p.noise_scheduler, DDPMScheduler)


def test_policy_unset_cli_values_keep_config(light_policy):
    p = light_policy("ckpt.pth", "cpu", num_inference_steps=None, scheduler={"timestep_spacing": None})
    assert p.num_inference_steps == 25
    assert p.noise_scheduler.config.timestep_spacing == "leading"
