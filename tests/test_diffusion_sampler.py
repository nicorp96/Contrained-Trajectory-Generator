"""ctg/diffusion/sampler.py"""

import pytest
import torch

from ctg.data.layout import make_layout
from ctg.data.normalization import LayoutNormalizer, PlanCodec
from ctg.diffusion.inpainting import known_mask
from ctg.diffusion.sampler import DiffusionSampler
from ctg.diffusion.scheduler import build_noise_scheduler, predicted_x0
from ctg.physics.constraints import JointConstraints

LAYOUT = make_layout({"qpos": {"size": 9}, "qvel": {"size": 9}})
CODEC = PlanCodec(LAYOUT, LayoutNormalizer(LAYOUT))


class Oracle(torch.nn.Module):
    """Predicts the exact noise toward a fixed target plan; records the goals it saw."""

    def __init__(self, target, scheduler):
        super().__init__()
        self.target, self.ab = target, scheduler.alphas_cumprod
        self.calls = []

    def forward(self, x, k, goal, history, goal_drop=None):
        self.calls.append((x.shape[0], goal_drop))
        ab = self.ab[k.long()].view(-1, 1, 1) if k.ndim else self.ab[k.long()]
        return (x - ab.sqrt() * self.target) / (1 - ab).sqrt()


def setup(sampler="ddim", **kw):
    s = build_noise_scheduler(sampler=sampler, num_train_timesteps=100)
    target = torch.full((8, 18), 0.5)  # broadcasts over the (doubled) batch
    return DiffusionSampler(s, num_inference_steps=10, **kw), Oracle(target, s), target


def run(sampler, model, **kw):
    return sampler.sample(model, (2, 8, 18), torch.zeros(2, 3), torch.zeros(2, 4), CODEC,
                          generator=torch.Generator().manual_seed(0), **kw)


def test_oracle_model_gives_its_target():
    sampler, model, target = setup()
    torch.testing.assert_close(run(sampler, model), target.expand(2, 8, 18), atol=1e-4, rtol=0)
    assert all(drop is None for _, drop in model.calls)


def test_classifier_free_guidance_doubles_the_batch():
    sampler, model, target = setup(guidance_scale=2.0)
    torch.testing.assert_close(run(sampler, model), target.expand(2, 8, 18), atol=1e-4, rtol=0)  # cond == uncond
    batch, drop = model.calls[0]
    assert batch == 4 and drop.tolist() == [True, True, False, False]


def test_inpainting_keeps_known_entries():
    sampler, model, _ = setup(sampler="ddpm")
    known = known_mask((2, 8, 18), start_dims=range(9))
    values = torch.full((2, 8, 18), -3.0)
    out = run(sampler, model, known=known, known_values=values)
    assert torch.all(out[:, 0, :9] == -3.0)


def test_physics_step_moves_the_clean_estimate_toward_feasibility():
    limits = JointConstraints(q_upper=[0.1] * 9, weights={"kinematic": 0.0})
    sampler, model, _ = setup(constraints=limits, guidance_eta=1.0, guidance_grad_clip=10.0)
    x, k = torch.randn(2, 8, 18), torch.tensor(30)
    out = model(x, k, None, None)
    guided = sampler.physics_step(out, x, k, CODEC)
    excess = lambda o: limits.energy(CODEC.decode(predicted_x0(sampler.scheduler, o, x, k.expand(2))))["total"]
    assert torch.all(excess(guided) < excess(out))


def test_guidance_only_in_late_steps():
    sampler, _, _ = setup(constraints=JointConstraints(), guidance_eta=1.0, guidance_start=0.3)
    assert sampler._physics_active(torch.tensor(20)) and not sampler._physics_active(torch.tensor(50))
    assert not setup(constraints=JointConstraints())[0]._physics_active(torch.tensor(0))  # eta = 0
    assert not setup(guidance_eta=1.0)[0]._physics_active(torch.tensor(0))  # no constraints


@pytest.mark.parametrize("kind", ["epsilon", "v_prediction", "sample"])
def test_physics_step_without_gradient_keeps_the_output(kind):
    s = build_noise_scheduler(num_train_timesteps=100, prediction_type=kind)
    sampler = DiffusionSampler(s, constraints=JointConstraints(q_lower=[-9] * 9, q_upper=[9] * 9,
                                                               weights={"kinematic": 0.0, "velocity_limits": 0.0,
                                                                        "acceleration_limits": 0.0}),
                               guidance_eta=1.0)
    x, out = torch.randn(2, 8, 18), torch.randn(2, 8, 18) * 0.1
    known = known_mask((2, 8, 18), start_dims=[0])
    torch.testing.assert_close(sampler.physics_step(out, x, torch.tensor(40), CODEC, known), out, atol=1e-4, rtol=1e-4)


def test_final_projection_enforces_limits():
    sampler, model, _ = setup(constraints=JointConstraints(q_upper=[0.1] * 9), project_final=True)
    out = CODEC.decode(run(sampler, model))
    assert out["qpos"].max() <= 0.1 + 1e-6
