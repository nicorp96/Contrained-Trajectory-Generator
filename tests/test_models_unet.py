"""ctg/models/unet.py and the denoiser interface shared with ctg/models/dit.py."""

import pytest
import torch

from ctg.models.common import SinusoidalPosEmb, apply_null_goal
from ctg.models.dit import DiTDenoiser
from ctg.models.unet import UNetDenoiser

DIMS = dict(plan_dim=8, goal_dim=7, history_dim=35, history_len=2)


def unet():
    return UNetDenoiser(**DIMS, down_dims=(16, 32, 64), step_embed_dim=16)


def dit():
    return DiTDenoiser(**DIMS, d_model=32, n_heads=2, n_blocks=2, dropout=0.0)


def inputs(B=3, T=16):
    g = torch.Generator().manual_seed(1)
    return (torch.randn(B, T, 8, generator=g), torch.randint(0, 100, (B,), generator=g),
            torch.randn(B, 7, generator=g), torch.randn(B, 2, 35, generator=g))


def perturb_adaln(model):
    """adaLN-Zero starts with zero output; give the conditioning paths non-zero weights."""
    for name, p in model.named_parameters():
        if p.abs().sum() == 0 and "null_goal" not in name:
            torch.nn.init.normal_(p, std=0.1)
    return model


@pytest.fixture(params=[unet, dit], ids=["unet", "dit"])
def model(request):
    torch.manual_seed(0)
    return perturb_adaln(request.param()).eval()


def test_output_shape(model):
    x, t, goal, hist = inputs()
    assert model(x, t, goal, hist).shape == x.shape


def test_scalar_timestep(model):
    x, _, goal, hist = inputs()
    assert model(x, torch.tensor(50), goal, hist).shape == x.shape


def test_goal_and_history_condition_the_output(model):
    x, t, goal, hist = inputs()
    out = model(x, t, goal, hist)
    assert not torch.allclose(out, model(x, t, goal + 1.0, hist))
    assert not torch.allclose(out, model(x, t, goal, hist + 1.0))


def test_dropped_goal_uses_the_null_token(model):
    x, t, goal, hist = inputs()
    drop = torch.tensor([True, False, True])
    a = model(x, t, goal, hist, goal_drop=drop)
    b = model(x, t, goal + 5.0, hist, goal_drop=drop)
    torch.testing.assert_close(a[drop], b[drop])  # goal ignored where dropped
    assert not torch.allclose(a[~drop], b[~drop])


def test_null_goal_is_trained(model):
    model.train()
    x, t, goal, hist = inputs()
    model(x, t, goal, hist, goal_drop=torch.ones(3, dtype=torch.bool)).pow(2).mean().backward()
    assert model.null_goal.grad is not None and model.null_goal.grad.abs().sum() > 0


def test_unet_ignores_the_padding_mask():
    model = unet().eval()
    x, t, goal, hist = inputs()
    torch.testing.assert_close(model(x, t, goal, hist, mask=torch.ones(3, 16, dtype=torch.bool)),
                               model(x, t, goal, hist))


def test_unet_length_must_fit_downsampling():
    x, t, goal, hist = inputs(T=18)
    with pytest.raises(ValueError, match="divisible by 4"):
        unet()(x, t, goal, hist)


def test_unet_history_size_checked():
    x, t, goal, _ = inputs()
    with pytest.raises(ValueError, match="70 features"):
        unet()(x, t, goal, torch.randn(3, 3, 35))


def test_unet_reference_size():
    model = UNetDenoiser(**DIMS)
    assert model.down_dims == [64, 128, 256]
    assert 3e6 < sum(p.numel() for p in model.parameters()) < 6e6  # ManiSkill: "about ~4.5M params"


def test_apply_null_goal():
    goal, null = torch.ones(2, 3), torch.zeros(3)
    assert apply_null_goal(goal, null, None) is goal
    out = apply_null_goal(goal, null, torch.tensor([True, False]))
    assert out[0].sum() == 0 and out[1].sum() == 3


def test_sinusoidal_embedding():
    emb = SinusoidalPosEmb(8)(torch.tensor([0, 5]))
    assert emb.shape == (2, 8)
    torch.testing.assert_close(emb[0], torch.tensor([0.0] * 4 + [1.0] * 4))
