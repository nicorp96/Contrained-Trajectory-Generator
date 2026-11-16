"""models/diffusion_unet1d.py: ConditionalUnet1D behind the DiTTrjCC interface."""

import pytest
import torch

from models.diffusion_unet1d import UNetTrjCC

CFG = {"input_dim": 8, "cond_dim": 7, "history_shape": 35, "history_len": 2,
       "unet_dims": [16, 32, 64], "diffusion_step_embed_dim": 16}


@pytest.fixture
def model():
    torch.manual_seed(0)
    return UNetTrjCC(CFG).eval()


def inputs(B=3, T=16):
    g = torch.Generator().manual_seed(1)
    return (torch.randn(B, T, 8, generator=g), torch.randint(0, 1000, (B,), generator=g),
            torch.randn(B, 7, generator=g), torch.randn(B, 2, 35, generator=g))


def test_output_shape(model):
    x, t, cond, hist = inputs()
    assert model(x, t, cond, state_hist=hist).shape == x.shape


@pytest.mark.parametrize("t", [torch.tensor(500), 500])
def test_scalar_timestep(model, t):
    x, _, cond, hist = inputs()
    assert model(x, t, cond, state_hist=hist).shape == x.shape


def test_goal_and_history_condition_the_output(model):
    x, t, cond, hist = inputs()
    out = model(x, t, cond, state_hist=hist)
    assert not torch.allclose(out, model(x, t, cond + 1.0, state_hist=hist))
    assert not torch.allclose(out, model(x, t, cond, state_hist=hist + 1.0))


def test_dropped_goal_uses_the_null_token(model):
    x, t, cond, hist = inputs()
    drop = torch.tensor([True, False, True])
    a = model(x, t, cond, cond_drop=drop, state_hist=hist)
    b = model(x, t, cond + 5.0, cond_drop=drop, state_hist=hist)
    torch.testing.assert_close(a[drop], b[drop])  # goal ignored where dropped
    assert not torch.allclose(a[~drop], b[~drop])


def test_null_token_is_trained():
    model = UNetTrjCC(CFG)
    x, t, cond, hist = inputs()
    model(x, t, cond, cond_drop=torch.ones(3, dtype=torch.bool), state_hist=hist).pow(2).mean().backward()
    assert model.null_cond.grad is not None and model.null_cond.grad.abs().sum() > 0


def test_padding_mask_is_accepted_and_ignored(model):
    x, t, cond, hist = inputs()
    mask = torch.zeros(3, 16, dtype=torch.bool)
    torch.testing.assert_close(model(x, t, cond, mask=mask, state_hist=hist), model(x, t, cond, state_hist=hist))


def test_sequence_length_must_fit_downsampling(model):
    x, t, cond, hist = inputs(T=18)
    with pytest.raises(ValueError, match="divisible by 4"):
        model(x, t, cond, state_hist=hist)


@pytest.mark.parametrize("hist", [None, torch.randn(3, 3, 35)])
def test_history_size_checked(model, hist):
    x, t, cond, _ = inputs()
    with pytest.raises(ValueError, match="70 features"):
        model(x, t, cond, state_hist=hist)


def test_reference_defaults():
    model = UNetTrjCC({k: v for k, v in CFG.items() if k not in ("unet_dims", "diffusion_step_embed_dim")})
    assert model.down_dims == [64, 128, 256]
    assert 3e6 < sum(p.numel() for p in model.parameters()) < 6e6  # ManiSkill: "about ~4.5M params"
