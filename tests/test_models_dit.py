"""ctg/models/dit.py (the shared denoiser interface is tested in test_models_unet.py)."""

import torch

from ctg.models.dit import DiTDenoiser, GoalEmbedder, HistoryEncoder, modulate


def test_adaln_zero_initial_output_is_zero():
    model = DiTDenoiser(plan_dim=4, goal_dim=3, history_dim=5, d_model=16, n_heads=2, n_blocks=1)
    out = model(torch.randn(2, 8, 4), torch.tensor([1, 2]), torch.randn(2, 3), torch.randn(2, 2, 5))
    assert out.shape == (2, 8, 4) and torch.all(out == 0)


def test_padded_steps_do_not_influence_real_ones():
    torch.manual_seed(0)
    model = DiTDenoiser(plan_dim=4, goal_dim=3, history_dim=5, d_model=16, n_heads=2, n_blocks=2, dropout=0.0)
    for p in model.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)
    model.eval()
    x, t, goal, hist = torch.randn(1, 8, 4), torch.tensor([3]), torch.randn(1, 3), torch.randn(1, 2, 5)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, 5:] = True
    changed = x.clone()
    changed[:, 5:] += 10.0
    a, b = model(x, t, goal, hist, mask=mask), model(changed, t, goal, hist, mask=mask)
    torch.testing.assert_close(a[:, :5], b[:, :5])
    assert not torch.allclose(model(x, t, goal, hist)[:, :5], model(changed, t, goal, hist)[:, :5])


def test_components():
    assert GoalEmbedder(3, 16)(torch.randn(2, 3)).shape == (2, 16)
    assert HistoryEncoder(5, 16, dropout=0.0)(torch.randn(2, 4, 5)).shape == (2, 16)
    x = torch.ones(2, 3, 4)
    torch.testing.assert_close(modulate(x, torch.ones(2, 4), torch.ones(2, 4)), torch.full((2, 3, 4), 3.0))
