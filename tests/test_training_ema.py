"""ctg/training/ema.py"""

import pytest
import torch

from ctg.training.ema import EMA


def test_decay_warm_up_and_clipping():
    ema = EMA(torch.nn.Linear(1, 1), update_after_step=2, max_decay=0.9)
    assert ema.decay(0) == 0.0 and ema.decay(3) == 0.0
    assert ema.decay(4) == pytest.approx(1 - 2 ** -0.75)
    assert ema.decay(10 ** 6) == 0.9
    assert EMA(torch.nn.Linear(1, 1), min_decay=0.5).decay(2) == 0.5


def test_step_averages_float_and_copies_integer_buffers():
    model = torch.nn.BatchNorm1d(2)
    ema = EMA(torch.nn.BatchNorm1d(2), inv_gamma=1e9)  # decay ~ 0 at first
    assert not ema.model.training and not any(p.requires_grad for p in ema.model.parameters())
    with torch.no_grad():
        model.weight.fill_(3.0)
    model.num_batches_tracked += 5
    ema.step(model)
    assert torch.equal(ema.model.weight, model.weight)
    assert ema.model.num_batches_tracked == 5 and ema.optimization_step == 1

    ema.decay = lambda step: 0.5
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.step(model)
    torch.testing.assert_close(ema.model.weight, torch.full((2,), 2.0))
