"""ctg/diffusion/inpainting.py"""

import torch

from ctg.diffusion.inpainting import apply_known, known_mask


def test_no_known_entries():
    assert known_mask((2, 4, 3)) is None
    x = torch.randn(2, 4, 3)
    assert apply_known(x, None, torch.zeros_like(x)) is x


def test_start_and_goal_channels():
    mask = known_mask((2, 4, 3), start_dims=[0, 2], goal_dims=[1])
    assert mask.dtype == torch.bool and mask.sum() == 2 * 3
    assert mask[:, 0].tolist() == [[True, False, True]] * 2
    assert mask[:, -1].tolist() == [[False, True, False]] * 2
    assert not mask[:, 1:3].any()


def test_apply_known_replaces_only_masked_entries():
    mask = known_mask((1, 3, 2), start_dims=[1])
    x, known = torch.zeros(1, 3, 2), torch.ones(1, 3, 2)
    out = apply_known(x, mask, known)
    assert out.sum() == 1 and out[0, 0, 1] == 1
