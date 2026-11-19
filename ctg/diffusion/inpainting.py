"""
Inpainting: entries of the plan that are known (e.g. the current state at the first
step, or a goal state at the last step) are fixed during training and sampling.
"""

from typing import Optional, Sequence

import torch


def known_mask(shape, start_dims: Optional[Sequence[int]] = None, goal_dims: Optional[Sequence[int]] = None,
               device=None) -> Optional[torch.Tensor]:
    """Boolean mask [B, T, D]: True where the plan is known.
    start_dims: channels fixed at the first step; goal_dims: channels fixed at the last step."""
    if not start_dims and not goal_dims:
        return None
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    if start_dims:
        mask[:, 0, list(start_dims)] = True
    if goal_dims:
        mask[:, -1, list(goal_dims)] = True
    return mask


def apply_known(x: torch.Tensor, mask: Optional[torch.Tensor], known: torch.Tensor) -> torch.Tensor:
    """x with the masked entries replaced by `known`."""
    return x if mask is None else torch.where(mask, known, x)
