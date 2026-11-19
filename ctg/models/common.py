"""Building blocks shared by the denoisers."""

import math

import torch
import torch.nn as nn


def apply_null_goal(goal: torch.Tensor, null_goal: torch.Tensor, goal_drop) -> torch.Tensor:
    """Replace the goal by a learned null token where goal_drop is True (classifier-free guidance)."""
    if goal_drop is None:
        return goal
    return torch.where(goal_drop.view(-1, 1), null_goal.to(goal.dtype).expand_as(goal), goal)


class SinusoidalPosEmb(nn.Module):
    """Scalar (diffusion step, position) -> [sin, cos] features of `dim` channels."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=x.device) * -(math.log(10000) / (half - 1)))
        emb = x.float()[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
