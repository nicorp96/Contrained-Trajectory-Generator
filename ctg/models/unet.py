import torch
import torch.nn as nn

from ctg.models.common import apply_null_goal
from ctg.models.conditional_unet1d import ConditionalUnet1D


class UNetDenoiser(nn.Module):
    """
    Diffusion Policy's ConditionalUnet1D as a plan denoiser.

    The flattened observation history and the goal form the FiLM condition of every
    block. Where `goal_drop` is True the goal is replaced by a learned null token
    (classifier-free guidance). Padding masks are not needed: padded steps are
    excluded from the loss by the trainer.
    """

    def __init__(
        self,
        plan_dim: int,
        goal_dim: int,
        history_dim: int,
        history_len: int,
        down_dims=(64, 128, 256),
        step_embed_dim: int = 64,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.history_size = int(history_dim) * int(history_len)
        self.down_dims = list(down_dims)
        self.null_goal = nn.Parameter(torch.zeros(goal_dim))
        self.net = ConditionalUnet1D(
            input_dim=plan_dim,
            global_cond_dim=self.history_size + goal_dim,
            diffusion_step_embed_dim=step_embed_dim,
            down_dims=self.down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

    def forward(self, x, t, goal, history, goal_drop=None, mask=None):
        """x [B, T, plan_dim], t [B] or scalar, goal [B, goal_dim], history [B, H, F] -> [B, T, plan_dim]."""
        stride = 2 ** (len(self.down_dims) - 1)
        if x.shape[1] % stride:
            raise ValueError(f"plan length {x.shape[1]} must be divisible by {stride} (U-Net downsampling)")
        if history[0].numel() != self.history_size:
            raise ValueError(f"history must hold {self.history_size} features per sample, got {tuple(history.shape[1:])}")
        goal = apply_null_goal(goal, self.null_goal, goal_drop)
        return self.net(x, t, global_cond=torch.cat([history.flatten(1), goal], dim=-1))
