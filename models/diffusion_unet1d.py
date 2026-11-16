import torch
import torch.nn as nn

from models.conditional_unet1d import ConditionalUnet1D
from models.diffusion_transformer import apply_null_cond


class UNetTrjCC(nn.Module):
    """
    ConditionalUnet1D (Diffusion Policy) behind the DiTTrjCC interface.

    The flattened observation history and the goal form the FiLM condition of every
    block, as in ManiSkill's Diffusion Policy; the goal can be replaced by a learned
    null token for classifier-free guidance (cond_drop). Padding masks are ignored:
    padded steps are excluded from the loss by the trainer.

    config: input_dim, cond_dim, history_shape (features per history step),
            history_len (history steps), optional unet_dims, diffusion_step_embed_dim,
            kernel_size, n_groups (ManiSkill defaults).
    """

    def __init__(self, config):
        super().__init__()
        self.cond_dim = int(config["cond_dim"])
        self.history_dim = int(config["history_shape"]) * int(config["history_len"])
        self.down_dims = list(config.get("unet_dims", [64, 128, 256]))
        self.null_cond = nn.Parameter(torch.zeros(self.cond_dim))
        self.net = ConditionalUnet1D(
            input_dim=int(config["input_dim"]),
            global_cond_dim=self.history_dim + self.cond_dim,
            diffusion_step_embed_dim=int(config.get("diffusion_step_embed_dim", 64)),
            down_dims=self.down_dims,
            kernel_size=int(config.get("kernel_size", 5)),
            n_groups=int(config.get("n_groups", 8)),
        )

    def forward(self, x, t, cond, mask=None, cond_drop=None, state_hist=None, **kwargs):
        """x: [B, T, input_dim], t: [B] or scalar, cond: [B, cond_dim],
        state_hist: [B, history_len, history_shape] -> [B, T, input_dim]."""
        stride = 2 ** (len(self.down_dims) - 1)
        if x.shape[1] % stride:
            raise ValueError(f"sequence length {x.shape[1]} must be divisible by {stride} (U-Net downsampling)")
        if state_hist is None or state_hist[0].numel() != self.history_dim:
            got = None if state_hist is None else tuple(state_hist.shape[1:])
            raise ValueError(f"state_hist must hold {self.history_dim} features per sample, got {got}")
        cond = apply_null_cond(cond, self.null_cond, cond_drop)
        global_cond = torch.cat([state_hist.flatten(1), cond], dim=-1)
        return self.net(x, t, global_cond=global_cond)
