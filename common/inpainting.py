import torch
from typing import Optional, Iterable


def apply_inpainting(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,  # [B,H,D] 1=known
    x_known: torch.Tensor | None = None,  # [B,H,D]
    noise: bool = False,
):
    """
    If train_mode=True: replace known entries with 0.
    If train_mode=False: replace known entries with their provided values.
    """
    # --- dense mask-based inpainting (optional)
    if (mask is not None) and (x_known is not None):
        m = mask.float()
        if noise:
            x = x * (1.0 - m)  # known -> 0
        else:
            x = x * (1.0 - m) + x_known * m  # clamp known values
    return x


def make_inpainting_mask(
    trajectory: torch.Tensor,
    *,
    start_indices: Optional[Iterable[int]] = None,  # inpaint all features at t=0
    goal_indices: Optional[Iterable[int]] = None,  # inpaint these features at t=-1
) -> torch.Tensor:
    """
    trajectory: [B, T, D]
    start=True  -> mask[:, 0, :] = True
    goal_indices=[...] -> mask[:, -1, goal_indices] = True
    """
    if trajectory.ndim != 3:
        raise ValueError(f"Expected [B,T,D], got {tuple(trajectory.shape)}")

    device = trajectory.device
    mask = torch.zeros_like(trajectory, dtype=torch.bool, device=device)

    if start_indices is not None:
        start_idx = torch.as_tensor(list(start_indices), device=device)
        mask[:, 0, start_idx] = True

    if goal_indices is not None:
        goal_idx = torch.as_tensor(list(goal_indices), device=device)
        mask[:, -1, goal_idx] = True

    return mask
