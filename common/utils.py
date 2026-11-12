import torch
import yaml
from typing import Dict

AXIS_ID_TO_XYZ = ["x", "y", "z"]


def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")


def apply_overrides(config: dict, overrides) -> dict:
    """
    Apply dotted overrides in place, e.g. ["dataset.path=/data/x.h5", "training.batch_size=32"].
    Values are parsed as YAML (numbers, booleans, lists).
    """
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' must look like key.sub=value")
        node = config
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return config


def normalized_width(spec: dict) -> int:
    """Number of channels a state key occupies in normalized space."""
    width = int(spec["shape"])
    if spec.get("method_norm") == "sincos":
        width *= 2
    return width


def split_trajectory(state_tensor: torch.Tensor, param_shapes: dict) -> dict:
    """
    Split a normalized [B, T, D] tensor into per-key views.
    Keeps device and autograd graph (use for losses / guidance).
    """
    split_dict = {}
    start = 0
    for name, spec in param_shapes.items():
        end = start + normalized_width(spec)
        split_dict[name] = state_tensor[:, :, start:end]
        start = end
    return split_dict


def split_state_tensor(state_tensor: torch.Tensor, param_shapes: dict) -> dict:
    """Detached CPU variant of split_trajectory (for plotting / logging)."""
    return {
        k: v.detach().cpu()
        for k, v in split_trajectory(state_tensor, param_shapes).items()
    }


def extract_into_shape(
    x_1d: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor
):
    """
    x_1d: [T]
    timesteps: [B] int64
    target: tensor with batch dim first, e.g. [B, ...]
    returns: [B, 1, 1, ...] broadcastable to target
    """
    b = timesteps.shape[0]
    out = x_1d.gather(0, timesteps)  # [B]
    # reshape to [B, 1, 1, ...] to broadcast across target dims
    return out.view(b, *([1] * (target.ndim - 1)))


def dict_to_tensor_concat(
    dict_v: Dict[str, torch.Tensor], dim: int, device: torch.device
) -> torch.Tensor:
    tensor_concat = torch.cat(
        [dict_v[key] for key in dict_v.keys()],
        dim=dim,
    ).to(device)
    return tensor_concat
