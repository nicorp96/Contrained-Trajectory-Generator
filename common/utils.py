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


def split_state_tensor(
    state_tensor: torch.Tensor, param_shapes: dict, ignore_keys: list = [""]
) -> dict:
    split_dict = {}
    start = 0
    for name, value in param_shapes.items():
        if name not in ignore_keys:
            end = start + value["shape"]
            split_dict[name] = state_tensor[:, :, start:end]  # .detach().cpu()
            start = end
    return split_dict


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


def split_state_tensor(state_tensor: torch.Tensor, param_shapes: dict) -> dict:
    split_dict = {}
    start = 0
    for name in param_shapes.keys():
        end = start + param_shapes[name]["shape"]
        split_dict[name] = state_tensor[:, :, start:end].detach().cpu()
        start = end
    return split_dict
