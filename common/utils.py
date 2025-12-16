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


def safe_norm(x, dim=-1, eps=1e-8):
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim), min=eps))


def set_torch_dict_to(dict: Dict[str, torch.Tensor], device: torch.device):
    for key in dict:
        dict[key] = dict[key].to(device)
    return dict


def dict_to_tensor_concat(
    dict_v: Dict[str, torch.Tensor], dim: int, device: torch.device
) -> torch.Tensor:
    tensor_concat = torch.cat(
        [dict_v[key] for key in dict_v.keys()],
        dim=dim,
    ).to(device)
    return tensor_concat
