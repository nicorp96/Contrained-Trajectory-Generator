"""
Normalization of plan, goal and history fields, and of camera images.

Statistics come from stats.json (scripts/compute_stats.py) and are stored as
buffers, so they travel with the checkpoint (state_dict / load_state_dict).
"""

import json
from typing import Dict, Mapping

import torch
import torch.nn as nn

from ctg.data.layout import FieldSpec, Layout, concat, split

EPS = 1e-8


class FieldNormalizer(nn.Module):
    """none: identity; standard: (x - mean) / std; minmax: to [-1, 1]; sincos: angles -> (sin, cos)."""

    def __init__(self, spec: FieldSpec):
        super().__init__()
        self.method = spec.norm
        self.register_buffer("mean", torch.zeros(spec.size))
        self.register_buffer("std", torch.ones(spec.size))
        self.register_buffer("min", -torch.ones(spec.size))
        self.register_buffer("max", torch.ones(spec.size))

    def set_stats(self, stats: Mapping):
        for name in ("mean", "std", "min", "max"):
            if name in stats:
                getattr(self, name).copy_(torch.as_tensor(stats[name], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "standard":
            return (x - self.mean) / (self.std + EPS)
        if self.method == "minmax":
            return 2 * (x - self.min) / (self.max - self.min + EPS) - 1
        if self.method == "sincos":
            return torch.stack((torch.sin(x), torch.cos(x)), dim=-1).flatten(-2)
        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "standard":
            return x * (self.std + EPS) + self.mean
        if self.method == "minmax":
            return 0.5 * (x + 1) * (self.max - self.min + EPS) + self.min
        if self.method == "sincos":
            return torch.atan2(x[..., 0::2], x[..., 1::2])
        return x


class ImageNormalizer(nn.Module):
    """uint8 images [B, L, H, W, C] or [B, L, C, H, W] -> float [B, L, C, H, W] in [0, 1].
    The image encoders apply their own (ImageNet) normalization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B, L, ...] images, got shape {tuple(x.shape)}")
        if x.shape[2] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 1, 4, 2, 3)
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        return x.expand(-1, -1, 3, -1, -1) if x.shape[2] == 1 else x


class LayoutNormalizer(nn.Module):
    """Normalizes a dict of fields; cameras (if given) are converted with ImageNormalizer."""

    def __init__(self, layout: Layout, cameras=()):
        super().__init__()
        self.fields = nn.ModuleDict({name: FieldNormalizer(spec) for name, spec in layout.items()})
        self.cameras = tuple(cameras)
        self.image = ImageNormalizer()

    def load_stats(self, stats: Mapping):
        """stats: {field: {"mean": [...], "std": [...], "min": [...], "max": [...]}}."""
        for name, norm in self.fields.items():
            if norm.method in ("standard", "minmax"):
                if name not in stats:
                    raise KeyError(f"no statistics for field {name!r}")
                norm.set_stats(stats[name])

    def forward(self, x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {name: self.fields[name](x[name]) for name in self.fields}
        out.update({cam: self.image(x[cam]) for cam in self.cameras if cam in x})
        return out

    def unnormalize(self, x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: self.fields[name].unnormalize(x[name]) for name in self.fields if name in x}


class PlanCodec:
    """Converts between the normalized plan tensor [B, T, D] and physical fields {field: [B, T, d]}."""

    def __init__(self, layout: Layout, normalizer: LayoutNormalizer):
        self.layout = layout
        self.normalizer = normalizer

    def encode(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return concat(self.normalizer(fields), self.layout)

    def decode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.normalizer.unnormalize(split(x, self.layout))


def load_stats_file(path) -> dict:
    with open(path) as f:
        return json.load(f)
