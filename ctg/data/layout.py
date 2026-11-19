"""
Field layout: which quantities make up the plan, the goal and the history.

A layout maps field names (as in the ManiSkill observation, e.g. "qpos") to a
FieldSpec. Plans are stored as one tensor [B, T, D] whose last dimension is the
concatenation of the (normalized) fields in layout order.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch

NORM_METHODS = ("none", "standard", "minmax", "sincos")


@dataclass(frozen=True)
class FieldSpec:
    """One quantity, e.g. joint positions: size 9, min-max normalized."""

    size: int
    norm: str = "none"
    interp: str = "vector"  # resampling: "vector", "quaternion" or "pose" (xyz + quaternion)

    def __post_init__(self):
        if self.norm not in NORM_METHODS:
            raise ValueError(f"unknown normalization {self.norm!r}; choose from {NORM_METHODS}")

    @property
    def width(self) -> int:
        """Channels in normalized space (sin/cos doubles them)."""
        return 2 * self.size if self.norm == "sincos" else self.size


Layout = Dict[str, FieldSpec]


def make_layout(fields: Mapping) -> Layout:
    """{"qpos": {"size": 9, "norm": "minmax"}, ...} -> {"qpos": FieldSpec(...), ...}.
    Entries set to None are skipped, so a config can remove a field it inherits."""
    return {name: FieldSpec(**dict(spec)) for name, spec in (fields or {}).items() if spec is not None}


def total_width(layout: Layout) -> int:
    return sum(spec.width for spec in layout.values())


def split(x: torch.Tensor, layout: Layout) -> Dict[str, torch.Tensor]:
    """[..., D] -> {field: [..., width]} (views, keeps autograd)."""
    out, start = {}, 0
    for name, spec in layout.items():
        out[name] = x[..., start:start + spec.width]
        start += spec.width
    if start != x.shape[-1]:
        raise ValueError(f"layout width {start} != tensor width {x.shape[-1]}")
    return out


def concat(fields: Mapping[str, torch.Tensor], layout: Layout) -> torch.Tensor:
    """{field: [..., width]} -> [..., D] in layout order."""
    return torch.cat([fields[name] for name in layout], dim=-1)


@dataclass(frozen=True)
class DataSpec:
    """What a policy sees and plans, independent of the stored demonstrations
    (the dataset config without the file): enough to rebuild the networks."""

    plan_layout: Layout
    goal_layout: Layout
    history_layout: Layout
    cameras: Tuple[str, ...]
    plan_len: int
    history_len: int
    action_key: str = "actions"
    control_dt: float = 0.05

    @classmethod
    def from_config(cls, dataset_cfg: Mapping) -> "DataSpec":
        return cls(make_layout(dataset_cfg["plan"]), make_layout(dataset_cfg["goal"]),
                   make_layout(dataset_cfg["history"]), tuple(dataset_cfg.get("cameras") or ()),
                   int(dataset_cfg["plan_len"]), int(dataset_cfg["history_len"]),
                   dataset_cfg.get("action_key", "actions"), float(dataset_cfg.get("control_dt", 0.05)))
