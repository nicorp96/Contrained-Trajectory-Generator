"""
Temporal resampling of demonstration segments to a fixed number of points.

A resampler maps a segment (time stamps t [N] and fields {name: [N, d]}) to
num_points samples. The returned time stamps are part of the training sample,
so time-dependent physics (velocities, accelerations) stays well defined even
when the spacing is not uniform.

  TimeResampler        points uniform in time ("compress the rest of the episode")
  ArcLengthResampler   points uniform in path length of one field (e.g. qpos)
"""

from typing import Dict, Mapping, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

INTERP_METHODS = ("linear", "cubic", "pchip")


def interpolate(t: np.ndarray, y: np.ndarray, t_new: np.ndarray, method: str = "linear") -> np.ndarray:
    """Interpolate y [N, d] given at t [N] (strictly increasing) to t_new [K]."""
    if method not in INTERP_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {INTERP_METHODS}")
    if len(t) < 2:
        return np.repeat(y[:1], len(t_new), axis=0)
    if method == "linear" or len(t) < 3:
        return np.stack([np.interp(t_new, t, y[:, j]) for j in range(y.shape[1])], axis=1)
    if method == "cubic":
        return CubicSpline(t, y, axis=0, bc_type="natural")(t_new)
    return PchipInterpolator(t, y, axis=0)(t_new)


def slerp(t: np.ndarray, q: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Segment-wise spherical interpolation of unit quaternions q [N, 4] (any convention)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    idx = np.clip(np.searchsorted(t, t_new, side="right") - 1, 0, len(t) - 2)
    u = np.clip((t_new - t[idx]) / np.maximum(t[idx + 1] - t[idx], 1e-12), 0.0, 1.0)
    q0, q1 = q[idx], q[idx + 1]
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)  # shortest path
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    near = dot > 0.9995  # nearly identical: linear interpolation is exact enough
    sin_theta = np.where(near, 1.0, np.sin(theta))
    w0 = np.where(near, 1 - u, np.sin((1 - u) * theta) / sin_theta)
    w1 = np.where(near, u, np.sin(u * theta) / sin_theta)
    out = w0[:, None] * q0 + w1[:, None] * q1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def interpolate_field(t, y, t_new, kind: str = "vector", method: str = "linear") -> np.ndarray:
    """Interpolate one field according to its kind: vector, quaternion or pose (xyz + quaternion)."""
    if kind == "vector":
        return interpolate(t, y, t_new, method)
    if kind == "quaternion":
        return slerp(t, y, t_new)
    if kind == "pose":
        return np.concatenate([interpolate(t, y[:, :3], t_new, method), slerp(t, y[:, 3:7], t_new)], axis=1)
    raise ValueError(f"unknown field kind {kind!r}")


class TimeResampler:
    """num_points samples uniformly spaced in time over the segment."""

    def __init__(self, num_points: int, method: str = "linear"):
        if num_points < 2:
            raise ValueError("num_points must be >= 2")
        self.num_points = int(num_points)
        self.method = method

    def grid(self, t: np.ndarray, fields: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.linspace(t[0], t[-1], self.num_points)

    def __call__(
        self, t: np.ndarray, fields: Mapping[str, np.ndarray], kinds: Mapping[str, str] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        t = np.asarray(t, dtype=np.float64)
        if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing with at least 2 samples")
        t_new = self.grid(t, fields)
        kinds = kinds or {}
        out = {
            name: interpolate_field(t, np.asarray(y, dtype=np.float64), t_new, kinds.get(name, "vector"), self.method)
            for name, y in fields.items()
        }
        return t_new, out


class ArcLengthResampler(TimeResampler):
    """num_points samples uniformly spaced in path length of `path_field` (e.g. joint space).
    Pauses (no motion) get few samples, fast motion many; time stamps follow the path."""

    def __init__(self, num_points: int, path_field: str = "qpos", method: str = "linear", eps: float = 1e-9):
        super().__init__(num_points, method)
        self.path_field = path_field
        self.eps = eps

    def grid(self, t: np.ndarray, fields: Mapping[str, np.ndarray]) -> np.ndarray:
        path = np.asarray(fields[self.path_field], dtype=np.float64)
        length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        if length[-1] < self.eps:  # no motion at all: fall back to uniform time
            return super().grid(t, fields)
        # strictly increasing s (drop samples without motion) so t(s) is well defined
        keep = np.concatenate([[True], np.diff(length) > self.eps])
        s = length[keep] / length[-1]
        t_new = np.interp(np.linspace(0.0, 1.0, self.num_points), s, t[keep])
        t_new[-1] = t[-1]
        return np.maximum.accumulate(t_new)  # monotone despite rounding
