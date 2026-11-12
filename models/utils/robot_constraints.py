"""
Physical constraints for joint-space robot trajectories (Franka Panda defaults).

All functions work on *unnormalized* per-key tensors [B, T, d] and are
differentiable, so the same module is used for
  - the physics-informed training loss (on the denoised estimate x0_hat),
  - constraint guidance during sampling,
  - validation / rollout metrics,
  - the final hard projection of a sampled trajectory.

Terms (each returned per sample, masked over valid time steps):
  kinematic           wrap(q[t+1] - q[t]) - dt * (qd[t] + qd[t+1]) / 2   (needs qvel)
  joint_limits        distance of q outside [q_lower, q_upper]
  velocity_limits     |qd| above qvel_max (qd from qvel, else finite differences of q)
  acceleration_limits |qdd| above qacc_max (finite differences of qd)
  action_bounds       action outside [action_low, action_high]
  action_consistency  q[t+1] - q[t] - action_delta_scale * a[t]  (pd_joint_delta_pos only)
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi

# Franka Panda (7 revolute joints + 2 prismatic fingers), from the ManiSkill URDF
PANDA_DEFAULTS = {
    "q_lower": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0],
    "q_upper": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04],
    "qvel_max": [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.2, 0.2],
    "qacc_max": [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, math.inf, math.inf],
}

DEFAULT_WEIGHTS = {
    "kinematic": 1.0,
    "joint_limits": 1.0,
    "velocity_limits": 1.0,
    "acceleration_limits": 0.1,
    "action_bounds": 1.0,
    "action_consistency": 0.0,
}

DEFAULT_TOLERANCES = {
    "kinematic": 1e-2,  # rad
    "joint_limits": 1e-3,  # rad / m
    "velocity_limits": 1e-2,  # rad/s
    "acceleration_limits": 1e-1,  # rad/s^2
    "action_bounds": 1e-3,
    "action_consistency": 1e-2,
}


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def masked_time_mean(e: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """e: [B, T'] -> [B], averaged over valid steps."""
    if mask is None:
        return e.mean(dim=1)
    m = mask.to(e.dtype)
    return (e * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def pair_mask(mask: Optional[torch.Tensor], order: int = 1) -> Optional[torch.Tensor]:
    """Valid mask for finite differences of the given order."""
    if mask is None:
        return None
    m = mask.bool()
    for _ in range(order):
        m = m[:, 1:] & m[:, :-1]
    return m


class RobotConstraints:
    def __init__(self, config: Dict, state_shapes: Dict):
        config = dict(config or {})
        self.state_shapes = state_shapes
        self.qpos_key = config.get("qpos_key", "qpos")
        self.qvel_key = config.get("qvel_key", "qvel")
        self.action_key = config.get("action_key", "actions")
        if self.qpos_key not in state_shapes:
            raise ValueError(f"constraints need '{self.qpos_key}' in state_shapes")
        self.has_qvel = self.qvel_key in state_shapes
        self.has_action = self.action_key in state_shapes

        self.dt = float(config.get("dt", 0.05))
        self.n_revolute = int(config.get("n_revolute", 7))
        # angles that came out of a sin/cos representation live in [-pi, pi)
        self.wrap = bool(
            config.get(
                "wrap_angles",
                state_shapes[self.qpos_key].get("method_norm") == "sincos",
            )
        )

        def _vec(name):
            return torch.tensor(config.get(name, PANDA_DEFAULTS[name]), dtype=torch.float32)

        self.q_lower = _vec("q_lower")
        self.q_upper = _vec("q_upper")
        self.qvel_max = _vec("qvel_max")
        self.qacc_max = _vec("qacc_max")
        self.action_low = float(config.get("action_low", -1.0))
        self.action_high = float(config.get("action_high", 1.0))
        self.action_delta_scale = float(config.get("action_delta_scale", 0.1))

        self.weights = {**DEFAULT_WEIGHTS, **config.get("weights", {})}
        self.tolerances = {**DEFAULT_TOLERANCES, **config.get("tolerances", {})}
        if not self.has_qvel:
            self.weights["kinematic"] = 0.0

    # ------------------------------------------------------------------ helpers
    def _limits(self, ref: torch.Tensor):
        to = dict(device=ref.device, dtype=ref.dtype)
        return (
            self.q_lower.to(**to),
            self.q_upper.to(**to),
            self.qvel_max.to(**to),
            self.qacc_max.to(**to),
        )

    def _dq(self, q: torch.Tensor) -> torch.Tensor:
        dq = q[:, 1:] - q[:, :-1]
        if self.wrap:
            n = self.n_revolute
            dq = torch.cat([wrap_angle(dq[..., :n]), dq[..., n:]], dim=-1)
        return dq

    def _joint_excess(self, q, lower, upper):
        """Distance outside the joint interval; wrap-aware for revolute joints."""
        excess = F.relu(q - upper) + F.relu(lower - q)
        if self.wrap:
            n = self.n_revolute
            rev = q[..., :n]
            candidates = torch.stack([rev, rev + TWO_PI, rev - TWO_PI], dim=0)
            exc = F.relu(candidates - upper[:n]) + F.relu(lower[:n] - candidates)
            excess = torch.cat([exc.min(dim=0).values, excess[..., n:]], dim=-1)
        return excess

    def _velocity(self, x: Dict[str, torch.Tensor]):
        """Joint velocity [B, T, J] and its validity order (0 = measured, 1 = finite diff)."""
        if self.has_qvel:
            return x[self.qvel_key], 0
        return self._dq(x[self.qpos_key]) / self.dt, 1

    # ------------------------------------------------------------------ residuals
    def residuals(
        self, x: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> Dict[str, tuple]:
        """
        Returns {term: (per_step_violation [B, T'], per_step_mask [B, T'] | None)}.
        per_step_violation is the max over joints of the absolute violation.
        """
        q = x[self.qpos_key]
        lower, upper, vmax, amax = self._limits(q)
        out = {}

        out["joint_limits"] = (self._joint_excess(q, lower, upper), mask)

        qd, order = self._velocity(x)
        out["velocity_limits"] = (F.relu(qd.abs() - vmax), pair_mask(mask, order))

        qdd = (qd[:, 1:] - qd[:, :-1]) / self.dt
        out["acceleration_limits"] = (
            F.relu(qdd.abs() - amax),
            pair_mask(mask, order + 1),
        )

        if self.has_qvel:
            dq = self._dq(q)
            r = dq - 0.5 * self.dt * (qd[:, 1:] + qd[:, :-1])
            out["kinematic"] = (r.abs(), pair_mask(mask, 1))

        if self.has_action:
            a = x[self.action_key]
            out["action_bounds"] = (
                F.relu(a - self.action_high) + F.relu(self.action_low - a),
                mask,
            )
            n = min(self.n_revolute, a.shape[-1])
            r = self._dq(q)[..., :n] - self.action_delta_scale * a[:, :-1, :n]
            out["action_consistency"] = (r.abs(), pair_mask(mask, 1))
        return out

    def energy(
        self, x: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Per-sample squared-violation energies [B] per term and weighted 'total'."""
        terms = {}
        total = None
        for name, (viol, m) in self.residuals(x, mask).items():
            e = masked_time_mean((viol**2).sum(dim=-1), m)
            terms[name] = e
            w = self.weights.get(name, 0.0)
            if w > 0.0:
                total = w * e if total is None else total + w * e
        if total is None:
            total = torch.zeros(
                x[self.qpos_key].shape[0], device=x[self.qpos_key].device
            )
        terms["total"] = total
        return terms

    @torch.no_grad()
    def metrics(
        self, x: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Violation rate (fraction of valid steps above tolerance) and max violation per term."""
        metrics = {}
        for name, (viol, m) in self.residuals(x, mask).items():
            step_viol = viol.amax(dim=-1)  # [B, T']
            valid = torch.ones_like(step_viol, dtype=torch.bool) if m is None else m.bool()
            n_valid = valid.sum().clamp_min(1)
            bad = (step_viol > self.tolerances.get(name, 0.0)) & valid
            metrics[f"{name}_violation_rate"] = (bad.sum() / n_valid).item()
            metrics[f"{name}_max"] = (
                step_viol[valid].max().item() if valid.any() else 0.0
            )
            metrics[f"{name}_mean"] = ((step_viol * valid).sum() / n_valid).item()
        return metrics

    # ------------------------------------------------------------------ projection
    @torch.no_grad()
    def project(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clamp joint positions, velocities and actions into their feasible sets."""
        out = dict(x)
        q = x[self.qpos_key]
        lower, upper, vmax, _ = self._limits(q)
        if self.wrap:
            n = self.n_revolute
            rev = q[..., :n]
            candidates = torch.stack([rev, rev + TWO_PI, rev - TWO_PI], dim=0)
            exc = F.relu(candidates - upper[:n]) + F.relu(lower[:n] - candidates)
            best = exc.argmin(dim=0, keepdim=True)
            rev = candidates.gather(0, best).squeeze(0)
            q = torch.cat([rev, q[..., n:]], dim=-1)
        out[self.qpos_key] = torch.maximum(torch.minimum(q, upper), lower)
        if self.has_qvel:
            qd = x[self.qvel_key]
            out[self.qvel_key] = torch.maximum(torch.minimum(qd, vmax), -vmax)
        if self.has_action:
            out[self.action_key] = x[self.action_key].clamp(
                self.action_low, self.action_high
            )
        return out
