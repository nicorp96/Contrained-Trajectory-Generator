"""
Physical constraints for joint-space robot plans (Franka Panda defaults).

All terms work on unnormalized plans {field: [B, T, d]} and are differentiable, so
the same module serves the physics loss (on the denoised estimate), guidance during
sampling, metrics and the final projection.

  kinematic           q[t+1] - q[t] - dt * (qd[t] + qd[t+1]) / 2           (needs qpos, qvel)
  joint_limits        distance of q outside [q_lower, q_upper]                (needs qpos)
  velocity_limits     |qd| above qvel_max (qd from qvel, else differences of q)
  acceleration_limits |qdd| above qacc_max (differences of qd)
  action_bounds       action outside [action_low, action_high]                (needs actions)
  action_consistency  joint change vs. commanded change u = action_delta_scale * a   (pd_joint_delta_pos)
                      without a controller model: dq[t] - u[t]
                      with the identified first-order lag (ctg.physics.identification):
                      dq[t] - alpha * dq[t-1] - beta * u[t],   dq[t] = q[t+1] - q[t]

Terms whose fields are missing from the plan are skipped. The time step is `dt`,
or per sample from the plan time stamps (`time` [B, T]) for non-uniform plans.
The controller model is discrete at the control rate, so action_consistency is
skipped for plans whose time stamps are not spaced by dt (resampled plans).
"""

import math
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi

# Franka Panda (7 revolute joints + 2 prismatic fingers), from the ManiSkill URDF
PANDA_Q_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0]
PANDA_Q_UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04]
PANDA_QVEL_MAX = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.2, 0.2]
PANDA_QACC_MAX = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, math.inf, math.inf]

DEFAULT_WEIGHTS = {"kinematic": 1.0, "joint_limits": 1.0, "velocity_limits": 1.0, "acceleration_limits": 0.1,
                   "action_bounds": 1.0, "action_consistency": 0.0}
DEFAULT_TOLERANCES = {"kinematic": 1e-2, "joint_limits": 1e-3, "velocity_limits": 1e-2, "acceleration_limits": 1e-1,
                      "action_bounds": 1e-3, "action_consistency": 1e-2}


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def masked_time_mean(e: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """e [B, T'] -> [B], averaged over valid steps."""
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


class JointConstraints:
    def __init__(
        self,
        dt: float = 0.05,
        n_revolute: int = 7,
        q_lower: Sequence[float] = PANDA_Q_LOWER,
        q_upper: Sequence[float] = PANDA_Q_UPPER,
        qvel_max: Sequence[float] = PANDA_QVEL_MAX,
        qacc_max: Sequence[float] = PANDA_QACC_MAX,
        action_low: float = -1.0,
        action_high: float = 1.0,
        action_delta_scale: float = 0.1,
        controller_alpha: Optional[Sequence[float]] = None,
        controller_beta: Optional[Sequence[float]] = None,
        weights: Optional[Mapping[str, float]] = None,
        tolerances: Optional[Mapping[str, float]] = None,
        wrap_angles: bool = False,
        qpos_key: str = "qpos",
        qvel_key: str = "qvel",
        action_key: str = "actions",
    ):
        self.dt = float(dt)
        self.n_revolute = int(n_revolute)
        self.q_lower = torch.tensor(list(q_lower), dtype=torch.float32)
        self.q_upper = torch.tensor(list(q_upper), dtype=torch.float32)
        self.qvel_max = torch.tensor(list(qvel_max), dtype=torch.float32)
        self.qacc_max = torch.tensor([float(v) for v in qacc_max], dtype=torch.float32)
        self.action_low, self.action_high = float(action_low), float(action_high)
        self.action_delta_scale = float(action_delta_scale)
        if (controller_alpha is None) != (controller_beta is None):
            raise ValueError("controller_alpha and controller_beta must be given together")
        self.controller = None
        if controller_alpha is not None:
            alpha, beta = (torch.tensor([float(v) for v in c], dtype=torch.float32)
                           for c in (controller_alpha, controller_beta))
            if alpha.shape != beta.shape:
                raise ValueError("controller_alpha and controller_beta need one value per joint")
            self.controller = (alpha, beta)
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}
        self.wrap = bool(wrap_angles)
        self.qpos_key, self.qvel_key, self.action_key = qpos_key, qvel_key, action_key

    # ------------------------------------------------------------------ helpers
    def _to(self, ref: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v.to(device=ref.device, dtype=ref.dtype)

    def _dq(self, q: torch.Tensor) -> torch.Tensor:
        dq = q[:, 1:] - q[:, :-1]
        if self.wrap:
            n = self.n_revolute
            dq = torch.cat([wrap_angle(dq[..., :n]), dq[..., n:]], dim=-1)
        return dq

    def _dt(self, ref: torch.Tensor, time: Optional[torch.Tensor]):
        """Scalar dt or per-pair time steps [B, T-1, 1] from the plan time stamps."""
        if time is None:
            return self.dt
        return (time[:, 1:] - time[:, :-1]).to(ref.dtype).clamp_min(1e-6).unsqueeze(-1)

    def _uniform(self, time: Optional[torch.Tensor]) -> bool:
        return time is None or bool(torch.all((time[:, 1:] - time[:, :-1] - self.dt).abs() < 1e-5))

    def _action_consistency(self, q, a, mask):
        """Residual of the joint change against the commanded change u = action_delta_scale * a."""
        dq, u = self._dq(q), self.action_delta_scale * a
        if self.controller is None:
            n = min(self.n_revolute, a.shape[-1])
            return (dq[..., :n] - u[:, :-1, :n]).abs(), pair_mask(mask, 1)
        alpha, beta = (self._to(q, c) for c in self.controller)
        n = alpha.shape[0]
        pred = alpha * dq[:, :-1, :n] + beta * u[:, 1:-1, :n]
        return (dq[:, 1:, :n] - pred).abs(), pair_mask(mask, 2)

    def _joint_excess(self, q: torch.Tensor) -> torch.Tensor:
        lower, upper = self._to(q, self.q_lower), self._to(q, self.q_upper)
        excess = F.relu(q - upper) + F.relu(lower - q)
        if self.wrap:  # revolute joints may be represented +-2pi away
            n = self.n_revolute
            rev = q[..., :n]
            cand = torch.stack([rev, rev + TWO_PI, rev - TWO_PI])
            exc = (F.relu(cand - upper[:n]) + F.relu(lower[:n] - cand)).min(dim=0).values
            excess = torch.cat([exc, excess[..., n:]], dim=-1)
        return excess

    # ------------------------------------------------------------------ terms
    def residuals(self, x: Mapping[str, torch.Tensor], mask: Optional[torch.Tensor] = None,
                  time: Optional[torch.Tensor] = None) -> Dict[str, tuple]:
        """{term: (violation [B, T', d], step mask [B, T'] or None)} for the terms available in x."""
        out = {}
        q = x.get(self.qpos_key)
        qd = x.get(self.qvel_key)
        a = x.get(self.action_key)
        ref = next(v for v in (q, qd, a) if v is not None)
        dt = self._dt(ref, time)

        if q is not None:
            out["joint_limits"] = (self._joint_excess(q), mask)
        vel, order = (qd, 0) if qd is not None else ((self._dq(q) / dt, 1) if q is not None else (None, 0))
        if vel is not None:
            out["velocity_limits"] = (F.relu(vel.abs() - self._to(vel, self.qvel_max)), pair_mask(mask, order))
            if vel.shape[1] > 1:
                dt_acc = dt if not torch.is_tensor(dt) else dt[:, order:]
                acc = (vel[:, 1:] - vel[:, :-1]) / dt_acc
                out["acceleration_limits"] = (F.relu(acc.abs() - self._to(acc, self.qacc_max)), pair_mask(mask, order + 1))
        if q is not None and qd is not None:
            out["kinematic"] = ((self._dq(q) - 0.5 * dt * (qd[:, 1:] + qd[:, :-1])).abs(), pair_mask(mask, 1))
        if a is not None:
            out["action_bounds"] = (F.relu(a - self.action_high) + F.relu(self.action_low - a), mask)
            if q is not None and q.shape[1] > 2 and self._uniform(time):
                out["action_consistency"] = self._action_consistency(q, a, mask)
        return out

    def energy(self, x, mask=None, time=None) -> Dict[str, torch.Tensor]:
        """Per-sample squared-violation energies [B] per term and the weighted 'total'."""
        terms, total = {}, None
        for name, (viol, m) in self.residuals(x, mask, time).items():
            terms[name] = masked_time_mean((viol ** 2).sum(dim=-1), m)
            w = self.weights.get(name, 0.0)
            if w > 0.0:
                total = w * terms[name] if total is None else total + w * terms[name]
        if total is None:
            total = torch.zeros(next(iter(x.values())).shape[0], device=next(iter(x.values())).device)
        terms["total"] = total
        return terms

    @torch.no_grad()
    def metrics(self, x, mask=None, time=None) -> Dict[str, float]:
        """Violation rate (share of valid steps above tolerance), max and mean violation per term."""
        metrics = {}
        for name, (viol, m) in self.residuals(x, mask, time).items():
            step_viol = viol.amax(dim=-1)
            valid = torch.ones_like(step_viol, dtype=torch.bool) if m is None else m.bool()
            n_valid = valid.sum().clamp_min(1)
            bad = (step_viol > self.tolerances.get(name, 0.0)) & valid
            metrics[f"{name}_violation_rate"] = (bad.sum() / n_valid).item()
            metrics[f"{name}_max"] = step_viol[valid].max().item() if valid.any() else 0.0
            metrics[f"{name}_mean"] = ((step_viol * valid).sum() / n_valid).item()
        return metrics

    @torch.no_grad()
    def project(self, x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clamp joint positions, velocities and actions into their feasible sets."""
        out = dict(x)
        if self.qpos_key in x:
            q = x[self.qpos_key]
            if self.wrap:
                n = self.n_revolute
                rev = q[..., :n]
                cand = torch.stack([rev, rev + TWO_PI, rev - TWO_PI])
                lower, upper = self._to(q, self.q_lower)[:n], self._to(q, self.q_upper)[:n]
                best = (F.relu(cand - upper) + F.relu(lower - cand)).argmin(dim=0, keepdim=True)
                q = torch.cat([cand.gather(0, best).squeeze(0), q[..., n:]], dim=-1)
            out[self.qpos_key] = torch.clamp(q, self._to(q, self.q_lower), self._to(q, self.q_upper))
        if self.qvel_key in x:
            vmax = self._to(x[self.qvel_key], self.qvel_max)
            out[self.qvel_key] = torch.clamp(x[self.qvel_key], -vmax, vmax)
        if self.action_key in x:
            out[self.action_key] = x[self.action_key].clamp(self.action_low, self.action_high)
        return out
