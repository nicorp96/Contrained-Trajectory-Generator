"""
Identification of the joint controller response from demonstrations.

ManiSkill's pd_joint_delta_pos sets the joint target to the current position plus
action_delta_scale * a. The PD controller does not reach it within one control step:
per joint, the measured change follows a first-order lag

  dq[t] = alpha * dq[t-1] + beta * u[t],   dq[t] = q[t+1] - q[t],  u[t] = action_delta_scale * a[t]

fitted by least squares. beta / (1 - alpha) is the steady-state gain (1 = target reached).
The coefficients enter JointConstraints(controller_alpha, controller_beta).
"""

from typing import Dict, Iterable

import h5py
import numpy as np

from ctg.data.maniskill import find_dataset


def controller_samples(h5_path: str, episode_ids: Iterable[int], n_joints: int = 7, action_delta_scale: float = 0.1,
                       qpos_key: str = "qpos", action_key: str = "actions") -> Dict[str, np.ndarray]:
    """Regression samples {dq, dq_prev, u} [N, n_joints] from all steps of the given episodes."""
    chunks = {"dq": [], "dq_prev": [], "u": []}
    with h5py.File(h5_path, "r") as f:
        for ep in episode_ids:
            traj = f[f"traj_{ep}"]
            a = np.asarray(traj[action_key][:, :n_joints], dtype=np.float64)
            q = np.asarray(find_dataset(traj, qpos_key)[: len(a) + 1, :n_joints], dtype=np.float64)
            dq = np.diff(q, axis=0)
            chunks["dq"].append(dq[1:])
            chunks["dq_prev"].append(dq[:-1])
            chunks["u"].append(action_delta_scale * a[1:])
    return {k: np.concatenate(v) for k, v in chunks.items()}


def predict(samples: Dict[str, np.ndarray], alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return alpha * samples["dq_prev"] + beta * samples["u"]


def r2_score(samples: Dict[str, np.ndarray], alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Coefficient of determination per joint (1 = perfect, 0 = as good as the mean)."""
    y = samples["dq"]
    residual = ((y - predict(samples, alpha, beta)) ** 2).sum(axis=0)
    return 1.0 - residual / np.maximum(((y - y.mean(axis=0)) ** 2).sum(axis=0), 1e-12)


def fit_first_order(samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Least-squares alpha, beta per joint, with R^2 and steady-state gain."""
    n_joints = samples["dq"].shape[1]
    coef = np.array([np.linalg.lstsq(np.stack([samples["dq_prev"][:, j], samples["u"][:, j]], axis=1),
                                     samples["dq"][:, j], rcond=None)[0] for j in range(n_joints)])
    alpha, beta = coef[:, 0], coef[:, 1]
    return {"alpha": alpha, "beta": beta, "r2": r2_score(samples, alpha, beta), "gain": beta / (1.0 - alpha)}
