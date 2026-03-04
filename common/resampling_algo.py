import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, Any
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator

Kind = Literal["scalar", "vector", "delta_pose", "accel3", "tcp_pose", "copy"]


def _interp1(x_src, y_src, x_tgt):
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_tgt = np.asarray(x_tgt, dtype=float)
    if np.isnan(y_src).any():
        raise ValueError("NaNs detected in source data for interpolation.")
    if _HAS_PCHIP and len(x_src) >= 3:
        f = _Pchip(x_src, y_src, extrapolate=True)
        return f(x_tgt)
    else:
        return np.interp(x_tgt, x_src, y_src)


def unwrap_angles_over_time(a: np.ndarray) -> np.ndarray:
    """
    a: (N, D) angles in radians
    returns: (N, D) unwrapped along time axis
    """
    return np.unwrap(a, axis=0)


def _wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def interp_angles(t, a, t_new, kind="linear"):
    a_unw = np.unwrap(a)
    f = interp1d(
        t, a_unw, kind=kind, bounds_error=False, fill_value=(a_unw[0], a_unw[-1])
    )
    a_new = f(np.clip(t_new, t[0], t[-1]))
    return ((a_new + np.pi) % (2 * np.pi)) - np.pi


def _quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # q: (..., 4)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, eps, None)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    SLERP between q0 and q1 for interpolation fractions u in [0,1].
    q0,q1: (M,4)  u: (M,)
    returns: (M,4)
    """
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)

    # Ensure shortest path: if dot < 0, flip q1
    dot = np.sum(q0 * q1, axis=-1)
    flip = dot < 0.0
    q1 = np.where(flip[:, None], -q1, q1)
    dot = np.abs(dot)

    # If very close, use lerp then normalize
    DOT_THRESH = 0.9995
    close = dot > DOT_THRESH

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))  # (M,)
    sin_theta_0 = np.sin(theta_0)  # (M,)
    theta = theta_0 * u  # (M,)
    sin_theta = np.sin(theta)  # (M,)

    s0 = np.where(
        close, 1.0 - u, np.sin(theta_0 - theta) / np.clip(sin_theta_0, 1e-12, None)
    )
    s1 = np.where(close, u, sin_theta / np.clip(sin_theta_0, 1e-12, None))

    out = (s0[:, None] * q0) + (s1[:, None] * q1)
    return _quat_normalize(out)


def _interp_piecewise_linear(
    t: np.ndarray, y: np.ndarray, t_new: np.ndarray
) -> np.ndarray:
    """
    Vectorized 1D time interpolation for multi-dim signals using linear segment interpolation.
    t: (N,) strictly increasing
    y: (N,D)
    t_new: (K,)
    returns: (K,D)
    """
    t = np.asarray(t, np.float64)
    y = np.asarray(y, np.float64)
    t_new = np.asarray(t_new, np.float64)

    idx = np.searchsorted(t, t_new, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 2)

    t0 = t[idx]
    t1 = t[idx + 1]
    u = (t_new - t0) / np.clip(t1 - t0, 1e-12, None)

    y0 = y[idx]
    y1 = y[idx + 1]
    return y0 + (y1 - y0) * u[:, None]


def _interp_quat4(t: np.ndarray, q: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """
    Quaternion interpolation via segment-wise SLERP.
    """
    idx = np.searchsorted(t, t_new, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 2)

    t0 = t[idx]
    t1 = t[idx + 1]
    u = (t_new - t0) / np.clip(t1 - t0, 1e-12, None)

    q0 = q[idx]
    q1 = q[idx + 1]
    return _quat_slerp(q0, q1, u)


def interp_angle_vec(t: np.ndarray, a: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """
    Wrap-aware interpolation for a vector of joint angles.
    """
    a_unwrap = unwrap_angles_over_time(a)  # (N, D)
    a_new = _interp_piecewise_linear(t, a_unwrap, t_new)
    return _wrap_to_pi(a_new)


@dataclass
class FieldSpec:
    kind: Kind  # how to treat the field
    cols: Tuple[int, ...]  # column indices in `states` (e.g., (0,1,2))
    name: Optional[str] = None  # optional friendly name used in output
    # optional per-field config
    interp: str = "linear"  # for scalar/vector: "linear" or "cubic"
    required: bool = False  # if True and missing, raise


class ResamplingBase:
    def __init__(self, config):
        self.interp_kind = config.get("interp_kind", "linear")
        self.extrapolate_edge = bool(config.get("extrapolate_edge", True))
        self.K = int(config["K"])
        self.eps = float(config.get("eps", 1e-9))

    # ---- helpers (reused by all subclasses)
    def _safe_interp1d(self, x, y, x_new, kind="linear"):
        if not self.extrapolate_edge:
            # clamp queries within [x0, x1]
            x_new = np.clip(x_new, x[0], x[-1])
            fill = "extrapolate"
        else:
            # hold with end values (safer for control inputs)
            fill = (y[0], y[-1])

        f = interp1d(
            x, y, kind=kind, assume_sorted=True, bounds_error=False, fill_value=fill
        )
        return f(x_new)

    @staticmethod
    def _unwrap_angles(a: np.ndarray) -> np.ndarray:
        return np.unwrap(a)

    @staticmethod
    def _rewrap_angles(a: np.ndarray) -> np.ndarray:
        # Wrap to [-pi, pi)
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _interp_numeric(
        self, spec, t: np.ndarray, y: np.ndarray, t_new: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate numeric fields (scalars/vectors/positions/etc).
        spec.interp can be: "linear" (default), "cubic", "pchip" (optional if SciPy)
        """
        t = np.asarray(t, np.float64)
        y = np.asarray(y, np.float64)
        t_new = np.asarray(t_new, np.float64)

        if len(t) < 2:
            # degenerate: repeat the only sample
            return np.repeat(y[:1], repeats=len(t_new), axis=0)

        method = (getattr(spec, "interp", None) or "linear").lower()

        # Always safe + fast
        if method == "linear":
            return _interp_piecewise_linear(t, y, t_new)

        # Optional SciPy-based methods
        try:
            if method == "cubic":
                from scipy.interpolate import CubicSpline

                cs = CubicSpline(t, y, axis=0, bc_type="natural")
                return cs(t_new)

            if method == "pchip":
                from scipy.interpolate import PchipInterpolator

                pi = PchipInterpolator(t, y, axis=0)
                return pi(t_new)

        except Exception:
            # If SciPy isn't installed or method fails, fall back to linear
            return _interp_piecewise_linear(t, y, t_new)

        # Unknown method -> linear fallback (or raise if you prefer)
        return _interp_piecewise_linear(t, y, t_new)

    def _interp_field_by_kind(
        self, spec, t: np.ndarray, states: np.ndarray, t_grid: np.ndarray
    ) -> np.ndarray:
        cols = spec.cols
        x = states[:, cols]  # (N,dim)
        kind = spec.kind

        if kind in ("scalar", "vector", "delta_pose", "accel3", "tcp_pose"):
            # your existing interpolation (linear/cubic/etc) can stay here
            return self._interp_numeric(spec, t, x, t_grid)

        if kind == "angle":
            # keep your wrap-safe angle interpolation if you already have it
            return interp_angles(spec, t, x, t_grid)

        if kind == "quat4":
            if x.shape[1] != 4:
                raise ValueError(
                    f"{spec.name}: quat4 kind requires 4 columns, got {x.shape[1]}"
                )
            return _interp_quat4(t, x, t_grid)

        if kind == "tcp_pose":
            # assume [pos3, quat4]
            if x.shape[1] != 7:
                raise ValueError(
                    f"{spec.name}: pose7 kind requires 7 columns, got {x.shape[1]}"
                )
            pos = x[:, :3]
            quat = x[:, 3:7]
            pos_new = _interp_piecewise_linear(t, pos, t_grid)
            quat_new = _interp_quat4(t, quat, t_grid)
            return np.concatenate([pos_new, quat_new], axis=1)

        if kind == "angle_vec":
            return interp_angle_vec(t, x, t_grid)

        raise ValueError(f"Unknown field kind: {kind}")

    # Optional: convenience to build a schema from legacy args
    def _schema_from_legacy(
        self,
        states: np.ndarray,
        vector_cols: Optional[Tuple[int, ...]],
        angle_cols: Optional[Tuple[int, ...]],
    ) -> Dict[str, FieldSpec]:
        schema: Dict[str, FieldSpec] = {
            "position": FieldSpec(
                kind="position3", cols=(0, 1, 2), interp=self.interp_kind
            )
        }
        if vector_cols:
            schema["velocity"] = FieldSpec(
                kind="vector", cols=tuple(vector_cols), interp=self.interp_kind
            )
        if angle_cols:
            # support one angle; extend as needed
            schema["angle"] = FieldSpec(
                kind="angle", cols=(angle_cols[0],), interp=self.interp_kind
            )
        return schema

    def resample(
        self,
        t,
        states,
        dt=None,
        vector_cols=None,
        angle_cols=None,
        schema: Optional[Dict[str, FieldSpec]] = None,
    ):
        raise NotImplementedError


class AbsoluteTimeGrid(ResamplingBase):
    def __init__(self, config):
        super().__init__(config)

    def resample(self, t, states, dt=None, vector_cols=None, angle_cols=None):
        """
        Resample a trajectory to a fixed number of points or fixed time step.
        Args:
            t (np.ndarray): Original timestamps, shape (M,).
            states (np.ndarray): Original states, shape (M, D).
            T (float, optional): Total duration to resample to. If None, uses original duration.
            N (int, optional): Number of points to resample to if T is None. Defaults to 4000.
            dt (float, optional): Time step to resample to if T is None. If provided, overrides N.
            kind (str, optional): Interpolation method. Defaults to 'linear'.
            fill (str, optional): Fill method for out-of-bounds values. Defaults to "edge".
        """
        out_dict = {}
        t = np.asarray(t).astype(float)
        states = np.asarray(states).astype(float)
        assert (
            t.ndim == 1 and states.ndim == 2 and states.shape[0] == t.shape[0]
        ), "shapes are not aligned"

        # Shift to start at 0 for convenience (
        t0 = t[0]
        t_rel = t - t0
        T = t_rel[-1]
        K = self.K
        assert T > 0.0, "non-positive duration"

        if dt is not None and self.K is not None:
            # keep dt authority; recompute N so end aligns exactly with T
            K = int(np.floor(T / dt)) + 1
        if dt is None:
            dt = T / (self.K - 1)
        else:
            K = int(np.floor(T / dt)) + 1

        self.K = K
        t_new = np.linspace(0, T, K)
        # tq = np.clip(t_new, 0.0, T) if extrapolate_edge else t_new
        D = states.shape[1]
        states_new = np.zeros((K, D), dtype=states.dtype)
        # default column partitions
        used = np.zeros(D, dtype=bool)
        angle_cols = [] if angle_cols is None else list(angle_cols)
        used[angle_cols] = True
        if vector_cols is None:
            vector_cols = np.where(~used)[0].tolist()

        if len(vector_cols):
            f_vec = interp1d(
                t_rel,
                states[:, vector_cols],
                kind=self.interp_kind,
                axis=0,
                bounds_error=False,
                fill_value=(
                    (states[0, vector_cols], states[-1, vector_cols])
                    if self.extrapolate_edge
                    else np.nan
                ),
                assume_sorted=True,
            )
            states_new[:, vector_cols] = f_vec(t_new)

        for c in angle_cols:
            states_new[:, c] = interp_angles(
                t_rel, states[:, c], t_new=t_new, kind=self.interp_kind
            )

        t_new_abs = t0 + t_new
        out_dict["t_new"] = t_new_abs
        out_dict["X"] = states_new
        out_dict["T"] = T
        out_dict["s"] = dt
        return out_dict


class TimePhaseGrid(ResamplingBase):
    def __init__(self, config):
        super().__init__(config)
        self.angle_unit = config["angle_unit"]

    @staticmethod
    def _interpolate_angles(x_src, ang_src, x_tgt, unit="rad"):
        ang_src = np.asarray(ang_src, dtype=float)
        if unit == "deg":
            ang_src = np.deg2rad(ang_src)
        sin_src = np.sin(ang_src)
        cos_src = np.cos(ang_src)
        sin_tgt = _interp1(x_src, sin_src, x_tgt)
        cos_tgt = _interp1(x_src, cos_src, x_tgt)
        ang_tgt = np.arctan2(sin_tgt, cos_tgt)
        ang_tgt = _wrap_to_pi(ang_tgt)
        if unit == "deg":
            ang_tgt = np.rad2deg(ang_tgt)
        return ang_tgt

    @staticmethod
    def compute_time_phase(t: np.ndarray):
        t = np.asarray(t, dtype=float).reshape(-1)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("t must be a 1-D array with length >= 2")
        t0, t1 = float(t[0]), float(t[-1])
        T = t1 - t0
        if T <= 0:
            raise ValueError("Non-positive duration: ensure t is strictly increasing.")
        s = (t - t0) / T
        return s, T, t0, t1

    def resample(self, t, states, dt=None, vector_cols=None, angle_cols=None):
        out_dict = {}
        t = np.asarray(t, dtype=float).reshape(-1)
        X = np.asarray(states, dtype=float)
        K = self.K
        if X.ndim != 2 or X.shape[0] != t.shape[0]:
            raise ValueError("states must be (N, D) aligned with t")
        s_src, T, t0, t1 = self.compute_time_phase(t)
        s_grid = np.linspace(0.0, 1.0, int(K))
        angle_indices = set(angle_cols or [])
        D = X.shape[1]
        X_grid = np.empty((K, D), dtype=float)
        s_unique, idx = np.unique(s_src, return_index=True)
        X_unique = X[idx]
        for d in range(D):
            if d in angle_indices:
                X_grid[:, d] = self._interpolate_angles(
                    s_unique, X_unique[:, d], s_grid, unit=self.angle_unit
                )
            else:
                X_grid[:, d] = _interp1(s_unique, X_unique[:, d], s_grid)
        X_grid[0] = X[0]
        X_grid[-1] = X[-1]
        t_grid = t0 + s_grid * T
        out_dict["t_new"] = t_grid
        out_dict["X"] = X_grid
        out_dict["T"] = T
        out_dict["s"] = s_grid
        return out_dict


class ArcLengthGrid(ResamplingBase):
    def __init__(self, config):
        super().__init__(config)
        self.eps = config["eps"]

    def resample(
        self,
        t: np.ndarray,
        states: np.ndarray,
        dt: Optional[float] = None,
        vector_cols: Optional[Tuple[int, ...]] = None,
        angle_cols: Optional[Tuple[int, ...]] = None,
        schema: Optional[Dict[str, FieldSpec]] = None,
        compute_vel_from_pos_if_missing: bool = True,
    ):
        # --- choose a schema (explicit beats legacy)
        if schema is None:
            schema = self._schema_from_legacy(states, vector_cols, angle_cols)

        # find the position field
        pos_name = None
        for name, spec in schema.items():
            if spec.kind == "position3":
                pos_name = name
                break
        if pos_name is None:
            raise ValueError("Schema must include a 'position3' field (x,y,z columns).")

        pos = states[:, schema[pos_name].cols]
        K = self.K
        eps = self.eps

        # --- arc-length s from positions
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        ell = np.concatenate([[0.0], np.cumsum(seg)])
        L = max(ell[-1], eps)
        s_raw = ell / L
        s_grid = np.linspace(0.0, 1.0, K)

        # position splines on s
        pos_spl = [CubicSpline(s_raw, pos[:, k], bc_type="natural") for k in range(3)]
        X = np.stack([f(s_grid) for f in pos_spl], axis=1)  # (K,3)
        dX_ds = np.stack([f.derivative()(s_grid) for f in pos_spl], axis=1)

        # monotone t(s)
        t_of_s = PchipInterpolator(s_raw, t)
        t0, t1 = t[0], t[-1]
        t_grid = t_of_s(s_grid).astype(np.float64)
        # keep strictly inside original time span
        t_hi = np.nextafter(t1, -np.inf)
        t_lo = np.nextafter(t0, np.inf)
        t_grid = np.clip(t_grid, t_lo, t_hi)
        dt_ds = np.clip(t_of_s.derivative()(s_grid), eps, None)

        # chain-rule velocity (fallback)
        vel_chain = dX_ds / dt_ds[:, None]  # (K,3)

        # --- collect outputs per field
        fields: Dict[str, np.ndarray] = {}
        fields[pos_name] = X

        for name, spec in schema.items():
            if name == pos_name:
                continue
            if spec.kind == "position3":
                # if someone adds another position field, interpolate like base (time) or copy behaviour can be chosen
                fields[name] = self._interp_field_by_kind(spec, t, states, t_grid)
            else:
                fields[name] = self._interp_field_by_kind(spec, t, states, t_grid)

        # If a 'velocity' field exists and input is missing/NaN, fill with chain rule
        if "velocity" in schema and compute_vel_from_pos_if_missing:
            v_spec = schema["velocity"]
            if v_spec.kind in ("vector", "accel3") and len(v_spec.cols) == 3:
                v_raw = states[:, v_spec.cols]
                # treat "missing" as all-NaN or zero-length states array
                if np.isnan(v_raw).all():
                    fields["velocity"] = vel_chain

        # --- stacked output in schema order
        stacked = np.concatenate([fields[name] for name in schema.keys()], axis=1)

        out = {
            "X": stacked,  # concatenated states on the arc-length grid
            "t_new": stacked,  # kept for backward-compat if your downstream expects it
            "T": t_grid,  # time samples on the grid
            "dt": s_grid,  # normalized arc length
            "L": L,  # total length
            "s_raw": s_raw,
            "fields": fields,  # dict: each named field separately
        }
        return out


class TimeGrid(ResamplingBase):
    def __init__(self, config):
        super().__init__(config)
        # physics options
        self.enforce_tcp_vel_consistency = bool(
            config.get("enforce_tcp_vel_consistency", True)
        )

    def resample(
        self,
        t: np.ndarray,
        states: np.ndarray,
        dt: Optional[float] = None,
        vector_cols: Optional[Tuple[int, ...]] = None,
        angle_cols: Optional[Tuple[int, ...]] = None,
        schema: Optional[Dict[str, FieldSpec]] = None,
    ):
        t = np.asarray(t, dtype=np.float64)
        states = np.asarray(states, dtype=np.float64)
        assert t.ndim == 1 and states.ndim == 2 and states.shape[0] == t.shape[0]

        if schema is None:
            schema = self._schema_from_legacy(states, vector_cols, angle_cols)

        # ---- build time grid
        t0, t1 = float(t[0]), float(t[-1])
        T = t1 - t0
        assert T > 0.0, "non-positive duration"
        K = int(self.K)

        if dt is not None:
            K = int(np.floor(T / dt)) + 1
            dt_eff = dt
        else:
            dt_eff = T / (K - 1)

        # absolute time grid (avoid exact endpoints if desired)
        t_grid = np.linspace(t0, t1, K)
        t_hi = np.nextafter(t1, -np.inf)
        t_lo = np.nextafter(t0, np.inf)
        t_grid = np.clip(t_grid, t_lo, t_hi)

        # ---- interpolate fields
        fields: Dict[str, np.ndarray] = {}
        for name, spec in schema.items():
            fields[name] = self._interp_field_by_kind(spec, t, states, t_grid)

        # ---- stack output in schema order
        stacked = np.concatenate([fields[name] for name in schema.keys()], axis=1)

        return {
            "X": stacked,
            "t_new": t_grid,  # <-- fixed: time grid, not stacked
            "T": T,
            "dt": dt_eff,
            "fields": fields,
        }


def build_schema_from_layout(
    state_layout: Dict[str, Dict[str, Any]],
    kind_map: Optional[Dict[str, str]] = None,
    interp_kind: str = "linear",
    required: Optional[Dict[str, bool]] = None,
) -> Dict[str, FieldSpec]:
    if kind_map is None:
        kind_map = {}
    if required is None:
        required = {}

    schema: Dict[str, FieldSpec] = {}
    idx = 0

    def norm(s: str) -> str:
        return s.lower().replace(" ", "_")

    for name, meta in state_layout.items():
        if "shape" not in meta:
            raise ValueError(f"state_layout['{name}'] must contain a 'shape' key.")
        size = int(meta["shape"])
        if size <= 0:
            raise ValueError(f"state_layout['{name}']['shape'] must be > 0.")

        cols: Tuple[int, ...] = tuple(range(idx, idx + size))
        idx += size

        n = norm(name)

        # 1) explicit override wins
        if name in kind_map:
            kind = kind_map[name]
        elif n in kind_map:
            kind = kind_map[n]
        else:
            # ---- heuristics (add qpos)
            if "qpos" in n:
                # joint positions in radians -> wrap-aware interpolation
                kind = "angle_vec"

            elif "tcp_pos" in n:
                kind = "tcp_pose"

            elif "delta_pose" in n:
                kind = "delta_pose"

            elif "vel" in n:
                kind = "vector"

            elif "acc" in n:
                kind = "accel3" if size == 3 else "vector"

            elif any(k in n for k in ("psi_rad", "chi_rad", "phi_rad", "theta_rad")):
                # single angle (or could be angle_vec if size>1; your choice)
                kind = "angle" if size == 1 else "angle_vec"

            else:
                # IMPORTANT: default based on dimensionality
                kind = "scalar" if size == 1 else "vector"

        schema[name] = FieldSpec(
            kind=kind,
            cols=cols,
            interp=meta.get("interp", interp_kind),
            name=name,
            required=bool(required.get(name, False)),
        )

    return schema


def prepare_strictly_increasing_x(x, y):
    """
    x: shape (M,)
    y: shape (M, D) or (M,)
    Returns x_u, y_u with strictly increasing x (duplicates collapsed).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # sort by x (stable so equal x keep original order)
    order = np.argsort(x, kind="mergesort")
    x_s = x[order]
    y_s = y[order]

    # collapse duplicates by taking the mean (nan-safe)
    x_u, idx, counts = np.unique(x_s, return_index=True, return_counts=True)
    if np.any(counts > 1):
        # segment boundaries for reduceat
        y_s_nan = np.nan_to_num(y_s, nan=0.0)
        # sum over groups
        y_sum = np.add.reduceat(y_s_nan, idx, axis=0)
        # count non-nans in each group
        valid = ~np.isnan(y_s)
        n_sum = np.add.reduceat(valid.astype(np.int64), idx, axis=0)
        y_u = y_sum / np.clip(n_sum, 1, None)
    else:
        y_u = y_s[idx]

    # drop any residual NaNs or Infs in x
    good = np.isfinite(x_u)
    x_u = x_u[good]
    y_u = y_u[good]

    # also ensure strictly increasing (guard against numeric ties)
    keep = np.r_[True, np.diff(x_u) > 0]
    return x_u[keep], y_u[keep]


# Example:
if __name__ == "__main__":
    schema = {
        "position": FieldSpec(kind="position3", cols=(0, 1, 2)),
        # if you have measured velocities in cols 3:5; put NaNs if missing to auto-compute via chain rule
        "velocity": FieldSpec(kind="vector", cols=(3, 4, 5), interp="linear"),
        # accelerations, angles, etc.
        "accel": FieldSpec(kind="accel3", cols=(6, 7, 8), interp="linear"),
        "yaw": FieldSpec(kind="angle", cols=(9,), interp="linear"),
        "throttle": FieldSpec(kind="scalar", cols=(10,), interp="linear"),
    }
    state_layout = {
        "position": 3,
        "velocity": 3,
        "accel": 3,
        "yaw": 1,
        "throttle": 1,
    }

    kind_map = {"yaw": "angle"}  # override yaw to be an angle

    schema = build_schema_from_layout(state_layout, kind_map, interp_kind="linear")
    out = resampler.resample(t, states, schema=schema)
    T = out["T"]  # (K,)
    S = out["s"]  # (K,)
    X_all = out["X"]  # (K, sum(d))
    pos = out["fields"]["position"]
    vel = out["fields"]["velocity"]
