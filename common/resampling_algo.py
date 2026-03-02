import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, Any
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator


# __all__ = [
#     "compute_time_phase",
#     "time_phase_resample",
#     "reconstruct_timestamps",
#     "interp_angles",
#     "resample_to_absolute_grid",
# ]
#
Kind = Literal[
    "position3", "vector", "scalar", "angle", "copy", "accel3", "rho", "mass", "mdot"
]


#
#
# def resample_with_speed(x, t, v=None, M=128, eps=1e-3):
#     # x: (N,3), t: (N,), v: (N,3) optional measured velocity
#     seg = np.linalg.norm(np.diff(x, axis=0), axis=1)
#     ell = np.concatenate([[0.0], np.cumsum(seg)])
#     L = max(ell[-1], eps)
#     s_raw = ell / L
#     s_grid = np.linspace(0.0, 1.0, M)
#
#     # position spline over s
#     spl = [CubicSpline(s_raw, x[:, k]) for k in range(3)]
#     X = np.stack([f(s_grid) for f in spl], axis=1)  # (M,3)
#
#     # speed on s-grid
#     if v is not None:
#         v_spl = [CubicSpline(s_raw, v[:, k]) for k in range(3)]
#         V = np.stack([f(s_grid) for f in v_spl], axis=1)  # (M,3)
#         vmag = np.clip(np.linalg.norm(V, axis=1), eps, None)
#     else:  # recompute from t(s) and X'(s)
#         t_of_s = interp1d(s_raw, t, kind="linear", assume_sorted=True)
#         t_grid = t_of_s(s_grid)
#         dt = np.diff(t_grid, prepend=t_grid[0])
#         dt[0] = dt[1]
#         ds = np.gradient(s_grid)
#         vmag = np.clip(L * ds / np.maximum(dt, eps), eps, None)
#
#     log_v = np.log(vmag)
#     return X, log_v  # or return V if you choose vector velocity
#
#
# def compute_time_phase(t: np.ndarray):
#     t = np.asarray(t, dtype=float).reshape(-1)
#     if t.ndim != 1 or t.size < 2:
#         raise ValueError("t must be a 1-D array with length >= 2")
#     t0, t1 = float(t[0]), float(t[-1])
#     T = t1 - t0
#     if T <= 0:
#         raise ValueError("Non-positive duration: ensure t is strictly increasing.")
#     s = (t - t0) / T
#     return s, T, t0, t1
#
#
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


#
# def _wrap_to_pi(a):
#     return (a + np.pi) % (2 * np.pi) - np.pi
#
#
# def _interpolate_angles(x_src, ang_src, x_tgt, unit="rad"):
#     ang_src = np.asarray(ang_src, dtype=float)
#     if unit == "deg":
#         ang_src = np.deg2rad(ang_src)
#     sin_src = np.sin(ang_src)
#     cos_src = np.cos(ang_src)
#     sin_tgt = _interp1(x_src, sin_src, x_tgt)
#     cos_tgt = _interp1(x_src, cos_src, x_tgt)
#     ang_tgt = np.arctan2(sin_tgt, cos_tgt)
#     ang_tgt = _wrap_to_pi(ang_tgt)
#     if unit == "deg":
#         ang_tgt = np.rad2deg(ang_tgt)
#     return ang_tgt
#
#
# def time_phase_resample(
#     t: np.ndarray,
#     states: np.ndarray,
#     K: int = 256,
#     angle_indices=None,
#     angle_unit: str = "rad",
# ):
#     t = np.asarray(t, dtype=float).reshape(-1)
#     X = np.asarray(states, dtype=float)
#     if X.ndim != 2 or X.shape[0] != t.shape[0]:
#         raise ValueError("states must be (N, D) aligned with t")
#     s_src, T, t0, t1 = compute_time_phase(t)
#     s_grid = np.linspace(0.0, 1.0, int(K))
#     angle_indices = set(angle_indices or [])
#     D = X.shape[1]
#     X_grid = np.empty((K, D), dtype=float)
#     s_unique, idx = np.unique(s_src, return_index=True)
#     X_unique = X[idx]
#     for d in range(D):
#         if d in angle_indices:
#             X_grid[:, d] = _interpolate_angles(
#                 s_unique, X_unique[:, d], s_grid, unit=angle_unit
#             )
#         else:
#             X_grid[:, d] = _interp1(s_unique, X_unique[:, d], s_grid)
#     X_grid[0] = X[0]
#     X_grid[-1] = X[-1]
#     t_grid = t0 + s_grid * T
#     return s_grid, X_grid, t_grid, T
#
#
# def reconstruct_timestamps(T: float, K: int, t0: float = 0.0):
#     s_grid = np.linspace(0.0, 1.0, int(K))
#     t_grid = t0 + s_grid * float(T)
#     return s_grid, t_grid


def interp_angles(t, a, t_new, kind="linear"):
    a_unw = np.unwrap(a)
    f = interp1d(
        t, a_unw, kind=kind, bounds_error=False, fill_value=(a_unw[0], a_unw[-1])
    )
    a_new = f(np.clip(t_new, t[0], t[-1]))
    return ((a_new + np.pi) % (2 * np.pi)) - np.pi


def resample_to_absolute_grid(
    t,  # (M,) original timestamps (seconds), strictly increasing
    states,  # (M, D) states
    dt=None,  # fixed Δt (seconds). If None, set N and we compute dt
    N=None,  # number of steps, inclusive of both ends. If None, inferred from dt
    vector_cols=None,  # list of int indices to interpolate as vectors (default: all non-angle/quat)
    angle_cols=None,  # list of int indices (radians), each treated independently
    interp_kind="linear",  # 'linear', 'cubic' for vectors/angles
    extrapolate_edge=True,  # if True, clamp outside original to edge values
):
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
    t = np.asarray(t).astype(float)
    states = np.asarray(states).astype(float)
    assert (
        t.ndim == 1 and states.ndim == 2 and states.shape[0] == t.shape[0]
    ), "shapes are not aligned"

    # Shift to start at 0 for convenience (
    t0 = t[0]
    t_rel = t - t0
    T = t_rel[-1]
    assert T > 0.0, "non-positive duration"

    if dt is not None and N is not None:
        # keep dt authority; recompute N so end aligns exactly with T
        N = int(np.floor(T / dt)) + 1
    if dt is None:
        dt = T / (N - 1)
    else:
        N = int(np.floor(T / dt)) + 1
    t_new = np.linspace(0, T, N)
    # tq = np.clip(t_new, 0.0, T) if extrapolate_edge else t_new
    D = states.shape[1]
    states_new = np.zeros((N, D), dtype=states.dtype)
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
            kind=interp_kind,
            axis=0,
            bounds_error=False,
            fill_value=(
                (states[0, vector_cols], states[-1, vector_cols])
                if extrapolate_edge
                else np.nan
            ),
            assume_sorted=True,
        )
        states_new[:, vector_cols] = f_vec(t_new)

    for c in angle_cols:
        states_new[:, c] = interp_angles(
            t_rel, states[:, c], t_new=t_new, kind=interp_kind
        )

    t_new_abs = t0 + t_new
    return t_new_abs, states_new, dt, T


def isa_density(h_m: np.ndarray) -> np.ndarray:
    """Simple ISA density model up to ~20km."""
    h = np.asarray(h_m, dtype=np.float64)

    T0 = 288.15
    p0 = 101325.0
    g = 9.80665
    R = 287.05287
    L = 0.0065
    h11 = 11000.0
    T11 = T0 - L * h11
    p11 = p0 * (T11 / T0) ** (g / (R * L))

    tro = h <= h11
    T = np.empty_like(h)
    p = np.empty_like(h)

    T[tro] = T0 - L * h[tro]
    p[tro] = p0 * (T[tro] / T0) ** (g / (R * L))

    if np.any(~tro):
        p[~tro] = p11 * np.exp(-g * (h[~tro] - h11) / (R * T11))
        T[~tro] = T11

    rho = p / (R * T)
    return rho


# TODO: does not work correctly yet
def integrate_mass_from_mdot(
    t: np.ndarray, mdot_fuel: np.ndarray, m0: float
) -> np.ndarray:
    """
    Enforce dm/dt = -mdot_fuel (mdot_fuel >= 0) with trapezoid integration.
    t: (K,), mdot_fuel: (K,)
    """
    t = np.asarray(t, dtype=np.float64)
    md = np.asarray(mdot_fuel, dtype=np.float64)

    K = len(t)
    m = np.empty(K, dtype=np.float64)
    m[0] = float(m0)
    dt = np.diff(t)

    for k in range(K - 1):
        m[k + 1] = m[k] - 0.5 * (md[k] + md[k + 1]) * dt[k]
    return m


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

    def _interp_field_by_kind(
        self,
        spec: FieldSpec,
        t: np.ndarray,
        states: np.ndarray,
        t_grid: np.ndarray,
    ) -> np.ndarray:
        cols = spec.cols
        kind = spec.kind

        if kind == "position3":
            Y = states[:, cols]  # (N,3)
            comps = [
                self._safe_interp1d(t, Y[:, i], t_grid, kind=spec.interp)
                for i in range(3)
            ]
            return np.stack(comps, axis=1)

        if kind in ("scalar", "mass", "mdot"):  # treat as scalar interpolation
            y = states[:, cols[0]]
            return self._safe_interp1d(t, y, t_grid, kind=spec.interp)[:, None]

        if kind == "rho":
            # We'll derive rho from altitude later; here return NaNs as placeholder
            return np.full((len(t_grid), 1), np.nan, dtype=np.float64)

        if kind == "vector":
            Y = states[:, cols]
            comps = [
                self._safe_interp1d(t, Y[:, i], t_grid, kind=spec.interp)
                for i in range(len(cols))
            ]
            return np.stack(comps, axis=1)
        if kind == "accel3":
            Y = states[:, cols]
            comps = [
                self._safe_interp1d(t, Y[:, i], t_grid, kind=spec.interp)
                for i in range(3)
            ]
            return np.stack(comps, axis=1)

        if kind == "scalar":
            y = states[:, cols[0]]
            return self._safe_interp1d(t, y, t_grid, kind=spec.interp)[:, None]

        if kind == "angle":
            y = states[:, cols[0]]
            y_u = self._unwrap_angles(y)
            y_new = self._safe_interp1d(t, y_u, t_grid, kind=spec.interp)
            return self._rewrap_angles(y_new)[:, None]

        if kind == "copy":
            return np.tile(states[0, cols], (len(t_grid), 1))

        raise ValueError(f"Unknown field kind for base interpolation: {kind}")

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
        self.mass_flow_is_fuel_burn_positive = bool(
            config.get("mass_flow_is_fuel_burn_positive", True)
        )
        self.derive_rho_from_altitude = bool(
            config.get("derive_rho_from_altitude", True)
        )
        self.enforce_mass_mdot_consistency = bool(
            config.get("enforce_mass_mdot_consistency", True)
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

        # ---- physics repair: rho from altitude
        if self.derive_rho_from_altitude and ("rho" in schema):
            # find a position3 field to get altitude; prefer "position" if present
            pos_name = None
            if "position" in schema and schema["position"].kind == "position3":
                pos_name = "position"
            else:
                for nm, sp in schema.items():
                    if sp.kind == "position3":
                        pos_name = nm
                        break
            if pos_name is None:
                raise ValueError(
                    "derive_rho_from_altitude=True requires a position3 field."
                )

            z = fields[pos_name][:, 2]  # altitude [m] assumed in 3rd component
            rho = isa_density(z)
            fields["rho"] = rho[:, None]

        # ---- physics repair: mass <-> mdot consistency
        if self.enforce_mass_mdot_consistency and (
            ("m_fuel" in schema) or ("d_m_fuel_cmd" in schema)
        ):
            have_m = "m_fuel" in schema
            have_mdot = "d_m_fuel_cmd" in schema

            # pull interpolated versions (may be NaN if missing/derived)
            m_itp = fields["m_fuel"][:, 0] if have_m else None
            mdot_itp = fields["d_m_fuel_cmd"][:, 0] if have_mdot else None

            # Decide which is primary:
            # Prefer mdot -> integrate mass (more stable) if mdot exists and not all NaN
            use_mdot = (
                have_mdot and (mdot_itp is not None) and (not np.isnan(mdot_itp).all())
            )
            use_m = have_m and (m_itp is not None) and (not np.isnan(m_itp).all())

            dt_vec = np.diff(t_grid)

            if use_mdot:
                # Ensure sign convention: mdot_fuel >= 0
                if self.mass_flow_is_fuel_burn_positive:
                    mdot_fuel = np.clip(mdot_itp, 0.0, None)
                else:
                    # if mdot stored as dm/dt (typically negative), convert to fuel burn positive
                    mdot_fuel = np.clip(-mdot_itp, 0.0, None)

                # choose initial mass:
                if use_m:
                    m0 = float(m_itp[0])
                else:
                    raise ValueError(
                        "To integrate mass from mdot you need mass at t0 (mass field present) or provide it externally."
                    )

                m_cons = integrate_mass_from_mdot(t_grid, mdot_fuel, m0=m0)

                if have_m:
                    fields["m_fuel"] = m_cons[:, None]
                if have_mdot:
                    # store back in original convention
                    if self.mass_flow_is_fuel_burn_positive:
                        fields["d_m_fuel_cmd"] = mdot_fuel[:, None]
                    else:
                        fields["d_m_fuel_cmd"] = (-mdot_fuel)[:, None]

            elif use_m and have_mdot:
                # derive mdot from mass by finite differences (noisy but workable)
                dm = np.diff(m_itp)
                mdot_fuel_mid = np.clip(-dm / dt_vec, 0.0, None)  # (K-1,)
                # make length K: pad last with last
                mdot_fuel = np.concatenate([mdot_fuel_mid, mdot_fuel_mid[-1:]], axis=0)

                if self.mass_flow_is_fuel_burn_positive:
                    fields["d_m_fuel_cmd"] = mdot_fuel[:, None]
                else:
                    fields["d_m_fuel_cmd"] = (-mdot_fuel)[:, None]

        # ---- stack output in schema order
        stacked = np.concatenate([fields[name] for name in schema.keys()], axis=1)

        return {
            "X": stacked,
            "t_new": t_grid,  # <-- fixed: time grid, not stacked
            "T": T,
            "dt": dt_eff,
            "fields": fields,
        }


#
# def build_schema_from_layout(state_layout, kind_map=None, interp_kind="linear"):
#     """
#     Build a schema dictionary (name → FieldSpec) given a dict of
#     {state_name: size}.
#
#     Parameters
#     ----------
#     state_layout : Dict[str, int]
#         e.g. {"position":3, "velocity":3, "yaw":1}
#     kind_map : Optional[Dict[str, str]]
#         Optional override mapping from state name to FieldSpec.kind
#         (e.g. {"yaw":"angle"}). If not given, defaults will be guessed.
#     interp_kind : str
#         Default interpolation kind ("linear" or "cubic")
#
#     Returns
#     -------
#     schema : Dict[str, FieldSpec]
#     """
#     if kind_map is None:
#         kind_map = {}
#
#     schema = {}
#     idx = 0
#     for name in state_layout.keys():
#         cols = tuple(range(idx, idx + state_layout[name]["shape"]))
#         idx += state_layout[name]["shape"]
#
#         # pick kind automatically if not in kind_map
#         if name in kind_map:
#             kind = kind_map[name]
#         elif "position" in name or name == "position":
#             kind = "position3"
#         elif "vel_ned" in name:
#             kind = "vector"
#         elif "acc_ned" in name:
#             kind = "vector"
#         elif "deg" in name or "yaw" in name or "heading" in name:
#             kind = "angle"
#         else:
#             kind = "scalar"
#
#         schema[name] = FieldSpec(kind=kind, cols=cols, interp=interp_kind, name=name)
#     return schema


def build_schema_from_layout(
    state_layout: Dict[str, Dict[str, Any]],
    kind_map: Optional[Dict[str, str]] = None,
    interp_kind: str = "linear",
    required: Optional[Dict[str, bool]] = None,
) -> Dict[str, FieldSpec]:
    """
    Build a schema dict (name -> FieldSpec) from a layout of:
        state_layout = { "name": {"shape": int, ...}, ... }

    Adds aircraft-aware defaults for:
      - mass: kind="mass"
      - mdot / mass_flow / fuel_flow: kind="mdot"
      - rho / density: kind="rho"  (typically derived later from altitude)

    Parameters
    ----------
    state_layout : Dict[str, Dict[str, Any]]
        Each entry must include {"shape": int}. Optional extra metadata allowed.
    kind_map : Optional[Dict[str, str]]
        Override mapping name -> kind
    interp_kind : str
        Default interpolation kind for fields ("linear", "cubic", "pchip" if you support it)
    required : Optional[Dict[str, bool]]
        Optional mapping name -> required flag

    Returns
    -------
    schema : Dict[str, FieldSpec]
    """
    if kind_map is None:
        kind_map = {}
    if required is None:
        required = {}

    schema: Dict[str, FieldSpec] = {}
    idx = 0

    # helper for robust name matching
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
            # 2) aircraft-aware heuristics
            if "position" in n:
                # if it isn't 3, fall back to vector
                kind = "position3" if size == 3 else "vector"

            elif "delta_pos" in n:
                # if it isn't 3, fall back to vector
                kind = "position3" if size == 3 else "vector"

            elif "vel" in n:
                kind = "vector"

            elif "acc" in n:
                # if you want "accel3" special handling, keep it
                kind = "accel3" if size == 3 else "vector"

            elif "m_fuel" in n:
                kind = "mass"

            elif "d_m_fuel_cmd" in n:
                kind = "mdot"

            elif "rho" in n:
                kind = "rho"

            elif any(k in n for k in ("psi_rad", "chi_rad", "phi_rad", "theta_rad")):
                # adjust list as your convention dictates
                kind = "angle"
            else:
                kind = "scalar"

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
