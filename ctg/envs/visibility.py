"""
How well can each camera see the hole (and the peg head) during the demonstrations?

  python scripts/camera_visibility.py --h5 <trajectory.h5> --episodes 100 --out <dir>

Replays the demonstrations from their saved env states and, per camera and frame,
projects the hole entrance (centre + rim) and the peg head into the image. A point
counts as visible when it is inside the image and the rendered depth at its pixel
is not in front of it (nothing occludes it; for the peg head, its own end face may be
up to one peg radius in front). Frames are split into phases: before the grasp (peg on
the table), carry (peg lifted, head >= 5 cm from the hole channel), align (head < 5 cm,
still in front of the entrance) and inserted (head past the entrance).
Writes summary.json and example images of the align phase.
"""

import json
import os

import h5py
import numpy as np

INSERTION_DIST = 0.05  # m, peg head to the hole channel
LIFT = 0.01  # m above its resting height = peg is held (contacts are not restored with a state)
DEPTH_TOL = 0.01  # m, rendered depth may be this much in front of the point
RIM_POINTS = 8
PHASES = ("before_grasp", "carry", "align", "inserted")

# eye / target in world coordinates; the box sits at x in [-0.05, 0.05], y in [0.2, 0.4]
# and the peg enters its hole from -y (the default base camera looks along +y).
CANDIDATES = {
    "side_right": dict(eye=[0.40, 0.15, 0.20], target=[0.0, 0.22, 0.10], fov=1.0),
    "side_low": dict(eye=[0.45, 0.20, 0.12], target=[0.0, 0.20, 0.10], fov=1.0),
    "top": dict(eye=[0.0, 0.12, 0.65], target=[0.0, 0.22, 0.10], fov=1.0),
    "front_right": dict(eye=[0.30, -0.15, 0.30], target=[0.0, 0.20, 0.10], fov=1.0),
}


def project(points, intrinsic, extrinsic):
    """World points [N, 3] -> pixel coordinates [N, 2] (u, v) and camera depth [N]
    with OpenCV conventions (extrinsic: 3x4 world -> camera)."""
    cam = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (cam @ intrinsic.T)[:, :2] / z[:, None]
    return uv, z


def visible(uv, z, depth, tol=DEPTH_TOL):
    """Visibility of projected points given the rendered depth image [H, W] in metres
    (0 = nothing rendered, i.e. background); tol: scalar or one value per point."""
    h, w = depth.shape
    uv = np.where(np.isfinite(uv), uv, -1.0)  # points at z = 0 never land in the image
    u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.zeros(len(z), dtype=bool)
    d = depth[v[inside], u[inside]]
    out[inside] = (d == 0) | (d >= z[inside] - np.broadcast_to(tol, z.shape)[inside])
    return out


def rim_points(center, rotation, radius, n=RIM_POINTS):
    """n points on the hole rim; the hole axis is the local x axis of `rotation` [3, 3]."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    local = np.stack([np.zeros(n), radius * np.cos(a), radius * np.sin(a)], axis=1)
    return center + local @ rotation.T


def distance_to_hole(point, hole_p, hole_r, half_len):
    """Distance of `point` to the hole channel (the segment from the entrance at
    local x = -half_len to the centre at x = 0) and whether it is past the entrance."""
    local = hole_r.T @ (np.asarray(point) - hole_p)
    nearest = np.array([np.clip(local[0], -half_len, 0.0), 0.0, 0.0])
    return float(np.linalg.norm(local - nearest)), bool(local[0] >= -half_len)


def pixel_diameter(radius, fx, z):
    return 2 * radius * fx / z if z > 0 else 0.0


def phase(lifted, dist, past_entrance):
    if not lifted:
        return "before_grasp"
    if dist >= INSERTION_DIST:
        return "carry"
    return "inserted" if past_entrance else "align"


def summarize(records):
    """records: dicts with camera, phase, hole_center, rim_fraction, peg_head, hole_px."""
    out = {}
    for cam in sorted({r["camera"] for r in records}):
        out[cam] = {}
        for ph in PHASES:
            rs = [r for r in records if r["camera"] == cam and r["phase"] == ph]
            if not rs:
                continue
            centre = np.array([r["hole_center"] for r in rs])
            head = np.array([r["peg_head"] for r in rs])
            out[cam][ph] = {
                "frames": len(rs),
                "hole_center_visible": float(centre.mean()),
                "hole_rim_visible": float(np.mean([r["rim_fraction"] for r in rs])),
                "peg_head_visible": float(head.mean()),
                "hole_and_peg_head_visible": float((centre & head).mean()),
                "hole_px_median": float(np.median([r["hole_px"] for r in rs])),
            }
    return out


def quat_to_matrix(q):
    """wxyz quaternion -> rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def frame_records(env, obs, cameras):
    """Visibility records for the current env state, one per camera."""
    base = env.unwrapped
    hole = base.box_hole_pose
    hole_p, hole_r = hole.p[0].cpu().numpy(), quat_to_matrix(hole.q[0].cpu().numpy())
    half_len = float(base.peg_half_sizes[0, 0])
    radius = float(base.box_hole_radii[0])
    entrance = hole_p + hole_r @ np.array([-half_len, 0.0, 0.0])
    head = base.peg_head_pose.p[0].cpu().numpy()  # (peg_head_pos ignores the peg rotation)
    lifted = float(base.peg.pose.p[0, 2]) > float(base.peg_half_sizes[0, 2]) + LIFT
    ph = phase(lifted, *distance_to_hole(head, hole_p, hole_r, half_len))
    points = np.vstack([entrance[None], head[None], rim_points(entrance, hole_r, radius)])
    tol = np.full(len(points), DEPTH_TOL)
    tol[1] += float(base.peg_half_sizes[0, 1])  # the head point lies inside the peg's end face

    records = []
    for cam in cameras:
        param = obs["sensor_param"][cam]
        intrinsic = param["intrinsic_cv"][0].cpu().numpy()
        extrinsic = param["extrinsic_cv"][0].cpu().numpy()
        depth = obs["sensor_data"][cam]["depth"][0, ..., 0].cpu().numpy().astype(np.float64) / 1000.0
        uv, z = project(points, intrinsic, extrinsic)
        vis = visible(uv, z, depth, tol)
        records.append({
            "camera": cam, "phase": ph,
            "hole_center": bool(vis[0]), "peg_head": bool(vis[1]),
            "rim_fraction": float(vis[2:].mean()),
            "hole_px": pixel_diameter(radius, intrinsic[0, 0], z[0]),
        })
    return records


def measure_visibility(h5_path: str, out_dir: str, episodes: int = 100, every: int = 2, resolution: int = 256,
                       examples: int = 4) -> dict:
    """Visibility statistics of all default and candidate cameras over the first `episodes`
    successful demonstrations (every `every`-th frame); writes summary.json and example
    images of the align phase to out_dir and returns the summary."""
    import gymnasium as gym
    import imageio.v3 as iio
    from mani_skill.trajectory import utils as trajectory_utils

    import ctg.envs.peg_insertion  # noqa: F401  (registers PegInsertionSide-Extended)

    with open(h5_path.replace(".h5", ".json")) as f:
        meta = json.load(f)
    demos = [e for e in meta["episodes"] if e.get("success", True)][:episodes]

    extra = [dict(uid=k, width=resolution, height=resolution, **v) for k, v in CANDIDATES.items()]
    env = gym.make(
        "PegInsertionSide-Extended", obs_mode="rgb+depth", control_mode="pd_joint_pos",
        sensor_configs=dict(width=resolution, height=resolution),
        extra_cameras=extra, num_envs=1, sim_backend="physx_cpu",
    )
    cameras = ["base_camera", "hand_camera", *CANDIDATES]
    os.makedirs(out_dir, exist_ok=True)

    records, final_success, saved = [], [], 0
    with h5py.File(h5_path, "r") as h5:
        for i, ep in enumerate(demos):
            env.reset(seed=ep["episode_seed"])
            states = trajectory_utils.dict_to_list_of_dicts(h5[f"traj_{ep['episode_id']}"]["env_states"])
            saved_example = False
            for t in range(0, len(states), every):
                env.unwrapped.set_state_dict(states[t])
                obs = env.unwrapped.get_obs()
                frame = frame_records(env, obs, cameras)
                records += [dict(r, episode=ep["episode_id"], t=t) for r in frame]
                if saved < examples and not saved_example and frame[0]["phase"] == "align":
                    for cam in cameras:
                        rgb = obs["sensor_data"][cam]["rgb"][0].cpu().numpy()
                        iio.imwrite(os.path.join(out_dir, f"ep{ep['episode_id']}_t{t}_{cam}.png"), rgb)
                    saved_example, saved = True, saved + 1
            env.unwrapped.set_state_dict(states[-1])
            final_success.append(bool(env.unwrapped.evaluate()["success"][0]))
            print(f"episode {i + 1}/{len(demos)}: {len(states)} states, final success {final_success[-1]}")
    env.close()

    summary = {
        "episodes": len(demos), "every": every, "resolution": resolution,
        "final_state_success_rate": float(np.mean(final_success)),
        "cameras": {c: CANDIDATES.get(c, "default") for c in cameras},
        "visibility": summarize(records),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
