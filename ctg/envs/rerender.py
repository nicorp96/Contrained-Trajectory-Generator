"""
Re-render a ManiSkill demo dataset with other cameras; the actions stay identical.

  python scripts/replay_cameras.py --src <trajectory.state_dict+rgb.pd_joint_delta_pos.physx_cpu.h5> \
      --out-dir <dir> --resolution 256 --cameras top --num-procs 16

Replays the saved env states of an already converted dataset in
PegInsertionSide-Extended with all cameras at the given resolution plus the selected
extra cameras. The control mode is kept, so the recorded actions are unchanged;
observations are re-simulated one step from each saved state and can differ from the
source by a few 1e-3 rad while the gripper is in contact. The camera setup
is written into the new dataset's json (env_info.env_kwargs), where the evaluation
reads it from (ctg.envs.evaluation.camera_kwargs).
"""

import json
import multiprocessing as mp
import os

ENV_ID = "PegInsertionSide-Extended"
OBS_MODE = "state_dict+rgb"

# chosen with ctg/envs/visibility.py: sees hole and peg head together in
# 80 % of the frames before insertion (base camera 0 %, hand camera 0 %)
EXTRA_CAMERAS = {
    "top": dict(uid="top_camera", eye=[0.0, 0.12, 0.65], target=[0.0, 0.22, 0.10], fov=1.0),
}


def camera_env_info(env_info, resolution, cameras):
    """env_info of the source dataset -> env_info for the re-rendered one."""
    kwargs = dict(env_info["env_kwargs"])
    kwargs["sensor_configs"] = {**kwargs.get("sensor_configs", {}), "width": resolution, "height": resolution}
    kwargs["extra_cameras"] = [dict(EXTRA_CAMERAS[c], width=resolution, height=resolution) for c in cameras]
    return {**env_info, "env_id": ENV_ID, "env_kwargs": kwargs}


def prepare(src_h5, out_dir, resolution, cameras):
    """Links the source states into out_dir with a json describing the new cameras;
    returns the h5 path to replay."""
    os.makedirs(out_dir, exist_ok=True)
    with open(src_h5.replace(".h5", ".json")) as f:
        meta = json.load(f)
    meta["env_info"] = camera_env_info(meta["env_info"], resolution, cameras)
    link = os.path.join(out_dir, "trajectory.h5")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(src_h5), link)
    with open(link.replace(".h5", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return link


def output_path(link):
    """File name the ManiSkill replay writes for `link`."""
    with open(link.replace(".h5", ".json")) as f:
        kwargs = json.load(f)["env_info"]["env_kwargs"]
    return link.replace(".h5", f".{OBS_MODE}.{kwargs['control_mode']}.{kwargs['sim_backend']}.h5")


def rerender(src: str, out_dir: str, resolution: int = 256, cameras=("top",), num_procs: int = 16,
             count=None) -> str:
    """Replay `src` with the new camera setup into out_dir; returns the new dataset path."""
    from mani_skill.trajectory import replay_trajectory

    import ctg.envs.peg_insertion  # noqa: F401  (registers PegInsertionSide-Extended)

    link = prepare(src, out_dir, resolution, cameras)
    # the replay's worker pool must inherit the registration above
    mp.set_start_method("fork", force=True)
    replay_trajectory.main(replay_trajectory.Args(
        traj_path=link, obs_mode=OBS_MODE, use_env_states=True, save_traj=True,
        num_envs=num_procs, count=count, sim_backend="physx_cpu",
    ))
    print(f"saved {output_path(link)}")
    return output_path(link)
