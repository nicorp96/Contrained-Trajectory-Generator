"""
Re-render a converted demonstration dataset with other cameras (actions unchanged).

  python scripts/replay_cameras.py --src <trajectory.state_dict+rgb.pd_joint_delta_pos.physx_cpu.h5> \
      --out-dir <dir> --resolution 256 --cameras top --num-procs 16
"""

import argparse

from ctg.envs.rerender import EXTRA_CAMERAS, rerender

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", required=True, help="converted dataset with env_states")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--cameras", nargs="*", default=["top"], choices=sorted(EXTRA_CAMERAS))
    parser.add_argument("--num-procs", type=int, default=16)
    parser.add_argument("--count", type=int, default=None, help="only the first n episodes")
    a = parser.parse_args()
    rerender(a.src, a.out_dir, a.resolution, a.cameras, a.num_procs, a.count)
