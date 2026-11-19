"""
How well can each camera see the hole (and the peg head) during the demonstrations?

  python scripts/camera_visibility.py --h5 <motionplanning/trajectory.h5> --episodes 100 --out <dir>
"""

import argparse
import json

from ctg.envs.visibility import measure_visibility

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--h5", required=True, help="state-only demo file (with env_states)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--every", type=int, default=2, help="use every n-th frame")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--examples", type=int, default=4, help="episodes with saved example images")
    a = parser.parse_args()
    summary = measure_visibility(a.h5, a.out, a.episodes, a.every, a.resolution, a.examples)
    print(json.dumps(summary["visibility"], indent=1))
