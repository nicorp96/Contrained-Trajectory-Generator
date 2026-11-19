"""
Normalization statistics of a data option, from its training episodes only.

  python scripts/compute_stats.py data=peg_state          # writes the data option's stats path
  python scripts/compute_stats.py experiment=e8 --out stats.json

Statistics cover every plan, goal and history field of the dataset (not the cameras),
using the same seeded episode split as training.
"""

import argparse
import json
from pathlib import Path

from hydra.utils import instantiate

from ctg.config import compose_config
from ctg.data.stats import compute_stats


def main(args):
    cfg = compose_config(args.overrides)
    dataset = instantiate(cfg.data.dataset)
    keys = list(dataset.plan_layout) + list(dataset.goal_layout) + list(dataset.history_layout)
    stats = compute_stats(dataset, keys, cfg.data.val_ratio, cfg.data.split_seed)
    out = Path(args.out or cfg.data.stats)
    out.write_text(json.dumps(stats, indent=2))
    print(f"saved {out}: {', '.join(stats)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. data=peg_image")
    parser.add_argument("--out", default=None, help="default: the data option's stats path")
    main(parser.parse_args())
