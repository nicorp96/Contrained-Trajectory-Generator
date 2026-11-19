"""
Identify the joint controller response (first-order lag per joint) from the training
episodes of a data option and print the coefficients for configs/physics/.

  python scripts/identify_controller.py                  # data=peg_state
  python scripts/identify_controller.py data=peg_image --out controller.json

See ctg/physics/identification.py for the model.
"""

import argparse
import json
from pathlib import Path

from hydra.utils import instantiate

from ctg.config import compose_config
from ctg.data.splits import split_episodes
from ctg.physics.identification import controller_samples, fit_first_order, r2_score


def main(args) -> dict:
    cfg = compose_config(args.overrides)
    dataset = instantiate(cfg.data.dataset)
    train, val = split_episodes(len(dataset), cfg.data.val_ratio, cfg.data.split_seed)
    ids = lambda idx: [dataset.episodes[i]["episode_id"] for i in idx]
    kwargs = dict(n_joints=args.joints, action_delta_scale=args.action_delta_scale, action_key=dataset.action_key)
    fit = fit_first_order(controller_samples(dataset.path, ids(train), **kwargs))
    result = {k: [round(float(v), 4) for v in fit[k]] for k in ("alpha", "beta", "gain")}
    result["r2_train"] = [round(float(v), 4) for v in fit["r2"]]
    if val:
        val_r2 = r2_score(controller_samples(dataset.path, ids(val), **kwargs), fit["alpha"], fit["beta"])
        result["r2_val"] = [round(float(v), 4) for v in val_r2]
    print(json.dumps(result, indent=1))
    print(f"controller_alpha: {result['alpha']}\ncontroller_beta: {result['beta']}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. data=peg_image")
    parser.add_argument("--joints", type=int, default=7, help="revolute joints (the gripper is not identified)")
    parser.add_argument("--action-delta-scale", type=float, default=0.1, help="rad per unit action")
    parser.add_argument("--out", default=None, help="also write the result as JSON")
    main(parser.parse_args())
