"""scripts/compute_stats.py and scripts/identify_controller.py on the synthetic demonstrations."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_stats(demo_h5, tmp_path):
    out = tmp_path / "stats.json"
    load("compute_stats").main(Namespace(overrides=[f"data.dataset.path={demo_h5}"], out=str(out)))
    stats = json.loads(out.read_text())
    assert set(stats) == {"actions", "box_hole_pose", "qpos", "qvel", "tcp_pose", "peg_pose", "peg_half_size"}


def test_identify_controller(demo_h5, tmp_path):
    out = tmp_path / "controller.json"
    result = load("identify_controller").main(Namespace(overrides=[f"data.dataset.path={demo_h5}"], joints=7,
                                                        action_delta_scale=0.1, out=str(out)))
    assert json.loads(out.read_text()) == result
    assert len(result["alpha"]) == 7 and len(result["r2_val"]) == 7
