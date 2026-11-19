"""Every experiment composes and builds networks whose widths fit the data layout."""

import pytest
import torch
from hydra.utils import instantiate

from ctg.config import CONFIG_DIR, compose_config
from ctg.data.layout import DataSpec, total_width
from ctg.training.trainer import build_components

EXPERIMENTS = sorted(p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml"))
IMAGE = 32  # small camera images keep the ResNet forward pass cheap


def test_experiments_found():
    assert {"e7b", "e8", "e9", "e9_inpainting", "e9_resampled_time", "e9_resampled_arclength"} <= set(EXPERIMENTS)


@pytest.fixture(params=EXPERIMENTS)
def cfg(request):
    return compose_config([f"experiment={request.param}"])


def test_run_name_matches_the_file(cfg, request):
    assert cfg.name == request.node.callspec.params["cfg"]


def test_components_fit_the_layout(cfg):
    spec = DataSpec.from_config(cfg.data.dataset)
    parts = build_components(cfg, spec, pretrained_encoders=False)
    model, encoders = parts["model"].eval(), parts["encoders"].eval()
    B, T = 1, spec.plan_len
    history = {k: torch.zeros(B, spec.history_len, s.width) for k, s in spec.history_layout.items()}
    history.update({cam: torch.rand(B, spec.history_len, 3, IMAGE, IMAGE) for cam in spec.cameras})
    encoded = encoders(history)
    hist = torch.cat([encoded[k] for k in list(spec.history_layout) + list(spec.cameras)], dim=-1)
    x = torch.zeros(B, T, total_width(spec.plan_layout))
    with torch.no_grad():
        out = model(x, torch.tensor([1]), torch.zeros(B, total_width(spec.goal_layout)), hist)
    assert out.shape == x.shape
    assert set(parts["normalizers"]) == {"plan", "goal", "history"}
    if cfg.physics.constraints is not None:
        assert parts["constraints"] is parts["sampler"].constraints


def test_inpainting_fields_are_planned(cfg):
    plan = set(cfg.data.dataset.plan)
    assert set(cfg.diffusion.inpainting.start_fields) <= plan
    assert set(cfg.diffusion.inpainting.goal_fields) <= plan


def test_qpos_is_not_sincos(cfg):
    """Panda joint limits stay within +-pi, sin/cos only compresses the signal."""
    for group in ("plan", "history"):
        spec = (cfg.data.dataset.get(group) or {}).get("qpos")
        assert spec is None or spec.norm != "sincos"


def test_default_is_e7b():
    default, e7b = compose_config(), compose_config(["experiment=e7b"])
    assert default == e7b


@pytest.mark.parametrize("name", ["e9_resampled_time", "e9_resampled_arclength"])
def test_resampler_follows_plan_len(name):
    cfg = compose_config([f"experiment={name}", "data.dataset.plan_len=32"])
    assert instantiate(cfg.data.dataset.resampler).num_points == 32


def test_paths_from_the_environment(monkeypatch):
    monkeypatch.setenv("CTG_DATA_DIR", "/data")
    assert compose_config().data.dataset.path.startswith("/data/mani_skill_data/")
