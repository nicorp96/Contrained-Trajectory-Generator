"""Consistency of model widths with the dataset shapes in every config."""

from pathlib import Path

import pytest

from common.utils import load_config, normalized_width

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))

# feature size of one camera after the image encoder
ENCODER_DIMS = {
    "resnet18": 512,
    "facebook/dinov2-small": 384,
    "facebook/dinov2-base": 768,
}


def encoder_dim(spec):
    if spec.get("pool") == "spatial_softmax":
        return 2 * spec.get("num_keypoints", 32)
    key = spec.get("name") if spec["type"] == "dinov2" else spec["type"]
    if key not in ENCODER_DIMS:
        pytest.fail(f"unknown image encoder {key!r}: add its feature size to ENCODER_DIMS")
    return ENCODER_DIMS[key]


@pytest.fixture(params=CONFIGS, ids=lambda p: p.stem)
def config(request):
    return load_config(str(request.param))


def test_configs_found():
    assert CONFIGS, f"no configs in {CONFIG_DIR}"


def test_input_dim_matches_state_shapes(config):
    expected = sum(normalized_width(s) for s in config["dataset"]["state_shapes"].values())
    assert config["model"]["input_dim"] == expected


def test_history_shape_matches_obs_shapes(config):
    if not config["model"].get("history"):
        pytest.skip("model without history")
    obs = config["dataset"].get("obs_shapes") or {}
    encoders = config["model"].get("image_encoders") or {}
    expected = sum(normalized_width(s) for k, s in obs.items() if "camera" not in k)
    expected += sum(encoder_dim(encoders[k]) for k in obs if "camera" in k)
    assert config["model"]["history_shape"] == expected


def test_qpos_not_sincos(config):
    """Panda joint limits stay within +-pi, sincos only compresses the signal."""
    for group in ("state_shapes", "obs_shapes"):
        spec = (config["dataset"].get(group) or {}).get("qpos")
        if spec is not None:
            assert spec["method_norm"] != "sincos"


@pytest.mark.parametrize(
    "spec, width",
    [
        ({"shape": 9, "method_norm": "minmax"}, 9),
        ({"shape": 9, "method_norm": "standard"}, 9),
        ({"shape": 9, "method_norm": "sincos"}, 18),
        ({"shape": "7"}, 7),
    ],
)
def test_normalized_width(spec, width):
    assert normalized_width(spec) == width


def test_history_len_matches_dataset(config):
    """Models that flatten the history (UNetTrjCC) need its length."""
    if "history_len" not in config["model"]:
        pytest.skip("model without a fixed history length")
    assert config["model"]["history_len"] == config["dataset"]["sequence_hist"]
