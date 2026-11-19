"""ctg/data/normalization.py"""

import json
import math

import pytest
import torch

from ctg.data.layout import FieldSpec, make_layout
from ctg.data.normalization import (FieldNormalizer, ImageNormalizer, LayoutNormalizer, PlanCodec,
                                    load_stats_file)

STATS = {"mean": [1.0, -2.0], "std": [2.0, 0.5], "min": [0.0, -4.0], "max": [2.0, 0.0]}


@pytest.mark.parametrize("norm", ["none", "standard", "minmax"])
def test_round_trip(norm):
    n = FieldNormalizer(FieldSpec(2, norm))
    n.set_stats(STATS)
    x = torch.randn(5, 2) * 3
    torch.testing.assert_close(n.unnormalize(n(x)), x)


def test_standard_and_minmax_values():
    std, mm = FieldNormalizer(FieldSpec(2, "standard")), FieldNormalizer(FieldSpec(2, "minmax"))
    std.set_stats(STATS)
    mm.set_stats(STATS)
    x = torch.tensor([[3.0, -1.5]])
    torch.testing.assert_close(std(x), torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(mm(torch.tensor([[0.0, 0.0], [2.0, -4.0]])), torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))


def test_none_is_identity():
    n = FieldNormalizer(FieldSpec(2))
    x = torch.randn(3, 2)
    assert torch.equal(n(x), x) and torch.equal(n.unnormalize(x), x)


def test_sincos_interleaves_and_wraps():
    n = FieldNormalizer(FieldSpec(2, "sincos"))
    x = torch.tensor([[0.5, 3.0]])
    y = n(x)
    assert y.shape == (1, 4)
    torch.testing.assert_close(y[0, :2], torch.tensor([math.sin(0.5), math.cos(0.5)]))
    torch.testing.assert_close(n.unnormalize(y), x)
    torch.testing.assert_close(n.unnormalize(n(x + 2 * math.pi)), x, atol=1e-5, rtol=0)


def test_image_normalizer_layouts_and_grayscale():
    img = ImageNormalizer()
    hwc = torch.full((2, 3, 8, 8, 3), 255, dtype=torch.uint8)
    out = img(hwc)
    assert out.shape == (2, 3, 3, 8, 8) and out.dtype == torch.float32 and out.max() == 1.0
    assert img(hwc.permute(0, 1, 4, 2, 3)).shape == (2, 3, 3, 8, 8)
    assert img(torch.rand(1, 2, 1, 8, 8)).shape == (1, 2, 3, 8, 8)
    with pytest.raises(ValueError, match="expected"):
        img(torch.zeros(2, 8, 8, 3))


def test_layout_normalizer_needs_stats_only_for_fitted_methods():
    layout = make_layout({"a": {"size": 2, "norm": "standard"}, "b": {"size": 2}})
    norm = LayoutNormalizer(layout, cameras=["cam"])
    with pytest.raises(KeyError, match="'a'"):
        norm.load_stats({})
    norm.load_stats({"a": STATS})
    x = {"a": torch.tensor([[3.0, -1.5]]), "b": torch.ones(1, 2),
         "cam": torch.zeros(1, 1, 4, 4, 3, dtype=torch.uint8)}
    out = norm(x)
    assert set(out) == {"a", "b", "cam"} and out["cam"].shape == (1, 1, 3, 4, 4)
    assert set(norm.unnormalize(out)) == {"a", "b"}


def test_stats_travel_with_the_state_dict():
    layout = make_layout({"a": {"size": 2, "norm": "minmax"}})
    src, dst = LayoutNormalizer(layout), LayoutNormalizer(layout)
    src.load_stats({"a": STATS})
    dst.load_state_dict(src.state_dict())
    assert torch.equal(dst.fields["a"].max, torch.tensor(STATS["max"]))


def test_plan_codec_round_trip():
    layout = make_layout({"a": {"size": 2, "norm": "standard"}, "b": {"size": 1, "norm": "sincos"}})
    norm = LayoutNormalizer(layout)
    norm.load_stats({"a": STATS})
    codec = PlanCodec(layout, norm)
    fields = {"a": torch.randn(2, 4, 2), "b": torch.rand(2, 4, 1)}
    x = codec.encode(fields)
    assert x.shape == (2, 4, 4)
    decoded = codec.decode(x)
    for k in fields:
        torch.testing.assert_close(decoded[k], fields[k])


def test_load_stats_file(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"a": STATS}))
    assert load_stats_file(tmp_path / "s.json") == {"a": STATS}
