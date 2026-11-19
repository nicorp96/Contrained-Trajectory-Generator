"""ctg/data/layout.py"""

import pytest
import torch

from ctg.data.layout import DataSpec, FieldSpec, concat, make_layout, split, total_width


def test_field_spec_width():
    assert FieldSpec(3).width == 3
    assert FieldSpec(3, norm="sincos").width == 6


def test_unknown_normalization_rejected():
    with pytest.raises(ValueError, match="unknown normalization"):
        FieldSpec(3, norm="robust")


def test_make_layout_keeps_order_and_skips_removed_fields():
    layout = make_layout({"b": {"size": 2}, "gone": None, "a": {"size": 1, "norm": "minmax"}})
    assert list(layout) == ["b", "a"]
    assert layout["a"] == FieldSpec(1, "minmax")
    assert make_layout(None) == {}


def test_split_and_concat_are_inverse():
    layout = make_layout({"a": {"size": 2}, "b": {"size": 1, "norm": "sincos"}})
    x = torch.arange(8.0).reshape(2, 4)
    parts = split(x, layout)
    assert parts["a"].shape == (2, 2) and parts["b"].shape == (2, 2)
    assert torch.equal(concat(parts, layout), x)
    assert total_width(layout) == 4


def test_split_checks_width():
    with pytest.raises(ValueError, match="layout width"):
        split(torch.zeros(2, 5), make_layout({"a": {"size": 2}}))


def test_data_spec_from_config():
    spec = DataSpec.from_config({"plan": {"actions": {"size": 8}}, "goal": {"g": {"size": 7}},
                                 "history": {"qpos": {"size": 9}}, "cameras": ["base_camera"],
                                 "plan_len": 16, "history_len": 2})
    assert list(spec.plan_layout) == ["actions"] and spec.cameras == ("base_camera",)
    assert spec.plan_len == 16 and spec.history_len == 2
    assert spec.action_key == "actions" and spec.control_dt == 0.05
