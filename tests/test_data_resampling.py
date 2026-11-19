"""ctg/data/resampling.py"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ctg.data.resampling import ArcLengthResampler, TimeResampler, interpolate, interpolate_field, slerp

T = np.linspace(0.0, 1.0, 11)


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip"])
def test_interpolate_reproduces_samples_and_lines(method):
    y = np.stack([2 * T + 1, -T], axis=1)
    np.testing.assert_allclose(interpolate(T, y, T, method), y, atol=1e-12)
    np.testing.assert_allclose(interpolate(T, y, np.array([0.25]), method), [[1.5, -0.25]], atol=1e-12)


def test_interpolate_edge_cases():
    with pytest.raises(ValueError, match="unknown method"):
        interpolate(T, T[:, None], T, "nearest")
    np.testing.assert_array_equal(interpolate(T[:1], np.array([[3.0]]), np.zeros(4)), np.full((4, 1), 3.0))
    # two samples: cubic falls back to linear
    np.testing.assert_allclose(interpolate(np.array([0.0, 1.0]), np.array([[0.0], [2.0]]), np.array([0.5]), "cubic"),
                               [[1.0]])


def test_slerp_matches_scipy_and_takes_shortest_path():
    r = Rotation.from_euler("z", [[0.0], [90.0]], degrees=True)
    q = r.as_quat()
    out = slerp(np.array([0.0, 1.0]), q, np.array([0.5]))
    expected = Rotation.from_euler("z", 45.0, degrees=True).as_quat()
    assert abs(abs(np.dot(out[0], expected)) - 1) < 1e-9
    flipped = np.stack([q[0], -q[1]])  # same rotation, opposite sign
    out2 = slerp(np.array([0.0, 1.0]), flipped, np.array([0.5]))
    assert abs(abs(np.dot(out2[0], expected)) - 1) < 1e-9
    near = slerp(np.array([0.0, 1.0]), np.stack([q[0], q[0]]), np.array([0.3]))  # identical: linear branch
    np.testing.assert_allclose(near[0], q[0])


def test_interpolate_field_kinds():
    q = np.tile([0.0, 0.0, 0.0, 1.0], (11, 1))
    pose = np.concatenate([np.stack([T, T, T], axis=1), q], axis=1)
    out = interpolate_field(T, pose, np.array([0.5]), "pose")
    np.testing.assert_allclose(out[0], [0.5, 0.5, 0.5, 0, 0, 0, 1], atol=1e-12)
    assert interpolate_field(T, q, np.array([0.5]), "quaternion").shape == (1, 4)
    with pytest.raises(ValueError, match="unknown field kind"):
        interpolate_field(T, q, T, "euler")


def test_time_resampler_uniform_grid_and_kinds():
    r = TimeResampler(5)
    t_new, out = r(T, {"x": T[:, None] * 3}, {"x": "vector"})
    np.testing.assert_allclose(t_new, np.linspace(0, 1, 5))
    np.testing.assert_allclose(out["x"][:, 0], 3 * t_new)


@pytest.mark.parametrize("t", [np.array([0.0]), np.array([0.0, 0.0, 1.0]), np.zeros((2, 2))])
def test_time_resampler_validates_time(t):
    with pytest.raises(ValueError, match="strictly increasing"):
        TimeResampler(3)(t, {"x": np.zeros((len(t), 1))})


def test_resampler_needs_two_points():
    with pytest.raises(ValueError, match="num_points"):
        TimeResampler(1)


def test_arc_length_puts_points_where_the_robot_moves():
    t = np.linspace(0, 2, 21)
    q = np.where(t < 1, t, 1.0)[:, None]  # moves during the first second, then rests
    t_new, out = ArcLengthResampler(5, path_field="q")(t, {"q": q})
    np.testing.assert_allclose(out["q"][:, 0], np.linspace(0, 1, 5), atol=1e-9)
    assert t_new[-1] == 2.0 and np.all(np.diff(t_new) >= 0)
    assert t_new[-2] <= 1.0  # no samples spent on the pause except the end point


def test_arc_length_without_motion_falls_back_to_time():
    t_new, _ = ArcLengthResampler(4, path_field="q")(T, {"q": np.zeros((11, 2))})
    np.testing.assert_allclose(t_new, np.linspace(0, 1, 4))
