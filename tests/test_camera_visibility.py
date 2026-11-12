"""Geometry of environment/camera_visibility.py on synthetic cameras."""

import json

import numpy as np
import pytest

from environment.camera_visibility import (
    distance_to_hole, phase, pixel_diameter, project, quat_to_matrix, rim_points, summarize, visible,
)

# camera at the origin looking along +z (OpenCV), 100x100 image, fx = fy = 50
K = np.array([[50.0, 0, 50], [0, 50.0, 50], [0, 0, 1]])
E = np.hstack([np.eye(3), np.zeros((3, 1))])


def test_project_centre_and_offset():
    uv, z = project(np.array([[0, 0, 2.0], [1, 0, 2.0]]), K, E)
    np.testing.assert_allclose(uv, [[50, 50], [75, 50]])
    np.testing.assert_allclose(z, [2, 2])


def test_project_applies_extrinsic_translation():
    shifted = np.hstack([np.eye(3), [[0], [0], [1.0]]])  # world origin 1 m in front of the camera
    _, z = project(np.zeros((1, 3)), K, shifted)
    assert z[0] == pytest.approx(1.0)


def test_visible_free_line_of_sight():
    depth = np.full((100, 100), 5.0)  # wall behind the point
    uv, z = project(np.array([[0, 0, 2.0]]), K, E)
    assert visible(uv, z, depth).tolist() == [True]


def test_visible_background_counts_as_free():
    uv, z = project(np.array([[0, 0, 2.0]]), K, E)
    assert visible(uv, z, np.zeros((100, 100))).tolist() == [True]


def test_occluded_point():
    depth = np.full((100, 100), 1.0)  # object 1 m in front of a point at 2 m
    uv, z = project(np.array([[0, 0, 2.0]]), K, E)
    assert visible(uv, z, depth).tolist() == [False]


def test_depth_tolerance():
    uv, z = project(np.array([[0, 0, 2.0]]), K, E)
    assert visible(uv, z, np.full((100, 100), 1.995)).tolist() == [True]  # surface of the point itself
    assert visible(uv, z, np.full((100, 100), 1.95)).tolist() == [False]


def test_per_point_tolerance():
    uv, z = project(np.array([[0, 0, 2.0], [0, 0, 2.0]]), K, E)
    depth = np.full((100, 100), 1.97)  # 3 cm in front of both points
    assert visible(uv, z, depth, np.array([0.01, 0.05])).tolist() == [False, True]


def test_outside_image_and_behind_camera():
    pts = np.array([[10, 0, 2.0], [0, 0, -2.0], [0, 0, 0.0], [0, 0, 2.0]])
    uv, z = project(pts, K, E)
    assert visible(uv, z, np.zeros((100, 100))).tolist() == [False, False, False, True]


def test_rim_points_lie_on_circle_orthogonal_to_axis():
    rot = quat_to_matrix([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])  # local x -> world y
    centre = np.array([0.1, 0.2, 0.3])
    pts = rim_points(centre, rot, 0.02, n=8)
    assert pts.shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(pts - centre, axis=1), 0.02)
    np.testing.assert_allclose(pts[:, 1], 0.2, atol=1e-12)  # all in the plane normal to the hole axis


def test_quat_identity_and_rotation():
    np.testing.assert_allclose(quat_to_matrix([1, 0, 0, 0]), np.eye(3))
    r = quat_to_matrix([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)


@pytest.mark.parametrize("point, expected, past", [
    ([-0.3, 0, 0], 0.2, False),      # in front of the entrance, on the axis
    ([-0.05, 0.03, 0], 0.03, True),  # inside the channel, off-axis
    ([0.04, 0, 0], 0.04, True),      # past the centre
    ([-0.1, 0, 0.0], 0.0, True),     # at the entrance
])
def test_distance_to_hole(point, expected, past):
    dist, is_past = distance_to_hole(np.array(point), np.zeros(3), np.eye(3), half_len=0.1)
    assert dist == pytest.approx(expected) and is_past == past


def test_distance_to_hole_rotated():
    rot = quat_to_matrix([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])  # hole axis along world y
    dist, past = distance_to_hole(np.array([1.0, 0.95, 2.0]), np.array([1.0, 1.0, 2.0]), rot, 0.1)
    assert dist == pytest.approx(0.0) and past


def test_pixel_diameter():
    assert pixel_diameter(0.02, 50, 1.0) == pytest.approx(2.0)
    assert pixel_diameter(0.02, 50, 0.0) == 0.0


@pytest.mark.parametrize("lifted, dist, past, expected", [
    (False, 0.01, True, "before_grasp"), (True, 0.2, False, "carry"), (True, 0.05, False, "carry"),
    (True, 0.049, False, "align"), (True, 0.0, True, "inserted"),
])
def test_phase(lifted, dist, past, expected):
    assert phase(lifted, dist, past) == expected


def test_summarize():
    rec = lambda cam, ph, c, h, rim, px: dict(camera=cam, phase=ph, hole_center=c, peg_head=h, rim_fraction=rim, hole_px=px)
    out = summarize([
        rec("a", "align", True, True, 1.0, 4), rec("a", "align", True, False, 0.5, 6),
        rec("a", "carry", False, True, 0.0, 2), rec("b", "align", False, False, 0.0, 3),
    ])
    ins = out["a"]["align"]
    assert ins["frames"] == 2
    assert ins["hole_center_visible"] == 1.0 and ins["peg_head_visible"] == 0.5
    assert ins["hole_and_peg_head_visible"] == 0.5
    assert ins["hole_rim_visible"] == 0.75 and ins["hole_px_median"] == 5.0
    assert "before_grasp" not in out["a"] and out["b"]["align"]["hole_center_visible"] == 0.0


# --- with the simulator -------------------------------------------------------

@pytest.fixture(scope="module")
def sim_env():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")
    import environment.extended_peg_insertion  # noqa: F401

    e = gym.make(
        "PegInsertionSide-Extended", obs_mode="rgb+depth", num_envs=1, sim_backend="physx_cpu",
        sensor_configs=dict(width=96, height=96),
        extra_cameras=[dict(uid="top", eye=[0.0, 0.12, 0.65], target=[0.0, 0.22, 0.10])],
    )
    yield e
    e.close()


def test_frame_records_at_reset(sim_env):
    from environment.camera_visibility import frame_records

    sim_env.reset(seed=3)
    recs = frame_records(sim_env, sim_env.unwrapped.get_obs(), ["base_camera", "hand_camera", "top"])
    assert [r["camera"] for r in recs] == ["base_camera", "hand_camera", "top"]
    assert {r["phase"] for r in recs} == {"before_grasp"}  # the peg lies on the table after a reset
    base = recs[0]
    assert base["hole_center"] and base["rim_fraction"] > 0.5  # the hole faces the base camera
    assert 1 < base["hole_px"] < 20


def test_main_on_one_episode(tmp_path):
    pytest.importorskip("mani_skill")
    from types import SimpleNamespace

    from environment.camera_visibility import main
    from global_parameters import ConfigGlobalP
    import os

    h5 = os.path.join(ConfigGlobalP().DATA_DIR, "mani_skill_data/demos/PegInsertionSide-v1/motionplanning/trajectory.h5")
    if not os.path.exists(h5):
        pytest.skip("demonstrations not downloaded")
    main(SimpleNamespace(h5=h5, episodes=1, every=10, resolution=64, examples=1, out=str(tmp_path)))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["final_state_success_rate"] == 1.0  # restored states reproduce the demo
    assert set(summary["visibility"]) == {"base_camera", "hand_camera", "side_right", "side_low", "top", "front_right"}
    assert any(p.suffix == ".png" for p in tmp_path.iterdir())
