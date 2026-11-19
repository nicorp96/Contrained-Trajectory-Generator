"""ctg/physics/constraints.py"""

import math

import pytest
import torch

from ctg.physics.constraints import JointConstraints, masked_time_mean, pair_mask, wrap_angle

DT = 0.05


def consistent_plan(B=2, T=10, speed=0.5):
    """qpos moving at constant speed with matching qvel and delta actions."""
    t = torch.arange(T, dtype=torch.float32) * DT
    q = 0.1 + speed * t[None, :, None].expand(B, T, 9).clone()
    q[..., 3] -= 1.0  # joint 4 must stay negative
    q[..., 7:] = 0.02  # fingers (limit 0.04 m) hold still
    qd = torch.full((B, T, 9), speed)
    qd[..., 7:] = 0.0
    a = torch.full((B, T, 8), speed * DT / 0.1)
    return {"qpos": q, "qvel": qd, "actions": a}


def test_feasible_plan_has_no_violations():
    c = JointConstraints(qvel_max=[2.0] * 9)
    res = c.residuals(consistent_plan())
    assert set(res) == {"joint_limits", "velocity_limits", "acceleration_limits", "kinematic", "action_bounds",
                        "action_consistency"}
    for name, (viol, _) in res.items():
        assert viol.abs().max() < 1e-5, name
    assert c.energy(consistent_plan())["total"].abs().max() < 1e-8


def test_each_limit_is_detected():
    c = JointConstraints(qvel_max=[2.0] * 9)
    plan = consistent_plan()
    plan["qpos"][0, 3, 0] = 5.0  # beyond 2.8973
    plan["qvel"][0, 5, 1] = 3.0
    plan["actions"][1, 2, 2] = 1.5
    res = c.residuals(plan)
    assert res["joint_limits"][0][0, 3, 0] == pytest.approx(5.0 - 2.8973, abs=1e-5)
    assert res["velocity_limits"][0][0, 5, 1] == pytest.approx(1.0)
    assert res["action_bounds"][0][1, 2, 2] == pytest.approx(0.5)
    assert res["kinematic"][0][0, 2:4, 0].min() > 1.0
    m = c.metrics(plan)
    assert m["joint_limits_violation_rate"] > 0 and m["joint_limits_max"] == pytest.approx(5.0 - 2.8973, abs=1e-5)


def test_velocities_from_positions_without_qvel():
    c = JointConstraints(qvel_max=[0.1] * 9)
    plan = consistent_plan(speed=0.5)
    del plan["qvel"]
    res = c.residuals(plan)
    assert res["velocity_limits"][0].shape == (2, 9, 9)
    assert res["velocity_limits"][0].max() == pytest.approx(0.4, abs=1e-4)
    assert res["acceleration_limits"][0].shape == (2, 8, 9)
    assert "kinematic" not in res


def test_actions_only_plan_still_gets_action_bounds():
    res = JointConstraints().residuals({"actions": torch.full((1, 4, 8), 2.0)})
    assert set(res) == {"action_bounds"}
    assert JointConstraints().energy({"actions": torch.zeros(1, 4, 8)}, )["total"].shape == (1,)


def test_energy_without_weighted_terms_is_zero():
    c = JointConstraints(weights={"action_bounds": 0.0})
    e = c.energy({"actions": torch.full((3, 4, 8), 2.0)})
    assert torch.equal(e["total"], torch.zeros(3)) and e["action_bounds"].min() > 0


def test_masked_steps_are_ignored():
    c = JointConstraints()
    plan = consistent_plan(B=1, T=6)
    plan["qpos"][0, 4:, 0] = 9.0
    mask = torch.tensor([[True] * 4 + [False] * 2])
    assert c.energy(plan, mask)["joint_limits"].item() == 0.0
    assert c.metrics(plan, mask)["joint_limits_violation_rate"] == 0.0
    assert c.energy(plan)["joint_limits"].item() > 0


def test_non_uniform_time_stamps():
    """Resampled plans: the kinematic residual uses the time stamps, not dt."""
    c = JointConstraints(qvel_max=[2.0] * 9)
    time = torch.tensor([[0.0, 0.1, 0.4, 0.5, 1.0]])
    q = 0.1 + 0.5 * time[..., None].expand(1, 5, 9).clone()
    q[..., 3] -= 1.0
    plan = {"qpos": q, "qvel": torch.full((1, 5, 9), 0.5)}
    assert c.residuals(plan, time=time)["kinematic"][0].max() < 1e-6
    assert c.residuals(plan)["kinematic"][0].max() > 1e-3  # wrong with the fixed dt
    del plan["qvel"]
    assert c.residuals(plan, time=time)["acceleration_limits"][0].max() < 1e-4


def test_wrapped_angles():
    c = JointConstraints(wrap_angles=True)
    q = torch.zeros(1, 3, 9)
    q[..., 3] = -1.0
    q[0, 1, 0] = 2 * math.pi  # same joint angle as 0
    q[0, 2, 0] = 0.1
    res = c.residuals({"qpos": q})
    assert res["joint_limits"][0].max() < 1e-6
    assert res["velocity_limits"][0].max() < 1e-4  # 2pi jumps are no motion
    projected = c.project({"qpos": q})["qpos"]
    assert projected[0, 1, 0].abs() < 1e-6


def test_project_clamps_into_limits():
    c = JointConstraints()
    plan = {"qpos": torch.full((1, 2, 9), 10.0), "qvel": torch.full((1, 2, 9), -10.0),
            "actions": torch.full((1, 2, 8), 3.0), "other": torch.ones(1)}
    out = c.project(plan)
    assert torch.equal(out["qpos"][0, 0], c.q_upper) and torch.equal(out["qvel"][0, 0], -c.qvel_max)
    assert out["actions"].max() == 1.0 and out["other"] is plan["other"]
    assert c.residuals(out)["joint_limits"][0].max() == 0


def test_energy_is_differentiable():
    plan = consistent_plan()
    plan["qpos"] = plan["qpos"].clone().requires_grad_(True)
    plan["qpos"].data[0, 0, 0] = 5.0
    JointConstraints().energy(plan)["total"].sum().backward()
    assert plan["qpos"].grad[0, 0, 0] > 0


def test_helpers():
    torch.testing.assert_close(wrap_angle(torch.tensor([3 * math.pi / 2])), torch.tensor([-math.pi / 2]))
    e = torch.tensor([[1.0, 3.0, 100.0]])
    assert masked_time_mean(e, None).item() == pytest.approx(104 / 3)
    assert masked_time_mean(e, torch.tensor([[True, True, False]])).item() == 2.0
    assert pair_mask(None) is None
    assert pair_mask(torch.tensor([[True, True, True, False]]), 2).tolist() == [[True, False]]


ALPHA, BETA = [0.5] * 7, [0.2] * 7


def lag_plan(T=12, seed=0):
    """Joint motion generated by the first-order controller model from random actions."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(1, T, 8, generator=g) * 2 - 1
    q = torch.zeros(1, T, 9)
    q[..., 3] = -1.0
    dq_prev = torch.zeros(7)
    for t in range(T - 1):
        dq = 0.5 * dq_prev + 0.2 * 0.1 * a[0, t, :7]
        q[0, t + 1, :7] = q[0, t, :7] + dq
        dq_prev = dq
    return {"qpos": q, "actions": a}


def test_controller_model_explains_lagged_motion():
    plan = lag_plan()
    with_model = JointConstraints(controller_alpha=ALPHA, controller_beta=BETA).residuals(plan)
    viol, mask = with_model["action_consistency"]
    assert viol.shape == (1, 10, 7) and mask is None and viol.max() < 1e-6
    without = JointConstraints().residuals(plan)["action_consistency"][0]
    assert without.max() > 1e-2  # dq = 0.1 a does not hold under the lag


def test_controller_model_mask_and_gradient():
    plan = lag_plan(T=6)
    plan["actions"] = plan["actions"].clone().requires_grad_(True)
    c = JointConstraints(controller_alpha=ALPHA, controller_beta=BETA, weights={"action_consistency": 1.0})
    mask = torch.tensor([[True] * 4 + [False] * 2])
    assert c.residuals(plan, mask)["action_consistency"][1].tolist() == [[True, True, False, False]]
    plan["qpos"] = plan["qpos"] + 0.01 * torch.arange(6).view(1, 6, 1)
    c.energy(plan)["total"].sum().backward()
    assert plan["actions"].grad[0, 1:5, :7].abs().sum() > 0  # the energy moves the actions too


def test_action_consistency_skipped_for_resampled_plans():
    plan = lag_plan(T=5)
    c = JointConstraints(controller_alpha=ALPHA, controller_beta=BETA)
    uniform = torch.arange(5.0).unsqueeze(0) * DT
    assert "action_consistency" in c.residuals(plan, time=uniform)
    assert "action_consistency" not in c.residuals(plan, time=torch.tensor([[0.0, 0.1, 0.4, 0.5, 1.0]]))
    short = {k: v[:, :2] for k, v in plan.items()}
    assert "action_consistency" not in c.residuals(short)


@pytest.mark.parametrize("kw", [{"controller_alpha": ALPHA}, {"controller_alpha": ALPHA, "controller_beta": [0.2]}])
def test_controller_coefficients_validated(kw):
    with pytest.raises(ValueError, match="controller"):
        JointConstraints(**kw)
