"""ctg/physics/identification.py"""

import numpy as np
import pytest

from ctg.physics.identification import controller_samples, fit_first_order, predict, r2_score

from conftest import LENGTHS


def lag_samples(alpha, beta, n=2000, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    u = 0.1 * rng.uniform(-1, 1, (n, len(alpha)))
    dq = np.zeros_like(u)
    for t in range(1, n):
        dq[t] = alpha * dq[t - 1] + beta * u[t] + noise * rng.standard_normal(len(alpha))
    return {"dq": dq[1:], "dq_prev": dq[:-1], "u": u[1:]}


def test_fit_recovers_the_lag():
    alpha, beta = np.array([0.9, 0.35, 0.6]), np.array([0.02, 0.25, 0.14])
    fit = fit_first_order(lag_samples(alpha, beta))
    np.testing.assert_allclose(fit["alpha"], alpha, atol=1e-8)
    np.testing.assert_allclose(fit["beta"], beta, atol=1e-8)
    np.testing.assert_allclose(fit["r2"], 1.0)
    np.testing.assert_allclose(fit["gain"], beta / (1 - alpha))


def test_r2_with_noise_and_trivial_model():
    alpha, beta = np.array([0.5]), np.array([0.2])
    samples = lag_samples(alpha, beta, noise=1e-3)
    assert 0.5 < r2_score(samples, alpha, beta)[0] < 1.0
    np.testing.assert_allclose(predict(samples, np.zeros(1), np.zeros(1)), 0.0)
    assert r2_score(samples, np.zeros(1), np.zeros(1))[0] == pytest.approx(0.0, abs=0.05)


def test_samples_from_demonstrations(demo_h5):
    samples = controller_samples(demo_h5, [0, 1], n_joints=7)
    n = (LENGTHS[0] - 1) + (LENGTHS[1] - 1)  # one sample per step after the first
    assert {k: v.shape for k, v in samples.items()} == {"dq": (n, 7), "dq_prev": (n, 7), "u": (n, 7)}
    # synthetic actions are dq / 0.1 clipped to [-1, 1]: u equals the next joint change
    np.testing.assert_allclose(samples["u"][:5], samples["dq"][:5], atol=1e-6)
