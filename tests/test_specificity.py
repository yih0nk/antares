"""Tests for antares.specificity with a toy servo instead of MuJoCo."""

from __future__ import annotations

import math

import pytest

from antares.specificity import summarize_cross

A, B = "lab/arm_a", "lab/arm_b"


def matrix(scores):
    return {"rows": [A, B], "cols": ["lock_a", "lock_b", "default"], "scores": scores}


LOCK_DATASETS = {"lock_a": A, "lock_b": B}


def test_summarize_cross_diagonal_wins():
    s = summarize_cross(matrix([[1.0, 2.5, 6.0], [3.0, 0.8, 2.7]]), LOCK_DATASETS)
    assert s["all_specific"] is True
    a, b = s["rows"]
    assert (a["own_lock"], a["best_other_lock"]) == ("lock_a", "lock_b")
    assert a["margin"] == pytest.approx(2.5)
    assert a["default_score"] == 6.0
    assert b["margin"] == pytest.approx(3.0 / 0.8)


def test_summarize_cross_foreign_lock_better_is_not_specific():
    s = summarize_cross(matrix([[1.0, 0.9, 6.0], [3.0, 0.8, 2.7]]), LOCK_DATASETS)
    assert s["rows"][0]["specific"] is False
    assert s["rows"][1]["specific"] is True
    assert s["all_specific"] is False


def test_summarize_cross_picks_best_of_several_own_locks():
    m = {"rows": [A], "cols": ["a1", "a2", "lock_b", "default"], "scores": [[1.5, 1.0, 2.0, 5.0]]}
    s = summarize_cross(m, {"a1": A, "a2": A, "lock_b": B})
    assert s["rows"][0]["own_lock"] == "a2"
    assert s["rows"][0]["margin"] == pytest.approx(2.0)


def test_summarize_cross_undecided_without_a_foreign_lock():
    m = {"rows": [A], "cols": ["lock_a", "default"], "scores": [[1.0, 5.0]]}
    s = summarize_cross(m, {"lock_a": A})
    assert s["rows"][0]["specific"] is None
    assert s["rows"][0]["margin"] is None
    assert s["all_specific"] is False


def test_summarize_cross_zero_own_score():
    s = summarize_cross(matrix([[0.0, 1.0, 2.0], [1.0, 0.5, 2.0]]), LOCK_DATASETS)
    assert math.isinf(s["rows"][0]["margin"])


# --------------------------------------------------------------------------
# identifiability
# --------------------------------------------------------------------------

import numpy as np  # noqa: E402

from antares.calibrate import SimParams, encode  # noqa: E402
from antares.data import Episode  # noqa: E402
from antares.specificity import CONTINUOUS, identifiability, jacobian, residuals  # noqa: E402

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TRUE = SimParams(kp=400.0, kv=4.0, damping=2.0, armature=0.1, frictionloss=0.05, latency_frames=2)


def flat_episodes(n=3, T=50):
    rng = np.random.default_rng(0)
    return [Episode(index=i, fps=30.0, joint_names=JOINTS,
                    state=rng.normal(size=(T, 6)), action=np.zeros((T, 6))) for i in range(n)]


def separable_simulate(weights):
    """sim = real + sum_k w_k (x_k - x*_k) * pattern_k, one joint per param.

    Patterns are orthogonal, so H is diagonal with H_kk = w_k^2 * T * n_eps
    (in radians -> degrees units) and nothing trades off.
    """
    x_true = encode(TRUE)

    def simulate(params, ep):
        x = encode(params)
        sim = ep.state.copy()
        for k, name in enumerate(CONTINUOUS):
            i = list(SimParams().to_dict()).index(name)
            sim[:, k] += np.radians(weights[k] * (x[i] - x_true[i]))
        return ep.state, sim

    return simulate


def test_residuals_match_overall_rms():
    eps = flat_episodes()
    sim = separable_simulate([1.0] * 5)
    p = TRUE.replace(kp=800.0)
    r = residuals(p, eps, sim)
    assert r.size == 3 * 50 * 5  # gripper excluded
    from antares.calibrate import score
    assert float(np.sqrt(np.mean(r**2))) == pytest.approx(score(p, eps, sim).overall_deg)


def test_jacobian_of_separable_model():
    eps = flat_episodes()
    w = [3.0, 1.0, 0.5, 2.0, 0.1]
    _, J = jacobian(TRUE, eps, separable_simulate(w))
    assert J.shape == (3 * 50 * 5, 5)
    # column k is w_k on joint k's entries and zero elsewhere
    for k in range(5):
        assert np.abs(J[:, k]).max() == pytest.approx(w[k], rel=1e-6)
        assert np.count_nonzero(np.abs(J[:, k]) > 1e-9) == 3 * 50


def test_identifiability_separable_has_no_tradeoff():
    eps = flat_episodes()
    w = [3.0, 1.0, 0.5, 2.0, 0.1]
    rep = identifiability(TRUE, eps, separable_simulate(w), delta_deg=0.1)
    assert rep.rms_deg == pytest.approx(0.0, abs=1e-9)
    n = rep.n_residuals
    for k, name in enumerate(CONTINUOUS):
        expected = 0.1 * np.sqrt(n / (w[k] ** 2 * 150))
        assert rep.tolerance_fixed[name] == pytest.approx(expected, rel=1e-6)
        assert rep.tradeoff[name] == pytest.approx(1.0, rel=1e-6)
    assert rep.condition_number == pytest.approx((3.0 / 0.1) ** 2, rel=1e-6)
    # the least-constrained direction is the param with the smallest weight
    assert max(rep.sloppiest, key=lambda k: abs(rep.sloppiest[k])) == "frictionloss"
    corr = np.array(rep.correlation)
    assert np.allclose(corr, np.eye(5), atol=1e-9)
    d = rep.to_dict()
    assert d["names"] == list(CONTINUOUS) and d["factor_fixed"]["kp"] > 1.0


def coupled_simulate(params, ep):
    """Only kp * kv matters: kp and kv trade off exactly in log space."""
    x = encode(params)
    x_true = encode(TRUE)
    s = (x[0] - x_true[0]) + (x[1] - x_true[1])
    sim = ep.state.copy()
    sim[:, 0] += np.radians(s)
    for k, i in ((1, 2), (2, 3), (3, 4)):  # damping, armature, frictionloss independent
        sim[:, k] += np.radians(x[i] - x_true[i])
    return ep.state, sim


def test_identifiability_flags_perfect_tradeoff():
    rep = identifiability(TRUE, flat_episodes(), coupled_simulate)
    assert np.isinf(rep.tolerance_free["kp"]) or rep.tradeoff["kp"] > 1e3
    assert rep.tradeoff["damping"] == pytest.approx(1.0, rel=1e-6)
    assert rep.condition_number > 1e6
    slop = rep.sloppiest
    assert slop["kp"] == pytest.approx(-slop["kv"], abs=1e-6)
    assert abs(slop["kp"]) == pytest.approx(1 / np.sqrt(2), abs=1e-6)


def test_residuals_raise_on_divergence():
    def bad(params, ep):
        return ep.state, np.full_like(ep.state, np.nan)

    with pytest.raises(FloatingPointError):
        residuals(TRUE, flat_episodes(1), bad)


# --------------------------------------------------------------------------
# direct checks, on a toy first-order servo
# --------------------------------------------------------------------------

from antares.specificity import interpolate, latency_profile, pinned_refit  # noqa: E402


def toy_response(p, action, fps):
    """First-order lag with time constant from kp/kv, plus a pure delay."""
    tau = 20.0 * p.kv / p.kp + p.armature
    alpha = 1.0 - np.exp(-1.0 / fps / tau)
    lag = int(p.latency_frames)
    q = np.empty_like(action)
    q[0] = action[0]
    for k in range(1, len(action)):
        q[k] = q[k - 1] + alpha * (action[max(k - 1 - lag, 0)] - q[k - 1])
    return q


def toy_episodes(truth, n=4, T=90, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 30.0
    eps = []
    for i in range(n):
        action = np.zeros((T, 6))
        for j in range(6):
            f = rng.uniform(0.3, 1.5)
            action[:, j] = 0.5 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS,
                           state=toy_response(truth, action, 30.0), action=action))
    return eps


def toy_simulate(params, ep):
    return ep.state, toy_response(params, ep.action, ep.fps)


def test_latency_profile_minimum_at_truth():
    prof = latency_profile(TRUE, toy_episodes(TRUE), toy_simulate)
    assert list(prof) == [0, 1, 2, 3, 4, 5]
    assert min(prof, key=prof.get) == 2
    assert prof[2] == pytest.approx(0.0, abs=1e-9)


def test_interpolate_endpoints_and_ridge():
    eps = toy_episodes(TRUE)
    far = TRUE.replace(kp=4000.0, latency_frames=0)
    path = interpolate(TRUE, far, eps, toy_simulate, n=5)
    assert [t for t, _ in path] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert path[0][1] == pytest.approx(0.0, abs=1e-9)
    assert path[-1][1] > path[0][1]


def test_pinned_refit_compensates_along_a_tradeoff():
    """Only kv/kp matters in the toy, so pinning kp lets kv absorb it."""
    eps = toy_episodes(TRUE)
    out = pinned_refit(eps[:3], eps[3:], {"kp": 1600.0}, toy_simulate, max_evaluations=300)
    assert out["params"]["kp"] == 1600.0
    assert out["fixed"] == {"kp": 1600.0}
    assert out["eval_score"] < 0.5  # the default start is several degrees off
    with pytest.raises(ValueError):
        pinned_refit(eps[:3], eps[3:], {"stiffness": 1.0}, toy_simulate)


def test_checks_refuse_per_joint_locks():
    pj = TRUE.replace(kp=(400.0,) * 6)
    with pytest.raises(ValueError, match="shared"):
        identifiability(pj, flat_episodes(1), separable_simulate([1.0] * 5))
    with pytest.raises(ValueError, match="shared"):
        interpolate(TRUE, pj, flat_episodes(1), separable_simulate([1.0] * 5))


def test_perturbations_keep_fields_outside_the_search():
    seen = []

    def sim(params, ep):
        seen.append(params.forcerange)
        return ep.state, ep.state

    jacobian(TRUE.replace(forcerange=7.5), flat_episodes(1), sim)
    assert set(seen) == {7.5}
