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


def test_residuals_reject_an_empty_split():
    with pytest.raises(ValueError, match="no episodes"):
        residuals(TRUE, [], separable_simulate([1.0] * 5))


def test_identifiability_has_no_tradeoff_for_an_unconstrained_param():
    # kv does nothing to the residuals: unbounded alone *and* with the others free,
    # so the ratio is undefined rather than inf/inf = nan by accident
    rep = identifiability(TRUE, flat_episodes(), separable_simulate([3.0, 0.0, 0.5, 2.0, 0.1]))
    assert math.isinf(rep.tolerance_fixed["kv"]) and math.isinf(rep.tolerance_free["kv"])
    assert math.isnan(rep.tradeoff["kv"])
    assert rep.to_dict()["tradeoff"]["kv"] is None
    assert rep.tradeoff["kp"] == pytest.approx(1.0, rel=1e-6)


def test_residuals_raise_on_divergence():
    def bad(params, ep):
        return ep.state, np.full_like(ep.state, np.nan)

    with pytest.raises(FloatingPointError):
        residuals(TRUE, flat_episodes(1), bad)


# --------------------------------------------------------------------------
# direct checks, on a toy first-order servo
# --------------------------------------------------------------------------

from antares.specificity import interpolate, latency_profile, pinned_refit, seed_restarts  # noqa: E402


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


def test_seed_restarts_report_spread():
    eps = toy_episodes(TRUE)
    out = seed_restarts(eps[:3], eps[3:], [0, 1], toy_simulate, max_evaluations=40)
    assert [r["seed"] for r in out["runs"]] == [0, 1]
    from antares.calibrate import DIMS
    assert set(out["spread"]) == set(DIMS)
    assert all(v >= 1.0 for k, v in out["spread"].items() if k != "latency_frames")
    assert out["spread"]["latency_frames"] >= 0


# --------------------------------------------------------------------------
# the whole study, on two toy arms
# --------------------------------------------------------------------------

from antares.specificity import study  # noqa: E402

FAST = TRUE.replace(kp=1500.0, latency_frames=0)


def test_study_end_to_end_on_two_toy_arms():
    data = {"toy/slow": toy_episodes(TRUE, n=8, seed=1), "toy/fast": toy_episodes(FAST, n=8, seed=2)}
    locks = {"slow": ("toy/slow", TRUE), "fast": ("toy/fast", FAST)}
    logs = []
    out = study(data, locks, toy_simulate, n_train=3, n_eval=3, restart_seeds=(1,),
                max_evaluations=30, truth={"toy/slow": TRUE}, log=logs.append)
    assert out["matrix"]["cols"] == ["slow", "fast", "default"]
    assert out["specificity"]["all_specific"] is True
    slow = out["datasets"]["toy/slow"]
    assert slow["other_lock"] == "fast"
    assert len(slow["seed_restarts"]["runs"]) == 2  # the lock itself + one restart
    assert slow["same_arm_control"]["n_train"] == 2  # 8 - 3 eval - 3 train
    assert slow["swap"]["params"]["kp"] == FAST.kp
    assert slow["path_to_other"][0][1] == pytest.approx(0.0, abs=1e-9)
    assert slow["truth_eval_score"] == pytest.approx(0.0, abs=1e-9)
    assert "truth" not in out["datasets"]["toy/fast"]
    assert any("swap" in m for m in logs)
    import json
    json.dumps(out)  # serializable as-is


def test_study_scores_the_split_its_lock_was_fitted_on():
    data = {"toy/slow": toy_episodes(TRUE, n=8, seed=1), "toy/fast": toy_episodes(FAST, n=8, seed=2)}
    locks = {"slow": ("toy/slow", TRUE), "fast": ("toy/fast", FAST)}
    logs = []
    out = study(data, locks, toy_simulate, n_train=3, n_eval=3, restart_seeds=(), max_evaluations=10,
                splits={"toy/slow": ([5, 6, 7], [0, 1]), "toy/fast": ([0, 1], [9, 10])}, log=logs.append)
    assert out["splits"]["toy/slow"] == {"train": [5, 6, 7], "eval": [0, 1]}
    # the fresh same-arm control never touches the lock's own episodes
    assert out["datasets"]["toy/slow"]["same_arm_control"]["n_train"] == 3  # 8 - 3 - 2
    # a split that names episodes the dataset lacks falls back to the seeded one
    assert any("toy/fast: lock split does not match" in m for m in logs)
    assert out["splits"]["toy/fast"]["train"] != [0, 1]
    assert len(out["splits"]["toy/fast"]["eval"]) == 3


def test_study_refuses_a_per_joint_lock_up_front():
    data = {"toy/slow": toy_episodes(TRUE, n=4, seed=1), "toy/fast": toy_episodes(FAST, n=4, seed=2)}
    locks = {"slow": ("toy/slow", TRUE), "fast": ("toy/fast", FAST.replace(kp=(1500.0,) * 6))}
    calls = []
    with pytest.raises(ValueError, match="lock fast: .*shared"):
        study(data, locks, lambda p, ep: calls.append(1) or toy_simulate(p, ep), n_train=1, n_eval=1)
    assert not calls  # before any simulation


def test_study_refuses_an_empty_eval_split():
    data = {"toy/slow": toy_episodes(TRUE, n=2, seed=1)}
    with pytest.raises(ValueError, match="no held-out episodes"):
        study(data, {"slow": ("toy/slow", TRUE)}, toy_simulate, n_train=2, n_eval=1)


def test_study_survives_a_diverging_lock():
    from antares.calibrate import PENALTY

    data = {"toy/slow": toy_episodes(TRUE, n=4, seed=1)}

    def simulate(params, ep):
        if params.kp == FAST.kp:
            raise FloatingPointError("boom")
        return toy_simulate(params, ep)

    out = study(data, {"slow": ("toy/slow", TRUE), "fast": ("toy/other", FAST)}, simulate,
                n_train=2, n_eval=2, restart_seeds=(), max_evaluations=10)
    assert out["matrix"]["scores"][0][1] == PENALTY
    assert out["specificity"]["rows"][0]["specific"] is True


def test_render_markdown_has_every_section():
    from antares.specificity import render_markdown

    data = {"toy/slow": toy_episodes(TRUE, n=8, seed=1), "toy/fast": toy_episodes(FAST, n=8, seed=2)}
    locks = {"slow": ("toy/slow", TRUE), "fast": ("toy/fast", FAST)}
    out = study(data, locks, toy_simulate, n_train=3, n_eval=3, restart_seeds=(1,),
                max_evaluations=20, truth={"toy/slow": TRUE})
    md = render_markdown(out)
    for needle in ("## Robot specificity", "| dataset | slow | fast | default |", "### toy/slow",
                   "same arm, refit on 2 other episodes", "kp/kv pinned", "true parameters",
                   "| spread |", "Least constrained direction", "Latency: 0 frames", "Path from own lock"):
        assert needle in md, needle


def test_names_outside_the_locks_space_are_refused_up_front():
    """A servo-model lock's kp/kv are derived, so the position-actuator
    CONTINUOUS names do not fit it: a ValueError naming the space, not a
    KeyError from inside the search space."""
    from antares.calibrate import default_simulate
    from antares.sim import SimParams
    from antares.synthetic import make_dataset

    p = SimParams.sts3215(latency_frames=2)
    eps = list(make_dataset("x/y", p, n_episodes=1, T=30).episodes)
    with pytest.raises(ValueError, match="'kp', 'kv': not in the lock's servo-model search space"):
        identifiability(p, eps, default_simulate(), names=CONTINUOUS)
    with pytest.raises(ValueError, match="servo-model search space"):
        jacobian(p, eps, default_simulate(), names=["kp"])
    with pytest.raises(ValueError, match="integer dimension"):
        jacobian(p, eps, default_simulate(), names=["latency_frames"])
    with pytest.raises(ValueError, match="position-actuator search space"):
        jacobian(SimParams(), eps, default_simulate(), names=["servo_gain"])


def test_identifiability_of_a_servo_lock_uses_the_servo_space():
    """kp/kv are derived from the servo gain on a servo-model lock, so the
    check perturbs the fields that lock was actually fitted on."""
    from antares.calibrate import default_simulate
    from antares.sim import SimParams
    from antares.synthetic import make_dataset

    p = SimParams.sts3215(latency_frames=2)
    eps = list(make_dataset("x/y", p, n_episodes=1, T=60).episodes)
    rep = identifiability(p, eps, default_simulate())
    assert "servo_gain" in rep.names and "kp" not in rep.names
    assert all(math.isfinite(rep.factor_fixed[n]) for n in rep.names)
