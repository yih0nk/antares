"""Tests for antares.calibrate using a toy first-order servo instead of MuJoCo.

The fake ``simulate`` treats dataset values as radians and runs a discrete
first-order lag whose time constant depends on kp/kv/damping/armature, with
a Coulomb-style deadband from frictionloss and a pure delay of
``latency_frames``. "Real" trajectories come from a hidden set of true
parameters; candidate parameters produce the "sim" trajectory. A perfect fit
exists, so CMA-ES must be able to beat the default starting point.
"""

from __future__ import annotations

import importlib.util
import math
import time

import numpy as np
import pytest

from antares.calibrate import (
    DIMS,
    PENALTY,
    SPACE,
    CalibrationResult,
    ProgressEvent,
    SimParams,
    calibrate,
    decode,
    encode,
    score,
    split_episodes,
)
from antares.data import Episode

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TRUE = SimParams(kp=400.0, kv=4.0, damping=2.0, armature=0.1, frictionloss=0.05, latency_frames=2)


def make_episodes(n: int, T: int = 120, fps: float = 30.0, seed: int = 0) -> list[Episode]:
    """Smooth random multi-sine commands; ``state`` is filled from TRUE."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    eps = []
    for i in range(n):
        action = np.zeros((T, 6))
        for j in range(6):
            for _ in range(3):
                f = rng.uniform(0.2, 1.2)
                action[:, j] += rng.uniform(-0.6, 0.6) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        action[:, 5] = np.abs(action[:, 5])  # gripper is one-sided
        state = toy_response(TRUE, action, fps)
        eps.append(Episode(index=i, fps=fps, joint_names=JOINTS, state=state, action=action))
    return eps


def toy_response(p: SimParams, action: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / fps
    tau = 20.0 * (p.kv + p.damping) / p.kp + p.armature
    alpha = 1.0 - math.exp(-dt / tau)
    deadband = 0.05 * p.frictionloss
    lag = int(p.latency_frames)
    q = np.empty_like(action)
    q[0] = action[0]
    for k in range(1, len(action)):
        cmd = action[max(k - 1 - lag, 0)]
        err = cmd - q[k - 1]
        err = np.where(np.abs(err) < deadband, 0.0, err)
        q[k] = q[k - 1] + alpha * err
    return q


def fake_simulate(params: SimParams, ep: Episode) -> tuple[np.ndarray, np.ndarray]:
    return ep.state, toy_response(params, ep.action, ep.fps)


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------


def test_space_matches_simparams_fields():
    assert set(DIMS) == set(SimParams().to_dict())
    assert all(scale in ("log", "linear", "int") for _, _, scale in SPACE.values())


def test_encode_decode_round_trip():
    p = SimParams(kp=1234.5, kv=7.5, damping=0.3, armature=0.01, frictionloss=0.2, latency_frames=3)
    x = encode(p)
    assert x.shape == (6,)
    assert np.all(x >= 0) and np.all(x <= 1)
    q = decode(x)
    for name in DIMS:
        if SPACE[name][2] == "int":
            assert getattr(q, name) == getattr(p, name)
        else:
            assert getattr(q, name) == pytest.approx(getattr(p, name), rel=1e-9)
    # default start lies strictly inside the box for continuous dims
    x0 = encode(SimParams())
    assert np.all(x0[:5] > 0) and np.all(x0[:5] < 1)
    assert x0[5] == 0.0 and decode(x0).latency_frames == 0


def test_decode_clips_to_bounds():
    lo = decode(np.full(6, -3.0))
    hi = decode(np.full(6, 7.0))
    for name, (a, b, _) in SPACE.items():
        assert getattr(lo, name) == pytest.approx(a)
        assert getattr(hi, name) == pytest.approx(b)
    assert isinstance(lo.latency_frames, int) and lo.latency_frames == 0
    assert hi.latency_frames == 5


def test_encode_clips_out_of_range_params():
    x = encode(SimParams(kp=1e9, kv=1e-9, latency_frames=42))
    assert x[0] == 1.0 and x[1] == 0.0 and x[5] == 1.0


def test_latency_rounding():
    assert decode(np.array([0.5] * 5 + [0.09])).latency_frames == 0
    assert decode(np.array([0.5] * 5 + [0.11])).latency_frames == 1
    assert decode(np.array([0.5] * 5 + [0.5])).latency_frames in (2, 3)


def test_decode_rejects_wrong_dim():
    with pytest.raises(ValueError):
        decode(np.zeros(5))


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def test_score_true_params_is_zero_and_default_is_not():
    eps = make_episodes(3)
    perfect = score(TRUE, eps, fake_simulate)
    assert perfect.overall_deg == pytest.approx(0.0, abs=1e-9)
    assert "gripper" not in perfect.scored_joints
    assert perfect.n_frames == sum(len(e) for e in eps)
    off = score(SimParams(), eps, fake_simulate)
    assert off.overall_deg > 1.0
    assert set(off.per_joint_deg) >= set(JOINTS[:5])
    d = off.to_dict()
    assert d["overall_deg"] == pytest.approx(off.overall_deg)


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def test_calibrate_toy_improves_over_baseline():
    eps = make_episodes(8, seed=1)
    train, eval = split_episodes(eps, n_train=5, n_eval=3, seed=0)
    events: list[ProgressEvent] = []
    t0 = time.perf_counter()
    res = calibrate(
        train,
        eval,
        simulate=fake_simulate,
        max_evaluations=90,
        seed=0,
        on_progress=events.append,
    )
    elapsed = time.perf_counter() - t0
    print(f"\ntoy calibration: {res.n_evaluations} evals in {elapsed:.2f}s, "
          f"eval {res.baseline_eval_score:.3f} -> {res.eval_score:.3f} deg, params={res.params}")

    assert isinstance(res, CalibrationResult)
    assert res.baseline == SimParams()
    assert 0 < res.n_evaluations <= 90
    assert res.eval_score < res.baseline_eval_score
    assert res.train_score < res.baseline_train_score
    assert res.train_score <= 0.5 * res.baseline_train_score  # it should really lock on
    assert res.improvement > 0

    # progress: start point + one per evaluation, monotone best
    assert len(events) == res.n_evaluations + 1
    assert [e.evaluation for e in events] == list(range(res.n_evaluations + 1))
    assert events[0].generation == 0 and events[0].best_params == SimParams()
    assert events[0].candidate_score == pytest.approx(res.baseline_train_score)
    best = [e.best_score for e in events]
    assert all(b <= a for a, b in zip(best, best[1:]))
    assert events[-1].best_score == pytest.approx(res.train_score)
    assert events[-1].best_params == res.params
    assert all(e.elapsed_s >= 0 for e in events)
    assert events[-1].elapsed_s >= events[0].elapsed_s

    # history mirrors progress
    assert res.history == [(e.evaluation, e.best_score) for e in events]
    assert res.train_episodes == [e.index for e in train]
    assert res.eval_episodes == [e.index for e in eval]
    assert set(res.per_joint_after) == set(res.per_joint_before)

    # JSON-able round trip
    d = res.to_dict()
    assert d["params"] == res.params.to_dict()
    assert d["history"][0] == [0, pytest.approx(res.baseline_train_score)]
    back = CalibrationResult.from_dict(d)
    assert back.params == res.params and back.history == res.history
    ev = events[-1].to_dict()
    assert ev["best_params"] == res.params.to_dict() and ev["evaluation"] == res.n_evaluations


def test_calibrate_is_deterministic_for_seed():
    eps = make_episodes(3, T=60, seed=2)
    kw = dict(simulate=fake_simulate, max_evaluations=12, popsize=4)
    a = calibrate(eps[:2], eps[2:], seed=7, **kw)
    b = calibrate(eps[:2], eps[2:], seed=7, **kw)
    assert a.params == b.params and a.history == b.history


def test_calibrate_respects_popsize_and_budget():
    eps = make_episodes(2, T=60)
    res = calibrate(eps, [], simulate=fake_simulate, max_evaluations=10, popsize=4)
    assert res.n_evaluations == 10  # 2 full generations + a partial one
    assert res.eval_episodes == []
    assert res.eval_score == pytest.approx(res.train_score)  # no eval set: train stands in


def test_calibrate_should_stop_cancels_early():
    eps = make_episodes(2, T=60)
    seen = []

    def should_stop():
        return len(seen) >= 5

    res = calibrate(
        eps[:1], eps[1:], simulate=fake_simulate, max_evaluations=100, popsize=6,
        on_progress=seen.append, should_stop=should_stop,
    )
    assert res.n_evaluations < 100
    assert res.n_evaluations <= 6
    assert len(seen) == res.n_evaluations + 1


def test_calibrate_penalizes_nan_sim():
    eps = make_episodes(2, T=60)

    def flaky(params, ep):
        real, sim = fake_simulate(params, ep)
        if params.kp > 1500:
            sim = np.full_like(sim, np.nan)
        return real, sim

    events = []
    res = calibrate(eps[:1], eps[1:], simulate=flaky, max_evaluations=20, popsize=5,
                    sigma0=0.6, on_progress=events.append)
    assert all(math.isfinite(e.candidate_score) for e in events)
    assert any(e.candidate_score == PENALTY for e in events)  # some candidate blew up
    assert res.params.kp <= 1500 and math.isfinite(res.eval_score)
    # JSON form never carries NaN
    assert all(v is None or math.isfinite(v) for v in res.to_dict()["per_joint_after"].values())


def test_calibrate_requires_training_episodes():
    with pytest.raises(ValueError):
        calibrate([], [], simulate=fake_simulate)


# ---------------------------------------------------------------------------
# split_episodes
# ---------------------------------------------------------------------------


def test_split_episodes_deterministic_and_disjoint():
    eps = make_episodes(10, T=5)
    tr1, ev1 = split_episodes(eps, 6, 3, seed=0)
    tr2, ev2 = split_episodes(eps, 6, 3, seed=0)
    assert [e.index for e in tr1] == [e.index for e in tr2]
    assert [e.index for e in ev1] == [e.index for e in ev2]
    assert len(tr1) == 6 and len(ev1) == 3
    assert not set(e.index for e in tr1) & set(e.index for e in ev1)
    assert [e.index for e in tr1] == sorted(e.index for e in tr1)
    tr3, _ = split_episodes(eps, 6, 3, seed=1)
    assert [e.index for e in tr3] != [e.index for e in tr1]


def test_split_episodes_clamps_to_available():
    eps = make_episodes(4, T=5)
    tr, ev = split_episodes(eps, 10, 10)
    assert len(tr) == 4 and ev == []
    tr, ev = split_episodes(eps, 3, 10)
    assert len(tr) == 3 and len(ev) == 1
    with pytest.raises(ValueError):
        split_episodes([], 1, 1)


# ---------------------------------------------------------------------------
# optional integration against the real MuJoCo arm
# ---------------------------------------------------------------------------


def _sim_available() -> bool:
    spec = importlib.util.find_spec("antares.sim")
    if spec is None:
        return False
    try:
        import antares.sim as sim

        return hasattr(sim, "Arm") and hasattr(sim, "SimParams")
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _sim_available(), reason="antares.sim not available yet")
def test_calibrate_with_real_arm_smoke():
    rng = np.random.default_rng(0)
    T, fps = 60, 30.0
    t = np.arange(T) / fps
    eps = []
    for i in range(2):
        action = np.zeros((T, 6))
        for j in range(5):
            action[:, j] = 30 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t + rng.uniform(0, 6))
        action[:, 5] = 20 + 10 * np.sin(2 * np.pi * 0.4 * t)
        state = np.roll(action, 2, axis=0)
        state[:2] = action[0]
        eps.append(Episode(index=i, fps=fps, joint_names=JOINTS, state=state, action=action))
    events = []
    res = calibrate(eps[:1], eps[1:], max_evaluations=6, popsize=3, on_progress=events.append)
    assert res.n_evaluations == 6 and len(events) == 7
    assert math.isfinite(res.eval_score) and math.isfinite(res.baseline_eval_score)
