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
import json
import math
import time

import numpy as np
import pytest

from antares.calibrate import (
    DIMS,
    PENALTY,
    PER_JOINT,
    PER_JOINT_SPACE,
    SHARED,
    SPACE,
    CalibrationResult,
    ProgressEvent,
    SearchSpace,
    SimParams,
    calibrate,
    decode,
    encode,
    score,
    split_episodes,
)
from antares.data import Episode
from antares.sim import SERVO_FIELDS

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TRUE = SimParams(kp=400.0, kv=4.0, damping=2.0, armature=0.1, frictionloss=0.05, latency_frames=2)
# A robot whose elbow is slower and laggier than the rest: no shared fit can match it.
TRUE_PER_JOINT = TRUE.replace(
    kp=(400.0, 400.0, 120.0, 400.0, 400.0, 400.0),
    latency_frames=(2, 2, 4, 2, 2, 2),
)


def make_episodes(
    n: int, T: int = 120, fps: float = 30.0, seed: int = 0, true: SimParams = TRUE
) -> list[Episode]:
    """Smooth random multi-sine commands; ``state`` is filled from ``true``."""
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
        state = toy_response(true, action, fps)
        eps.append(Episode(index=i, fps=fps, joint_names=JOINTS, state=state, action=action))
    return eps


def toy_response(p: SimParams, action: np.ndarray, fps: float) -> np.ndarray:
    """First-order lag per joint; scalar and per-joint params both work."""
    dt = 1.0 / fps
    kp, kv, damping, armature, frictionloss = (
        p.per_joint(n) for n in ("kp", "kv", "damping", "armature", "frictionloss")
    )
    tau = 20.0 * (kv + damping) / kp + armature  # (6,)
    alpha = 1.0 - np.exp(-dt / tau)
    deadband = 0.05 * frictionloss
    lag = p.per_joint("latency_frames")
    cols = np.arange(action.shape[1])
    q = np.empty_like(action)
    q[0] = action[0]
    for k in range(1, len(action)):
        cmd = action[np.maximum(k - 1 - lag, 0), cols]
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
    # the shared space fits every original scalar; forcerange keeps its MJCF
    # value there and is only searched per joint
    assert set(DIMS) == set(SimParams().to_dict()) - {"forcerange"}
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
    with pytest.raises(ValueError):
        PER_JOINT.decode(np.zeros(6))


def test_search_spaces():
    assert SHARED.name == "shared" and SHARED.dims == DIMS and SHARED.n_dims == 6
    assert PER_JOINT.name == "per_joint"
    # every per-joint field, incl. forcerange; gravity_scale and deadband are opt-in
    assert set(PER_JOINT_SPACE) == set(SimParams.field_names()) - {"gravity_scale", "deadband", "servo"} - set(
        SERVO_FIELDS
    )
    assert PER_JOINT.n_dims == 7 * 6 == len(PER_JOINT.dims)
    assert PER_JOINT.dims[:2] == ("kp.shoulder_pan", "kp.shoulder_lift")
    assert PER_JOINT.dims[-1] == "forcerange.gripper"
    # shared decode keeps unsearched fields from the base, e.g. forcerange
    base = SimParams(forcerange=7.0)
    assert SHARED.decode(SHARED.encode(base), base=base).forcerange == 7.0
    assert decode(encode(base)).forcerange == 2.94  # no base: defaults


def test_per_joint_encode_decode_round_trip():
    x = PER_JOINT.encode(TRUE_PER_JOINT)
    assert x.shape == (42,) and np.all(x >= 0) and np.all(x <= 1)
    back = PER_JOINT.decode(x)
    assert back.has_per_joint and all(back.is_per_joint(n) for n in PER_JOINT.fields)
    assert back.latency_frames == (2, 2, 4, 2, 2, 2)
    for name in SimParams.field_names():
        np.testing.assert_allclose(back.per_joint(name), TRUE_PER_JOINT.per_joint(name), rtol=1e-9)
    # a scalar start is broadcast, so the per-joint stage starts exactly there
    x0 = PER_JOINT.encode(TRUE)
    np.testing.assert_allclose(x0.reshape(7, 6), np.repeat(x0.reshape(7, 6)[:, :1], 6, axis=1))
    for name in SimParams.field_names():
        np.testing.assert_allclose(PER_JOINT.decode(x0).per_joint(name), TRUE.per_joint(name), rtol=1e-9)
    # the shared space summarizes a per-joint field by its mean in unit coords
    xs = SHARED.encode(TRUE_PER_JOINT)
    assert xs.shape == (6,)
    kp_unit = [SHARED.encode(TRUE.replace(kp=k))[0] for k in TRUE_PER_JOINT.kp]
    assert xs[0] == pytest.approx(np.mean(kp_unit))
    custom = SearchSpace({"kp": SPACE["kp"]}, per_joint=True)
    assert custom.n_dims == 6 and custom.decode(np.full(6, 0.5), base=TRUE).kv == TRUE.kv


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


def test_calibrate_default_is_shared_only():
    eps = make_episodes(2, T=60)
    res = calibrate(eps[:1], eps[1:], simulate=fake_simulate, max_evaluations=8, popsize=4)
    assert res.space == "shared" and not res.params.has_per_joint
    assert [s["name"] for s in res.stages] == ["shared"]
    assert res.stages[0]["n_evaluations"] == res.n_evaluations == 8
    assert res.stages[0]["train_score"] == pytest.approx(res.train_score)
    assert res.stages[0]["params"] == res.params.to_dict()
    # old result dicts without the new keys still load
    d = res.to_dict()
    del d["space"], d["stages"]
    back = CalibrationResult.from_dict(d)
    assert back.space == "shared" and back.stages == []


def test_zero_shared_budget_skips_the_shared_stage():
    eps = make_episodes(2, T=60)
    start = TRUE.replace(kp=(300.0,) * 6)
    res = calibrate(eps[:1], eps[1:], simulate=fake_simulate, start=start, max_evaluations=0,
                    per_joint=True, per_joint_max_evaluations=8, per_joint_popsize=4)
    assert [s["name"] for s in res.stages] == ["per_joint"]
    assert res.n_evaluations == 8 and res.space == "per_joint"
    # and with no per-joint stage either, the start is returned untouched
    res = calibrate(eps[:1], eps[1:], simulate=fake_simulate, start=TRUE, max_evaluations=0)
    assert res.stages == [] and res.n_evaluations == 0 and res.params == TRUE


def test_calibrate_per_joint_refines_shared_fit():
    eps = make_episodes(6, T=90, seed=3, true=TRUE_PER_JOINT)
    train, eval = split_episodes(eps, n_train=4, n_eval=2, seed=0)
    kw = dict(simulate=fake_simulate, max_evaluations=60, seed=0)
    shared = calibrate(train, eval, **kw)
    events: list[ProgressEvent] = []
    t0 = time.perf_counter()
    res = calibrate(
        train, eval, per_joint=True, per_joint_max_evaluations=240, on_progress=events.append, **kw
    )
    elapsed = time.perf_counter() - t0
    print(f"\nper-joint toy: eval shared {shared.eval_score:.4f} -> per-joint {res.eval_score:.4f} deg "
          f"in {elapsed:.2f}s; elbow {shared.per_joint_after['elbow_flex']:.4f} -> "
          f"{res.per_joint_after['elbow_flex']:.4f}; latency {res.params.latency_frames}")

    # same shared stage, then a refinement that can only help on train
    assert res.stages[0] == shared.stages[0]
    assert [s["name"] for s in res.stages] == ["shared", "per_joint"]
    assert res.stages[1]["n_evaluations"] == 240
    assert res.n_evaluations == 60 + 240
    assert res.space == "per_joint" and res.params.has_per_joint
    assert res.train_score <= shared.train_score
    # the elbow is where the shared fit was wrong, and the per-joint fit finds it
    assert res.eval_score < 0.8 * shared.eval_score
    assert res.per_joint_after["elbow_flex"] < 0.6 * shared.per_joint_after["elbow_flex"]
    # (the toy's elbow lag trades off against its gain, so only the error is
    # asserted, not which of the two parameters the fit used to explain it)

    # progress: one stream over both stages, labelled, monotone
    assert len(events) == res.n_evaluations + 1
    assert [e.evaluation for e in events] == list(range(res.n_evaluations + 1))
    assert [e.stage for e in events[:61]] == ["shared"] * 61
    assert [e.stage for e in events[61:]] == ["per_joint"] * 240
    assert events[61].generation > events[60].generation
    best = [e.best_score for e in events]
    assert all(b <= a for a, b in zip(best, best[1:]))
    assert events[-1].best_params == res.params
    assert events[-1].to_dict()["stage"] == "per_joint"
    assert res.history == [(e.evaluation, e.best_score) for e in events]

    # JSON round trip keeps the per-joint tuples, the stages and the space
    d = json.loads(json.dumps(res.to_dict()))
    assert isinstance(d["params"]["kp"], list) and len(d["params"]["kp"]) == 6
    assert d["space"] == "per_joint" and len(d["stages"]) == 2
    back = CalibrationResult.from_dict(d)
    assert back.params == res.params and back.stages == res.stages and back.space == "per_joint"


def test_calibrate_per_joint_skipped_when_cancelled():
    eps = make_episodes(2, T=60)
    seen = []
    res = calibrate(
        eps[:1], eps[1:], simulate=fake_simulate, max_evaluations=100, popsize=6, per_joint=True,
        on_progress=seen.append, should_stop=lambda: len(seen) >= 5,
    )
    assert res.n_evaluations <= 6 and res.space == "shared"
    assert [s["name"] for s in res.stages] == ["shared"]
    assert not res.params.has_per_joint


# ---------------------------------------------------------------------------
# best of N seeds
# ---------------------------------------------------------------------------


def test_calibrate_best_of_n_seeds_keeps_the_best_held_out_candidate():
    eps = make_episodes(5, T=60, seed=4)
    train, eval = eps[:3], eps[3:]
    kw = dict(simulate=fake_simulate, max_evaluations=12, popsize=4, seed=0)
    events: list[ProgressEvent] = []
    res = calibrate(train, eval, n_seeds=3, on_progress=events.append, **kw)

    # one shared stage per seed, seeds 0/100/200, every evaluation counted
    assert [s["name"] for s in res.stages] == ["shared"] * 3
    assert [s["seed"] for s in res.stages] == [0, 100, 200]
    assert all(s["n_evaluations"] == 12 for s in res.stages)
    assert res.n_evaluations == 36 and len(events) == 37
    assert [e.evaluation for e in events] == list(range(37))
    assert [e.seed for e in events[:13]] == [0] * 13
    assert [e.seed for e in events[13:25]] == [1] * 12 and [e.seed for e in events[25:]] == [2] * 12
    assert events[-1].to_dict()["seed"] == 2
    assert res.history == [(e.evaluation, e.best_score) for e in events]

    # each seed run is the plain single-seed fit with that seed, from the same start
    singles = [calibrate(train, eval, **{**kw, "seed": s}) for s in (0, 100, 200)]
    for stage, single in zip(res.stages, singles):
        assert stage["params"] == single.params.to_dict()
        assert stage["train_score"] == pytest.approx(single.train_score)

    # the candidate with the best held-out score wins, whatever its train score
    sd = res.seeds
    assert sd["n"] == 3 and sd["seeds"] == [0, 100, 200]
    assert sd["eval_scores"] == pytest.approx([s.eval_score for s in singles])
    assert sd["train_scores"] == pytest.approx([s.train_score for s in singles])
    assert sd["params"] == [s.params.to_dict() for s in singles]
    assert sd["chosen"] == int(np.argmin(sd["eval_scores"]))
    assert res.params == singles[sd["chosen"]].params
    assert res.eval_score == pytest.approx(min(sd["eval_scores"]))
    assert res.train_score == pytest.approx(sd["train_scores"][sd["chosen"]])
    assert res.space == "shared"

    # spread covers the shared search space, pinned follows the rule
    assert set(sd["spread"]) == set(SPACE)
    for name, (_, _, scale) in SPACE.items():
        sp = sd["spread"][name]
        vals = [p[name] for p in sd["params"]]
        assert sp["min"] == pytest.approx(min(vals)) and sp["max"] == pytest.approx(max(vals))
        if scale == "log":
            assert sp["ratio"] == pytest.approx(max(vals) / min(vals))
            assert (name in sd["pinned"]) == (sp["ratio"] <= 2.0)
        else:
            assert sp["range"] == max(vals) - min(vals)
            assert (name in sd["pinned"]) == (sp["range"] <= 1)

    # JSON round trip keeps the block; a single-seed result has none
    d = json.loads(json.dumps(res.to_dict()))
    back = CalibrationResult.from_dict(d)
    assert back.seeds == res.seeds and back.stages == res.stages
    assert singles[0].seeds is None and "seeds" not in singles[0].to_dict()
    assert [s["seed"] for s in singles[0].stages] == [0]
    del d["seeds"]
    assert CalibrationResult.from_dict(d).seeds is None


def test_calibrate_explicit_seeds_and_no_eval_episodes():
    eps = make_episodes(2, T=60)
    res = calibrate(eps, [], simulate=fake_simulate, max_evaluations=8, popsize=4, seeds=[3, 5])
    assert [s["seed"] for s in res.stages] == [3, 5] and res.seeds["seeds"] == [3, 5]
    # no eval set: the training score stands in and decides
    assert res.seeds["eval_scores"] == res.seeds["train_scores"]
    assert res.seeds["chosen"] == int(np.argmin(res.seeds["train_scores"]))
    assert res.eval_score == pytest.approx(res.train_score)
    assert res.n_evaluations == 16
    with pytest.raises(ValueError):
        calibrate(eps, [], simulate=fake_simulate, n_seeds=0)
    with pytest.raises(ValueError):
        calibrate(eps, [], simulate=fake_simulate, seeds=[])


def test_calibrate_seeds_then_per_joint_runs_once_from_the_winner():
    eps = make_episodes(3, T=60, seed=3, true=TRUE_PER_JOINT)
    res = calibrate(eps[:2], eps[2:], simulate=fake_simulate, max_evaluations=8, popsize=4, n_seeds=2,
                    per_joint=True, per_joint_max_evaluations=8, per_joint_popsize=4)
    assert [s["name"] for s in res.stages] == ["shared", "shared", "per_joint"]
    assert res.stages[2]["seed"] == 1  # base seed + 1, as before
    assert res.n_evaluations == 24 and res.space == "per_joint"
    chosen = res.seeds["chosen"]
    # the refinement can only improve on the chosen shared optimum
    assert res.train_score <= res.seeds["train_scores"][chosen] + 1e-12


def test_calibrate_seeds_should_stop_returns_the_best_so_far():
    eps = make_episodes(2, T=60)
    seen: list[ProgressEvent] = []
    res = calibrate(
        eps[:1], eps[1:], simulate=fake_simulate, max_evaluations=10, popsize=5, n_seeds=3,
        per_joint=True, on_progress=seen.append, should_stop=lambda: len(seen) >= 14,
    )
    # seed 0 finished (10 evaluations), seed 1 was cut short, seed 2 never ran
    assert [s["name"] for s in res.stages] == ["shared", "shared"]
    assert res.stages[0]["n_evaluations"] == 10 and res.stages[1]["n_evaluations"] < 10
    assert res.n_evaluations == len(seen) - 1 < 30
    assert res.seeds["n"] == 2 and res.seeds["seeds"] == [0, 100]
    assert res.space == "shared" and not res.params.has_per_joint
    assert res.eval_score == pytest.approx(min(res.seeds["eval_scores"]))


def test_param_spread_of_rule():
    from antares.calibrate import param_spread_of

    a = SimParams(kp=100.0, kv=1.0, damping=0.1, armature=0.01, frictionloss=0.1, latency_frames=2)
    b = a.replace(kp=190.0, kv=2.5, damping=0.1, latency_frames=3)
    c = a.replace(kp=120.0, kv=1.0, armature=(0.01, 0.01, 0.05, 0.01, 0.01, 0.01), latency_frames=2)
    spread, pinned = param_spread_of([a, b, c])
    assert spread["kp"] == {"min": 100.0, "max": 190.0, "ratio": pytest.approx(1.9)}
    assert spread["kv"]["ratio"] == pytest.approx(2.5)
    assert spread["armature"] == {"min": 0.01, "max": 0.05, "ratio": pytest.approx(5.0)}  # per-joint values count
    assert spread["latency_frames"] == {"min": 2.0, "max": 3.0, "range": 1.0}
    assert pinned == ["kp", "damping", "frictionloss", "latency_frames"]
    _, pinned = param_spread_of([a, a.replace(latency_frames=4)])
    assert "latency_frames" not in pinned
    with pytest.raises(ValueError):
        param_spread_of([a])


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


def test_fit_gravity_adds_one_shared_dimension():
    import antares.calibrate as cal
    from antares.calibrate import GRAVITY_SPACE, spaces

    shared, per_joint = spaces(fit_gravity=True)
    assert shared.name == "shared" and per_joint == PER_JOINT
    assert shared.dims == DIMS + ("gravity_scale",)
    p = shared.decode(shared.encode(SimParams(gravity_scale=0.4)))
    assert p.gravity_scale == pytest.approx(0.4)
    assert GRAVITY_SPACE["gravity_scale"][:2] == (0.0, 2.0)
    assert not hasattr(cal, "SHARED_GRAVITY")  # spaces() is the one source of the optional spaces


def test_calibrate_fit_gravity_searches_gravity_scale():
    eps = make_episodes(2)
    seen = set()

    def sim(params, ep):
        seen.add(params.gravity_scale)
        real, simq = fake_simulate(params, ep)
        return real, simq + (params.gravity_scale - 0.5)  # best at 0.5

    off = calibrate(eps[:1], eps[1:], simulate=sim, max_evaluations=12, popsize=4)
    assert seen == {1.0} and "gravity_scale" not in off.params.to_dict()
    res = calibrate(eps[:1], eps[1:], simulate=sim, max_evaluations=60, popsize=6, fit_gravity=True)
    assert len(seen) > 1
    assert res.params.gravity_scale == pytest.approx(0.5, abs=0.15)
    assert res.space == "shared" and "gravity_scale" in res.to_dict()["params"]


def test_spaces_add_optional_terms():
    from antares.calibrate import DEADBAND_SPACE, spaces

    assert spaces() == (SHARED, PER_JOINT)
    shared, pj = spaces(fit_deadband=True)
    assert shared.dims == DIMS + ("deadband",)
    assert "deadband" in pj.fields and pj.n_dims == PER_JOINT.n_dims + 6
    shared, pj = spaces(fit_gravity=True, fit_deadband=True)
    assert shared.dims == DIMS + ("gravity_scale", "deadband")
    assert "gravity_scale" not in pj.fields
    assert DEADBAND_SPACE["deadband"][0] == 0.0


def test_calibrate_fit_deadband_per_joint_stage():
    # Three joints want a 0.02 rad band and the others none, so the shared
    # stage can only compromise and the per-joint stage has something to win.
    # The residual depends on the deadband alone: with the toy servo in the
    # loop, a CMA-ES step in the other 42 per-joint dims costs far more than
    # any deadband gain and the stage never beats the shared optimum.
    eps = make_episodes(2)
    target = np.array([0.02, 0.02, 0.02, 0.0, 0.0, 0.0])

    def sim(params, ep):
        return ep.state, ep.state + (params.per_joint("deadband") - target)

    res = calibrate(eps[:1], eps[1:], simulate=sim, max_evaluations=40, popsize=6,
                    fit_deadband=True, per_joint=True, per_joint_max_evaluations=200)
    assert [s["name"] for s in res.stages] == ["shared", "per_joint"]
    shared = res.stages[0]["params"]["deadband"]
    assert isinstance(shared, float) and 0.0 < shared < 0.02  # one compromise value
    assert res.space == "per_joint" and res.params.is_per_joint("deadband")
    assert len(res.to_dict()["params"]["deadband"]) == 6
    assert res.train_score < res.stages[0]["train_score"]
    band = res.params.per_joint("deadband")
    # the gripper is left out of the score, so its band is unconstrained
    np.testing.assert_allclose(band[:5], target[:5], atol=0.003)


@pytest.mark.slow
@pytest.mark.skipif(not _sim_available(), reason="antares.sim not available yet")
def test_default_simulate_holds_a_command_map():
    from antares.calibrate import default_simulate
    from antares.sim import SERVO_FIELDS, CommandMap

    T = 30
    action = np.zeros((T, 6))
    action[:, 1] = 20.0
    ep = Episode(index=0, fps=30.0, joint_names=JOINTS, state=np.zeros((T, 6)), action=action)
    shifted = CommandMap(gain=np.ones(6), bias=[0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    _, plain = default_simulate()(SimParams(), ep)
    _, mapped = default_simulate(shifted)(SimParams(), ep)
    assert mapped[-1, 1] > plain[-1, 1]


def test_spaces_servo_swaps_the_actuator_dims():
    from antares.calibrate import SERVO_SPACE, spaces

    shared, pj = spaces(servo=True)
    assert shared.fields == tuple(SERVO_SPACE) and pj.fields == tuple(SERVO_SPACE) and pj.per_joint
    assert not {"kp", "kv", "forcerange"} & set(shared.fields)
    shared, pj = spaces(fit_gravity=True, fit_deadband=True, servo=True)
    assert shared.fields == tuple(SERVO_SPACE) + ("gravity_scale", "deadband")
    # every searched servo field exists on SimParams, and BAM's start sits inside the bounds
    start = SimParams.sts3215()
    for name, (lo, hi, _) in SERVO_SPACE.items():
        assert lo <= getattr(start, name) <= hi, name


def test_calibrate_from_a_servo_start_fits_the_servo_terms():
    # The toy residual depends only on servo_gain, so the fit must move it
    # (and only fields of the servo space) while keeping the servo model.
    eps = make_episodes(2)

    def sim(params, ep):
        return ep.state, ep.state + (np.log(params.per_joint("servo_gain")) - np.log(1.6))

    start = SimParams.sts3215()
    res = calibrate(eps[:1], eps[1:], simulate=sim, start=start, max_evaluations=60, popsize=6)
    assert res.params.servo == "sts3215"
    assert res.params.servo_gain == pytest.approx(1.6, rel=0.1)
    # kp is not searched: it is the PD equivalent derived from the fitted gain
    assert res.params.kp == pytest.approx(start.kp * res.params.servo_gain / start.servo_gain)
    assert res.params.forcerange == start.forcerange
    d = res.to_dict()["params"]
    assert d["servo"] == "sts3215" and "servo_gain" in d


def test_seed_spread_of_servo_fits_covers_the_servo_fields():
    from antares.calibrate import SERVO_SPACE, param_spread_of

    fits = [SimParams.sts3215(servo_gain=0.8, damping=0.004), SimParams.sts3215(servo_gain=0.82, damping=0.04)]
    spread, pinned = param_spread_of(fits)
    assert set(spread) == set(SERVO_SPACE)
    assert "servo_gain" in pinned and "damping" not in pinned and "kp" not in spread
