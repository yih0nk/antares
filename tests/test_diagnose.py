"""Tests for antares.diagnose with a toy first-order servo instead of MuJoCo.

"Real" recordings come from known parameters; drift is a known change on
top of the lock (extra lag, a weaker gain), so the diagnosis can be checked
against the truth. The slow tests at the bottom do the same with the real
MuJoCo arm on synthetic recordings.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from antares.calibrate import CalibrationResult, SimParams
from antares.data import Episode
from antares.diagnose import (
    FIELDS,
    ParamChange,
    apply_change,
    decode_change,
    describe_change,
    diagnose,
    encode_change,
    neutral,
    split_alternating,
)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TRUE = SimParams(kp=400.0, kv=4.0, damping=2.0, armature=0.02, frictionloss=0.05, latency_frames=1)


def toy_response(p: SimParams, action: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / fps
    tau = 20.0 * (p.kv + p.damping) / p.kp + p.armature
    alpha = 1.0 - math.exp(-dt / tau)
    lag = int(p.latency_frames)
    q = np.empty_like(action)
    q[0] = action[0]
    for k in range(1, len(action)):
        q[k] = q[k - 1] + alpha * (action[max(k - 1 - lag, 0)] - q[k - 1])
    return q


def make_episodes(truth: SimParams, n: int = 8, T: int = 120, fps: float = 30.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    eps = []
    for i in range(n):
        action = np.zeros((T, 6))
        for j in range(6):
            for _ in range(3):
                f = rng.uniform(0.2, 1.2)
                action[:, j] += rng.uniform(-0.6, 0.6) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        eps.append(Episode(index=i, fps=fps, joint_names=JOINTS, state=toy_response(truth, action, fps),
                           action=action))
    return eps


def toy_simulate(params: SimParams, ep: Episode):
    return ep.state, toy_response(params, ep.action, ep.fps)


def lock_of(params: SimParams = TRUE, eval_score: float = 0.0) -> dict:
    return {"name": "unit-a", "dataset": "fake/unit_a", "params": params.to_dict(),
            "result": {"eval_score": eval_score}}


# -- change space ---------------------------------------------------------------


def test_neutral_round_trips_and_changes_nothing():
    x = encode_change(neutral())
    assert decode_change(x) == pytest.approx(neutral())
    assert apply_change(TRUE, neutral()) == TRUE


def test_change_space_bounds():
    lo, hi = decode_change(np.zeros(6)), decode_change(np.ones(6))
    assert lo["kp"] == pytest.approx(0.25) and hi["kp"] == pytest.approx(4.0)
    assert lo["latency_frames"] == -2 and hi["latency_frames"] == 3
    with pytest.raises(ValueError):
        decode_change(np.zeros(5))


def test_encode_change_round_trip():
    c = {"kp": 0.7, "kv": 1.0, "damping": 2.5, "armature": 1.0, "frictionloss": 0.3, "latency_frames": 2}
    back = decode_change(encode_change(c))
    for f in FIELDS:
        assert back[f] == pytest.approx(c[f])


def test_apply_change_keeps_per_joint_locks_per_joint():
    p = SimParams(kp=(100.0, 200.0, 300.0, 400.0, 500.0, 600.0), latency_frames=2)
    q = apply_change(p, {"kp": 0.5, "latency_frames": 1})
    assert q.kp == (50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
    assert q.latency_frames == 3


def test_describe_change():
    assert describe_change("kp", 0.7) == "-30%"
    assert describe_change("frictionloss", 2.0) == "+100%"
    assert describe_change("latency_frames", 1) == "+1 frame"
    assert describe_change("latency_frames", -2) == "-2 frames"


def test_split_alternating_is_disjoint_and_ordered():
    eps = make_episodes(TRUE, n=5)
    a, b = split_alternating(list(reversed(eps)))
    assert [e.index for e in a] == [0, 2, 4] and [e.index for e in b] == [1, 3]


def test_param_change_to_dict_is_json():
    c = ParamChange("latency_frames", 1, alone_deg=0.5, stale_deg=2.0)
    d = json.loads(json.dumps(c.to_dict()))
    assert d["change"] == "+1 frame" and d["recovers_deg"] == pytest.approx(1.5)


# -- diagnose ---------------------------------------------------------------------


def test_extra_lag_is_named():
    eps = make_episodes(TRUE.replace(latency_frames=3), seed=1)
    d = diagnose(eps, lock_of(), simulate=toy_simulate, max_evaluations=80)
    assert d.stale_deg > 1.0
    assert d.refit_deg < 0.05 * d.stale_deg
    assert d.explained and d.recovered > 0.9
    assert [c.name for c in d.causes][0] == "latency_frames"
    by = {c.name: c for c in d.changes}
    assert by["latency_frames"].value == 2
    assert "likely cause: latency_frames +2 frames" in d.summary()


def test_weaker_gain_is_named():
    eps = make_episodes(TRUE.replace(kp=TRUE.kp * 0.5), seed=2)
    d = diagnose(eps, lock_of(), simulate=toy_simulate, max_evaluations=120)
    assert d.explained
    # in the toy servo kp, kv and damping only enter through (kv+damping)/kp,
    # so any of them can carry the change; it must be named and alone suffice
    top = d.causes[0]
    assert top.name in ("kp", "kv", "damping")
    assert top.recovers_deg > 0.5 * d.gap_deg


def test_unchanged_arm_names_nothing():
    eps = make_episodes(TRUE, seed=3)
    d = diagnose(eps, lock_of(), simulate=toy_simulate, max_evaluations=40)
    assert d.stale_deg == pytest.approx(0.0, abs=1e-9)
    assert d.gap_deg == pytest.approx(0.0, abs=1e-9)
    assert d.causes == []
    assert all(c.recovers_deg == 0.0 for c in d.changes)


def test_drift_outside_the_dynamics_is_not_explained():
    """A sensor offset is not something the servo parameters can absorb."""
    eps = [Episode(e.index, e.fps, e.joint_names, e.state + 0.05, e.action)
           for e in make_episodes(TRUE, seed=4)]
    d = diagnose(eps, lock_of(eval_score=0.0), simulate=toy_simulate, max_evaluations=60)
    assert d.stale_deg > 2.0
    assert not d.explained
    assert "not explained by the sim's dynamics parameters" in d.summary()


def test_refit_and_held_out_halves_are_disjoint():
    eps = make_episodes(TRUE.replace(latency_frames=2), n=6)
    d = diagnose(eps, lock_of(), simulate=toy_simulate, max_evaluations=20)
    assert d.refit_episodes == [0, 2, 4] and d.held_out_episodes == [1, 3, 5]
    assert d.n_evaluations <= 20 + 12  # one generation may overshoot the budget


def test_accepts_a_robot_lock_and_serializes():
    from antares.lock import RobotLock

    result = CalibrationResult(
        params=TRUE, baseline=SimParams(), train_score=0.0, eval_score=0.0,
        baseline_train_score=1.0, baseline_eval_score=1.0, per_joint_before={},
        per_joint_after={}, history=[], n_evaluations=0, elapsed_s=0.0,
    )
    lock = RobotLock(name="unit-b", dataset="fake/unit_b", robot_type="so101_follower",
                     data_sha256="0" * 64, params=TRUE, result=result,
                     created_at="2026-10-17T00:00:00Z", antares_version="0")
    eps = make_episodes(TRUE.replace(latency_frames=2), n=4)
    d = diagnose(eps, lock, simulate=toy_simulate, max_evaluations=30, dataset="fresh")
    out = json.loads(json.dumps(d.to_dict()))
    assert out["lock"] == "unit-b" and out["dataset"] == "fresh"
    assert out["causes"] == [c.name for c in d.causes]
    assert {c["name"] for c in out["changes"]} == set(FIELDS)
    assert out["refit_params"]["latency_frames"] == d.refit_params.latency_frames


def test_needs_two_episodes():
    with pytest.raises(ValueError, match="two episodes"):
        diagnose(make_episodes(TRUE, n=1), lock_of(), simulate=toy_simulate)


# -- real MuJoCo arm on synthetic recordings -------------------------------------


@pytest.fixture(scope="module")
def arm_lock():
    """The committed SO-101 lock and a MuJoCo simulate."""
    from pathlib import Path

    from antares.calibrate import default_simulate
    from antares.lock import RobotLock

    path = Path(__file__).resolve().parent.parent / "locks" / "svla_so101_pickplace.lock.json"
    if not path.is_file():
        pytest.skip(f"no committed lock at {path}")
    return RobotLock.load(path), default_simulate()


def _recorded(lock, change, seed):
    from antares import synth

    truth = apply_change(lock.params, change)
    return synth.synthesize(truth, n_episodes=8, n_frames=150, seed=seed).episodes


@pytest.mark.slow
def test_real_arm_extra_lag_is_named(arm_lock):
    lock, simulate = arm_lock
    d = diagnose(_recorded(lock, {"latency_frames": 1}, seed=11), lock, simulate=simulate,
                 max_evaluations=60)
    assert d.explained, d.summary()
    assert d.causes and d.causes[0].name == "latency_frames", d.summary()
    assert d.causes[0].value == 1


@pytest.mark.slow
def test_real_arm_unchanged_names_nothing(arm_lock):
    lock, simulate = arm_lock
    d = diagnose(_recorded(lock, {}, seed=12), lock, simulate=simulate, max_evaluations=40)
    # only sensor noise is left: the refit barely improves and nothing is named
    assert d.gap_deg < 0.05, d.summary()
    assert d.causes == []
