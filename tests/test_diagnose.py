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
    # so the change can be split between them in any proportion (and CMA-ES
    # splits it differently across platforms); a gain must lead the causes
    # and the refit must close the gap
    assert d.causes[0].name in ("kp", "kv", "damping")
    assert d.causes[0].recovers_deg > 0.3 * d.gap_deg
    assert d.refit_deg < 0.1 * d.stale_deg


def test_unchanged_arm_names_nothing():
    eps = make_episodes(TRUE, seed=3)
    d = diagnose(eps, lock_of(), simulate=toy_simulate, max_evaluations=40)
    assert d.stale_deg == pytest.approx(0.0, abs=1e-9)
    assert d.gap_deg == pytest.approx(0.0, abs=1e-9)
    assert d.causes == []
    assert all(c.recovers_deg == 0.0 for c in d.changes)
    assert d.unchanged and "no change found" in d.summary()
    assert d.to_dict()["unchanged"] is True


def test_lag_and_a_weaker_gain_together_are_both_named():
    eps = make_episodes(TRUE.replace(latency_frames=3, kp=TRUE.kp * 0.5), seed=5)
    d = diagnose(eps, lock_of(), simulate=toy_simulate)
    assert d.explained, d.summary()
    names = [c.name for c in d.causes]
    assert "latency_frames" in names and {c.name: c for c in d.changes}["latency_frames"].value == 2
    assert set(names) & {"kp", "kv", "damping"}  # the gain change is named too
    assert not d.unchanged


def test_a_change_too_small_to_matter_is_not_kept():
    """A gain that moved 0.2% lowers the error less than the step threshold."""
    eps = make_episodes(TRUE.replace(kp=TRUE.kp * 0.998), seed=6)
    d = diagnose(eps, lock_of(), simulate=toy_simulate)
    assert all(c.value == (0 if c.name == "latency_frames" else 1.0) for c in d.changes), d.changes
    assert d.unchanged and d.causes == []


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
    assert d.n_evaluations == 20  # the budget is exact: no generation overshoots it


def test_budget_is_exact_and_bounded_below():
    eps = make_episodes(TRUE.replace(latency_frames=2), n=4)
    calls = []
    d = diagnose(eps, lock_of(), simulate=lambda p, e: calls.append(1) or toy_simulate(p, e),
                 max_evaluations=37, on_progress=lambda n, best: calls.append(("p", n, best)))
    assert d.n_evaluations == 37
    progress = [c for c in calls if isinstance(c, tuple)]
    assert [n for _, n, _ in progress] == list(range(1, 38))
    assert all(b2 <= b1 for (_, _, b1), (_, _, b2) in zip(progress, progress[1:]))  # best never worsens
    with pytest.raises(ValueError, match="max_evaluations must be >= 10"):
        diagnose(eps, lock_of(), simulate=toy_simulate, max_evaluations=9)


def test_should_stop_ends_the_search_with_the_best_so_far():
    eps = make_episodes(TRUE.replace(latency_frames=3), n=4, seed=1)
    seen = {"n": 0}

    def counting(p, e):
        seen["n"] += 1
        return toy_simulate(p, e)

    # stop after the latency scan (6 shifts x 2 refit episodes)
    d = diagnose(eps, lock_of(), simulate=counting, max_evaluations=200, should_stop=lambda: seen["n"] >= 12)
    assert d.n_evaluations <= 7
    by = {c.name: c for c in d.changes}
    assert by["latency_frames"].value == 2  # the scan alone already found the lag
    assert all(c.value == 1.0 for c in d.changes if c.name != "latency_frames")


def test_causes_are_empty_when_not_explained():
    """Consumers that read only `causes` (the Action's output) must agree with the summary."""
    changes = [ParamChange(f, 0 if f == "latency_frames" else 1.0, alone_deg=4.0, stale_deg=4.0) for f in FIELDS]
    changes[0] = ParamChange("kp", 0.7, alone_deg=3.6, stale_deg=4.0)  # alone recovers 0.4 of a 1.0 gap
    from antares.diagnose import Diagnosis

    d = Diagnosis(lock="l", dataset="d", reference_deg=1.0, stale_deg=4.0, refit_deg=3.0,
                  per_joint_stale={}, per_joint_refit={}, changes=changes, refit_episodes=[0],
                  held_out_episodes=[1], n_evaluations=0, elapsed_s=0.0)
    assert d.recovered == pytest.approx(1.0 / 3.0) and not d.explained
    assert d.causes == [] and d.to_dict()["causes"] == []
    assert "not explained" in d.summary()
    explained = Diagnosis(**{**d.__dict__, "refit_deg": 2.5})  # gap 1.5 = half the excess
    assert explained.explained and [c.name for c in explained.causes] == ["kp"]


def test_lock_without_a_reference_cannot_be_judged():
    eps = make_episodes(TRUE.replace(latency_frames=3), seed=1)
    lock = {"name": "hand", "dataset": "fake/hand", "params": TRUE.to_dict(), "result": {}}
    d = diagnose(eps, lock, simulate=toy_simulate, max_evaluations=40)
    assert d.gap_deg > 1.0  # the refit itself works
    assert d.reference_deg is None and d.excess_deg is None and d.recovered is None
    assert not d.explained and d.causes == []
    assert "reference unknown" in d.summary() and "cannot be judged" in d.summary()
    out = json.loads(json.dumps(d.to_dict()))
    assert out["recovered"] is None and out["excess_deg"] is None and out["causes"] == []


def test_latency_shifts_the_lock_clamps_alike_are_scanned_once():
    from antares.diagnose import effective_shift

    p = TRUE.replace(latency_frames=1)
    assert [effective_shift(s, p) for s in range(-2, 4)] == [-1, -1, 0, 1, 2, 3]
    pj = TRUE.replace(latency_frames=(0, 1, 2, 3, 4, 0))
    assert effective_shift(-2, pj) == 0  # one joint is already at zero: no uniform shift down exists
    eps = make_episodes(p, n=4)
    scanned = []
    d = diagnose(eps, lock_of(p), simulate=lambda q, e: scanned.append(q.latency_frames) or toy_simulate(q, e),
                 max_evaluations=10)
    assert d.n_evaluations == 10  # 5 distinct shifts scanned, not 6, then the refit
    assert set(scanned[:10]) == {0, 2, 4, 6, 8} or sorted(set(scanned[:10])) == [0, 1, 2, 3, 4]


def test_causes_are_gated_on_the_lock_pinned_parameters():
    eps = make_episodes(TRUE.replace(latency_frames=3), seed=1)
    plain = diagnose(eps, lock_of(), simulate=toy_simulate, max_evaluations=80)
    assert plain.pinned is None and plain.causes and plain.causes[0].name == "latency_frames"
    assert plain.unpinned_causes == [] and plain.to_dict()["pinned"] is None

    def seeded(pinned):
        return {**lock_of(), "result": {"eval_score": 0.0, "seeds": {"n": 3, "pinned": list(pinned)}}}

    # the lock pins latency: the same diagnosis names it
    d = diagnose(eps, seeded(["latency_frames"]), simulate=toy_simulate, max_evaluations=80)
    assert d.pinned == ["latency_frames"] and d.explained
    assert [c.name for c in d.causes] == [c.name for c in plain.causes if c.name == "latency_frames"]
    assert "likely cause: latency_frames +2 frames" in d.summary()
    assert d.to_dict()["causes"] == ["latency_frames"] and d.to_dict()["pinned"] == ["latency_frames"]

    # the lock pins nothing of use: the change is found but no cause is named
    d = diagnose(eps, seeded(["kp"]), simulate=toy_simulate, max_evaluations=80)
    assert d.explained and d.candidate_causes and d.causes == []
    assert [c.name for c in d.unpinned_causes] == [c.name for c in d.candidate_causes]
    assert "no pinned parameter explains it alone" in d.summary()
    assert "latency_frames would explain it but move across seeds at equal error" in d.summary()
    out = json.loads(json.dumps(d.to_dict()))
    assert out["causes"] == [] and out["unpinned_causes"] == [c.name for c in d.candidate_causes]
    assert out["pinned"] == ["kp"] and out["explained"] is True


def test_weaker_gain_with_only_latency_pinned_names_nothing():
    eps = make_episodes(TRUE.replace(kp=TRUE.kp * 0.5), seed=2)
    lock = {**lock_of(), "result": {"eval_score": 0.0, "seeds": {"n": 3, "pinned": ["latency_frames"]}}}
    d = diagnose(eps, lock, simulate=toy_simulate, max_evaluations=120)
    assert d.explained and d.candidate_causes
    assert all(c.name in ("kp", "kv", "damping") for c in d.candidate_causes)
    assert d.causes == [] and "not read as wear" in d.summary()


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


# -- real SO-101 dataset ---------------------------------------------------------


@pytest.fixture(scope="module")
def real(arm_lock):
    """The lock's held-out episodes from the real SO-101 dataset.

    Skips when the dataset is not cached and cannot be downloaded.
    """
    from antares.data import load_dataset

    lock, simulate = arm_lock
    try:
        ds = load_dataset(lock.dataset)
    except Exception as exc:  # noqa: BLE001 - offline, no cache
        pytest.skip(f"{lock.dataset} unavailable: {exc}")
    held_out = [e for e in ds.episodes if e.index in set(lock.result.eval_episodes)]
    assert len(held_out) >= 4
    return lock, held_out, simulate


def _lagged(ep: Episode, frames: int) -> Episode:
    src = np.maximum(np.arange(len(ep)) - frames, 0)
    return Episode(ep.index, ep.fps, ep.joint_names, ep.state[src], ep.action)


def _weakened(ep: Episode, factor: float) -> Episode:
    return Episode(ep.index, ep.fps, ep.joint_names, ep.state[0] + factor * (ep.state - ep.state[0]), ep.action)


@pytest.mark.slow
def test_real_unchanged_recordings_report_no_change(real):
    lock, held_out, simulate = real
    d = diagnose(held_out, lock, simulate=simulate)
    assert d.causes == [], d.summary()
    assert d.unchanged, d.summary()


@pytest.mark.slow
def test_real_extra_lag_is_named_as_lag(real):
    """On the real arm a higher kv can mimic a frame of lag; the lag must win."""
    lock, held_out, simulate = real
    d = diagnose([_lagged(e, 2) for e in held_out], lock, simulate=simulate)
    assert d.explained, d.summary()
    assert d.causes[0].name == "latency_frames" and d.causes[0].value == 2, d.summary()
    kv = {c.name: c for c in d.changes}["kv"]
    assert 0.5 < kv.value < 2.0, d.summary()  # no large velocity-gain swing standing in for it


@pytest.mark.slow
def test_real_scaled_motion_is_not_explained(real):
    lock, held_out, simulate = real
    d = diagnose([_weakened(e, 0.9) for e in held_out], lock, simulate=simulate)
    assert not d.explained and d.causes == [], d.summary()
