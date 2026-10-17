"""Tests for the pre-deploy drift gate (``antares.check``).

The unit tests use a trivial ``simulate`` where the sim reproduces the
commanded trajectory exactly, so the tracking error is just the gap between
``state`` and ``action`` and can be dialled in directly. The slow tests at
the bottom run the real MuJoCo arm on the real SO-101 dataset against the
committed lock and inject synthetic drift.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from antares.calibrate import CalibrationResult, SimParams
from antares.check import (
    VERDICT_DRIFTED,
    VERDICT_INSUFFICIENT,
    VERDICT_OK,
    CheckResult,
    check,
    limit_for,
    select_episodes,
)
from antares.data import Episode
from antares.lock import RobotLock

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
REF = 1.0  # the lock's held-out score, degrees
PER_JOINT_REF = {j: 1.0 for j in JOINTS}


def exact_simulate(params, ep):
    """Sim == command; the error is whatever offset the episode carries."""
    return np.radians(ep.state), np.radians(ep.action)


def episodes(n=6, T=30, offset_deg=0.5, per_joint=None, start=0):
    """``state = action + offset`` so each joint's RMSE is exactly its offset."""
    off = np.full(6, float(offset_deg))
    if per_joint:
        for j, v in per_joint.items():
            off[JOINTS.index(j)] = v
    eps = []
    for i in range(start, start + n):
        action = np.linspace(0, 10, T)[:, None].repeat(6, axis=1) + i
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=action + off, action=action))
    return eps


def make_lock(eval_score=REF, per_joint_after=PER_JOINT_REF, name="unit-a") -> RobotLock:
    result = CalibrationResult(
        params=SimParams(kp=1200.0, latency_frames=1),
        baseline=SimParams(),
        train_score=0.9,
        eval_score=eval_score,
        baseline_train_score=4.0,
        baseline_eval_score=4.5,
        per_joint_before={j: 4.0 for j in JOINTS},
        per_joint_after=dict(per_joint_after),
        history=[(0, 4.0), (1, 0.9)],
        n_evaluations=1,
        elapsed_s=0.1,
        train_episodes=[0, 1],
        eval_episodes=[2],
    )
    return RobotLock.create("fake/so101_unit_a", "so101_follower", episodes(2), result, name=name)


# -- limit rule ---------------------------------------------------------------


def test_limit_is_max_of_threshold_and_ratio_times_reference():
    assert limit_for(1.0, None, 1.5) == pytest.approx(1.5)
    assert limit_for(1.0, 3.0, 1.5) == pytest.approx(3.0)
    assert limit_for(1.0, 0.2, 1.5) == pytest.approx(1.5)
    assert limit_for(None, 2.0, 1.5) == pytest.approx(2.0)
    assert limit_for(float("nan"), 2.0, 1.5) == pytest.approx(2.0)
    assert limit_for(None, None, 1.5) is None


# -- pass / fail ---------------------------------------------------------------


def test_passes_within_ratio_of_reference():
    res = check(episodes(offset_deg=1.2), make_lock(), simulate=exact_simulate)
    assert isinstance(res, CheckResult)
    assert res.passed is True
    assert res.verdict == VERDICT_OK
    assert res.score == pytest.approx(1.2)
    assert res.reference == pytest.approx(REF)
    assert res.limit == pytest.approx(1.5)
    assert res.n_episodes == 6 and res.n_frames == 6 * 30
    assert res.episodes == [0, 1, 2, 3, 4, 5]
    assert res.lock == "unit-a" and res.dataset == "fake/so101_unit_a"
    assert set(res.per_joint) == set(JOINTS)
    assert all(j.passed for j in res.per_joint.values())
    assert "gripper" not in res.scored_joints
    assert res.summary().startswith("OK:")
    assert "1.200 deg" in res.summary()


def test_fails_when_drifted_and_names_worst_joint():
    eps = episodes(offset_deg=1.2, per_joint={"elbow_flex": 4.0, "gripper": 9.0})
    res = check(eps, make_lock(), simulate=exact_simulate)
    assert res.passed is False
    assert res.verdict == VERDICT_DRIFTED
    # RMS over the five scored joints: sqrt((4*1.2^2 + 4^2)/5)
    assert res.score == pytest.approx(math.sqrt((4 * 1.2**2 + 16) / 5))
    assert res.score > res.limit
    assert res.per_joint["elbow_flex"].passed is False
    assert res.per_joint["elbow_flex"].limit == pytest.approx(1.5)
    assert res.per_joint["shoulder_pan"].passed is True
    # the gripper is the largest error but is not scored, so it is not "worst"
    assert res.per_joint["gripper"].passed is False
    assert res.worst_joint == "elbow_flex"
    assert res.summary().startswith("DRIFTED:")
    assert "worst joint elbow_flex" in res.summary()


def test_absolute_threshold_loosens_the_gate():
    eps = episodes(offset_deg=2.0)
    assert check(eps, make_lock(), simulate=exact_simulate).passed is False
    res = check(eps, make_lock(), threshold_deg=2.5, simulate=exact_simulate)
    assert res.passed is True
    assert res.limit == pytest.approx(2.5)
    assert res.threshold_deg == 2.5
    # a threshold below ratio*reference changes nothing
    assert check(eps, make_lock(), threshold_deg=0.1, simulate=exact_simulate).limit == pytest.approx(1.5)


def test_ratio_is_configurable():
    eps = episodes(offset_deg=2.0)
    assert check(eps, make_lock(), ratio=2.5, simulate=exact_simulate).passed is True
    assert check(eps, make_lock(), ratio=1.1, simulate=exact_simulate).passed is False


# -- episode selection ----------------------------------------------------------


def test_insufficient_data_does_not_simulate():
    calls = []

    def spy(params, ep):
        calls.append(ep.index)
        return exact_simulate(params, ep)

    res = check(episodes(n=3), make_lock(), min_episodes=5, simulate=spy)
    assert res.passed is False
    assert res.verdict == VERDICT_INSUFFICIENT
    assert res.score is None and res.per_joint == {} and res.worst_joint is None
    assert res.n_episodes == 3
    assert res.limit == pytest.approx(1.5)  # still reported, so the CLI can show it
    assert calls == []
    assert res.summary().startswith("INSUFFICIENT DATA")
    json.dumps(res.to_dict())


def test_last_n_takes_the_newest_episodes():
    eps = episodes(n=10, offset_deg=0.5)
    # the newest three carry drift; diluted over all ten the RMS stays under
    # the limit (sqrt((7*0.5^2 + 3*2^2)/10) = 1.17 < 1.5)
    eps[-3:] = episodes(n=3, offset_deg=2.0, start=7)
    assert check(eps, make_lock(), simulate=exact_simulate).passed is True
    res = check(list(reversed(eps)), make_lock(), last=3, min_episodes=3, simulate=exact_simulate)
    assert res.episodes == [7, 8, 9]
    assert res.passed is False
    assert check(eps, make_lock(), last=3, min_episodes=5, simulate=exact_simulate).verdict == VERDICT_INSUFFICIENT
    assert [e.index for e in select_episodes(eps, last=2)] == [8, 9]
    with pytest.raises(ValueError):
        select_episodes(eps, last=0)


# -- lock handling --------------------------------------------------------------


def test_accepts_lock_dict():
    lock = make_lock()
    as_dict = json.loads(json.dumps(lock.to_dict()))
    eps = episodes(offset_deg=1.2)
    a = check(eps, lock, simulate=exact_simulate)
    b = check(eps, as_dict, simulate=exact_simulate)
    assert a.to_dict() == b.to_dict()


def test_lock_without_eval_score_needs_threshold():
    lock = make_lock(eval_score=float("nan"))
    # calibrate() copies the train score into eval_score when there is no
    # held-out split; a NaN reference falls back the same way.
    res = check(episodes(offset_deg=1.0), lock, simulate=exact_simulate)
    assert res.reference == pytest.approx(0.9)
    broken = json.loads(json.dumps(lock.to_dict()))
    broken["result"]["train_score"] = None
    with pytest.raises(ValueError, match="threshold_deg"):
        check(episodes(), broken, simulate=exact_simulate)
    res = check(episodes(offset_deg=1.0), broken, threshold_deg=2.0, simulate=exact_simulate)
    assert res.passed is True and res.reference is None and res.limit == 2.0


def test_missing_per_joint_reference_is_informational():
    lock = make_lock(per_joint_after={"shoulder_pan": 1.0})
    res = check(episodes(offset_deg=1.2), lock, simulate=exact_simulate)
    assert res.per_joint["shoulder_pan"].limit == pytest.approx(1.5)
    assert res.per_joint["elbow_flex"].reference is None
    assert res.per_joint["elbow_flex"].limit is None
    assert res.per_joint["elbow_flex"].passed is True
    assert res.passed is True


def test_input_validation():
    with pytest.raises(ValueError):
        check(episodes(), make_lock(), ratio=0, simulate=exact_simulate)
    with pytest.raises(ValueError):
        check(episodes(), make_lock(), threshold_deg=-1, simulate=exact_simulate)
    with pytest.raises(ValueError):
        check(episodes(), make_lock(), min_episodes=0, simulate=exact_simulate)


def test_to_dict_is_json_and_round_trips_numbers():
    res = check(episodes(offset_deg=1.2), make_lock(), threshold_deg=2.0, simulate=exact_simulate)
    d = json.loads(json.dumps(res.to_dict()))
    assert d["passed"] is True and d["verdict"] == "ok"
    assert d["score"] == pytest.approx(1.2)
    assert d["reference"] == pytest.approx(1.0)
    assert d["limit"] == pytest.approx(2.0)
    assert d["per_joint"]["elbow_flex"] == {
        "score": pytest.approx(1.2),
        "reference": pytest.approx(1.0),
        "limit": pytest.approx(2.0),
        "passed": True,
    }
    assert d["worst_joint"] in JOINTS
    assert d["n_episodes"] == 6 and d["lock"] == "unit-a" and d["dataset"] == "fake/so101_unit_a"
    assert d["ratio"] == 1.5 and d["threshold_deg"] == 2.0


# -- real data: synthetic drift -------------------------------------------------

REAL_DATASET = "lerobot/svla_so101_pickplace"
REAL_LOCK = "locks/svla_so101_pickplace.lock.json"


@pytest.fixture(scope="module")
def real():
    """Real SO-101 dataset, the committed lock, and a MuJoCo simulate.

    Skips when the dataset is not cached and cannot be downloaded.
    """
    from pathlib import Path

    from antares.calibrate import default_simulate
    from antares.data import load_dataset

    lock_path = Path(__file__).resolve().parent.parent / REAL_LOCK
    if not lock_path.is_file():
        pytest.skip(f"no committed lock at {lock_path}")
    try:
        ds = load_dataset(REAL_DATASET)
    except Exception as exc:  # noqa: BLE001 - offline, no cache
        pytest.skip(f"{REAL_DATASET} unavailable: {exc}")
    lock = RobotLock.load(lock_path)
    held_out = [e for e in ds.episodes if e.index in set(lock.result.eval_episodes)]
    assert len(held_out) >= 5, "lock has too few held-out episodes for a check"
    return ds, lock, held_out, default_simulate()


def lagged(ep: Episode, frames: int) -> Episode:
    """The arm responds ``frames`` later than it did when the lock was fitted."""
    src = np.maximum(np.arange(len(ep)) - frames, 0)
    return Episode(ep.index, ep.fps, ep.joint_names, ep.state[src], ep.action)


def weakened(ep: Episode, factor: float) -> Episode:
    """A tired servo: every excursion from the start pose is scaled down."""
    state = ep.state[0] + factor * (ep.state - ep.state[0])
    return Episode(ep.index, ep.fps, ep.joint_names, state, ep.action)


@pytest.mark.slow
def test_real_held_out_episodes_pass(real):
    ds, lock, held_out, simulate = real
    res = check(held_out, lock, simulate=simulate)
    assert res.verdict == VERDICT_OK, res.summary()
    assert res.passed is True
    # Shipped locks predate the mj_setConst fix in Arm.apply (~1e-5 deg shift).
    assert res.score == pytest.approx(lock.result.eval_score, abs=1e-4)
    assert res.limit == pytest.approx(1.5 * lock.result.eval_score)
    assert sorted(res.episodes) == sorted(lock.result.eval_episodes)
    # the whole dataset and just the newest episodes also pass
    assert check(ds.episodes, lock, simulate=simulate).passed is True
    assert check(ds.episodes, lock, last=5, simulate=simulate).passed is True


@pytest.mark.slow
def test_real_extra_lag_is_caught(real):
    ds, lock, held_out, simulate = real
    clean = check(held_out, lock, simulate=simulate)
    res = check([lagged(e, 3) for e in held_out], lock, simulate=simulate)
    assert res.verdict == VERDICT_DRIFTED, res.summary()
    assert res.passed is False
    assert res.score > res.limit > clean.score
    assert res.score > 2 * clean.score
    assert res.worst_joint in res.scored_joints
    assert not res.per_joint[res.worst_joint].passed


@pytest.mark.slow
def test_real_weak_servo_is_caught(real):
    ds, lock, held_out, simulate = real
    res = check([weakened(e, 0.9) for e in held_out], lock, simulate=simulate)
    assert res.verdict == VERDICT_DRIFTED, res.summary()
    assert res.score > res.limit
    # drift concentrated in the newest recordings is caught by --last
    mixed = list(ds.episodes[:-5]) + [weakened(e, 0.9) for e in ds.episodes[-5:]]
    assert check(mixed, lock, last=5, simulate=simulate).passed is False
