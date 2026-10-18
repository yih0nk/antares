"""Tests for antares.diagnose with a toy first-order servo instead of MuJoCo.

"Real" recordings come from known parameters; drift is a known change on
top of the lock (extra lag, a weaker gain), so the diagnosis can be checked
against the truth. The slow tests at the bottom do the same with the real
MuJoCo arm on synthetic recordings.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from antares.calibrate import SimParams
from antares.data import Episode
from antares.diagnose import (
    FIELDS,
    apply_change,
    decode_change,
    describe_change,
    encode_change,
    neutral,
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
