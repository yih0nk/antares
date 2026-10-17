"""Tests for antares.drift with the toy first-order servo from test_calibrate.

The "real" robot here has a stiffness that decays linearly over the session,
so a lock fitted at the start should score worse and worse on later windows.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from antares.calibrate import SimParams
from antares.data import Episode
from antares.drift import (
    DriftScan,
    DriftWindow,
    Trend,
    compute_trends,
    drift_scan,
    fit_trend,
    window_slices,
)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
START = SimParams(kp=400.0, kv=4.0, damping=2.0, armature=0.1, frictionloss=0.05, latency_frames=2)


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


def fake_simulate(params: SimParams, ep: Episode):
    return ep.state, toy_response(params, ep.action, ep.fps)


def session(n: int = 30, decay: float = 0.4, T: int = 100, fps: float = 30.0, seed: int = 0) -> list[Episode]:
    """Episodes whose true kp falls linearly by ``decay`` from first to last."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    eps = []
    for i in range(n):
        action = np.zeros((T, 6))
        for j in range(6):
            for _ in range(3):
                f = rng.uniform(0.2, 1.2)
                action[:, j] += rng.uniform(-0.6, 0.6) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        action[:, 5] = np.abs(action[:, 5])
        true = START.replace(kp=START.kp * (1.0 - decay * i / max(n - 1, 1)))
        state = toy_response(true, action, fps)
        eps.append(Episode(index=i, fps=fps, joint_names=JOINTS, state=state, action=action))
    return eps


class FakeLock:
    def __init__(self, params=START, name="toy", dataset="toy/session"):
        self.params, self.name, self.dataset = params, name, dataset


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


def test_window_slices():
    s = window_slices(50, 10, 5)
    assert len(s) == 9
    assert list(s[0]) == list(range(0, 10)) and list(s[-1]) == list(range(40, 50))
    assert len(window_slices(50, 15, 5)) == 8
    assert len(window_slices(10, 10, 5)) == 1
    with pytest.raises(ValueError):
        window_slices(5, 10, 5)
    with pytest.raises(ValueError):
        window_slices(50, 0, 5)
    with pytest.raises(ValueError):
        window_slices(50, 10, 0)


# ---------------------------------------------------------------------------
# trends
# ---------------------------------------------------------------------------


def test_fit_trend_exact_line():
    x = np.arange(9) * 5 + 4.5
    t = fit_trend(x, 2.0 + 0.5 * x)
    assert t.slope == pytest.approx(0.5)
    assert t.intercept == pytest.approx(2.0)
    assert t.r2 == pytest.approx(1.0)
    assert t.ci_lo == pytest.approx(0.5, abs=1e-9) and t.ci_hi == pytest.approx(0.5, abs=1e-9)
    assert t.significant and t.drifting
    first, last = 2.0 + 0.5 * x[0], 2.0 + 0.5 * x[-1]
    assert t.total_change == pytest.approx((last - first) / first)
    assert t.n == 9 and t.log_space is False


def test_fit_trend_log_space():
    x = np.arange(9) * 5.0
    y = 100.0 * 10.0 ** (-0.01 * x)  # 40 episodes -> 10**-0.4 = -60%
    t = fit_trend(x, y, log_space=True)
    assert t.log_space
    assert t.slope == pytest.approx(-0.01)
    assert t.total_change == pytest.approx(10 ** (-0.4) - 1)
    assert t.drifting
    # the same relative change is not drifting if the threshold is above it
    assert not fit_trend(x, y, log_space=True, threshold=0.9).drifting
    # non-positive values are dropped before the log
    t2 = fit_trend([0, 1, 2, 3], [1.0, 0.0, 10.0, 100.0], log_space=True)
    assert t2.n == 3


def test_fit_trend_constant_and_degenerate():
    t = fit_trend(np.arange(5), np.full(5, 3.0))
    assert t.slope == pytest.approx(0.0, abs=1e-12) and t.r2 == 0.0
    assert t.total_change == pytest.approx(0.0, abs=1e-12)
    assert not t.significant and not t.drifting
    nan = fit_trend([1.0], [2.0])
    assert math.isnan(nan.slope) and not nan.drifting and nan.n == 1
    same_x = fit_trend([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    assert math.isnan(same_x.slope) and not same_x.drifting
    two = fit_trend([0.0, 1.0], [0.0, 1.0])  # slope defined, no bootstrap
    assert two.slope == pytest.approx(1.0) and math.isnan(two.ci_lo) and not two.drifting
    with pytest.raises(ValueError):
        fit_trend([0, 1], [0])


def test_fit_trend_noise_is_not_drift():
    rng = np.random.default_rng(3)
    x = np.arange(9) * 5.0
    y = 1.0 + 0.3 * rng.normal(size=9)
    t = fit_trend(x, y, seed=1)
    assert not t.significant and not t.drifting
    assert t.ci_lo < 0 < t.ci_hi


def test_fit_trend_relative_change_from_zero():
    # a linear quantity starting at 0 measures its change against the end value
    t = fit_trend(np.arange(5), [0.0, 0.25, 0.5, 0.75, 1.0])
    assert t.total_change == pytest.approx(1.0)


def _windows(kp_factor, lock_scores, other=START):
    out = []
    for i, (f, s) in enumerate(zip(kp_factor, lock_scores)):
        eps = list(range(5 * i, 5 * i + 10))
        out.append(DriftWindow(eps, other.replace(kp=other.kp * f), 0.5, s))
    return out


def test_compute_trends_flags_only_the_moving_quantities():
    rng = np.random.default_rng(0)
    n = 9
    kp_factor = np.linspace(1.0, 0.75, n) * (1 + 0.01 * rng.normal(size=n))
    lock_scores = np.linspace(1.0, 1.6, n) + 0.02 * rng.normal(size=n)
    tr = compute_trends(_windows(kp_factor, lock_scores))
    assert set(tr) == set(START.to_dict()) | {"lock_score", "train_score"}
    assert tr["kp"].drifting and tr["kp"].log_space and tr["kp"].slope < 0
    assert tr["kp"].total_change == pytest.approx(-0.25, abs=0.05)
    assert tr["lock_score"].drifting and tr["lock_score"].slope > 0
    for k in ("kv", "damping", "armature", "frictionloss", "latency_frames", "train_score"):
        assert not tr[k].drifting, k
    assert tr["latency_frames"].slope == pytest.approx(0.0, abs=1e-12)
    assert not tr["latency_frames"].log_space
    assert compute_trends([]) == {}


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------


def test_drift_scan_sees_the_lock_degrade():
    eps = session(n=30, decay=0.4)
    lock = FakeLock()
    seen = []
    scan = drift_scan(
        eps, lock, window=6, step=3, max_evaluations=24, seed=0,
        on_progress=seen.append, simulate=fake_simulate, n_boot=300,
    )
    assert scan.lock == "toy" and scan.dataset == "toy/session"
    assert scan.window == 6 and scan.step == 3 and scan.n_windows_planned == 9
    assert len(scan.windows) == 9 and not scan.cancelled
    assert [w.first for w in scan.windows] == list(range(0, 25, 3))
    assert all(w.last - w.first == 5 and len(w.episodes) == 6 for w in scan.windows)
    assert scan.windows[0].center == pytest.approx(2.5)

    # progress: one event per window, in order
    assert [p.index for p in seen] == list(range(9))
    assert all(p.n_windows == 9 for p in seen)
    assert seen[-1].window is scan.windows[-1]

    # the first window is where the lock was "fitted": the lock is nearly perfect there
    assert scan.windows[0].lock_score < 0.5
    # ... and gets worse along the session; the refit always does at least as well
    lock_scores = [w.lock_score for w in scan.windows]
    assert lock_scores[-1] > 3 * lock_scores[0]
    assert all(w.train_score <= w.lock_score + 1e-12 for w in scan.windows)
    assert scan.windows[-1].train_score < 0.5 * scan.windows[-1].lock_score
    assert all(w.n_evaluations == 24 for w in scan.windows)

    t = scan.trends["lock_score"]
    assert t.slope > 0 and t.drifting and t.r2 > 0.8
    assert scan.drifting["lock_score"] is True
    assert "lock_score" in scan.drifting_keys
    assert scan.verdict().startswith("drift:") and "lock_score" in scan.verdict()
    # the toy only identifies its time constant (kp is confounded with
    # kv/damping), so check that the refit made the arm slower, as the data did
    def tau(p):
        return 20.0 * (p.kv + p.damping) / p.kp + p.armature

    assert tau(scan.windows[-1].params) > 1.2 * tau(scan.windows[0].params)

    d = scan.to_dict()
    json.dumps(d)  # strictly serialisable
    assert d["drifting"]["lock_score"] is True
    assert len(d["windows"]) == 9 and d["windows"][0]["episodes"] == list(range(6))
    assert d["windows"][0]["params"]["kp"] > 0
    assert set(d["trends"]) == set(scan.trends)
    assert d["verdict"] == scan.verdict()


def test_drift_scan_static_session_has_no_drift():
    eps = session(n=20, decay=0.0)
    scan = drift_scan(eps, FakeLock(), window=5, step=5, max_evaluations=6,
                      simulate=fake_simulate, n_boot=200)
    assert len(scan.windows) == 4
    # the lock is the true model: score ~0 everywhere, nothing significant
    assert all(w.lock_score < 1e-6 for w in scan.windows)
    assert not any(scan.drifting.values())
    assert scan.verdict().startswith("no parameter or score drifts")


def test_drift_scan_save_load_round_trip(tmp_path):
    eps = session(n=12, decay=0.2)
    scan = drift_scan(eps, FakeLock(name="unit-a"), window=4, step=4, max_evaluations=4,
                      simulate=fake_simulate, n_boot=50)
    path = scan.save(tmp_path / "unit-a.drift.json")
    assert path.name == "unit-a.drift.json"
    assert scan.default_path(tmp_path) == path
    back = DriftScan.load(path)
    assert back.to_dict() == scan.to_dict()
    assert back.windows[1].params == scan.windows[1].params
    assert isinstance(back.trends["kp"], Trend)


def test_drift_scan_should_stop_and_errors():
    eps = session(n=12, decay=0.2)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 1  # let the first window run, then stop

    scan = drift_scan(eps, FakeLock(), window=4, step=4, max_evaluations=3,
                      simulate=fake_simulate, should_stop=stop, n_boot=10)
    assert scan.cancelled and scan.n_windows_planned == 3
    assert len(scan.windows) <= 1
    assert not any(scan.drifting.values())
    assert scan.verdict() == "too few windows to estimate a trend"
    with pytest.raises(ValueError):
        drift_scan(eps[:3], FakeLock(), window=4, simulate=fake_simulate)
