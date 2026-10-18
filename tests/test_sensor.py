"""Tests for antares.sensor with a toy servo instead of MuJoCo.

Commands go through the command map, then a first-order lag whose time
constant comes from kp/kv, with a pure delay of ``latency_frames``. The
"real" arm uses hidden true dynamics and a hidden command map, so a
dynamics-only fit cannot reach zero error but the joint fit can.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from antares.calibrate import SimParams
from antares.data import Episode
from antares.sensor import FIT_JOINTS, calibrate_with_sensor, fit_command_map, prefit_command_map
from antares.sim import GRIPPER_INDEX, JOINTS, CommandMap

FPS = 30.0
RAD_PER_UNIT = 0.015
TRUE_PARAMS = SimParams(kp=400.0, kv=4.0, damping=0.6, armature=0.028, frictionloss=0.052, latency_frames=2)
TRUE_MAP = CommandMap(
    gain=[1.04, 0.95, 1.06, 1.0, 0.97, 1.0],
    bias=[2.0, -4.0, 3.0, -1.5, 1.0, 0.0],
)


def toy_q(p: SimParams, cmap: CommandMap, action: np.ndarray, q0: np.ndarray) -> np.ndarray:
    pj = p.per_joint
    tau = 20.0 * (pj("kv") + pj("damping")) / pj("kp") + pj("armature")
    alpha = 1.0 - np.exp(-1.0 / FPS / tau)
    ctrl = cmap.apply(action) * RAD_PER_UNIT
    src = np.maximum(np.arange(len(action))[:, None] - pj("latency_frames")[None, :], 0)
    ctrl = np.take_along_axis(ctrl, src, axis=0)
    q = np.empty_like(ctrl)
    q[0] = q0
    for t in range(1, len(ctrl)):
        q[t] = q[t - 1] + alpha * (ctrl[t] - q[t - 1])
    return q


def make_episodes(n: int, T: int = 150, seed: int = 0) -> list[Episode]:
    rng = np.random.default_rng(seed)
    t = np.arange(T) / FPS
    eps = []
    for i in range(n):
        action = np.zeros((T, 6))
        for j in range(6):
            for _ in range(3):
                f = rng.uniform(0.1, 0.8)
                action[:, j] += rng.uniform(-30, 30) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        q = toy_q(TRUE_PARAMS, TRUE_MAP, action, TRUE_MAP.apply(action[0]) * RAD_PER_UNIT)
        eps.append(Episode(index=i, fps=FPS, joint_names=JOINTS, state=q / RAD_PER_UNIT, action=action))
    return eps


def toy_simulate(params: SimParams, cmap: CommandMap, ep: Episode):
    real = ep.state * RAD_PER_UNIT
    return real, toy_q(params, cmap, ep.action, real[0])


def test_fit_command_map_recovers_truth_with_true_dynamics():
    eps = make_episodes(3)
    fit = fit_command_map(TRUE_PARAMS, eps, toy_simulate)
    assert fit.rms_after < 0.01 < fit.rms_before
    for j in FIT_JOINTS:
        assert fit.command_map.bias[j] == pytest.approx(TRUE_MAP.bias[j], abs=0.05)
        assert fit.command_map.gain[j] == pytest.approx(TRUE_MAP.gain[j], abs=0.002)
    # gripper is never fitted
    assert fit.command_map.bias[GRIPPER_INDEX] == 0.0
    assert fit.command_map.gain[GRIPPER_INDEX] == 1.0


def test_fit_command_map_bias_only():
    eps = make_episodes(2)
    fit = fit_command_map(TRUE_PARAMS, eps, toy_simulate, fit_gain=False)
    np.testing.assert_array_equal(fit.command_map.gain, np.ones(6))
    assert fit.rms_after < fit.rms_before


def test_fit_command_map_needs_episodes():
    with pytest.raises(ValueError):
        fit_command_map(TRUE_PARAMS, [], toy_simulate)


def test_calibrate_with_sensor_beats_dynamics_only():
    eps = make_episodes(6, seed=1)
    train, eval_ = eps[:4], eps[4:]
    events = []
    out = calibrate_with_sensor(
        train, eval_, simulate=toy_simulate, max_evaluations=60, on_progress=events.append
    )
    assert out.dynamics_only.eval_score < out.dynamics_only.baseline_eval_score
    assert out.result.eval_score < 0.5 * out.dynamics_only.eval_score
    # baseline of the combined result is still the default sim
    assert out.result.baseline_eval_score == out.dynamics_only.baseline_eval_score
    # fair baseline: the dynamics-only side gets the same budget, map fits included
    assert out.n_map_simulations > 0
    assert out.result.n_evaluations == out.dynamics_only.n_evaluations
    assert sum(s["n_evaluations"] for s in out.result.stages) == out.result.n_evaluations
    assert sum(s["n_evaluations"] for s in out.dynamics_only.stages) == out.dynamics_only.n_evaluations
    names = [s["name"] for s in out.result.stages]
    assert names == ["shared", "command_map", "shared", "command_map"]
    assert sum(s["n_evaluations"] for s in out.result.stages if s["name"] == "command_map") == out.n_map_simulations
    assert [s["name"] for s in out.dynamics_only.stages] == ["shared", "shared", "shared"]
    assert out.dynamics_only.stages[-1]["n_evaluations"] == out.n_map_simulations
    assert out.dynamics_only.elapsed_s > out.dynamics_only.stages[0].get("elapsed_s", 0)
    unfair = calibrate_with_sensor(train, eval_, simulate=toy_simulate, max_evaluations=60, fair_baseline=False)
    assert unfair.dynamics_only.n_evaluations < unfair.result.n_evaluations
    evals = [e.evaluation for e in events]
    assert evals == sorted(evals) and evals[-1] == out.result.n_evaluations
    assert {e.stage for e in events} == {"shared", "command_map"}
    d = out.to_dict()
    assert set(d) == {"result", "command_map", "dynamics_only", "n_map_simulations"}
    for j in FIT_JOINTS:
        assert out.command_map.bias[j] == pytest.approx(TRUE_MAP.bias[j], abs=1.0)


def test_calibrate_with_sensor_stops_early():
    eps = make_episodes(3)
    out = calibrate_with_sensor(eps[:2], eps[2:], simulate=toy_simulate, max_evaluations=30, should_stop=lambda: True)
    assert out.result.n_evaluations == 0


def test_calibrate_with_sensor_stops_inside_the_map_fit():
    eps = make_episodes(3)
    calls = {"n": 0}

    def counting(params, cmap, ep):
        calls["n"] += 1
        return toy_simulate(params, cmap, ep)

    # let round 0 finish (30 evaluations + the start), then stop during the first map fit
    out = calibrate_with_sensor(eps[:2], eps[2:], simulate=counting, max_evaluations=30,
                                should_stop=lambda: calls["n"] >= 2 * 33)
    assert out.result.n_evaluations < 40  # 30 + the start, a few map-fit passes, no refit
    assert calls["n"] < 2 * 50  # nothing like the 71 passes an unstopped map fit could take
    assert [s["name"] for s in out.result.stages][:2] == ["shared", "command_map"]
    assert out.command_map.is_identity() or out.result.n_evaluations > 31


def test_fit_command_map_stop_returns_best_so_far():
    eps = make_episodes(2)
    n = {"sims": 0}

    def counting(params, cmap, ep):
        n["sims"] += 1
        return toy_simulate(params, cmap, ep)

    fit = fit_command_map(TRUE_PARAMS, eps, counting, should_stop=lambda: n["sims"] >= 3 * len(eps))
    assert fit.n_simulations <= 3
    assert fit.rms_after <= fit.rms_before


def test_fit_command_map_coupled_jacobian_matches_block_diagonal():
    eps = make_episodes(3)
    fast = fit_command_map(TRUE_PARAMS, eps, toy_simulate)
    slow = fit_command_map(TRUE_PARAMS, eps, toy_simulate, coupled=True)
    assert fast.n_simulations < slow.n_simulations
    for j in FIT_JOINTS:
        assert fast.command_map.bias[j] == pytest.approx(slow.command_map.bias[j], abs=0.05)
        assert fast.command_map.gain[j] == pytest.approx(slow.command_map.gain[j], abs=0.005)


def test_calibrate_with_sensor_accepts_sigma0_for_round_zero():
    eps = make_episodes(3)
    out = calibrate_with_sensor(eps[:2], eps[2:], simulate=toy_simulate, max_evaluations=10, sigma0=0.2)
    assert out.result.n_evaluations > 10


@pytest.mark.slow
def test_fit_command_map_on_real_arm_recovers_offsets():
    """MuJoCo arm, true dynamics known: the fit recovers a hidden command offset."""
    from antares.sensor import sensor_simulate
    from antares.sim import Arm

    hidden_map = CommandMap(gain=[1.0, 0.97, 1.03, 1.0, 1.0, 1.0], bias=[2.0, -3.0, 2.0, -1.0, 1.0, 0.0])
    params = SimParams(kp=200.0, latency_frames=2)
    hidden = Arm(params=params, command_map=hidden_map)
    rng = np.random.default_rng(0)
    T = 90
    t = np.arange(T) / FPS
    eps = []
    for i in range(2):
        action = np.zeros((T, 6))
        for j in range(5):
            action[:, j] = 25 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t + rng.uniform(0, 6))
        action[:, 5] = 40.0
        q0 = hidden.unit_map.to_rad(hidden_map.apply(action[0]))
        seed = Episode(i, FPS, JOINTS, hidden.unit_map.to_units(np.tile(q0, (T, 1))), action)
        state = hidden.unit_map.to_units(hidden.replay(seed))
        eps.append(Episode(i, FPS, JOINTS, state, action))

    fit = fit_command_map(params, eps, sensor_simulate())
    assert fit.rms_after < 0.1 * fit.rms_before
    for j in FIT_JOINTS:
        assert fit.command_map.bias[j] == pytest.approx(hidden_map.bias[j], abs=0.3)
        assert fit.command_map.gain[j] == pytest.approx(hidden_map.gain[j], abs=0.01)


def make_step_episodes(n: int, T: int = 240, hold: int = 40, seed: int = 0) -> list[Episode]:
    """Piecewise-constant commands, so the arm settles between moves."""
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n):
        levels = rng.uniform(-40, 40, size=(T // hold + 1, 6))
        action = np.repeat(levels, hold, axis=0)[:T]
        q = toy_q(TRUE_PARAMS, TRUE_MAP, action, TRUE_MAP.apply(action[0]) * RAD_PER_UNIT)
        eps.append(Episode(index=i, fps=FPS, joint_names=JOINTS, state=q / RAD_PER_UNIT, action=action))
    return eps


def test_prefit_command_map_from_still_frames():
    cmap = prefit_command_map(make_step_episodes(3))
    for j in FIT_JOINTS:
        assert cmap.bias[j] == pytest.approx(TRUE_MAP.bias[j], abs=0.1)
        assert cmap.gain[j] == pytest.approx(TRUE_MAP.gain[j], abs=0.005)
    assert cmap.gain[GRIPPER_INDEX] == 1.0 and cmap.bias[GRIPPER_INDEX] == 0.0


def test_prefit_keeps_identity_without_still_frames():
    # the multi-sine episodes never hold still
    assert prefit_command_map(make_episodes(2)).is_identity()


def test_calibrate_with_sensor_kinematic_mode():
    eps = make_step_episodes(6, seed=2)
    out = calibrate_with_sensor(eps[:4], eps[4:], simulate=toy_simulate, max_evaluations=60, mode="kinematic")
    assert out.result.eval_score < 0.5 * out.dynamics_only.eval_score
    for j in FIT_JOINTS:
        assert out.command_map.bias[j] == pytest.approx(TRUE_MAP.bias[j], abs=0.2)
    with pytest.raises(ValueError):
        calibrate_with_sensor(eps[:1], [], simulate=toy_simulate, mode="nope")


def test_calibrate_with_sensor_per_joint():
    eps = make_episodes(4, seed=3)
    out = calibrate_with_sensor(eps[:3], eps[3:], simulate=toy_simulate, max_evaluations=30,
                                per_joint=True, per_joint_max_evaluations=30)
    # the per-joint stage ran in every dynamics fit (here it never beat the shared
    # optimum, so each refit starts shared); the equal-budget extra refit is shared too
    assert [st["name"] for st in out.result.stages] == \
        ["shared", "per_joint", "command_map", "shared", "per_joint", "command_map"]
    assert [st["name"] for st in out.dynamics_only.stages] == \
        ["shared", "per_joint", "shared", "per_joint", "shared"]
    assert out.result.n_evaluations == out.dynamics_only.n_evaluations
    assert out.result.eval_score < out.dynamics_only.baseline_eval_score


def test_per_joint_fit_is_refit_in_its_own_space(monkeypatch):
    """When a round ends on per-joint parameters, the next dynamics refit must
    not go back through the shared stage (which would average them)."""
    import antares.sensor as sensor
    from antares.calibrate import CalibrationResult

    pj = TRUE_PARAMS.replace(kp=(400.0, 410.0, 390.0, 400.0, 400.0, 400.0))
    calls = []

    def fake_calibrate(train, eval, simulate=None, start=None, max_evaluations=150, **kw):
        calls.append({"start": start, "max_evaluations": max_evaluations, **kw})
        n = max_evaluations + (kw.get("per_joint_max_evaluations", 50) if kw.get("per_joint") else 0)
        stages = ([{"name": "shared", "n_evaluations": max_evaluations, "train_score": 1.0, "params": {}}]
                  if max_evaluations else [])
        if kw.get("per_joint"):
            stages.append({"name": "per_joint", "n_evaluations": n - max_evaluations, "train_score": 1.0, "params": {}})
        return CalibrationResult(params=pj, baseline=start or TRUE_PARAMS, train_score=1.0, eval_score=1.0,
                                 baseline_train_score=2.0, baseline_eval_score=2.0, per_joint_before={},
                                 per_joint_after={}, history=[(0, 2.0)], n_evaluations=n, elapsed_s=0.1,
                                 space="per_joint", stages=stages)

    monkeypatch.setattr(sensor, "calibrate", fake_calibrate)
    eps = make_episodes(3)
    out = calibrate_with_sensor(eps[:2], eps[2:], simulate=toy_simulate, max_evaluations=30,
                                per_joint=True, per_joint_max_evaluations=20, sigma0=0.4)
    # round 0: the caller's sigma0 and both stages; refits: per-joint only, refit sigma
    assert calls[0]["max_evaluations"] == 30 and calls[0]["sigma0"] == 0.4 and calls[0]["per_joint"]
    refits = calls[1:]
    assert all(c["max_evaluations"] == 0 and c["per_joint"] and c["sigma0"] == 0.15 for c in refits)
    assert all(c["start"].has_per_joint for c in refits)
    # the equal-budget extra refit for the dynamics-only side spends the map simulations
    assert refits[-1]["per_joint_max_evaluations"] == out.n_map_simulations > 0
    assert out.result.n_evaluations == out.dynamics_only.n_evaluations
    assert out.result.space == "per_joint"
    assert [s["name"] for s in out.result.stages] == \
        ["shared", "per_joint", "command_map", "per_joint", "command_map"]
    assert [s["name"] for s in out.dynamics_only.stages] == ["shared", "per_joint", "per_joint", "per_joint"]
