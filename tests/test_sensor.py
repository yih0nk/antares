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
from antares.sensor import FIT_JOINTS, fit_command_map
from antares.sim import GRIPPER_INDEX, JOINTS, CommandMap

FPS = 30.0
RAD_PER_UNIT = 0.015
TRUE_PARAMS = SimParams(kp=400.0, kv=4.0, damping=0.6, armature=0.028, frictionloss=0.052, latency_frames=2)
TRUE_MAP = CommandMap(
    gain=[1.04, 0.95, 1.06, 1.0, 0.97, 1.0],
    bias=[2.0, -4.0, 3.0, -1.5, 1.0, 0.0],
)


def toy_q(p: SimParams, cmap: CommandMap, action: np.ndarray, q0: np.ndarray) -> np.ndarray:
    tau = 20.0 * (p.kv + p.damping) / p.kp + p.armature
    alpha = 1.0 - math.exp(-1.0 / FPS / tau)
    ctrl = cmap.apply(action) * RAD_PER_UNIT
    src = np.maximum(np.arange(len(action)) - p.latency_frames, 0)
    ctrl = ctrl[src]
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
