"""Fit each arm's sensor calibration: the per-joint :class:`~antares.sim.CommandMap`.

A LeRobot SO-100/101 dataset normalizes the commands (the leader arm's
readings) and the measured state (the follower's) with two separate
per-arm calibrations. Any disagreement between them shows up in the sim as
a steady per-joint offset or a range mismatch that no dynamics parameter
can model, so the CMA-ES fit bends kp/latency to hide it instead.

This module fits ``u' = gain*u + bias`` per body joint *through the sim*
(Levenberg-Marquardt on the replay residual, finite-difference Jacobian),
holding the dynamics fixed, and alternates that with the dynamics fit of
:func:`antares.calibrate.calibrate`. Fitting through the sim rather than
regressing state on commands keeps gravity sag, which the sim already
models with kp, out of the bias.

The gripper is left at identity: it is excluded from the score, and its
error is dominated by unlogged object contacts.

Simulation is injected as ``simulate(params, command_map, episode) ->
(real_q, sim_q)`` so tests do not need MuJoCo.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from antares.calibrate import CalibrationResult, ProgressEvent, SimParams, calibrate
from antares.data import Episode
from antares.sim import GRIPPER_INDEX, JOINTS, CommandMap

SensorSimulateFn = Callable[[SimParams, CommandMap, Episode], tuple[np.ndarray, np.ndarray]]

#: Joints whose command map is fitted (the scored body joints).
FIT_JOINTS: tuple[int, ...] = tuple(i for i in range(len(JOINTS)) if i != GRIPPER_INDEX)

#: Bounds, in dataset units for bias. Wider than any sane calibration error.
BIAS_BOUND = 25.0
GAIN_BOUNDS = (0.75, 1.33)


def sensor_simulate() -> SensorSimulateFn:
    """One MuJoCo :class:`~antares.sim.Arm` wrapped as a ``SensorSimulateFn``."""
    from antares.sim import Arm

    arm = Arm()
    real_cache: dict[int, tuple[Episode, np.ndarray]] = {}

    def simulate(params: SimParams, cmap: CommandMap, episode: Episode) -> tuple[np.ndarray, np.ndarray]:
        arm.apply(params)
        arm.command_map = cmap
        hit = real_cache.get(id(episode))
        if hit is None or hit[0] is not episode:
            hit = (episode, arm.real_q(episode))
            real_cache[id(episode)] = hit
        return hit[1], arm.replay(episode)

    return simulate


# --------------------------------------------------------------------------
# Command-map fit (dynamics held fixed)
# --------------------------------------------------------------------------


def _pack(cmap: CommandMap, joints: Sequence[int], fit_gain: bool) -> np.ndarray:
    theta = [cmap.bias[j] for j in joints]
    if fit_gain:
        theta += [cmap.gain[j] for j in joints]
    return np.array(theta, dtype=np.float64)


def _unpack(theta: np.ndarray, base: CommandMap, joints: Sequence[int], fit_gain: bool) -> CommandMap:
    bias, gain = base.bias.copy(), base.gain.copy()
    n = len(joints)
    for k, j in enumerate(joints):
        bias[j] = np.clip(theta[k], -BIAS_BOUND, BIAS_BOUND)
        if fit_gain:
            gain[j] = np.clip(theta[n + k], *GAIN_BOUNDS)
    return CommandMap(gain=gain, bias=bias)


def _residual(
    params: SimParams, cmap: CommandMap, episodes: Sequence[Episode], simulate: SensorSimulateFn, joints
) -> np.ndarray:
    """Stacked (sim - real) in degrees over the fitted joints of every frame."""
    parts = []
    for ep in episodes:
        real_q, sim_q = simulate(params, cmap, ep)
        parts.append(np.degrees(np.asarray(sim_q)[:, joints] - np.asarray(real_q)[:, joints]).ravel())
    return np.concatenate(parts)


def _rms(r: np.ndarray) -> float:
    return float(np.sqrt(np.mean(r**2))) if r.size and np.all(np.isfinite(r)) else float("inf")


@dataclass(frozen=True)
class CommandMapFit:
    command_map: CommandMap
    rms_before: float  # RMS over fitted joints and frames, degrees
    rms_after: float
    n_simulations: int  # full passes over ``episodes``


def fit_command_map(
    params: SimParams,
    episodes: Sequence[Episode],
    simulate: SensorSimulateFn,
    start: CommandMap | None = None,
    fit_gain: bool = True,
    iterations: int = 5,
    joints: Sequence[int] = FIT_JOINTS,
) -> CommandMapFit:
    """Levenberg-Marquardt fit of the command map with ``params`` fixed.

    The residual is nearly linear in bias and gain (a command shift moves
    the servo's set point), so a few iterations converge. Each iteration
    costs one simulation per fitted scalar plus one.
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("fit_command_map needs at least one episode")
    joints = tuple(joints)
    base = start if start is not None else CommandMap.identity()
    theta = _pack(base, joints, fit_gain)
    steps = np.array([0.5] * len(joints) + ([0.01] * len(joints) if fit_gain else []))

    def res(th: np.ndarray) -> np.ndarray:
        return _residual(params, _unpack(th, base, joints, fit_gain), episodes, simulate, joints)

    r = res(theta)
    n_sim = 1
    rms0 = cur = _rms(r)
    lam = 1e-3
    for _ in range(iterations):
        if not np.isfinite(cur):
            break
        J = np.empty((r.size, theta.size))
        for k in range(theta.size):
            th = theta.copy()
            th[k] += steps[k]
            J[:, k] = (res(th) - r) / steps[k]
            n_sim += 1
        JtJ, Jtr = J.T @ J, J.T @ r
        improved = False
        for _ in range(4):
            delta = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-9), -Jtr)
            cand = _pack(_unpack(theta + delta, base, joints, fit_gain), joints, fit_gain)
            r_c = res(cand)
            n_sim += 1
            if _rms(r_c) < cur:
                theta, r, cur = cand, r_c, _rms(r_c)
                lam = max(lam / 10, 1e-7)
                improved = True
                break
            lam *= 10
        if not improved or np.max(np.abs(delta) / steps) < 0.02:
            break
    return CommandMapFit(_unpack(theta, base, joints, fit_gain), rms0, cur, n_sim)


# --------------------------------------------------------------------------
# Kinematic pre-fit (no sim)
# --------------------------------------------------------------------------


def _static_mask(x: np.ndarray, window: int, tol: float) -> np.ndarray:
    """Frames where ``x`` (T,) moved less than ``tol`` over the last ``window``."""
    T = len(x)
    mask = np.zeros(T, dtype=bool)
    for t in range(window, T):
        seg = x[t - window : t + 1]
        mask[t] = seg.max() - seg.min() < tol
    return mask


def prefit_command_map(
    episodes: Sequence[Episode],
    fit_gain: bool = True,
    window: int = 10,
    tol: float = 0.5,
    min_frames: int = 30,
    min_spread: float = 15.0,
    joints: Sequence[int] = FIT_JOINTS,
) -> CommandMap:
    """Fit the command map from kinematics alone, without the sim.

    Uses frames where both the command and the measured joint have been
    still for ``window`` frames (moved less than ``tol`` dataset units), so
    the servo has settled and latency and damping drop out; there the
    servo sits where it was told, and ``state ~ gain*action + bias`` is a
    least-squares line per joint. A joint with fewer than ``min_frames``
    such frames keeps identity; one whose still commands span less than
    ``min_spread`` units fits bias only.

    Caveat: a still servo under gravity sags below its command by an amount
    the sim models through kp, so this bias also absorbs that sag. Fitting
    through the sim (:func:`fit_command_map`) does not.
    """
    gain, bias = np.ones(len(JOINTS)), np.zeros(len(JOINTS))
    for j in joints:
        a_parts, s_parts = [], []
        for ep in episodes:
            m = _static_mask(ep.action[:, j], window, tol) & _static_mask(ep.state[:, j], window, tol)
            a_parts.append(ep.action[m, j])
            s_parts.append(ep.state[m, j])
        a, st = np.concatenate(a_parts), np.concatenate(s_parts)
        if a.size < min_frames:
            continue
        if fit_gain and np.ptp(a) >= min_spread:
            g, b = np.polyfit(a, st, 1)
            gain[j] = float(np.clip(g, *GAIN_BOUNDS))
            bias[j] = float(np.clip(b, -BIAS_BOUND, BIAS_BOUND))
        else:
            bias[j] = float(np.clip(np.mean(st - a), -BIAS_BOUND, BIAS_BOUND))
    return CommandMap(gain=gain, bias=bias)


# --------------------------------------------------------------------------
# Joint fit: dynamics + command map
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorCalibrationResult:
    """Outcome of :func:`calibrate_with_sensor`.

    ``result`` is the final fit, scored from the default sim (``baseline_*``)
    to fitted dynamics plus command map (``train_score``/``eval_score``).
    ``dynamics_only`` is the plain dynamics fit on the same split, the
    "without" side of the comparison.
    """

    result: CalibrationResult
    command_map: CommandMap
    dynamics_only: CalibrationResult

    def to_dict(self) -> dict:
        return {
            "result": self.result.to_dict(),
            "command_map": self.command_map.to_dict(),
            "dynamics_only": self.dynamics_only.to_dict(),
        }


def calibrate_with_sensor(
    train: Sequence[Episode],
    eval: Sequence[Episode],
    simulate: SensorSimulateFn | None = None,
    start: SimParams | None = None,
    max_evaluations: int = 150,
    rounds: int = 2,
    fit_gain: bool = True,
    mode: str = "sim",
    fair_baseline: bool = True,
    seed: int = 0,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    **calibrate_kwargs: Any,
) -> SensorCalibrationResult:
    """Fit dynamics and command map together by alternation.

    Round 0 is the ordinary dynamics fit with an identity command map (kept
    as ``dynamics_only``). Each later round fits the command map with the
    current dynamics fixed, then refits the dynamics from where they were,
    under the new command map, with ``max_evaluations`` CMA-ES evaluations.
    A final command-map fit follows the last round.

    ``mode="kinematic"`` instead pre-fits the map from still frames
    (:func:`prefit_command_map`), freezes it, and refits the dynamics once
    under it, so the map cannot absorb dynamics error.

    With ``fair_baseline`` (default) the dynamics-only side gets the same
    budget: it is refit round for round under the identity map, so any gain
    reported for the command map is not just extra CMA-ES evaluations.
    ``calibrate_kwargs``
    (e.g. ``per_joint=True``) go to every dynamics fit.
    """
    if mode not in ("sim", "kinematic"):
        raise ValueError(f"mode must be 'sim' or 'kinematic', got {mode!r}")
    t0 = time.perf_counter()
    train, eval = list(train), list(eval)
    if simulate is None:
        simulate = sensor_simulate()
    stop = should_stop or (lambda: False)
    offset = {"n": 0}

    def progress(ev: ProgressEvent) -> None:
        if on_progress is not None:
            on_progress(replace(ev, evaluation=ev.evaluation + offset["n"]))

    def with_map(cmap: CommandMap):
        return lambda p, ep: simulate(p, cmap, ep)

    ident = CommandMap.identity()
    first = calibrate(
        train, eval, simulate=with_map(ident), start=start, max_evaluations=max_evaluations,
        seed=seed, on_progress=progress, should_stop=stop, **calibrate_kwargs,
    )
    history = list(first.history)
    n_evals = first.n_evaluations
    params, cmap = first.params, ident

    for k in range(1, rounds if mode == "sim" else 2):
        if stop():
            break
        if mode == "kinematic":
            cmap = prefit_command_map(train, fit_gain=fit_gain)
        else:
            cmap = fit_command_map(params, train, simulate, start=cmap, fit_gain=fit_gain).command_map
        offset["n"] = n_evals
        nxt = calibrate(
            train, eval, simulate=with_map(cmap), start=params, max_evaluations=max_evaluations,
            sigma0=0.15, seed=seed + k, on_progress=progress, should_stop=stop, **calibrate_kwargs,
        )
        history += [(i + n_evals, s) for i, s in nxt.history[1:]]
        n_evals += nxt.n_evaluations
        params = nxt.params
    if mode == "sim":
        cmap = fit_command_map(params, train, simulate, start=cmap, fit_gain=fit_gain).command_map

    only = first
    if fair_baseline:
        n_rounds = rounds if mode == "sim" else 2
        for k in range(1, n_rounds):
            if stop():
                break
            nxt = calibrate(
                train, eval, simulate=with_map(ident), start=only.params, max_evaluations=max_evaluations,
                sigma0=0.15, seed=seed + k, should_stop=stop, **calibrate_kwargs,
            )
            only = replace(
                nxt,
                baseline=first.baseline,
                baseline_train_score=first.baseline_train_score,
                baseline_eval_score=first.baseline_eval_score,
                per_joint_before=first.per_joint_before,
                history=list(only.history) + [(i + only.n_evaluations, s) for i, s in nxt.history[1:]],
                n_evaluations=only.n_evaluations + nxt.n_evaluations,
            )

    from antares.calibrate import _per_joint, _safe_score

    sim = with_map(cmap)
    train_report, train_score = _safe_score(params, train, sim)
    if eval:
        eval_report, eval_score = _safe_score(params, eval, sim)
    else:
        eval_report, eval_score = train_report, train_score

    final = replace(
        first,
        params=params,
        train_score=train_score,
        eval_score=eval_score,
        per_joint_after=_per_joint(eval_report),
        history=history,
        n_evaluations=n_evals,
        elapsed_s=time.perf_counter() - t0,
    )
    return SensorCalibrationResult(result=final, command_map=cmap, dynamics_only=only)


__all__ = [
    "FIT_JOINTS",
    "CommandMapFit",
    "SensorCalibrationResult",
    "SensorSimulateFn",
    "calibrate_with_sensor",
    "fit_command_map",
    "prefit_command_map",
    "sensor_simulate",
]
