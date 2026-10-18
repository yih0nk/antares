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

from antares import factors
from antares.calibrate import CalibrationResult, ProgressEvent, SimParams, calibrate
from antares.data import Episode
from antares.sim import GRIPPER_INDEX, JOINTS, CommandMap

SensorSimulateFn = Callable[[SimParams, CommandMap, Episode], tuple[np.ndarray, np.ndarray]]

#: Joints whose command map is fitted (the scored body joints).
FIT_JOINTS: tuple[int, ...] = tuple(i for i in range(len(JOINTS)) if i != GRIPPER_INDEX)

#: Bounds, in dataset units for bias. Wider than any sane calibration error.
BIAS_BOUND = factors.get("sensor_bias").bounds[1]
GAIN_BOUNDS = factors.get("sensor_gain").bounds


def sensor_simulate(robot: str | None = None) -> SensorSimulateFn:
    """One MuJoCo :class:`~antares.sim.Arm` wrapped as a ``SensorSimulateFn``.

    ``robot`` picks the model from :data:`antares.robots.REGISTRY` (default
    SO-101).
    """
    from antares.sim import Arm

    arm = Arm(robot=robot)
    real_cache: dict[int, tuple[Episode, np.ndarray]] = {}

    def simulate(params: SimParams, cmap: CommandMap, episode: Episode) -> tuple[np.ndarray, np.ndarray]:
        arm.apply(params)
        arm.command_map = cmap
        hit = real_cache.get(id(episode))
        if hit is None or hit[0] is not episode:
            hit = (episode, arm.real_q(episode))
            real_cache[id(episode)] = hit
        return hit[1], arm.replay(episode)

    from antares.calibrate import describe_simulate

    describe_simulate(simulate, arm)
    simulate.unit_scale = arm.unit_map.scale.copy()  # type: ignore[attr-defined]
    return simulate


def _robot_fit_args(simulate: Any) -> dict[str, Any]:
    """Column indices, count and unit rescale for a tagged simulate of a
    robot other than the SO arms; empty (the old defaults) otherwise."""
    from antares.calibrate import robot_of

    r = robot_of(simulate)
    if r is None or (tuple(r.joints) == tuple(JOINTS) and r.units is None):
        return {}
    scale = getattr(simulate, "unit_scale", None)
    unit_k = None if scale is None else np.abs((np.pi / 180.0) / np.asarray(scale, dtype=np.float64))
    return {"joints": tuple(r.body_indices), "n_joints": r.n_joints, "unit_k": unit_k}


# --------------------------------------------------------------------------
# Command-map fit (dynamics held fixed)
# --------------------------------------------------------------------------


def _pack(cmap: CommandMap, joints: Sequence[int], fit_gain: bool) -> np.ndarray:
    theta = [cmap.bias[j] for j in joints]
    if fit_gain:
        theta += [cmap.gain[j] for j in joints]
    return np.array(theta, dtype=np.float64)


def _unpack(
    theta: np.ndarray, base: CommandMap, joints: Sequence[int], fit_gain: bool, unit_k: np.ndarray | None = None
) -> CommandMap:
    bias, gain = base.bias.copy(), base.gain.copy()
    n = len(joints)
    for k, j in enumerate(joints):
        bound = BIAS_BOUND if unit_k is None else BIAS_BOUND * float(unit_k[j])
        bias[j] = np.clip(theta[k], -bound, bound)
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
    coupled: bool = False,
    should_stop: Callable[[], bool] | None = None,
    on_simulation: Callable[[int, float, float], None] | None = None,
    unit_k: np.ndarray | None = None,
) -> CommandMapFit:
    """Levenberg-Marquardt fit of the command map with ``params`` fixed.

    The residual is nearly linear in bias and gain (a command shift moves
    the servo's set point), so a few iterations converge. A joint's bias
    and gain move only that joint's set point, so by default the Jacobian
    is taken block-diagonally: all biases are perturbed in one simulation
    and all gains in another, two simulations per iteration (the line
    search still checks the true residual, so any cross-joint coupling the
    dynamics add only slows convergence). ``coupled=True`` perturbs one
    scalar at a time instead, one simulation per fitted scalar.

    ``should_stop`` is polled between simulations; when it returns True the
    best map so far is returned. ``on_simulation(n, candidate_rms, best_rms)``
    is called after every full pass over ``episodes``.

    Bias bounds and steps are in dataset units sized for LeRobot's
    normalized SO units (one unit is about a degree). ``unit_k`` (dataset
    units per degree, one per column) rescales them for robots logged in
    other units, such as radians.
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("fit_command_map needs at least one episode")
    joints = tuple(joints)
    n_j = len(joints)
    stop = should_stop or (lambda: False)
    base = start if start is not None else CommandMap.identity()
    theta = _pack(base, joints, fit_gain)
    steps = np.array([0.5] * n_j + ([0.01] * n_j if fit_gain else []))
    if unit_k is not None:
        steps[:n_j] *= np.asarray(unit_k, dtype=np.float64)[list(joints)]
    n_sim = 0

    def res(th: np.ndarray, cur_best: float) -> np.ndarray:
        nonlocal n_sim
        out = _residual(params, _unpack(th, base, joints, fit_gain, unit_k), episodes, simulate, joints)
        n_sim += 1
        if on_simulation is not None:
            on_simulation(n_sim, _rms(out), min(cur_best, _rms(out)))
        return out

    r = res(theta, float("inf"))
    rms0 = cur = _rms(r)
    lam = 1e-3
    for _ in range(iterations):
        if not np.isfinite(cur) or stop():
            break
        J = np.empty((r.size, theta.size))
        if coupled:
            for k in range(theta.size):
                if stop():
                    break
                th = theta.copy()
                th[k] += steps[k]
                J[:, k] = (res(th, cur) - r) / steps[k]
            if stop():
                break
        else:
            # Residual rows are (frame, joint) row-major, so row i belongs to
            # joint position i % n_j; each block of scalars touches its own rows.
            row_joint = np.arange(r.size) % n_j
            for blk in range(theta.size // n_j):
                if stop():
                    break
                th = theta.copy()
                sl = slice(blk * n_j, (blk + 1) * n_j)
                th[sl] += steps[sl]
                dr = (res(th, cur) - r)
                for k in range(n_j):
                    col = blk * n_j + k
                    J[:, col] = np.where(row_joint == k, dr / steps[col], 0.0)
            if stop():
                break
        JtJ, Jtr = J.T @ J, J.T @ r
        improved = False
        for _ in range(4):
            if stop():
                break
            delta = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-9), -Jtr)
            cand = _pack(_unpack(theta + delta, base, joints, fit_gain, unit_k), joints, fit_gain)
            r_c = res(cand, cur)
            if _rms(r_c) < cur:
                theta, r, cur = cand, r_c, _rms(r_c)
                lam = max(lam / 10, 1e-7)
                improved = True
                break
            lam *= 10
        if not improved or np.max(np.abs(delta) / steps) < 0.02:
            break
    return CommandMapFit(_unpack(theta, base, joints, fit_gain, unit_k), rms0, cur, n_sim)


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
    n_joints: int | None = None,
    unit_k: np.ndarray | None = None,
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
    n = len(JOINTS) if n_joints is None else int(n_joints)
    gain, bias = np.ones(n), np.zeros(n)
    for j in joints:
        k = 1.0 if unit_k is None else float(unit_k[j])
        bound = BIAS_BOUND * k
        a_parts, s_parts = [], []
        for ep in episodes:
            m = _static_mask(ep.action[:, j], window, tol * k) & _static_mask(ep.state[:, j], window, tol * k)
            a_parts.append(ep.action[m, j])
            s_parts.append(ep.state[m, j])
        a, st = np.concatenate(a_parts), np.concatenate(s_parts)
        if a.size < min_frames:
            continue
        if fit_gain and np.ptp(a) >= min_spread * k:
            g, b = np.polyfit(a, st, 1)
            gain[j] = float(np.clip(g, *GAIN_BOUNDS))
            bias[j] = float(np.clip(b, -bound, bound))
        else:
            bias[j] = float(np.clip(np.mean(st - a), -bound, bound))
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
    #: Full passes over the training episodes spent by the command-map fits.
    #: They are included in ``result.n_evaluations`` (and its ``stages`` as
    #: ``command_map`` entries) and matched by ``dynamics_only``'s budget.
    n_map_simulations: int = 0

    def to_dict(self) -> dict:
        return {
            "result": self.result.to_dict(),
            "command_map": self.command_map.to_dict(),
            "dynamics_only": self.dynamics_only.to_dict(),
            "n_map_simulations": int(self.n_map_simulations),
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
    refit_sigma0: float = 0.15,
    **calibrate_kwargs: Any,
) -> SensorCalibrationResult:
    """Fit dynamics and command map together by alternation.

    Round 0 is the ordinary dynamics fit with an identity command map (kept
    as ``dynamics_only``). Each later round fits the command map with the
    current dynamics fixed, then refits the dynamics from where they were,
    under the new command map, with ``max_evaluations`` CMA-ES evaluations
    and step size ``refit_sigma0``. A final command-map fit follows the
    last round. A per-joint fit (``per_joint=True``) is refit in the
    per-joint space directly: the shared stage would only average its
    values back together, so later rounds skip it.

    ``mode="kinematic"`` instead pre-fits the map from still frames
    (:func:`prefit_command_map`), freezes it, and refits the dynamics once
    under it, so the map cannot absorb dynamics error.

    Every full pass over the training episodes counts as one evaluation,
    including the simulations the command-map fits spend (recorded as
    ``command_map`` stages and as ``n_map_simulations``). With
    ``fair_baseline`` (default) the dynamics-only side gets the same
    total: it is refit round for round under the identity map, plus one
    more refit with the map fits' simulation count, so any gain reported
    for the command map is not just extra evaluations. ``calibrate_kwargs``
    (e.g. ``per_joint=True``, ``sigma0`` for round 0) go to every dynamics
    fit. ``should_stop`` is polled inside every stage, the map fits too.
    """
    if mode not in ("sim", "kinematic"):
        raise ValueError(f"mode must be 'sim' or 'kinematic', got {mode!r}")
    t0 = time.perf_counter()
    train, eval = list(train), list(eval)
    if simulate is None:
        simulate = sensor_simulate()
    stop = should_stop or (lambda: False)
    offset = {"n": 0}
    refit_kwargs = {**calibrate_kwargs, "sigma0": refit_sigma0}

    def progress(ev: ProgressEvent) -> None:
        if on_progress is not None:
            on_progress(replace(ev, evaluation=ev.evaluation + offset["n"]))

    rargs = _robot_fit_args(simulate)
    n_cols = rargs.get("n_joints", len(JOINTS))

    def with_map(cmap: CommandMap):
        fn = lambda p, ep: simulate(p, cmap, ep)  # noqa: E731
        for attr in ("robot", "default_params"):
            if hasattr(simulate, attr):
                setattr(fn, attr, getattr(simulate, attr))
        return fn

    def refit(start_p: SimParams, sim, k: int, budget: int, extra_budget: bool = False, on_prog=None):
        """One dynamics refit from ``start_p``. ``budget`` is the shared-stage
        budget; a per-joint start skips the shared stage and refits in the
        per-joint space. ``extra_budget`` spends exactly ``budget``
        evaluations in whichever space the start lives in."""
        kw = dict(refit_kwargs)
        shared_budget = budget
        if getattr(start_p, "has_per_joint", False):
            kw["per_joint"] = True
            shared_budget = 0
            if extra_budget:
                kw["per_joint_max_evaluations"] = budget
        elif extra_budget:
            kw["per_joint"] = False
        return calibrate(
            train, eval, simulate=sim, start=start_p, max_evaluations=shared_budget,
            seed=seed + k, on_progress=on_prog, should_stop=stop, **kw,
        )

    ident = CommandMap.identity(n_cols)
    first = calibrate(
        train, eval, simulate=with_map(ident), start=start, max_evaluations=max_evaluations,
        seed=seed, on_progress=progress, should_stop=stop, **calibrate_kwargs,
    )
    history = list(first.history)
    stages = list(first.stages)
    n_evals = first.n_evaluations
    map_sims = 0
    params, cmap = first.params, ident

    def fit_map(p: SimParams, start_map: CommandMap) -> CommandMap:
        """One command-map fit, accounted as a ``command_map`` stage."""
        nonlocal n_evals, map_sims

        def on_sim(n: int, cand: float, best: float) -> None:
            history.append((n_evals + n, best))
            if on_progress is not None:
                on_progress(ProgressEvent(
                    evaluation=n_evals + n, generation=0, candidate_score=cand, best_score=best,
                    best_params=p, best_per_joint_deg={}, elapsed_s=time.perf_counter() - t0,
                    stage="command_map",
                ))

        fit = fit_command_map(p, train, simulate, start=start_map, fit_gain=fit_gain,
                              should_stop=stop, on_simulation=on_sim,
                              **{k: v for k, v in rargs.items() if k in ("joints", "unit_k")})
        stages.append({
            "name": "command_map",
            "n_evaluations": fit.n_simulations,
            "train_score": fit.rms_after,
            "params": p.to_dict(),
            "command_map": fit.command_map.to_dict(),
        })
        n_evals += fit.n_simulations
        map_sims += fit.n_simulations
        return fit.command_map

    for k in range(1, rounds if mode == "sim" else 2):
        if stop():
            break
        if mode == "kinematic":
            cmap = prefit_command_map(train, fit_gain=fit_gain, **rargs)
        else:
            cmap = fit_map(params, cmap)
        if stop():
            break
        offset["n"] = n_evals
        nxt = refit(params, with_map(cmap), k, max_evaluations, on_prog=progress)
        history += [(i + n_evals, s) for i, s in nxt.history[1:]]
        stages += nxt.stages
        n_evals += nxt.n_evaluations
        params = nxt.params
    if mode == "sim" and not stop():
        cmap = fit_map(params, cmap)

    only = first
    if fair_baseline:
        n_rounds = rounds if mode == "sim" else 2
        budgets = [(k, max_evaluations, False) for k in range(1, n_rounds)]
        if map_sims > 0:
            budgets.append((n_rounds, map_sims, True))
        for k, budget, extra in budgets:
            if stop():
                break
            nxt = refit(only.params, with_map(ident), k, budget, extra_budget=extra)
            only = replace(
                nxt,
                baseline=first.baseline,
                baseline_train_score=first.baseline_train_score,
                baseline_eval_score=first.baseline_eval_score,
                per_joint_before=first.per_joint_before,
                history=list(only.history) + [(i + only.n_evaluations, s) for i, s in nxt.history[1:]],
                n_evaluations=only.n_evaluations + nxt.n_evaluations,
                elapsed_s=only.elapsed_s + nxt.elapsed_s,
                stages=list(only.stages) + nxt.stages,
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
        space="per_joint" if getattr(params, "has_per_joint", False) else first.space,
        stages=stages,
    )
    return SensorCalibrationResult(result=final, command_map=cmap, dynamics_only=only,
                                   n_map_simulations=map_sims)


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
