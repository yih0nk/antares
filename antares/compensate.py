"""Lag compensation: commands that make the calibrated sim track a trajectory.

A calibrated lock says the servo responds with some frames of latency and a
soft position gain, so the commanded trajectory *leads* the real arm. Given
that model, this module computes the commands to send so the arm actually
follows a desired trajectory:

* :func:`shift_compensate` -- send ``q_des[t + L]`` at frame ``t`` (the
  pure-latency baseline).
* :func:`ilc_compensate` -- iterative learning control in the calibrated
  sim: start from the shifted commands, roll out, add ``gain`` times the
  latency-aligned tracking error, repeat while the error keeps improving.

Validation is **sim-only**. The desired trajectory is the real measured
trajectory of a held-out episode, and tracking is measured against the
calibrated MuJoCo model, never against hardware. The raw-command case
therefore reproduces the lock's gap number by construction; the shifted
and ILC numbers say how much of that gap the compensator removes *in the
sim*. Whether the real servo agrees is exactly the sim-to-real question the
lock cannot answer on its own.

Angles are radians internally and degrees in every report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from antares.data import Dataset, Episode
from antares.gap import GapReport, aggregate, tracking_error
from antares.sim import JOINTS, Arm, SimParams

METHODS = ("shift", "ilc")

#: Compensated commands demanding more than this multiple of the fastest
#: joint motion seen in the real data are flagged as infeasible.
VELOCITY_FACTOR = 1.5


# --------------------------------------------------------------------------
# Compensators
# --------------------------------------------------------------------------


def shift_compensate(q_des: np.ndarray, latency_frames: int) -> np.ndarray:
    """Send ``q_des[t + L]`` at frame ``t``; the last frame is held at the end.

    With exactly ``L`` frames of pure delay and a stiff servo this cancels
    the lag. It does nothing about a soft gain.
    """
    q_des = np.asarray(q_des, dtype=np.float64)
    if q_des.ndim != 2:
        raise ValueError(f"q_des must be (T, J), got {q_des.shape}")
    L = int(latency_frames)
    if L < 0:
        raise ValueError("latency_frames must be >= 0")
    T = q_des.shape[0]
    src = np.minimum(np.arange(T) + L, T - 1)
    return q_des[src].copy()


def _rms_deg(q_des: np.ndarray, q_sim: np.ndarray) -> float:
    return float(tracking_error(q_des, q_sim, joint_names=JOINTS).overall_deg)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along axis 0 with edge replication (window 1 = identity)."""
    if window <= 1:
        return x
    half = window // 2
    pad = np.concatenate([np.repeat(x[:1], half, axis=0), x, np.repeat(x[-1:], half, axis=0)])
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.stack([np.convolve(pad[:, j], kernel, mode="valid") for j in range(x.shape[1])], axis=1)


@dataclass(frozen=True)
class ILCResult:
    ctrl: np.ndarray  # (T,6) rad: best commands found (clipped to ctrlrange)
    q_sim: np.ndarray  # (T,6) rad: calibrated-sim rollout of ``ctrl``
    history_deg: tuple[float, ...]  # RMS deg per rollout; [0] is the shifted baseline
    iterations: int  # accepted update steps (0 = the shift alone was best)
    converged: bool  # stopped before ``iters`` because the error stopped improving

    @property
    def rms_deg(self) -> float:
        """RMS error (deg, scored joints) of the returned ``ctrl``.

        This is the last *accepted* entry of ``history_deg``; a final step
        rejected for improving by less than ``rel_tol`` may sit marginally
        below it but its commands are not the ones returned.
        """
        return self.history_deg[self.iterations]


def ilc(
    arm: Arm,
    q_des: np.ndarray,
    q0: np.ndarray,
    fps: float,
    iters: int = 10,
    gain: float = 0.5,
    latency: int | None = None,
    smooth: int = 3,
    rel_tol: float = 1e-3,
) -> ILCResult:
    """Iterative learning control in the calibrated sim; see :func:`ilc_compensate`.

    Each step rolls the current commands out, takes the tracking error
    ``q_des - q_sim``, aligns it by ``latency`` (the error at ``t + L`` is
    what the command sent at ``t`` can fix), optionally low-pass filters it
    with a ``smooth``-frame moving average (a Q-filter: the plant lags by
    more than the pure delay near Nyquist, so an unfiltered P-type update
    amplifies the highest frequencies), scales it by ``gain`` and adds it
    to the commands. A step that fails to lower the RMS error (scored
    joints, degrees) by more than ``rel_tol`` relative ends the loop and the
    best commands so far are returned.
    """
    q_des = np.asarray(q_des, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    if q_des.ndim != 2 or q_des.shape[1] != len(JOINTS):
        raise ValueError(f"q_des must be (T,{len(JOINTS)}), got {q_des.shape}")
    if iters < 0:
        raise ValueError("iters must be >= 0")
    if gain <= 0:
        raise ValueError("gain must be positive")
    L = arm.params.latency_frames if latency is None else int(latency)
    lo, hi = arm.ctrl_range

    ctrl = np.clip(shift_compensate(q_des, L), lo, hi)
    q_sim = arm.rollout(ctrl, q0, fps)
    err = _rms_deg(q_des, q_sim)
    history = [err]
    best = (ctrl, q_sim)
    accepted = 0
    converged = False
    for _ in range(iters):
        correction = shift_compensate(q_des - q_sim, L)
        correction = _smooth(correction, smooth)
        ctrl_new = np.clip(ctrl + gain * correction, lo, hi)
        q_new = arm.rollout(ctrl_new, q0, fps)
        err_new = _rms_deg(q_des, q_new)
        history.append(err_new)
        if not np.isfinite(err_new) or err_new >= err * (1.0 - rel_tol):
            converged = True
            break
        ctrl, q_sim, err = ctrl_new, q_new, err_new
        best = (ctrl, q_sim)
        accepted += 1
    return ILCResult(
        ctrl=best[0],
        q_sim=best[1],
        history_deg=tuple(float(h) for h in history),
        iterations=accepted,
        converged=converged,
    )


def ilc_compensate(
    arm: Arm,
    q_des: np.ndarray,
    q0: np.ndarray,
    fps: float,
    iters: int = 10,
    gain: float = 0.5,
    latency: int | None = None,
    smooth: int = 3,
) -> np.ndarray:
    """Commands (T,6) rad that make the calibrated sim track ``q_des`` from ``q0``.

    Starts from :func:`shift_compensate`, then repeats ``ctrl <- ctrl + gain *
    (q_des - q_sim)`` aligned by ``latency`` (default: the arm's calibrated
    latency) for up to ``iters`` rollouts, stopping early once the error
    stops improving. The result is clipped to the actuator ctrlrange. Use
    :func:`ilc` for the iteration history and the final sim trajectory.
    """
    return ilc(arm, q_des, q0, fps, iters=iters, gain=gain, latency=latency, smooth=smooth).ctrl


# --------------------------------------------------------------------------
# Feasibility
# --------------------------------------------------------------------------


def max_joint_velocity(episodes: Sequence[Episode], arm: Arm) -> np.ndarray:
    """Fastest measured joint motion per joint, deg/s, over ``episodes``."""
    vmax = np.zeros(len(JOINTS))
    for ep in episodes:
        if len(ep) < 2:
            continue
        q = np.degrees(arm.real_q(ep))
        vmax = np.maximum(vmax, np.max(np.abs(np.diff(q, axis=0)), axis=0) * ep.fps)
    return vmax


def _cmd_velocity(ctrl_deg: np.ndarray, fps: float) -> np.ndarray:
    if ctrl_deg.shape[0] < 2:
        return np.zeros(ctrl_deg.shape[1])
    return np.max(np.abs(np.diff(ctrl_deg, axis=0)), axis=0) * fps


def feasibility(
    ctrl_raw: np.ndarray,
    ctrl_comp: np.ndarray,
    fps: float,
    real_vmax_deg_s: np.ndarray,
    ctrl_range: tuple[np.ndarray, np.ndarray],
    factor: float = VELOCITY_FACTOR,
) -> dict[str, Any]:
    """Can the servo be asked to do this? All inputs in radians except ``real_vmax_deg_s``.

    Reports, per joint and in degrees: how far the compensated commands go
    outside the range the original commands used, how many frames sit on
    the actuator ctrlrange limit (i.e. were clipped) before and after
    compensation, the peak command velocity before and after, the fastest
    joint motion seen in the real data, and whether the compensated
    commands demand more than ``factor`` times that. ``ok`` is the velocity
    verdict alone: commands are always clipped to the ctrlrange, so they
    can never exceed it, and a desired trajectory that rides a joint limit
    legitimately pins its commands there (the raw count shows when that is
    the case).
    """
    raw = np.degrees(np.asarray(ctrl_raw, dtype=np.float64))
    comp = np.degrees(np.asarray(ctrl_comp, dtype=np.float64))
    if raw.shape != comp.shape or raw.ndim != 2:
        raise ValueError(f"ctrl_raw {raw.shape} and ctrl_comp {comp.shape} must be matching (T,J)")
    lo = np.degrees(np.asarray(ctrl_range[0], dtype=np.float64))
    hi = np.degrees(np.asarray(ctrl_range[1], dtype=np.float64))
    real_vmax = np.asarray(real_vmax_deg_s, dtype=np.float64)

    raw_min, raw_max = raw.min(axis=0), raw.max(axis=0)
    excess = np.maximum(0.0, np.maximum(comp.max(axis=0) - raw_max, raw_min - comp.min(axis=0)))
    tol = 1e-6
    at_limit = lambda c: np.sum((c <= lo + tol) | (c >= hi - tol), axis=0)  # noqa: E731
    clipped_raw = at_limit(raw)
    clipped = at_limit(comp)
    v_raw = _cmd_velocity(raw, fps)
    v_comp = _cmd_velocity(comp, fps)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(real_vmax > 0, v_comp / real_vmax, np.inf * (v_comp > 0))
    flagged = [j for j, r in zip(JOINTS, ratio) if r > factor]
    return {
        "velocity_factor": float(factor),
        "ok": not flagged,
        "flagged_joints": flagged,
        "max_range_excess_deg": float(excess.max()),
        "clipped_frames": int(clipped.sum()),
        "raw_clipped_frames": int(clipped_raw.sum()),
        "per_joint": {
            j: {
                "raw_cmd_range_deg": [float(raw_min[i]), float(raw_max[i])],
                "comp_cmd_range_deg": [float(comp.min(axis=0)[i]), float(comp.max(axis=0)[i])],
                "range_excess_deg": float(excess[i]),
                "clipped_frames": int(clipped[i]),
                "raw_clipped_frames": int(clipped_raw[i]),
                "raw_cmd_vmax_deg_s": float(v_raw[i]),
                "comp_cmd_vmax_deg_s": float(v_comp[i]),
                "real_vmax_deg_s": float(real_vmax[i]),
                "velocity_ratio": float(ratio[i]) if np.isfinite(ratio[i]) else None,
                "flagged": j in flagged,
            }
            for i, j in enumerate(JOINTS)
        },
    }


# --------------------------------------------------------------------------
# Evaluation on logged episodes (sim-only)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompensationReport:
    """One episode: raw vs shift vs ILC commands, all judged in the calibrated sim."""

    episode: int
    fps: float
    params: SimParams
    method: str
    q_des: np.ndarray  # (T,6) rad: the real measured trajectory
    ctrl_raw: np.ndarray  # (T,6) rad: the logged commands
    ctrl_shift: np.ndarray
    ctrl_ilc: np.ndarray
    q_sim_raw: np.ndarray
    q_sim_shift: np.ndarray
    q_sim_ilc: np.ndarray
    gap_raw: GapReport
    gap_shift: GapReport
    gap_ilc: GapReport
    ilc_iterations: int
    ilc_history_deg: tuple[float, ...]
    feasibility: dict[str, Any] = field(default_factory=dict)

    @property
    def ctrl_comp(self) -> np.ndarray:
        return self.ctrl_ilc if self.method == "ilc" else self.ctrl_shift

    @property
    def q_sim_comp(self) -> np.ndarray:
        return self.q_sim_ilc if self.method == "ilc" else self.q_sim_shift

    @property
    def rms(self) -> dict[str, float]:
        return {
            "raw": float(self.gap_raw.overall_deg),
            "shift": float(self.gap_shift.overall_deg),
            "ilc": float(self.gap_ilc.overall_deg),
        }


def compensate_episode(
    arm: Arm,
    episode: Episode,
    params: SimParams,
    method: str = "ilc",
    real_vmax_deg_s: np.ndarray | None = None,
    iters: int = 10,
    gain: float = 0.5,
    smooth: int = 3,
) -> CompensationReport:
    """Treat the real measured trajectory as the desired one and compensate it.

    ``real_vmax_deg_s`` is the per-joint velocity reference for the
    feasibility check; it defaults to this episode's own real motion. The
    arm is left with ``params`` applied.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    arm.apply(params)
    fps = float(episode.fps)
    L = params.latency_frames
    lo, hi = arm.ctrl_range

    q_des = arm.real_q(episode)
    q0 = q_des[0]
    action = episode.action
    cmap = getattr(arm, "command_map", None)
    if cmap is not None:  # the lock's sensor calibration, if any
        action = cmap.apply(action)
    ctrl_raw = np.clip(arm.unit_map.to_rad(action), lo, hi)
    q_sim_raw = arm.rollout(ctrl_raw, q0, fps)

    ctrl_shift = np.clip(shift_compensate(q_des, L), lo, hi)
    q_sim_shift = arm.rollout(ctrl_shift, q0, fps)

    res = ilc(arm, q_des, q0, fps, iters=iters, gain=gain, latency=L, smooth=smooth)

    if real_vmax_deg_s is None:
        real_vmax_deg_s = max_joint_velocity([episode], arm)
    ctrl_comp = res.ctrl if method == "ilc" else ctrl_shift
    feas = feasibility(ctrl_raw, ctrl_comp, fps, real_vmax_deg_s, (lo, hi))

    return CompensationReport(
        episode=episode.index,
        fps=fps,
        params=params,
        method=method,
        q_des=q_des,
        ctrl_raw=ctrl_raw,
        ctrl_shift=ctrl_shift,
        ctrl_ilc=res.ctrl,
        q_sim_raw=q_sim_raw,
        q_sim_shift=q_sim_shift,
        q_sim_ilc=res.q_sim,
        gap_raw=tracking_error(q_des, q_sim_raw, joint_names=JOINTS),
        gap_shift=tracking_error(q_des, q_sim_shift, joint_names=JOINTS),
        gap_ilc=tracking_error(q_des, res.q_sim, joint_names=JOINTS),
        ilc_iterations=res.iterations,
        ilc_history_deg=res.history_deg,
        feasibility=feas,
    )


@dataclass(frozen=True)
class HeldOutSummary:
    """Frame-weighted RMS over several episodes plus the worst-case feasibility."""

    episodes: tuple[int, ...]
    rms: dict[str, float]
    per_joint_deg: dict[str, dict[str, float]]
    ilc_iterations: tuple[int, ...]
    n_frames: int
    feasibility: dict[str, Any]
    reports: tuple[CompensationReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": list(self.episodes),
            "rms": dict(self.rms),
            "per_joint_deg": {k: dict(v) for k, v in self.per_joint_deg.items()},
            "ilc_iterations": list(self.ilc_iterations),
            "n_frames": int(self.n_frames),
            "feasibility": self.feasibility,
        }


def compensate_episodes(
    arm: Arm,
    episodes: Sequence[Episode],
    params: SimParams,
    method: str = "ilc",
    real_vmax_deg_s: np.ndarray | None = None,
    **kwargs: Any,
) -> HeldOutSummary:
    """Run :func:`compensate_episode` on each episode and aggregate (sim-only)."""
    episodes = list(episodes)
    if not episodes:
        raise ValueError("no episodes to compensate")
    if real_vmax_deg_s is None:
        real_vmax_deg_s = max_joint_velocity(episodes, arm)
    reports = [
        compensate_episode(arm, ep, params, method=method, real_vmax_deg_s=real_vmax_deg_s, **kwargs)
        for ep in episodes
    ]
    gaps = {
        "raw": aggregate([r.gap_raw for r in reports]),
        "shift": aggregate([r.gap_shift for r in reports]),
        "ilc": aggregate([r.gap_ilc for r in reports]),
    }
    worst = max(reports, key=lambda r: (len(r.feasibility["flagged_joints"]), r.feasibility["clipped_frames"]))
    per_joint = {k: {j: float(v) for j, v in g.per_joint_deg.items()} for k, g in gaps.items()}
    flagged = sorted({j for r in reports for j in r.feasibility["flagged_joints"]})
    feas = {
        "velocity_factor": float(VELOCITY_FACTOR),
        "ok": all(r.feasibility["ok"] for r in reports),
        "flagged_joints": flagged,
        "episodes_flagged": [r.episode for r in reports if not r.feasibility["ok"]],
        "max_range_excess_deg": max(r.feasibility["max_range_excess_deg"] for r in reports),
        "clipped_frames": sum(r.feasibility["clipped_frames"] for r in reports),
        "raw_clipped_frames": sum(r.feasibility["raw_clipped_frames"] for r in reports),
        "real_vmax_deg_s": {j: float(v) for j, v in zip(JOINTS, real_vmax_deg_s)},
        "comp_cmd_vmax_deg_s": {
            j: max(r.feasibility["per_joint"][j]["comp_cmd_vmax_deg_s"] for r in reports) for j in JOINTS
        },
        "raw_cmd_vmax_deg_s": {
            j: max(r.feasibility["per_joint"][j]["raw_cmd_vmax_deg_s"] for r in reports) for j in JOINTS
        },
        "worst_episode": worst.episode,
    }
    return HeldOutSummary(
        episodes=tuple(r.episode for r in reports),
        rms={k: float(g.overall_deg) for k, g in gaps.items()},
        per_joint_deg=per_joint,
        ilc_iterations=tuple(r.ilc_iterations for r in reports),
        n_frames=int(gaps["raw"].n_frames),
        feasibility=feas,
        reports=tuple(reports),
    )


def held_out_episodes(ds: Dataset, eval_indices: Sequence[int]) -> list[Episode]:
    """The dataset's episodes with the given indices (the lock's eval split), in order."""
    by_index = {ep.index: ep for ep in ds.episodes}
    missing = [i for i in eval_indices if i not in by_index]
    if missing:
        raise ValueError(f"episodes {missing} not in {ds.repo_id}")
    return [by_index[i] for i in eval_indices]


__all__ = [
    "METHODS",
    "VELOCITY_FACTOR",
    "ILCResult",
    "CompensationReport",
    "HeldOutSummary",
    "shift_compensate",
    "ilc",
    "ilc_compensate",
    "max_joint_velocity",
    "feasibility",
    "compensate_episode",
    "compensate_episodes",
    "held_out_episodes",
]
