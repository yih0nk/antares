"""Is the leftover real-vs-sim error pose dependent? Measure it against gravity.

A position servo with gain ``kp`` settles short of its command by about
``g(q) / kp``, where ``g(q)`` is the torque it needs to hold the arm still in
pose ``q`` against gravity. If the sim gets that static error wrong (its
links weigh more or less than the real ones, or the real servo is stiffer
under steady load than its tracking speed suggests), the residual
``sim_q - real_q`` grows with the holding torque: the arm "sags"
differently when it is stretched out.

:func:`sag_report` replays episodes with a lock's parameters and, per joint,
regresses the residual on the holding torque at the real pose, with the real
joint velocity as a second regressor so that ordinary tracking lag is not
mistaken for sag. The slope converts into the :attr:`SimParams.gravity_scale`
that would cancel it (``calibrate --fit-gravity`` fits that directly).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from antares.data import Episode
from antares.sim import GRIPPER_INDEX, JOINTS, Arm

#: Joints whose holding torque varies less than this (N*m, std) have too
#: little load signal to regress on: the pan axis is vertical, the roll axis
#: nearly, and in pick-and-place logs the elbow's load barely changes (std
#: about 0.02-0.04 N*m on the demo arms), so its slope is mostly noise.
MIN_TORQUE_STD = 0.05

#: Quantile bins of holding torque shown per joint.
N_BINS = 5

#: A joint counts as still below this speed (deg/s) when splitting the
#: residual by motion.
STILL_DEG_S = 2.0


def holding_torque(arm: Arm, q: np.ndarray) -> np.ndarray:
    """Torque (N*m) each joint needs to hold pose ``q`` (rad) still against gravity.

    ``q`` is (6,) or (T,6). Uses the arm's current gravity (so its
    ``gravity_scale``); this is MuJoCo's bias force at zero velocity.
    """
    q = np.asarray(q, dtype=np.float64)
    one = q.ndim == 1
    qs = q[None] if one else q
    m = arm.model
    d = mujoco.MjData(m)
    out = np.empty_like(qs)
    for t in range(qs.shape[0]):
        d.qpos[:] = qs[t]
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_rne(m, d, 0, out[t])
    return out[0] if one else out


@dataclass(frozen=True)
class JointSag:
    joint: str
    #: Residual (sim - real, deg) per N*m of holding torque, velocity held fixed.
    slope_deg_per_nm: float
    #: Share of the residual variance the holding torque explains beyond the
    #: intercept and velocity (partial R^2, 0..1).
    partial_r2: float
    rms_deg: float  # residual RMS as replayed
    #: After removing the fitted torque term around the mean torque. A
    #: constant sag at a nearly constant load is indistinguishable from an
    #: offset, so this only credits the part that varies with pose.
    rms_corrected_deg: float
    torque_std_nm: float
    #: ``gravity_scale`` that would cancel the slope on this joint alone.
    implied_gravity_scale: float | None
    #: ``[mean torque N*m, mean residual deg, frames]`` per torque quantile bin.
    bins: list[list[float]]
    #: Mean residual (deg) while the real joint is ``still``, moving ``up``
    #: (positive velocity) or ``down``. Sag tracks the torque bins above; a
    #: deadband or backlash shows as a jump between these instead.
    by_motion: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "slope_deg_per_nm": self.slope_deg_per_nm,
            "partial_r2": self.partial_r2,
            "rms_deg": self.rms_deg,
            "rms_corrected_deg": self.rms_corrected_deg,
            "torque_std_nm": self.torque_std_nm,
            "implied_gravity_scale": self.implied_gravity_scale,
            "bins": self.bins,
            "by_motion": dict(self.by_motion),
        }


@dataclass(frozen=True)
class SagReport:
    n_episodes: int
    n_frames: int
    gravity_scale: float  # of the parameters replayed
    joints: dict[str, JointSag]
    #: Torque-weighted mean of the per-joint implied scales (joints with signal).
    implied_gravity_scale: float | None

    @property
    def worst_joint(self) -> str | None:
        """Joint whose residual the holding torque explains most."""
        with_signal = [j for j in self.joints.values() if j.implied_gravity_scale is not None]
        return max(with_signal, key=lambda j: j.partial_r2).joint if with_signal else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "n_frames": self.n_frames,
            "gravity_scale": self.gravity_scale,
            "joints": {k: v.to_dict() for k, v in self.joints.items()},
            "implied_gravity_scale": self.implied_gravity_scale,
            "worst_joint": self.worst_joint,
        }


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares coefficients and residual sum of squares."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return coef, float(r @ r)


def sag_report(arm: Arm, episodes: Sequence[Episode]) -> SagReport:
    """Regress the replay residual on holding torque for every joint but the gripper.

    ``arm`` carries the parameters under test (``arm.apply(lock_params)``).
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("sag_report needs at least one episode")
    res, tau, vel = [], [], []
    for ep in episodes:
        real = arm.real_q(ep)
        sim = arm.replay(ep)
        res.append(sim - real)
        tau.append(holding_torque(arm, real))
        vel.append(np.gradient(real, 1.0 / ep.fps, axis=0) if len(real) > 1 else np.zeros_like(real))
    res_a, tau_a, vel_a = (np.concatenate(x) for x in (res, tau, vel))
    n = res_a.shape[0]
    kp = arm.params.per_joint("kp")
    scale = float(arm.params.per_joint("gravity_scale")[0])

    joints: dict[str, JointSag] = {}
    weights, implied = [], []
    for j, name in enumerate(JOINTS):
        if j == GRIPPER_INDEX:
            continue
        y = np.degrees(res_a[:, j])
        t = tau_a[:, j]
        ones = np.ones(n)
        base_X = np.column_stack([ones, vel_a[:, j]])
        _, rss0 = _ols(base_X, y)
        coef, rss1 = _ols(np.column_stack([base_X, t]), y)
        slope = float(coef[2])
        t_std = float(np.std(t))
        partial = 0.0 if rss0 <= 0 else max(0.0, 1.0 - rss1 / rss0)
        corrected = y - slope * (t - t.mean())
        # At rest sim_q ~= cmd - tau / kp with tau measured at ``scale``, and the
        # real arm settles at cmd - (s_real / scale) * tau / kp, so the residual
        # slope b (rad per N*m) is (s_real - scale) / (scale * kp).
        imp = None
        if t_std >= MIN_TORQUE_STD:
            imp = float(scale * (1.0 + np.radians(slope) * kp[j]))
            weights.append(t_std)
            implied.append(imp)
        edges = np.quantile(t, np.linspace(0, 1, N_BINS + 1))
        idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, N_BINS - 1)
        bins = [
            [float(t[idx == b].mean()), float(y[idx == b].mean()), int(np.sum(idx == b))]
            for b in range(N_BINS)
            if np.any(idx == b)
        ]
        v = np.degrees(vel_a[:, j])
        masks = {"still": np.abs(v) < STILL_DEG_S, "up": v >= STILL_DEG_S, "down": v <= -STILL_DEG_S}
        by_motion = {k: (float(y[mk].mean()) if np.any(mk) else None) for k, mk in masks.items()}
        joints[name] = JointSag(
            joint=name,
            slope_deg_per_nm=slope,
            partial_r2=float(partial),
            rms_deg=float(np.sqrt(np.mean(y**2))),
            rms_corrected_deg=float(np.sqrt(np.mean(corrected**2))),
            torque_std_nm=t_std,
            implied_gravity_scale=imp,
            bins=bins,
            by_motion=by_motion,
        )
    overall = float(np.average(implied, weights=weights)) if implied else None
    return SagReport(
        n_episodes=len(episodes),
        n_frames=n,
        gravity_scale=scale,
        joints=joints,
        implied_gravity_scale=overall,
    )


__all__ = ["JointSag", "SagReport", "holding_torque", "sag_report", "MIN_TORQUE_STD", "STILL_DEG_S"]
