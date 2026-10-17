"""Sim-to-real gap metrics: per-joint and overall tracking error between a
real episode and its simulated replay.

Inputs are joint trajectories in radians, (T, J); errors are reported in
degrees. The gripper is excluded from the headline score by default because
object contacts are not logged, so the sim cannot be expected to match it.
Pure numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from antares.sim import JOINTS


@dataclass(frozen=True)
class GapReport:
    per_joint_deg: dict[str, float]  # RMSE per joint in degrees, all joints
    overall_deg: float  # RMS over the scored joints
    scored_joints: tuple[str, ...]
    n_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_joint_deg": {k: float(v) for k, v in self.per_joint_deg.items()},
            "overall_deg": float(self.overall_deg),
            "scored_joints": list(self.scored_joints),
            "n_frames": int(self.n_frames),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GapReport:
        return cls(
            per_joint_deg={k: float(v) for k, v in d["per_joint_deg"].items()},
            overall_deg=float(d["overall_deg"]),
            scored_joints=tuple(d["scored_joints"]),
            n_frames=int(d["n_frames"]),
        )


def _is_sequence_of_arrays(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and (len(x) == 0 or np.ndim(x[0]) == 2)


def tracking_error(
    real_q: np.ndarray | Sequence[np.ndarray],
    sim_q: np.ndarray | Sequence[np.ndarray],
    joint_names: Sequence[str] = JOINTS,
    exclude: Sequence[str] = ("gripper",),
) -> GapReport:
    """RMSE between real and simulated joint trajectories, in degrees.

    ``real_q``/``sim_q`` are (T, J) arrays in radians, or matching lists of
    such arrays (one per episode), in which case the result is the
    frame-weighted combination over all episodes.
    """
    if _is_sequence_of_arrays(real_q) or _is_sequence_of_arrays(sim_q):
        if len(real_q) != len(sim_q):
            raise ValueError("real_q and sim_q must have the same number of episodes")
        if len(real_q) == 0:
            raise ValueError("no episodes to score")
        return aggregate([tracking_error(r, s, joint_names, exclude) for r, s in zip(real_q, sim_q)])

    real = np.asarray(real_q, dtype=np.float64)
    sim = np.asarray(sim_q, dtype=np.float64)
    if real.ndim != 2 or sim.ndim != 2:
        raise ValueError(f"expected (T, J) arrays, got {real.shape} and {sim.shape}")
    if real.shape != sim.shape:
        raise ValueError(f"shape mismatch: real {real.shape} vs sim {sim.shape}")
    joint_names = tuple(joint_names)
    if real.shape[1] != len(joint_names):
        raise ValueError(f"{real.shape[1]} columns but {len(joint_names)} joint names")
    n = real.shape[0]
    if n == 0:
        raise ValueError("empty trajectory")

    err_deg = np.degrees(sim - real)
    mse = np.mean(err_deg**2, axis=0)  # (J,)
    per_joint = {name: float(np.sqrt(mse[j])) for j, name in enumerate(joint_names)}

    scored = tuple(name for name in joint_names if name not in set(exclude))
    if not scored:
        raise ValueError("all joints excluded; nothing to score")
    scored_idx = [joint_names.index(name) for name in scored]
    overall = float(np.sqrt(np.mean(mse[scored_idx])))
    return GapReport(per_joint_deg=per_joint, overall_deg=overall, scored_joints=scored, n_frames=n)


def aggregate(reports: Sequence[GapReport]) -> GapReport:
    """Frame-weighted RMS combination of several reports.

    Equivalent to scoring the concatenated trajectories, as long as every
    report has the same joints and the same scored set.
    """
    reports = list(reports)
    if not reports:
        raise ValueError("no reports to aggregate")
    first = reports[0]
    names = tuple(first.per_joint_deg)
    scored = first.scored_joints
    for r in reports[1:]:
        if tuple(r.per_joint_deg) != names:
            raise ValueError("reports have different joints")
        if r.scored_joints != scored:
            raise ValueError("reports have different scored joints")

    weights = np.array([r.n_frames for r in reports], dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        raise ValueError("reports contain no frames")
    per_joint = {}
    for name in names:
        mse = np.array([r.per_joint_deg[name] ** 2 for r in reports])
        per_joint[name] = float(np.sqrt(np.sum(weights * mse) / total))
    overall_mse = np.array([r.overall_deg**2 for r in reports])
    overall = float(np.sqrt(np.sum(weights * overall_mse) / total))
    return GapReport(per_joint_deg=per_joint, overall_deg=overall, scored_joints=scored, n_frames=int(total))
