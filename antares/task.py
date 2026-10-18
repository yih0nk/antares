"""The task a robot was actually asked to do, read off its own logs.

A policy trained in sim sees whatever object positions and start poses the
environment hands it. The logs already record the real ones: when the
gripper command closes, the gripper tip is at the object, and when it opens
again the object is where it was put down. Forward kinematics of the
measured joint angles at those frames gives a cloud of pick and place
points in the arm's base frame; the first frame of every episode gives the
start poses.

Grasps and releases come from :func:`antares.events.gripper_transitions`
(the same hysteresis on the gripper command the contact detector uses). The
tip is the MJCF's ``gripperframe`` site, between the jaws.

Whether a grasp caught anything is read from the jaw: with an object between
them the jaws stop short of the closed command, so the measured gripper sits
well above the command; on a miss they close all the way. The jaw angle it
stops at is a proxy for the object's width.

Pure kinematics: no dynamics and no lock are needed, only the dataset-unit
-> radian map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from antares.data import Episode
from antares.events import GripperThresholds, gripper_thresholds, gripper_transitions
from antares.sim import GRIPPER_INDEX, UnitMap, load_model

#: MJCF site at the gripper tip, between the jaws.
TIP_SITE = "gripperframe"

#: Measured jaw this far above the closed command (dataset units, 0..100)
#: means something is between the jaws.
HELD_GAP = 5.0

#: The jaw is still closing for this long after the command flips; the stop
#: position is read after it.
SETTLE_S = 0.3


class Tip:
    """Gripper-tip position (base frame, metres) from joint angles in radians."""

    def __init__(self, model: mujoco.MjModel | None = None, site: str = TIP_SITE):
        self.model = model if model is not None else load_model()
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site < 0:
            raise ValueError(f"MJCF has no site {site!r}")

    def __call__(self, q: np.ndarray) -> np.ndarray:
        """(6,) -> (3,), or (T,6) -> (T,3)."""
        q = np.asarray(q, dtype=np.float64)
        if q.ndim == 1:
            return self._one(q)
        return np.stack([self._one(row) for row in q])

    def _one(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy()


@dataclass(frozen=True)
class Grasp:
    """One close of the gripper and the release that follows it (if any)."""

    episode: int
    frame: int  # gripper command flips to closed
    xyz: tuple[float, float, float]  # tip position at ``frame``, metres
    release_frame: int | None  # next flip back to open
    release_xyz: tuple[float, float, float] | None
    jaw_gap: float  # median measured - commanded gripper while closed, dataset units
    jaw_deg: float  # median measured jaw angle while closed, degrees
    held: bool  # jaw_gap >= the held threshold: an object stopped the jaw

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "frame": self.frame,
            "xyz": list(self.xyz),
            "release_frame": self.release_frame,
            "release_xyz": None if self.release_xyz is None else list(self.release_xyz),
            "jaw_gap": self.jaw_gap,
            "jaw_deg": self.jaw_deg,
            "held": self.held,
        }


def dataset_thresholds(episodes: Sequence[Episode]) -> GripperThresholds:
    """Gripper thresholds fitted on every episode's command at once.

    One pair for the whole dataset, so a short episode that never opens the
    gripper far does not get thresholds of its own.
    """
    cmd = np.concatenate([np.asarray(e.action[:, GRIPPER_INDEX], dtype=np.float64) for e in episodes])
    return gripper_thresholds(cmd)


def find_grasps(
    episode: Episode,
    thr: GripperThresholds,
    tip: Tip,
    unit_map: UnitMap,
    held_gap: float = HELD_GAP,
) -> list[Grasp]:
    """Every grasp in ``episode``, paired with the release that follows it.

    The jaw is read over the closed stretch (grasp to release, or to the end
    of the episode) after :data:`SETTLE_S`; a stretch shorter than that
    uses its last frame.
    """
    cmd = episode.action[:, GRIPPER_INDEX]
    jaw = episode.state[:, GRIPPER_INDEX]
    grasps, releases = gripper_transitions(cmd, thr)
    q = unit_map.to_rad(episode.state)
    settle = int(round(SETTLE_S * episode.fps))
    out = []
    for g in grasps:
        later = releases[releases > g]
        r = int(later[0]) if later.size else None
        end = r if r is not None else len(episode)
        start = min(g + settle, end - 1)
        gap = float(np.median(jaw[start:end] - cmd[start:end]))
        jaw_rad = float(np.median(q[start:end, GRIPPER_INDEX]))
        out.append(
            Grasp(
                episode=int(episode.index),
                frame=int(g),
                xyz=tuple(float(v) for v in tip(q[g])),
                release_frame=r,
                release_xyz=None if r is None else tuple(float(v) for v in tip(q[r])),
                jaw_gap=gap,
                jaw_deg=float(np.degrees(jaw_rad)),
                held=gap >= held_gap,
            )
        )
    return out
