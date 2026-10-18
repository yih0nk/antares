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

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from antares import __version__
from antares.data import Episode
from antares.events import GripperThresholds, gripper_thresholds, gripper_transitions
from antares.lock import LOCK_DIR, slug
from antares.sim import GRIPPER_INDEX, UnitMap, load_model

TASK_SUFFIX = ".task.json"

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stats(points: np.ndarray) -> dict[str, Any] | None:
    """Mean, standard deviation and min/max per axis of an (N,3) cloud."""
    if len(points) == 0:
        return None
    return {
        "n": int(len(points)),
        "mean": points.mean(axis=0).tolist(),
        "std": points.std(axis=0).tolist(),
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
    }


@dataclass
class TaskDistribution:
    """Where one dataset's objects were picked and placed, and where episodes start.

    Positions are the gripper tip in the arm's base frame, metres. Start
    poses are joint angles in radians (all six joints, gripper included).
    """

    name: str
    dataset: str
    fps: float
    thresholds: GripperThresholds
    grasps: list[Grasp]
    start_q: np.ndarray  # (E, 6) rad
    episodes: list[int]
    created_at: str = field(default_factory=_now_iso)
    antares_version: str = __version__

    @classmethod
    def from_episodes(
        cls,
        episodes: Sequence[Episode],
        name: str,
        dataset: str = "",
        unit_map: UnitMap | None = None,
        held_gap: float = HELD_GAP,
    ) -> TaskDistribution:
        if not episodes:
            raise ValueError("no episodes")
        tip = Tip()
        unit_map = unit_map if unit_map is not None else UnitMap.from_model(tip.model)
        thr = dataset_thresholds(episodes)
        grasps = [g for e in episodes for g in find_grasps(e, thr, tip, unit_map, held_gap)]
        return cls(
            name=name,
            dataset=dataset,
            fps=float(episodes[0].fps),
            thresholds=thr,
            grasps=grasps,
            start_q=np.stack([unit_map.to_rad(e.state[0]) for e in episodes]),
            episodes=[int(e.index) for e in episodes],
        )

    # -- the clouds -----------------------------------------------------------

    @property
    def held(self) -> list[Grasp]:
        return [g for g in self.grasps if g.held]

    @property
    def picks(self) -> np.ndarray:
        """(N,3) tip positions of the held grasps."""
        return np.array([g.xyz for g in self.held], dtype=np.float64).reshape(-1, 3)

    @property
    def places(self) -> np.ndarray:
        """(N,3) release positions of the held grasps that were released."""
        pts = [g.release_xyz for g in self.held if g.release_xyz is not None]
        return np.array(pts, dtype=np.float64).reshape(-1, 3)

    @property
    def missed_episodes(self) -> list[int]:
        """Episodes with at least one grasp that closed on nothing."""
        return sorted({g.episode for g in self.grasps if not g.held})

    def summary(self) -> dict[str, Any]:
        held = self.held
        return {
            "episodes": len(self.episodes),
            "grasps": len(self.grasps),
            "held": len(held),
            "missed": len(self.grasps) - len(held),
            "missed_episodes": self.missed_episodes,
            "pick": _stats(self.picks),
            "place": _stats(self.places),
            "jaw_deg_median": float(np.median([g.jaw_deg for g in held])) if held else None,
        }

    # -- sampling -------------------------------------------------------------

    def sample(self, n: int, rng: np.random.Generator | None = None, jitter_m: float = 0.01) -> list[dict[str, Any]]:
        """Draw ``n`` task instances for a sim environment.

        Each picks one held grasp at random and returns its pick and place
        positions, both with independent Gaussian jitter of ``jitter_m`` per
        axis (a kernel density estimate of the logged clouds), plus a start
        pose drawn from the logged ones. ``place`` is None for a grasp that
        was never released.
        """
        rng = rng if rng is not None else np.random.default_rng()
        held = self.held
        if not held:
            raise ValueError("no held grasps to sample from")
        out = []
        for _ in range(int(n)):
            g = held[int(rng.integers(len(held)))]
            pick = np.asarray(g.xyz) + rng.normal(0.0, jitter_m, 3)
            place = None if g.release_xyz is None else np.asarray(g.release_xyz) + rng.normal(0.0, jitter_m, 3)
            out.append(
                {
                    "pick": pick.tolist(),
                    "place": None if place is None else place.tolist(),
                    "start_q": self.start_q[int(rng.integers(len(self.start_q)))].tolist(),
                }
            )
        return out

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "fps": self.fps,
            "tip_site": TIP_SITE,
            "thresholds": {
                "closed_below": self.thresholds.closed_below,
                "open_above": self.thresholds.open_above,
            },
            "grasps": [g.to_dict() for g in self.grasps],
            "start_q": self.start_q.tolist(),
            "episodes": self.episodes,
            "summary": self.summary(),
            "created_at": self.created_at,
            "antares_version": self.antares_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskDistribution:
        def xyz(v):
            return None if v is None else tuple(float(x) for x in v)

        return cls(
            name=str(d["name"]),
            dataset=str(d.get("dataset", "")),
            fps=float(d["fps"]),
            thresholds=GripperThresholds(
                closed_below=float(d["thresholds"]["closed_below"]),
                open_above=float(d["thresholds"]["open_above"]),
            ),
            grasps=[
                Grasp(
                    episode=int(g["episode"]),
                    frame=int(g["frame"]),
                    xyz=xyz(g["xyz"]),
                    release_frame=None if g["release_frame"] is None else int(g["release_frame"]),
                    release_xyz=xyz(g["release_xyz"]),
                    jaw_gap=float(g["jaw_gap"]),
                    jaw_deg=float(g["jaw_deg"]),
                    held=bool(g["held"]),
                )
                for g in d["grasps"]
            ],
            start_q=np.asarray(d["start_q"], dtype=np.float64).reshape(-1, 6),
            episodes=[int(i) for i in d["episodes"]],
            created_at=str(d.get("created_at", "")),
            antares_version=str(d.get("antares_version", "")),
        )

    def default_path(self, lock_dir: Path = LOCK_DIR) -> Path:
        return Path(lock_dir) / f"{slug(self.name)}{TASK_SUFFIX}"

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path is not None else self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> TaskDistribution:
        return cls.from_dict(json.loads(Path(path).read_text()))
