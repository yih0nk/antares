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
them the jaws stop open by the object's width; on a miss they close all the
way. "All the way" is the dataset's own empty-closed level (the 1st
percentile of the measured jaw and of the closed command), and the jaw is
read just after it settles, before the operator has a chance to let go.
The angle it stops at is a proxy for the object's width.

Each close is then one of three kinds: a *pick* (it held something), a
*miss* (it closed on nothing before anything was picked in that episode),
or *idle* (it closed on nothing after a pick, typically the operator
shutting the gripper after letting go). Only picks that moved the object
more than :data:`CARRY_M` before letting go give a place position.

Pure kinematics: no dynamics and no lock are needed, only the dataset-unit
-> radian map.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
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

#: Names of what :func:`scene_xml` adds to an exported scene.
OBJECT_BODY = "antares_object"
OBJECT_JOINT = "antares_object_free"
PLACE_SITE = "antares_place"

#: Default object: a cube this wide (m) and this heavy (kg).
OBJECT_SIZE = 0.03
OBJECT_MASS = 0.03

#: MJCF site at the gripper tip, between the jaws.
TIP_SITE = "gripperframe"

#: Measured jaw this far above the dataset's empty-closed level (dataset
#: units, 0..100) means something is between the jaws.
HELD_GAP = 3.0

#: The jaw is still closing for this long after the command flips; the stop
#: position is read over the next ``JAW_WINDOW_S`` after it.
SETTLE_S = 0.3
JAW_WINDOW_S = 0.5

#: A held object that ends up farther than this from where it was picked
#: was carried (a re-grab in place is not a place).
CARRY_M = 0.03


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
    jaw_gap: float  # measured jaw above the empty-closed level just after settling, dataset units
    jaw_deg: float  # measured jaw angle just after settling, degrees
    held: bool  # jaw_gap >= the held threshold: an object stopped the jaw
    carried: bool = False  # held, released, and moved more than CARRY_M
    kind: str = "miss"  # "pick", "miss" or "idle"; see the module docstring

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
            "carried": self.carried,
            "kind": self.kind,
        }


def dataset_thresholds(episodes: Sequence[Episode]) -> GripperThresholds:
    """Gripper thresholds fitted on every episode's command at once, around the midpoint.

    One pair for the whole dataset, so a short episode that never opens the
    gripper far does not get thresholds of its own. The contact detector's
    thresholds sit at 1/3 and 2/3 between the open and closed clusters;
    operators often close the leader only part way on an object (SO-100
    logs hold a cube at commands 3-11 with the closed cluster near 0), so
    here the band is narrowed to the middle sixth: 5/12 and 7/12.
    """
    cmd = np.concatenate([np.asarray(e.action[:, GRIPPER_INDEX], dtype=np.float64) for e in episodes])
    t = gripper_thresholds(cmd)

    mid, half = 0.5 * (t.closed_below + t.open_above), (t.open_above - t.closed_below) / 4.0
    return GripperThresholds(closed_below=mid - half, open_above=mid + half)


def empty_level(episodes: Sequence[Episode], thr: GripperThresholds) -> float:
    """Where the jaw sits when it closes on nothing, dataset units.

    The 1st percentile of the measured jaw and of the closed commands, pooled:
    the measured jaw covers datasets with empty closes, the command covers
    ones where every close held something but the leader was squeezed shut.
    """
    vals = []
    for e in episodes:
        cmd = np.asarray(e.action[:, GRIPPER_INDEX], dtype=np.float64)
        vals.append(np.asarray(e.state[:, GRIPPER_INDEX], dtype=np.float64))
        vals.append(cmd[cmd < thr.closed_below])
    return float(np.percentile(np.concatenate(vals), 1.0))


def find_grasps(
    episode: Episode,
    thr: GripperThresholds,
    tip: Tip,
    unit_map: UnitMap,
    empty: float,
    held_gap: float = HELD_GAP,
) -> list[Grasp]:
    """Every grasp in ``episode``, paired with the release that follows it.

    The release is the next open flip of the command or, for a held grasp,
    the first frame the jaw opens ``held_gap`` past where the object held
    it, whichever comes first.

    The jaw is read over :data:`JAW_WINDOW_S` starting :data:`SETTLE_S`
    after the close, cut short by the release or the end of the episode
    (at least one frame), and compared with ``empty`` (see :func:`empty_level`).
    """
    jaw = episode.state[:, GRIPPER_INDEX]
    grasps, releases = gripper_transitions(episode.action[:, GRIPPER_INDEX], thr)
    q = unit_map.to_rad(episode.state)
    settle = int(round(SETTLE_S * episode.fps))
    window = max(1, int(round(JAW_WINDOW_S * episode.fps)))
    out = []
    for g in grasps:
        later = releases[releases > g]
        r = int(later[0]) if later.size else None
        end = r if r is not None else len(episode)
        start = min(g + settle, end - 1)
        stop = min(start + window, end)
        hold = float(np.median(jaw[start:stop]))
        gap = hold - empty
        jaw_rad = float(np.median(q[start:stop, GRIPPER_INDEX]))
        if gap >= held_gap:
            # Letting go opens the jaw past where the object held it, even
            # when the leader opens too little to cross the thresholds.
            opened = np.flatnonzero(jaw[start:end] > hold + held_gap)
            if opened.size:
                r = start + int(opened[0])
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
    picked = False
    for i, gr in enumerate(out):
        carried = (
            gr.held
            and gr.release_xyz is not None
            and float(np.linalg.norm(np.subtract(gr.release_xyz, gr.xyz))) > CARRY_M
        )
        kind = "pick" if gr.held else ("idle" if picked else "miss")
        picked = picked or gr.held
        out[i] = replace(gr, carried=carried, kind=kind)
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
        empty = empty_level(episodes, thr)
        grasps = [g for e in episodes for g in find_grasps(e, thr, tip, unit_map, empty, held_gap)]
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
    def carried(self) -> list[Grasp]:
        return [g for g in self.grasps if g.carried]

    @property
    def places(self) -> np.ndarray:
        """(N,3) release positions of the carried picks."""
        return np.array([g.release_xyz for g in self.carried], dtype=np.float64).reshape(-1, 3)

    @property
    def missed_episodes(self) -> list[int]:
        """Episodes with a close on nothing before anything was picked."""
        return sorted({g.episode for g in self.grasps if g.kind == "miss"})

    @property
    def episodes_without_pick(self) -> list[int]:
        picked = {g.episode for g in self.held}
        return [i for i in self.episodes if i not in picked]

    def summary(self) -> dict[str, Any]:
        held = self.held
        return {
            "episodes": len(self.episodes),
            "grasps": len(self.grasps),
            "held": len(held),
            "carried": len(self.carried),
            "missed": sum(g.kind == "miss" for g in self.grasps),
            "idle": sum(g.kind == "idle" for g in self.grasps),
            "missed_episodes": self.missed_episodes,
            "episodes_without_pick": self.episodes_without_pick,
            "pick": _stats(self.picks),
            "place": _stats(self.places),
            "jaw_deg_median": float(np.median([g.jaw_deg for g in held])) if held else None,
        }

    # -- sampling -------------------------------------------------------------

    def sample(self, n: int, rng: np.random.Generator | None = None, jitter_m: float = 0.01) -> list[dict[str, Any]]:
        """Draw ``n`` task instances for a sim environment.

        Each picks one carried pick at random (any held grasp if none was
        carried) and returns its pick and place positions, both with
        independent Gaussian jitter of ``jitter_m`` per axis (a kernel
        density estimate of the logged clouds), plus a start pose drawn from
        the logged ones. ``place`` is None when the grasp was not carried.
        """
        rng = rng if rng is not None else np.random.default_rng()
        pool = self.carried or self.held
        if not pool:
            raise ValueError("no held grasps to sample from")
        out = []
        for _ in range(int(n)):
            g = pool[int(rng.integers(len(pool)))]
            pick = np.asarray(g.xyz) + rng.normal(0.0, jitter_m, 3)
            place = np.asarray(g.release_xyz) + rng.normal(0.0, jitter_m, 3) if g.carried else None
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
                    carried=bool(g.get("carried", False)),
                    kind=str(g.get("kind", "pick" if g["held"] else "miss")),
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


# --------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------


def _resting(xyz: Sequence[float], size: float) -> np.ndarray:
    """``xyz`` with z raised so a cube of ``size`` centred there is not below the floor."""
    p = np.asarray(xyz, dtype=np.float64).copy()
    p[2] = max(p[2], size / 2.0)
    return p


def scene_xml(td: TaskDistribution, size: float = OBJECT_SIZE, mass: float = OBJECT_MASS) -> str:
    """``<worldbody>`` children for a free cube at the mean pick and a marker at the mean place.

    The arm's base frame is the scene's world frame, so logged positions go
    in unchanged. Use :func:`set_task` to move both to a sampled instance.
    """
    picks, places = td.picks, td.places
    if len(picks) == 0:
        raise ValueError("no held grasps: nothing to place in the scene")
    pick = _resting(picks.mean(axis=0), size)
    place = places.mean(axis=0) if len(places) else pick
    h = size / 2.0
    f = lambda v: " ".join(f"{x:.6g}" for x in v)  # noqa: E731
    return (
        f'    <body name="{OBJECT_BODY}" pos="{f(pick)}">\n'
        f'      <freejoint name="{OBJECT_JOINT}"/>\n'
        f'      <geom type="box" size="{h:.6g} {h:.6g} {h:.6g}" mass="{mass:.6g}" '
        f'friction="1 5e-3 1e-4" rgba="0.85 0.25 0.2 1"/>\n'
        f"    </body>\n"
        f'    <site name="{PLACE_SITE}" pos="{f(place)}" size="{h:.6g}" type="sphere" rgba="0.2 0.8 0.3 0.4"/>\n'
    )


def set_task(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    instance: dict[str, Any],
    unit_map: UnitMap | None = None,
    size: float = OBJECT_SIZE,
) -> None:
    """Reset a scene with :func:`scene_xml` objects to one :meth:`TaskDistribution.sample` draw.

    Moves the cube (upright, at rest) to ``instance["pick"]``, the place
    marker to ``instance["place"]`` (left where it is when None) and the arm
    to ``instance["start_q"]``, then runs ``mj_forward``.
    """
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PLACE_SITE)
    if jnt < 0 or site < 0:
        raise ValueError("model has no antares task objects; export it with a task")
    a = model.jnt_qposadr[jnt]
    data.qpos[a : a + 3] = _resting(instance["pick"], size)
    data.qpos[a + 3 : a + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[model.jnt_dofadr[jnt] : model.jnt_dofadr[jnt] + 6] = 0.0
    if instance.get("place") is not None:
        model.site_pos[site] = instance["place"]
    data.qpos[:6] = instance["start_q"]
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)
