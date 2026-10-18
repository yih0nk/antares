"""Robot locks: the calibrated simulator for one physical robot, saved like a
lockfile.

A lock records which dataset (and which episodes, by content digest) the
fit was trained on, the fitted :class:`~antares.sim.SimParams`, and the full
:class:`~antares.calibrate.CalibrationResult`, so a sim can be pinned to a
specific arm and the fit reproduced or audited later.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Self
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from antares import __version__
from antares.calibrate import CalibrationResult, SimParams
from antares.data import Episode
from antares.robots import DEFAULT_ROBOT

LOCK_DIR = Path("locks")
LOCK_SUFFIX = ".lock.json"


def slug(repo_id: str) -> str:
    """``"lerobot/svla_so101_pickplace"`` -> ``"svla_so101_pickplace"``."""
    name = repo_id.rstrip("/").rsplit("/", 1)[-1].strip().lower()
    name = re.sub(r"[^a-z0-9_.-]+", "_", name).strip("_.-")
    return name or "robot"


def dataset_digest(episodes: Sequence[Episode]) -> str:
    """SHA-256 over the episode indices and the state/action arrays.

    Independent of the order the episodes are passed in; sensitive to any
    change in the recorded values, shapes, fps or joint names.
    """
    h = hashlib.sha256()
    for ep in sorted(episodes, key=lambda e: e.index):
        header = f"{ep.index}|{ep.fps}|{','.join(ep.joint_names)}|{ep.state.shape}|{ep.action.shape}\n"
        h.update(header.encode())
        h.update(np.ascontiguousarray(ep.state, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(ep.action, dtype=np.float64).tobytes())
    return h.hexdigest()


def now_iso() -> str:
    """The current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``, the ``created_at`` stamp of every saved record."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class JsonFile:
    """``default_path``/``save``/``load`` for a record with ``to_dict``/``from_dict``.

    The file is ``<dir>/<file_stem><FILE_SUFFIX>`` (``FILE_DIR`` unless a
    directory is passed), written as indented JSON with sorted keys and a
    trailing newline. ``file_stem`` is the record's ``name``; override it
    to slug or otherwise derive the file name.
    """

    FILE_SUFFIX: ClassVar[str] = ".json"
    FILE_DIR: ClassVar[Path] = LOCK_DIR

    @property
    def file_stem(self) -> str:
        return self.name  # type: ignore[attr-defined]

    def default_path(self, lock_dir: Path | None = None) -> Path:
        return Path(lock_dir if lock_dir is not None else self.FILE_DIR) / f"{self.file_stem}{self.FILE_SUFFIX}"

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path is not None else self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")  # type: ignore[attr-defined]
        return path

    @classmethod
    def load(cls, path: Path | str) -> Self:
        return cls.from_dict(json.loads(Path(path).read_text()))  # type: ignore[attr-defined]


@dataclass(frozen=True)
class RobotLock(JsonFile):
    FILE_SUFFIX: ClassVar[str] = LOCK_SUFFIX

    name: str
    dataset: str
    robot_type: str
    data_sha256: str
    params: SimParams
    result: CalibrationResult
    created_at: str
    antares_version: str
    command_map: dict[str, Any] | None = None  # CommandMap.to_dict(); None = identity
    dynamics_only: dict[str, Any] | None = None  # CalibrationResult.to_dict()
    #: Which simulator model (``antares.robots.REGISTRY`` key) the fit ran on.
    #: Locks written before the registry existed were all SO-101.
    robot: str = DEFAULT_ROBOT
    # Fitted fixed cameras by short name (antares.camera.CameraFit.to_dict());
    # locks saved before camera fits existed load with none.
    cameras: dict[str, Any] = field(default_factory=dict)
    #: The follower's LeRobot motor calibration the fit used for its unit map
    #: (antares.lerobot_calib.calibration_record()); None = the model's map.
    calibration: dict[str, Any] | None = None
    # The static scene recovered from the logs (antares.scene.SceneFit.lock_entry():
    # table height, workspace box, backdrop paths); None until `scene-fit` runs.
    scene: dict[str, Any] | None = None
    # The manipulated object recovered from the video and the residual
    # (antares.objects.ObjectFit.lock_entry(): footprint, start/end pose
    # distribution, mass estimate); None until `objects-fit` runs.
    objects: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        dataset: str,
        robot_type: str,
        train_episodes: Sequence[Episode],
        result: CalibrationResult,
        name: str | None = None,
        command_map: dict[str, Any] | None = None,
        dynamics_only: dict[str, Any] | None = None,
        robot: str = DEFAULT_ROBOT,
        calibration: dict[str, Any] | None = None,
    ) -> "RobotLock":
        """Build a lock for a finished calibration, stamping time and version."""
        return cls(
            name=name or slug(dataset),
            dataset=dataset,
            robot_type=robot_type,
            data_sha256=dataset_digest(train_episodes),
            params=result.params,
            result=result,
            created_at=now_iso(),
            antares_version=__version__,
            command_map=command_map,
            dynamics_only=dynamics_only,
            robot=robot,
            calibration=calibration,
        )

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "dataset": self.dataset,
            "robot_type": self.robot_type,
            "robot": self.robot,
            "data_sha256": self.data_sha256,
            "params": self.params.to_dict(),
            "result": self.result.to_dict(),
            "created_at": self.created_at,
            "antares_version": self.antares_version,
        }
        if self.command_map is not None:
            d["command_map"] = self.command_map
        if self.dynamics_only is not None:
            d["dynamics_only"] = self.dynamics_only
        if self.cameras:
            d["cameras"] = dict(self.cameras)
        if self.calibration is not None:
            d["calibration"] = self.calibration
        if self.scene is not None:
            d["scene"] = dict(self.scene)
        if self.objects is not None:
            d["objects"] = dict(self.objects)
        return d

    def replace(self, **changes: Any) -> "RobotLock":
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, d: dict) -> "RobotLock":
        result = d["result"]
        if isinstance(result, dict):
            result = CalibrationResult.from_dict(result)
        params = d.get("params")
        params = SimParams.from_dict(params) if isinstance(params, dict) else result.params
        return cls(
            name=str(d["name"]),
            dataset=str(d["dataset"]),
            robot_type=str(d.get("robot_type", "unknown")),
            data_sha256=str(d["data_sha256"]),
            params=params,
            result=result,
            created_at=str(d["created_at"]),
            antares_version=str(d.get("antares_version", "")),
            command_map=d.get("command_map"),
            dynamics_only=d.get("dynamics_only"),
            robot=str(d.get("robot") or DEFAULT_ROBOT),
            cameras=dict(d.get("cameras") or {}),
            calibration=d.get("calibration"),
            scene=dict(d["scene"]) if isinstance(d.get("scene"), dict) else None,
            objects=dict(d["objects"]) if isinstance(d.get("objects"), dict) else None,
        )


#: Parameters taken as physical when a lock carries no seed spread. One frame
#: of latency either way costs 0.3-0.9 deg on the demo arms with everything
#: else held fixed, so it is the one parameter a single fit does pin down;
#: kp/kv/damping/armature/frictionloss spread 2-125x across seeds at equal error.
DEFAULT_PINNED: tuple[str, ...] = ("latency_frames",)


def seed_info(lock: Any) -> dict | None:
    """The ``result.seeds`` block of a lock (:class:`RobotLock`, a lock dict,
    or a :class:`~antares.calibrate.CalibrationResult`), or None when the
    lock was fitted with a single seed or before seeds were recorded."""
    if isinstance(lock, dict):
        result = lock.get("result")
        seeds = (result if isinstance(result, dict) else lock).get("seeds")
    else:
        result = getattr(lock, "result", lock)
        seeds = getattr(result, "seeds", None)
    return seeds if isinstance(seeds, dict) else None


def pinned_params(lock: Any) -> list[str]:
    """Parameters whose fitted value can be read as a measurement of the arm.

    With seed spread recorded (``calibrate --seeds 3``) these are the
    parameters that agreed across the seeds (``seeds.pinned``); otherwise
    the conservative :data:`DEFAULT_PINNED`. Drift, diagnose and fleet read
    only these as physical; the others trade off against each other at the
    same tracking error.
    """
    seeds = seed_info(lock)
    pinned = seeds.get("pinned") if seeds is not None else None
    return [str(p) for p in pinned] if pinned is not None else list(DEFAULT_PINNED)


def param_spread(lock: Any) -> dict | None:
    """``seeds.spread`` of a lock: per shared parameter ``{"min", "max",
    "ratio"}`` (log-scaled) or ``{"min", "max", "range"}`` (latency) across
    the seeds, or None when the lock carries no seed spread."""
    seeds = seed_info(lock)
    spread = seeds.get("spread") if seeds is not None else None
    return dict(spread) if isinstance(spread, dict) else None


def list_locks(lock_dir: Path = LOCK_DIR) -> list[RobotLock]:
    """All locks under ``lock_dir``, sorted by name. Unreadable files are
    skipped with a warning rather than hiding the rest."""
    lock_dir = Path(lock_dir)
    if not lock_dir.is_dir():
        return []
    locks = []
    for path in sorted(lock_dir.glob(f"*{LOCK_SUFFIX}")):
        try:
            locks.append(RobotLock.load(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.warn(f"skipping unreadable lock {path}: {exc}", stacklevel=2)
    return sorted(locks, key=lambda lk: lk.name)


__all__ = [
    "DEFAULT_PINNED",
    "LOCK_DIR",
    "RobotLock",
    "dataset_digest",
    "list_locks",
    "param_spread",
    "pinned_params",
    "seed_info",
    "slug",
]
