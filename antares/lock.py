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
from typing import Any
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from antares import __version__
from antares.calibrate import CalibrationResult, SimParams
from antares.data import Episode

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RobotLock:
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
    # Fitted fixed cameras by short name (antares.camera.CameraFit.to_dict());
    # locks saved before camera fits existed load with none.
    cameras: dict[str, Any] = field(default_factory=dict)

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
    ) -> "RobotLock":
        """Build a lock for a finished calibration, stamping time and version."""
        return cls(
            name=name or slug(dataset),
            dataset=dataset,
            robot_type=robot_type,
            data_sha256=dataset_digest(train_episodes),
            params=result.params,
            result=result,
            created_at=_now_iso(),
            antares_version=__version__,
            command_map=command_map,
            dynamics_only=dynamics_only,
        )

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "dataset": self.dataset,
            "robot_type": self.robot_type,
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
            cameras=dict(d.get("cameras") or {}),
        )

    def default_path(self, lock_dir: Path = LOCK_DIR) -> Path:
        return Path(lock_dir) / f"{self.name}{LOCK_SUFFIX}"

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path is not None else self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "RobotLock":
        return cls.from_dict(json.loads(Path(path).read_text()))


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


__all__ = ["LOCK_DIR", "RobotLock", "dataset_digest", "list_locks", "slug"]
