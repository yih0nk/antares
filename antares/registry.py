"""One identity per physical robot: the lock registry.

A :class:`~antares.lock.RobotLock` is a snapshot of one arm on one day.
Recalibrating means a new lock under a new name, and every ``check`` that
pointed at the old name has to be re-pointed. The registry gives each
physical robot a stable id (LeRobot's robot id, e.g. ``arm_a``), a versioned
lineage of the locks fitted to it, and one *active* lock that the rest of
Antares uses whenever it is handed the robot reference ``@arm_a`` in place
of a lock name.

The file is ``locks/registry.json``::

    {"version": 1,
     "robots": {"arm_a": {"robot_type": "so101_follower", "active": "arm_a-20261017",
                          "locks": [{"name": "arm_a-20261001", "created_at": "...",
                                     "dataset": "...", "eval_score": 1.05,
                                     "baseline_eval_score": 4.1, "note": null,
                                     "registered_at": "..."}, ...oldest first],
                          "created_at": "...", "updated_at": "..."}}}

Locks themselves are untouched; the registry only references them by name,
so an unregistered lock keeps working everywhere exactly as before. Writes
are atomic (temp file + rename) and a missing file is an empty registry.

Activation rule (:meth:`Registry.register`): the first lock of a robot is
active; a later one is activated when its held-out ``eval_score`` is not
worse than the active lock's by more than ``max_regression`` (10 % by
default), or when ``activate=True`` forces it. Otherwise it is registered
but left inactive, and the result says why. :meth:`Registry.rollback`
activates the lock before the active one in the lineage.

Pure stdlib: the lock module is imported lazily, only to read which
parameters a lock pins, so ``antares --help`` stays light and the server's
tests can stub it.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_FILENAME = "registry.json"
REGISTRY_VERSION = 1
LOCK_SUFFIX = ".lock.json"

#: ``@<robot id>`` names a robot wherever a lock name is accepted.
ROBOT_REF_PREFIX = "@"
#: A newer lock is activated by default unless its held-out score is worse
#: than the active one's by more than this fraction.
DEFAULT_MAX_REGRESSION = 0.10

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RegistryError(ValueError):
    """A malformed registry file, a bad id, or an operation that cannot apply."""


class UnknownRobot(LookupError):
    """``robot_id`` is not registered (or has no active lock)."""

    def __str__(self) -> str:  # LookupError would quote the message
        return str(self.args[0]) if self.args else ""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_robot_ref(ref: Any) -> bool:
    """True for ``@arm_a``; false for a lock name or a path."""
    return isinstance(ref, str) and ref.startswith(ROBOT_REF_PREFIX)


def robot_ref(robot_id: str) -> str:
    """``"arm_a"`` -> ``"@arm_a"``."""
    return f"{ROBOT_REF_PREFIX}{robot_id}"


def robot_id_of(ref: str) -> str:
    """``"@arm_a"`` -> ``"arm_a"``; validates the id."""
    rid = ref[len(ROBOT_REF_PREFIX):] if is_robot_ref(ref) else str(ref)
    return validate_robot_id(rid)


def validate_robot_id(robot_id: str) -> str:
    if not isinstance(robot_id, str) or not _ID_RE.match(robot_id) or ".." in robot_id:
        raise RegistryError(f"invalid robot id {robot_id!r}")
    return robot_id


def _score(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------
# entries
# --------------------------------------------------------------------------


@dataclass
class LockEntry:
    """One registered lock: what the registry remembers without opening the file."""

    name: str
    created_at: str | None = None
    dataset: str | None = None
    eval_score: float | None = None
    baseline_eval_score: float | None = None
    note: str | None = None
    registered_at: str = field(default_factory=now_iso)

    @classmethod
    def from_lock(cls, lock: dict, note: str | None = None) -> "LockEntry":
        res = lock.get("result") if isinstance(lock.get("result"), dict) else {}
        return cls(
            name=str(lock["name"]),
            created_at=lock.get("created_at"),
            dataset=lock.get("dataset"),
            eval_score=_score(res.get("eval_score")),
            baseline_eval_score=_score(res.get("baseline_eval_score")),
            note=note,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "LockEntry":
        if not isinstance(d, dict) or not d.get("name"):
            raise RegistryError("lock entry needs a name")
        return cls(
            name=str(d["name"]),
            created_at=d.get("created_at"),
            dataset=d.get("dataset"),
            eval_score=_score(d.get("eval_score")),
            baseline_eval_score=_score(d.get("baseline_eval_score")),
            note=d.get("note"),
            registered_at=str(d.get("registered_at") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "dataset": self.dataset,
            "eval_score": self.eval_score,
            "baseline_eval_score": self.baseline_eval_score,
            "note": self.note,
            "registered_at": self.registered_at,
        }


@dataclass
class RobotEntry:
    robot_type: str | None = None
    active: str | None = None
    locks: list[LockEntry] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.locks]

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise RegistryError(f"lock {name!r} is not in this robot's lineage") from None

    def version(self, name: str) -> int:
        """1-based position of ``name`` in the lineage (oldest is 1)."""
        return self.index(name) + 1

    def entry(self, name: str) -> LockEntry:
        return self.locks[self.index(name)]

    @property
    def active_entry(self) -> LockEntry | None:
        return self.entry(self.active) if self.active and self.active in self.names else None

    @classmethod
    def from_dict(cls, d: dict) -> "RobotEntry":
        if not isinstance(d, dict):
            raise RegistryError("robot entry is not an object")
        locks = [LockEntry.from_dict(x) for x in d.get("locks") or []]
        return cls(
            robot_type=d.get("robot_type"),
            active=d.get("active"),
            locks=locks,
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_type": self.robot_type,
            "active": self.active,
            "locks": [e.to_dict() for e in self.locks],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class RegistrationResult:
    robot_id: str
    lock: str
    version: int
    n_versions: int
    activated: bool
    active: str
    previous_active: str | None
    reason: str
    new: bool = True  # False when the name was already in the lineage and was refreshed

    def summary(self) -> str:
        what = "registered" if self.new else "re-registered"
        head = f"{what} lock {self.lock} as v{self.version} of {self.n_versions} for robot {self.robot_id}"
        if self.activated:
            return f"{head}; now active ({self.reason})"
        if self.active == self.lock:
            return f"{head}; still active"
        return f"{head}; not activated: {self.reason}; {self.active} stays active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "lock": self.lock,
            "version": self.version,
            "n_versions": self.n_versions,
            "activated": self.activated,
            "active": self.active,
            "previous_active": self.previous_active,
            "reason": self.reason,
            "new": self.new,
            "summary": self.summary(),
        }


# --------------------------------------------------------------------------
# lock files (read only; the registry never writes a lock)
# --------------------------------------------------------------------------


def lock_path(name: str, lock_dir: Path | str) -> Path:
    return Path(lock_dir) / f"{name}{LOCK_SUFFIX}"


def read_lock(lock: Any, lock_dir: Path | str) -> dict:
    """``lock`` as a dict: a dict, an object with ``to_dict()`` (a RobotLock),
    a ``.lock.json`` path, or a lock name under ``lock_dir``."""
    if isinstance(lock, dict):
        return lock
    if hasattr(lock, "to_dict"):
        return lock.to_dict()
    ref = str(lock)
    if is_robot_ref(ref):
        raise RegistryError(f"{ref!r} is a robot reference, not a lock")
    p = Path(ref)
    if not (p.is_file() and ref.endswith(".json")):
        if not _ID_RE.match(ref) or ".." in ref:
            raise RegistryError(f"invalid lock name {ref!r}")
        p = lock_path(ref, lock_dir)
    if not p.is_file():
        raise FileNotFoundError(f"no lock named {ref!r} in {Path(lock_dir)}")
    try:
        d = json.loads(p.read_text())
    except ValueError as e:
        raise RegistryError(f"lock {p} is not valid JSON: {e}") from None
    if not isinstance(d, dict) or not d.get("name"):
        raise RegistryError(f"lock {p} has no name")
    return d


def _lock_helpers() -> tuple[Any, Any]:
    """``(pinned_params, param_spread)`` from :mod:`antares.lock`, or dict
    fallbacks when the module is stubbed or unavailable."""

    def pinned_fallback(lock: dict) -> list[str]:
        res = lock.get("result") if isinstance(lock.get("result"), dict) else {}
        seeds = res.get("seeds") if isinstance(res.get("seeds"), dict) else None
        pinned = seeds.get("pinned") if seeds else None
        return [str(p) for p in pinned] if pinned is not None else ["latency_frames"]

    def spread_fallback(lock: dict) -> dict | None:
        res = lock.get("result") if isinstance(lock.get("result"), dict) else {}
        seeds = res.get("seeds") if isinstance(res.get("seeds"), dict) else None
        spread = seeds.get("spread") if seeds else None
        return dict(spread) if isinstance(spread, dict) else None

    try:
        m = importlib.import_module("antares.lock")
    except Exception:  # noqa: BLE001 - stubbed away in tests, or a broken install
        m = None
    pinned = getattr(m, "pinned_params", None) or pinned_fallback
    spread = getattr(m, "param_spread", None) or spread_fallback
    return pinned, spread


def lock_facts(lock: dict | None) -> dict[str, Any]:
    """``{"params", "pinned", "spread", "space"}`` of a lock dict (all None/empty when missing)."""
    if not lock:
        return {"params": None, "pinned": [], "spread": None, "space": None}
    pinned_params, param_spread = _lock_helpers()
    res = lock.get("result") if isinstance(lock.get("result"), dict) else {}
    try:
        pinned = list(pinned_params(lock))
    except Exception:  # noqa: BLE001
        pinned = []
    try:
        spread = param_spread(lock)
    except Exception:  # noqa: BLE001
        spread = None
    return {
        "params": lock.get("params") or res.get("params"),
        "pinned": pinned,
        "spread": spread,
        "space": res.get("space", "shared"),
    }


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


def decide_activation(
    active: LockEntry | None,
    new: LockEntry,
    activate: bool | None,
    max_regression: float,
) -> tuple[bool, str]:
    """The activation rule. Returns ``(activate?, reason)``."""
    if max_regression < 0:
        raise RegistryError("max_regression must be >= 0")
    if active is None or active.name == new.name:
        return True, "first lock of this robot" if active is None else "already the active lock"
    if activate is True:
        return True, "activation forced"
    if activate is False:
        return False, "registered without activation"
    if new.eval_score is None:
        return False, "the new lock has no held-out score to compare"
    if active.eval_score is None:
        return True, f"held-out {new.eval_score:.3f} deg; the active lock {active.name} has no held-out score"
    limit = active.eval_score * (1.0 + max_regression)
    if new.eval_score <= limit:
        return True, (
            f"held-out {new.eval_score:.3f} deg vs {active.eval_score:.3f} for {active.name} "
            f"(within {max_regression:.0%})"
        )
    worse = new.eval_score / active.eval_score - 1.0 if active.eval_score > 0 else float("inf")
    return False, (
        f"held-out {new.eval_score:.3f} deg is {worse:.0%} worse than {active.name}'s "
        f"{active.eval_score:.3f} (limit {max_regression:.0%}); "
        f"force with --activate or `antares robots activate`"
    )


@dataclass
class Registry:
    path: Path
    entries: dict[str, RobotEntry] = field(default_factory=dict)
    version: int = REGISTRY_VERSION

    # ---- file ------------------------------------------------------------

    @property
    def lock_dir(self) -> Path:
        """The lock directory the registry lives in."""
        return Path(self.path).parent

    @classmethod
    def load(cls, path: Path | str) -> "Registry":
        """Read ``path``; a missing file is an empty registry."""
        path = Path(path)
        if not path.is_file():
            return cls(path=path)
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            raise RegistryError(f"{path}: not a valid registry file ({e})") from None
        return cls.from_dict(d, path)

    @classmethod
    def from_dict(cls, d: Any, path: Path | str = Path(REGISTRY_FILENAME)) -> "Registry":
        if not isinstance(d, dict) or not isinstance(d.get("robots", {}), dict):
            raise RegistryError(f"{path}: registry must be an object with a 'robots' map")
        version = int(d.get("version") or REGISTRY_VERSION)
        if version > REGISTRY_VERSION:
            raise RegistryError(f"{path}: registry version {version} is newer than this antares ({REGISTRY_VERSION})")
        entries = {}
        for rid, body in (d.get("robots") or {}).items():
            validate_robot_id(str(rid))
            entries[str(rid)] = RobotEntry.from_dict(body)
        return cls(path=Path(path), entries=entries, version=REGISTRY_VERSION)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "robots": {rid: e.to_dict() for rid, e in sorted(self.entries.items())}}

    def save(self) -> Path:
        """Write atomically: a temp file in the same directory, then rename."""
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        os.replace(tmp, path)
        return path

    # ---- lookup ----------------------------------------------------------

    def robots(self) -> list[str]:
        """Registered robot ids, sorted."""
        return sorted(self.entries)

    def __contains__(self, robot_id: object) -> bool:
        return robot_id in self.entries

    def get(self, robot_id: str) -> RobotEntry:
        rid = robot_id_of(robot_id)
        try:
            return self.entries[rid]
        except KeyError:
            raise UnknownRobot(f"unknown robot {rid!r}; registered: {', '.join(self.robots()) or 'none'}") from None

    def active_lock(self, robot_id: str) -> str:
        """The active lock's name; :class:`UnknownRobot` when the robot has none."""
        rid = robot_id_of(robot_id)
        e = self.get(rid)
        if not e.active:
            raise UnknownRobot(f"robot {rid!r} has no active lock")
        return e.active

    def robot_for(self, lock_name: str) -> str | None:
        """The robot whose lineage holds ``lock_name``, or None."""
        for rid, e in self.entries.items():
            if lock_name in e.names:
                return rid
        return None

    def resolve(self, ref: str, lock_dir: Path | str | None = None) -> str:
        """A lock name or path passes through; ``@<robot>`` becomes the robot's
        active lock name. Raises :class:`UnknownRobot` for an unregistered robot."""
        if not is_robot_ref(ref):
            return ref
        return self.active_lock(robot_id_of(ref))

    # ---- mutation --------------------------------------------------------

    def register(
        self,
        robot_id: str,
        lock: Any,
        *,
        activate: bool | None = None,
        note: str | None = None,
        max_regression: float = DEFAULT_MAX_REGRESSION,
        lock_dir: Path | str | None = None,
        robot_type: str | None = None,
    ) -> RegistrationResult:
        """Add ``lock`` (a RobotLock, a lock dict, a name under the lock dir
        or a ``.lock.json`` path) to ``robot_id``'s lineage and apply the
        activation rule. Does not save; call :meth:`save`."""
        rid = robot_id_of(robot_id)
        lock_d = read_lock(lock, lock_dir if lock_dir is not None else self.lock_dir)
        new = LockEntry.from_lock(lock_d, note=note)
        entry = self.entries.get(rid)
        if entry is None:
            entry = RobotEntry()
            self.entries[rid] = entry
        if robot_type or (entry.robot_type is None and lock_d.get("robot_type")):
            entry.robot_type = robot_type or str(lock_d.get("robot_type"))
        is_new = new.name not in entry.names
        if is_new:
            entry.locks.append(new)
        else:
            old = entry.entry(new.name)
            if note is None:
                new.note = old.note
            entry.locks[entry.index(new.name)] = new
        previous = entry.active
        do_activate, reason = decide_activation(entry.active_entry, new, activate, max_regression)
        if do_activate:
            entry.active = new.name
        entry.updated_at = now_iso()
        return RegistrationResult(
            robot_id=rid,
            lock=new.name,
            version=entry.version(new.name),
            n_versions=len(entry.locks),
            activated=do_activate and previous != new.name,
            active=entry.active or new.name,
            previous_active=previous,
            reason=reason,
            new=is_new,
        )

    def activate(self, robot_id: str, name: str) -> str | None:
        """Make ``name`` (already in the lineage) active; returns the previous active name."""
        e = self.get(robot_id)
        e.index(name)  # RegistryError when not in the lineage
        previous, e.active = e.active, name
        e.updated_at = now_iso()
        return previous

    def rollback(self, robot_id: str) -> tuple[str, str]:
        """Activate the lock before the active one; returns ``(previous, now active)``."""
        e = self.get(robot_id)
        if not e.active or e.active not in e.names:
            raise RegistryError(f"robot {robot_id_of(robot_id)!r} has no active lock to roll back from")
        i = e.index(e.active)
        if i == 0:
            raise RegistryError(f"{e.active} is the oldest lock of robot {robot_id_of(robot_id)!r}; nothing to roll back to")
        previous = e.active
        e.active = e.locks[i - 1].name
        e.updated_at = now_iso()
        return previous, e.active

    def remove(self, robot_id: str) -> RobotEntry:
        """Forget a robot (its lock files stay)."""
        rid = robot_id_of(robot_id)
        e = self.get(rid)
        del self.entries[rid]
        return e

    # ---- views -----------------------------------------------------------

    def lineage(self, robot_id: str, lock_dir: Path | str | None = None) -> list[dict[str, Any]]:
        """Every registered lock of the robot, oldest first, joined with the
        lock file: ``version``, the entry fields, ``is_active``, ``exists``,
        ``params``, ``pinned`` (which parameters that fit can be read as
        measurements) and ``spread`` (the seed spread, when recorded)."""
        rid = robot_id_of(robot_id)
        e = self.get(rid)
        d = Path(lock_dir if lock_dir is not None else self.lock_dir)
        out = []
        for i, entry in enumerate(e.locks):
            p = lock_path(entry.name, d)
            lock = None
            if p.is_file():
                try:
                    lock = json.loads(p.read_text())
                except (OSError, ValueError):
                    lock = None
            row = {"version": i + 1, **entry.to_dict(), "is_active": entry.name == e.active,
                   "exists": lock is not None, **lock_facts(lock)}
            if lock is not None:
                res = lock.get("result") if isinstance(lock.get("result"), dict) else {}
                # prefer what the file says now over what was registered
                row["eval_score"] = _score(res.get("eval_score")) if res else entry.eval_score
                row["baseline_eval_score"] = _score(res.get("baseline_eval_score")) if res else entry.baseline_eval_score
                row["created_at"] = lock.get("created_at") or entry.created_at
                row["dataset"] = lock.get("dataset") or entry.dataset
            out.append(row)
        return out

    def summary(self, robot_id: str) -> dict[str, Any]:
        """One row per robot for tables and ``GET /api/registry``."""
        rid = robot_id_of(robot_id)
        e = self.get(rid)
        active = e.active_entry
        newest = e.locks[-1] if e.locks else None
        return {
            "robot_id": rid,
            "robot_type": e.robot_type,
            "active": e.active,
            "version": e.version(e.active) if active is not None else None,
            "n_versions": len(e.locks),
            "eval_score": active.eval_score if active else None,
            "baseline_eval_score": active.baseline_eval_score if active else None,
            "dataset": active.dataset if active else None,
            "last_calibrated": newest.created_at if newest else None,
            "locks": [x.to_dict() for x in e.locks],
            "created_at": e.created_at,
            "updated_at": e.updated_at,
        }

    def index_by_lock(self) -> dict[str, dict[str, Any]]:
        """lock name -> ``{"robot_id", "is_active", "version", "n_versions"}`` for every registered lock."""
        out: dict[str, dict[str, Any]] = {}
        for rid, e in self.entries.items():
            for i, entry in enumerate(e.locks):
                out[entry.name] = {
                    "robot_id": rid,
                    "is_active": entry.name == e.active,
                    "version": i + 1,
                    "n_versions": len(e.locks),
                }
        return out


def registry_path(lock_dir: Path | str) -> Path:
    """``<lock_dir>/registry.json``."""
    return Path(lock_dir) / REGISTRY_FILENAME


def param_drift(lineage: list[dict[str, Any]]) -> dict[str, Any]:
    """Parameter movement across the versions of a lineage, honest about pinning.

    A parameter is *comparable* when every version that has a lock file pins
    it; the others traded off against each other at equal tracking error and
    are listed as not comparable. Returns ``{"versions": [...], "comparable":
    {param: {"values": [...per version], "change": str}}, "not_comparable":
    {param: [versions that leave it unpinned]}}``.
    """
    present = [row for row in lineage if row.get("exists") and isinstance(row.get("params"), dict)]
    names: list[str] = []
    for row in present:
        for k in row["params"]:
            if k not in names:
                names.append(k)
    comparable: dict[str, Any] = {}
    not_comparable: dict[str, list[int]] = {}
    for k in names:
        unpinned = [row["version"] for row in present if k not in (row.get("pinned") or [])]
        if unpinned:
            not_comparable[k] = unpinned
            continue
        values = [row["params"].get(k) for row in present]
        comparable[k] = {"values": values, "change": _change(k, values)}
    return {"versions": [row["version"] for row in present], "comparable": comparable, "not_comparable": not_comparable}


def _flat(v: Any) -> float | None:
    """A scalar, or the mean of a per-joint list, as float."""
    if isinstance(v, (list, tuple)):
        xs = [_score(x) for x in v]
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    return _score(v)


def _change(param: str, values: list[Any]) -> str:
    xs = [_flat(v) for v in values]
    if len(xs) < 2 or xs[0] is None or xs[-1] is None:
        return "-"
    first, last = xs[0], xs[-1]
    if param == "latency_frames":
        if last == first:
            return "unchanged"
        return f"{last - first:+g} frame{'s' if abs(last - first) != 1 else ''}"
    if first <= 0:
        return f"{last - first:+.4g}"
    ratio = last / first
    return f"x{ratio:.2f}" if abs(ratio - 1) >= 0.005 else "unchanged"


__all__ = [
    "DEFAULT_MAX_REGRESSION",
    "REGISTRY_FILENAME",
    "REGISTRY_VERSION",
    "ROBOT_REF_PREFIX",
    "LockEntry",
    "RegistrationResult",
    "Registry",
    "RegistryError",
    "RobotEntry",
    "UnknownRobot",
    "decide_activation",
    "is_robot_ref",
    "lock_facts",
    "param_drift",
    "read_lock",
    "registry_path",
    "robot_id_of",
    "robot_ref",
    "validate_robot_id",
]
