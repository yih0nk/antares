"""Check history per robot, across sessions.

:mod:`antares.check` answers one question about one recording: has this arm
drifted from its lock *today*? A shop running arms for weeks also wants the
answer plotted over time. Every ``antares check`` (and ``POST /api/check``)
appends one line to ``locks/<lock>.history.jsonl`` next to the lock, and
:func:`trend` fits a least-squares line through the recorded scores against
days since the first check.

The file is append-only JSON Lines, one :class:`CheckRecord` per line::

    {"at": "2026-10-17T21:44:12Z", "lock": "so101_arm_a", "dataset": "...",
     "verdict": "ok", "passed": true, "score": 1.067, "reference": 1.054,
     "limit": 1.582, "n_episodes": 10, "episodes": [0, 1, ...],
     "worst_joint": "shoulder_pan", "per_joint": {"shoulder_pan": 0.405, ...},
     "antares_version": "0.0.1"}

Insufficient-data checks are recorded too (``verdict`` ``insufficient_data``,
``score`` null): a check that could not run is itself a signal. A corrupt
line is skipped with a warning rather than taking the whole history down.

Not to be confused with :mod:`antares.drift`, which looks for drift *within*
one session over windows of episodes. Pure stdlib + numpy; no MuJoCo.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from antares import __version__
from antares.check import VERDICT_DRIFTED

HISTORY_SUFFIX = ".history.jsonl"
LOCK_SUFFIX = ".lock.json"

_REQUIRED = ("at", "lock", "verdict")
_SECONDS_PER_DAY = 86400.0


def default_lock_dir() -> Path:
    """``$ANTARES_LOCK_DIR`` or ``locks``, the same default the server uses.

    Resolved at call time so the lock module (which pulls in the calibrator)
    is never imported here.
    """
    return Path(os.environ.get("ANTARES_LOCK_DIR", "locks"))


def now_iso() -> str:
    """UTC now as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s: str) -> datetime:
    """An ISO-8601 timestamp as an aware UTC datetime (``Z`` accepted)."""
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _json_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@dataclass(frozen=True)
class CheckRecord:
    """One recorded check: a :class:`~antares.check.CheckResult` plus when it ran."""

    at: str
    lock: str
    dataset: str
    verdict: str
    passed: bool
    score: float | None
    reference: float | None
    limit: float | None
    n_episodes: int
    episodes: list[int] = field(default_factory=list)
    worst_joint: str | None = None
    per_joint: dict[str, float | None] = field(default_factory=dict)
    antares_version: str = __version__

    @classmethod
    def from_result(cls, result: Any, at: str | None = None) -> "CheckRecord":
        """Build from a ``CheckResult`` (or its ``to_dict()``); ``at`` defaults to now."""
        d = result if isinstance(result, dict) else result.to_dict()
        per_joint = {
            str(j): _json_float(v["score"] if isinstance(v, dict) else v)
            for j, v in (d.get("per_joint") or {}).items()
        }
        return cls(
            at=at if at is not None else now_iso(),
            lock=str(d.get("lock")),
            dataset=str(d.get("dataset")),
            verdict=str(d.get("verdict")),
            passed=bool(d.get("passed")),
            score=_json_float(d.get("score")),
            reference=_json_float(d.get("reference")),
            limit=_json_float(d.get("limit")),
            n_episodes=int(d.get("n_episodes") or 0),
            episodes=[int(i) for i in d.get("episodes") or []],
            worst_joint=d.get("worst_joint"),
            per_joint=per_joint,
            antares_version=__version__,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "CheckRecord":
        """Parse one stored line; raises ``ValueError`` on a malformed one."""
        if not isinstance(d, dict):
            raise ValueError("record is not an object")
        missing = [k for k in _REQUIRED if k not in d]
        if missing:
            raise ValueError(f"record is missing {missing}")
        parse_time(d["at"])  # validate; keep the stored string
        return cls(
            at=str(d["at"]),
            lock=str(d["lock"]),
            dataset=str(d.get("dataset", "")),
            verdict=str(d["verdict"]),
            passed=bool(d.get("passed", False)),
            score=_json_float(d.get("score")),
            reference=_json_float(d.get("reference")),
            limit=_json_float(d.get("limit")),
            n_episodes=int(d.get("n_episodes") or 0),
            episodes=[int(i) for i in d.get("episodes") or []],
            worst_joint=d.get("worst_joint"),
            per_joint={str(j): _json_float(v) for j, v in (d.get("per_joint") or {}).items()},
            antares_version=str(d.get("antares_version", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def time(self) -> datetime:
        return parse_time(self.at)

    @property
    def date(self) -> str:
        """``YYYY-MM-DD`` of ``at``."""
        return self.at[:10]


def history_path(lock_name: str, lock_dir: Path | str | None = None) -> Path:
    """``<lock_dir>/<lock_name>.history.jsonl``."""
    name = str(lock_name)
    if not name or "/" in name or os.sep in name or name.startswith(".") or ".." in name:
        raise ValueError(f"invalid lock name {lock_name!r}")
    return Path(lock_dir if lock_dir is not None else default_lock_dir()) / f"{name}{HISTORY_SUFFIX}"


def append(
    result: Any,
    lock_dir: Path | str | None = None,
    *,
    at: str | None = None,
    name: str | None = None,
) -> Path:
    """Append ``result`` (a ``CheckResult``) as one line and return the file.

    ``name`` overrides the file's lock name (default ``result.lock``), for a
    lock loaded from a file whose name differs from the name inside it.
    """
    rec = CheckRecord.from_result(result, at=at)
    path = history_path(name if name is not None else rec.lock, lock_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec.to_dict(), separators=(",", ":")) + "\n")
    return path


def load(lock_name: str, lock_dir: Path | str | None = None) -> list[CheckRecord]:
    """Every readable record for ``lock_name``, oldest first.

    A missing file is an empty history. A line that is not JSON or not a
    record is skipped with a ``UserWarning`` naming the line number.
    """
    path = history_path(lock_name, lock_dir)
    if not path.is_file():
        return []
    out: list[CheckRecord] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(CheckRecord.from_dict(json.loads(line)))
            except (ValueError, TypeError, KeyError) as e:
                warnings.warn(f"{path}:{lineno}: skipping corrupt history line ({e})", stacklevel=2)
    out.sort(key=lambda r: r.time)
    return out


def trend(records: list[CheckRecord]) -> dict[str, Any]:
    """Least-squares slope of ``score`` (deg) against days since the first entry.

    Entries with a null score (insufficient data) count towards ``n`` and
    the dates but not the fit. ``slope_deg_per_day`` and ``r2`` are None
    with fewer than two scored entries or when all of them share one
    timestamp; ``r2`` is also None when the scores do not vary.
    """
    n = len(records)
    out: dict[str, Any] = {
        "n": n,
        "first": records[0].at if n else None,
        "last": records[-1].at if n else None,
        "slope_deg_per_day": None,
        "r2": None,
        "latest_verdict": records[-1].verdict if n else None,
        "n_drifted": sum(1 for r in records if r.verdict == VERDICT_DRIFTED),
    }
    if not n:
        return out
    t0 = records[0].time
    scored = [r for r in records if r.score is not None]
    if len(scored) < 2:
        return out
    x = np.array([(r.time - t0).total_seconds() / _SECONDS_PER_DAY for r in scored], dtype=float)
    y = np.array([r.score for r in scored], dtype=float)
    xm, ym = x.mean(), y.mean()
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0:
        return out
    slope = float(np.sum((x - xm) * (y - ym)) / sxx)
    pred = ym + slope * (x - xm)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - ym) ** 2))
    out["slope_deg_per_day"] = slope
    out["r2"] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    return out


def last_drifted(records: list[CheckRecord]) -> CheckRecord | None:
    """The newest record with verdict ``drifted``, if any."""
    for r in reversed(records):
        if r.verdict == VERDICT_DRIFTED:
            return r
    return None


def summary_line(records: list[CheckRecord]) -> str:
    """One line for a CLI log, e.g.
    ``history: 7 checks since 2026-09-30, +0.021 deg/day (r² 0.41), last drifted 2026-10-10``.

    The slope clause is omitted with fewer than two scored entries.
    """
    n = len(records)
    if n == 0:
        return "history: no checks recorded"
    t = trend(records)
    parts = [f"history: {n} check{'s' if n != 1 else ''} since {records[0].date}"]
    if t["slope_deg_per_day"] is not None:
        clause = f"{t['slope_deg_per_day']:+.3f} deg/day"
        if t["r2"] is not None:
            clause += f" (r² {t['r2']:.2f})"
        parts.append(clause)
    worst = last_drifted(records)
    if worst is not None:
        parts.append(f"last drifted {worst.date}")
    return ", ".join(parts)


def report(lock_name: str, lock_dir: Path | str | None = None, *, last: int | None = None) -> dict[str, Any]:
    """The ``GET /api/history/{lock}`` body: ``{"lock", "entries", "trend"}``.

    ``trend`` is over every record; ``last`` keeps only the newest N in
    ``entries``.
    """
    records = load(lock_name, lock_dir)
    shown = records[-last:] if last is not None and last > 0 else records
    return {"lock": lock_name, "entries": [r.to_dict() for r in shown], "trend": trend(records)}


def lock_names(lock_dir: Path | str | None = None) -> list[str]:
    """Names of the ``*.lock.json`` files in ``lock_dir``, sorted."""
    d = Path(lock_dir if lock_dir is not None else default_lock_dir())
    if not d.is_dir():
        return []
    return sorted(p.name[: -len(LOCK_SUFFIX)] for p in d.glob(f"*{LOCK_SUFFIX}"))


def summary_all(lock_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """For every lock in ``lock_dir``: ``{"lock", "trend", "latest"}`` (latest entry or None)."""
    out = []
    for name in lock_names(lock_dir):
        records = load(name, lock_dir)
        out.append({
            "lock": name,
            "trend": trend(records),
            "latest": records[-1].to_dict() if records else None,
        })
    return out


__all__ = [
    "HISTORY_SUFFIX",
    "CheckRecord",
    "append",
    "default_lock_dir",
    "history_path",
    "last_drifted",
    "load",
    "lock_names",
    "now_iso",
    "parse_time",
    "report",
    "summary_all",
    "summary_line",
    "trend",
]
