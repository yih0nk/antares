"""Fleet scan: find every public SO-100/101 dataset on the Hub and calibrate each.

Every community LeRobot dataset was recorded on somebody's own arm, so each
one is a different physical robot. :func:`discover` searches the Hugging Face
Hub, fetches only ``meta/info.json`` for each hit and keeps the datasets the
simulator can replay (SO-100/101 robot type, the six canonical joints, 30 or
50 fps, enough episodes, a tabular part small enough to download). Every
rejected candidate keeps the reason it was dropped.

:func:`scan` then calibrates each accepted dataset in turn, saves one lock
per robot under ``locks/fleet/`` and collects the results in a
:class:`FleetReport` (``locks/fleet/fleet.json``). One bad dataset is
recorded as a failure and the scan carries on; the scan stops when its time
budget is spent.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from antares import __version__
from antares.data import Dataset, Episode, canonical_joint_name
from antares.lock import LOCK_DIR, LOCK_SUFFIX, slug

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

FLEET_LOCK_DIR: Path = LOCK_DIR / "fleet"
FLEET_REPORT_NAME = "fleet.json"

#: Hub search strings and tag filters tried in turn; hits are merged.
SEARCH_TERMS: tuple[str, ...] = ("so100", "so101", "so-100", "so-101", "so_100", "so_101", "so_arm")
TAG_FILTERS: tuple[str, ...] = ("so100", "so101")

#: Always inspected first and exempt from the one-dataset-per-owner rule.
PINNED: tuple[str, ...] = ("lerobot/svla_so101_pickplace", "lerobot/svla_so100_pickplace")

MIN_EPISODES = 15
ALLOWED_FPS: tuple[float, ...] = (30.0, 50.0)
MAX_DATA_BYTES = 200 * 2**20  # tabular part only; videos are never downloaded

# ``so100``/``so101`` in any spelling, plus LeRobot's newer unified
# ``so_follower`` name for the same arm.
_ROBOT_RE = re.compile(r"so[-_ ]?(?:arm)?[-_ ]?10[01]|^so[-_]?follower$")
# Datasets recorded in a simulator are not physical robots, and merges of
# several people's recordings are not one robot.
_SIM_RE = re.compile(
    r"(?:^|[^a-z])(?:synthetic|synth|pybullet|isaac(?:sim|lab)?|mujoco|genesis|maniskill|gazebo)(?:[^a-z]|$)"
    r"|sim(?:[^a-z]|$)"
)
_MERGED_RE = re.compile(r"(?:^|[^a-z])(?:merged?|combined|community|aggregated|converted)(?:[^a-z]|$)")


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One Hub dataset that was inspected, accepted or not."""

    repo_id: str
    accepted: bool
    reason: str = ""  # why it was rejected; empty when accepted
    robot_type: str = ""
    fps: float = 0.0
    n_episodes: int = 0
    n_frames: int = 0
    data_bytes: int = 0
    downloads: int = 0
    codebase_version: str = ""

    @property
    def owner(self) -> str:
        return self.repo_id.split("/", 1)[0] if "/" in self.repo_id else ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Candidate:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def accepted(candidates: Iterable[Candidate]) -> list[Candidate]:
    return [c for c in candidates if c.accepted]


def _names(feature: Any) -> list[str] | None:
    """``features[x]["names"]`` as a flat list; v2 datasets nest it under a key."""
    if not isinstance(feature, dict):
        return None
    names = feature.get("names")
    if isinstance(names, dict):
        if len(names) != 1:
            return None
        names = next(iter(names.values()))
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return None
    return names


def check_info(
    repo_id: str,
    info: dict[str, Any],
    data_bytes: int | None = None,
    downloads: int = 0,
) -> Candidate:
    """Decide from ``meta/info.json`` whether a dataset can join the fleet.

    ``data_bytes`` is the size of the ``data/**/*.parquet`` files when known;
    ``None`` skips the size check (the caller may fill it in afterwards).
    """
    robot_type = str(info.get("robot_type") or "")
    fps = float(info.get("fps") or 0)
    n_episodes = int(info.get("total_episodes") or 0)
    n_frames = int(info.get("total_frames") or 0)
    version = str(info.get("codebase_version") or "")

    def reject(reason: str) -> Candidate:
        return Candidate(
            repo_id=repo_id,
            accepted=False,
            reason=reason,
            robot_type=robot_type,
            fps=fps,
            n_episodes=n_episodes,
            n_frames=n_frames,
            data_bytes=int(data_bytes or 0),
            downloads=downloads,
            codebase_version=version,
        )

    sim = _SIM_RE.search(repo_id.lower())
    if sim:
        return reject(f"looks like a simulation dataset (name contains {sim.group(0).strip('-_ /.')!r})")
    merged = _MERGED_RE.search(repo_id.lower())
    if merged:
        return reject(f"looks like a merge of several datasets (name contains {merged.group(0).strip('-_ /.')!r})")
    if not re.match(r"v[23]\.", version):
        return reject(f"codebase_version {version or 'missing'} (need the v2/v3 parquet layout)")
    if not _ROBOT_RE.search(robot_type.lower()):
        return reject(f"robot_type {robot_type or 'missing'!r} is not an SO-100/101")
    features = info.get("features") or {}
    for key in ("action", "observation.state"):
        names = _names(features.get(key))
        if names is None:
            return reject(f"{key} has no joint names")
        if len(names) != 6:
            return reject(f"{key} has {len(names)} joints, not 6")
        canonical = tuple(canonical_joint_name(n) for n in names)
        if canonical != JOINTS:
            return reject(f"{key} joints {list(canonical)} are not the SO-101 joints")
    if fps not in ALLOWED_FPS:
        return reject(f"fps {fps:g} not in {sorted(int(f) for f in ALLOWED_FPS)}")
    if n_episodes < MIN_EPISODES:
        return reject(f"{n_episodes} episodes < {MIN_EPISODES}")
    if data_bytes is not None:
        if data_bytes <= 0:
            return reject("no data/*.parquet files")
        if data_bytes > MAX_DATA_BYTES:
            return reject(f"tabular data {data_bytes / 2**20:.0f} MB > {MAX_DATA_BYTES // 2**20} MB")
    return Candidate(
        repo_id=repo_id,
        accepted=True,
        robot_type=robot_type,
        fps=fps,
        n_episodes=n_episodes,
        n_frames=n_frames,
        data_bytes=int(data_bytes or 0),
        downloads=downloads,
        codebase_version=version,
    )


# --------------------------------------------------------------------------
# Hub access (injectable for tests)
# --------------------------------------------------------------------------


def hub_search(
    api: Any = None,
    search_terms: Sequence[str] = SEARCH_TERMS,
    tag_filters: Sequence[str] = TAG_FILTERS,
    per_query: int = 300,
) -> list[tuple[str, int]]:
    """``(repo_id, downloads)`` for every public, ungated hit, most downloaded first."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    hits: dict[str, int] = {}
    queries: list[dict[str, Any]] = [{"search": t} for t in search_terms]
    queries += [{"filter": t} for t in tag_filters]
    for q in queries:
        try:
            found = list(api.list_datasets(sort="downloads", limit=per_query, **q))
        except Exception:  # noqa: BLE001 - one failing query must not end discovery
            continue
        for d in found:
            if getattr(d, "gated", False) or getattr(d, "private", False) or getattr(d, "disabled", False):
                continue
            hits[d.id] = max(hits.get(d.id, 0), int(getattr(d, "downloads", 0) or 0))
    return sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))


def fetch_info(repo_id: str) -> dict[str, Any]:
    """Download only ``meta/info.json`` (cached by huggingface_hub) and parse it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
    return json.loads(Path(path).read_text())


def fetch_data_bytes(repo_id: str, api: Any = None) -> int:
    """Total size of ``data/**/*.parquet`` from the repo's file metadata."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    info = api.dataset_info(repo_id, files_metadata=True)
    total = 0
    for s in info.siblings or []:
        name = s.rfilename
        if name.startswith("data/") and name.endswith(".parquet"):
            total += int(s.size or 0)
    return total


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def discover(
    limit: int = 60,
    *,
    max_inspect: int | None = None,
    one_per_owner: bool = True,
    pinned: Sequence[str] = PINNED,
    api: Any = None,
    search: Callable[..., list[tuple[str, int]]] | None = None,
    fetch_info: Callable[[str], dict[str, Any]] | None = None,
    fetch_size: Callable[[str], int] | None = None,
    on_progress: Callable[[Candidate], None] | None = None,
) -> list[Candidate]:
    """Inspect Hub hits most-downloaded first until ``limit`` are accepted.

    Returns every inspected candidate, accepted or rejected with a reason,
    in inspection order. ``max_inspect`` (default ``5 * limit``) bounds the
    number of ``meta/info.json`` fetches. With ``one_per_owner`` a second
    dataset from the same account is rejected, since it is most likely the
    same physical arm recorded twice; ``pinned`` datasets come first and are
    exempt (the two demo datasets are different arms of one organisation).
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if max_inspect is None:
        max_inspect = 5 * limit
    search = search or (lambda: hub_search(api))
    get_info = fetch_info or globals()["fetch_info"]
    get_size = fetch_size or (lambda rid: fetch_data_bytes(rid, api))

    hits = search()
    downloads_of = dict(hits)
    order = [(rid, downloads_of.get(rid, 0)) for rid in pinned]
    order += [(rid, dl) for rid, dl in hits if rid not in set(pinned)]

    out: list[Candidate] = []
    owners: dict[str, str] = {}
    n_accepted = 0
    for repo_id, downloads in order:
        if n_accepted >= limit or len(out) >= max_inspect:
            break
        owner = repo_id.split("/", 1)[0] if "/" in repo_id else ""
        if one_per_owner and owner in owners and repo_id not in pinned:
            cand = Candidate(repo_id, False, f"same owner as {owners[owner]} (likely the same arm)", downloads=downloads)
        else:
            try:
                info = get_info(repo_id)
            except Exception as e:  # noqa: BLE001 - record and move on
                cand = Candidate(repo_id, False, f"no readable meta/info.json ({type(e).__name__})", downloads=downloads)
            else:
                cand = check_info(repo_id, info, None, downloads)
                if cand.accepted:
                    try:
                        size = get_size(repo_id)
                    except Exception as e:  # noqa: BLE001
                        cand = Candidate(repo_id, False, f"could not list files ({type(e).__name__})", downloads=downloads)
                    else:
                        cand = check_info(repo_id, info, size, downloads)
        if cand.accepted:
            n_accepted += 1
            owners.setdefault(owner, repo_id)
        out.append(cand)
        if on_progress is not None:
            on_progress(cand)
    return out


# --------------------------------------------------------------------------
# Scan results
# --------------------------------------------------------------------------


def _json_float(v: Any) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def _parse_float(v: Any) -> float:
    return float("nan") if v is None else float(v)


@dataclass(frozen=True)
class RobotEntry:
    """One calibrated robot in the fleet report."""

    dataset: str
    robot_type: str
    n_episodes: int
    fps: float
    n_train: int
    n_eval: int
    baseline_eval: float  # held-out tracking RMSE (deg) with default params
    calibrated_eval: float  # same, with the fitted params
    improvement: float  # fraction of the held-out gap closed
    params: dict[str, Any]
    lock: str  # lock name under locks/fleet/
    per_joint_before: dict[str, float] = field(default_factory=dict)
    per_joint_after: dict[str, float] = field(default_factory=dict)
    n_evaluations: int = 0
    elapsed_s: float = 0.0
    frames_per_episode: float = 0.0
    data_bytes: int = 0
    downloads: int = 0

    @property
    def latency_ms(self) -> float:
        return 1000.0 * float(self.params.get("latency_frames", 0)) / self.fps if self.fps else float("nan")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("baseline_eval", "calibrated_eval", "improvement"):
            d[k] = _json_float(d[k])
        d["per_joint_before"] = {k: _json_float(v) for k, v in self.per_joint_before.items()}
        d["per_joint_after"] = {k: _json_float(v) for k, v in self.per_joint_after.items()}
        d["latency_ms"] = _json_float(self.latency_ms)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RobotEntry:
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        for k in ("baseline_eval", "calibrated_eval", "improvement"):
            kw[k] = _parse_float(kw.get(k))
        kw["per_joint_before"] = {k: _parse_float(v) for k, v in (kw.get("per_joint_before") or {}).items()}
        kw["per_joint_after"] = {k: _parse_float(v) for k, v in (kw.get("per_joint_after") or {}).items()}
        return cls(**kw)


@dataclass
class FleetReport:
    created_at: str
    antares_version: str
    settings: dict[str, Any]
    robots: list[RobotEntry] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)  # {dataset, error}
    skipped: list[dict[str, str]] = field(default_factory=list)  # {dataset, reason}
    rejected: list[dict[str, str]] = field(default_factory=list)  # {dataset, reason}
    stats: dict[str, Any] = field(default_factory=dict)
    outliers: list[dict[str, Any]] = field(default_factory=list)
    cross: dict[str, Any] | None = None
    elapsed_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        imp = [r.improvement for r in self.robots if math.isfinite(r.improvement)]
        return {
            "n_robots": len(self.robots),
            "n_failed": len(self.failures),
            "n_skipped": len(self.skipped),
            "n_rejected": len(self.rejected),
            "median_improvement": _json_float(np.median(imp)) if imp else None,
            "n_improved": sum(1 for v in imp if v > 0),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "antares_version": self.antares_version,
            "settings": dict(self.settings),
            "summary": self.summary(),
            "robots": [r.to_dict() for r in self.robots],
            "failures": list(self.failures),
            "skipped": list(self.skipped),
            "rejected": list(self.rejected),
            "stats": self.stats,
            "outliers": list(self.outliers),
            "cross": self.cross,
            "elapsed_s": float(self.elapsed_s),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FleetReport:
        return cls(
            created_at=str(d.get("created_at", "")),
            antares_version=str(d.get("antares_version", "")),
            settings=dict(d.get("settings") or {}),
            robots=[RobotEntry.from_dict(r) for r in d.get("robots", [])],
            failures=list(d.get("failures", [])),
            skipped=list(d.get("skipped", [])),
            rejected=list(d.get("rejected", [])),
            stats=dict(d.get("stats") or {}),
            outliers=list(d.get("outliers", [])),
            cross=d.get("cross"),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
        )

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path is not None else FLEET_LOCK_DIR / FLEET_REPORT_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> FleetReport:
        path = Path(path) if path is not None else FLEET_LOCK_DIR / FLEET_REPORT_NAME
        return cls.from_dict(json.loads(path.read_text()))


# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------


def lock_name(repo_id: str) -> str:
    """``"owner/name"`` -> ``"owner__name"`` (both halves slugged)."""
    owner, _, name = repo_id.rpartition("/")
    return f"{slug(owner)}__{slug(name)}" if owner else slug(name)


def plan_split(n_episodes: int, n_train: int, n_eval: int) -> tuple[int, int]:
    """Train/eval counts that always leave a held-out split.

    Datasets with at least ``n_train + n_eval`` episodes use the requested
    counts; smaller ones hold out about a third (at least one episode).
    """
    if n_episodes < 2:
        raise ValueError("need at least 2 episodes to split")
    if n_episodes >= n_train + n_eval:
        return n_train, n_eval
    n_ev = max(1, min(n_eval, n_episodes // 3))
    n_tr = min(n_train, n_episodes - n_ev)
    return n_tr, n_ev


def _usable_episodes(ds: Dataset) -> list[Episode]:
    """Episodes with at least two frames and only finite values."""
    out = []
    for ep in ds.episodes:
        if len(ep) < 2:
            continue
        if not (np.isfinite(ep.state).all() and np.isfinite(ep.action).all()):
            continue
        out.append(ep)
    return out


def load_with_retries(
    repo_id: str,
    load: Callable[[str], Dataset],
    retries: int = 3,
    wait_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Dataset:
    """Call ``load`` again after network errors (``OSError`` family, which
    includes ``requests``/``huggingface_hub`` failures); other errors mean
    the dataset itself is bad and are raised at once."""
    attempt = 0
    while True:
        try:
            return load(repo_id)
        except OSError:
            attempt += 1
            if attempt >= retries:
                raise
            sleep(wait_s * 2 ** (attempt - 1))


def scan(
    candidates: Sequence[Candidate],
    n_train: int = 20,
    n_eval: int = 10,
    max_evaluations: int = 150,
    time_budget_s: float = 2400.0,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    *,
    lock_dir: Path | str = FLEET_LOCK_DIR,
    seed: int = 0,
    load: Callable[[str], Dataset] | None = None,
    simulate: Any = None,
    retries: int = 3,
    retry_wait_s: float = 5.0,
    report_path: Path | str | None = None,
) -> FleetReport:
    """Calibrate every accepted candidate in turn and build the fleet report.

    Each dataset is loaded (tabular files only, with retries on network
    errors), split with :func:`antares.calibrate.split_episodes` at
    ``seed``, calibrated from the default parameters, and saved as a lock
    ``<lock_dir>/<owner>__<name>.lock.json``. Any exception from one dataset
    is recorded under ``failures`` and the scan continues. Candidates not
    reached before ``time_budget_s`` runs out are listed under ``skipped``.
    ``on_progress`` receives dict events (``start``, ``fit``, ``robot``,
    ``failed``, ``skipped``, ``done``). The report is saved to
    ``report_path`` (default ``<lock_dir>/fleet.json``) after every robot.
    """
    from antares.calibrate import calibrate, split_episodes
    from antares.lock import RobotLock
    from antares.sim import SimParams

    if load is None:
        from antares.data import load_dataset

        load = load_dataset
    lock_dir = Path(lock_dir)
    report_path = Path(report_path) if report_path is not None else lock_dir / FLEET_REPORT_NAME
    emit = on_progress or (lambda ev: None)

    t0 = time.monotonic()
    deadline = t0 + float(time_budget_s)
    todo = accepted(candidates)
    report = FleetReport(
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        antares_version=__version__,
        settings={
            "n_train": n_train,
            "n_eval": n_eval,
            "max_evaluations": max_evaluations,
            "time_budget_s": float(time_budget_s),
            "seed": seed,
            "n_candidates": len(todo),
        },
        rejected=[{"dataset": c.repo_id, "reason": c.reason} for c in candidates if not c.accepted],
    )

    for i, cand in enumerate(todo):
        if time.monotonic() > deadline:
            for rest in todo[i:]:
                report.skipped.append({"dataset": rest.repo_id, "reason": "time budget exhausted"})
                emit({"type": "skipped", "dataset": rest.repo_id, "reason": "time budget exhausted"})
            break
        emit({"type": "start", "dataset": cand.repo_id, "index": i, "total": len(todo)})
        t_robot = time.monotonic()
        try:
            ds = load_with_retries(cand.repo_id, load, retries=retries, wait_s=retry_wait_s)
            if tuple(ds.joint_names) != JOINTS:
                raise ValueError(f"joints {list(ds.joint_names)} are not the SO-101 joints")
            episodes = _usable_episodes(ds)
            n_tr, n_ev = plan_split(len(episodes), n_train, n_eval)
            train, eval_ = split_episodes(episodes, n_tr, n_ev, seed=seed)

            def on_fit(ev: Any, _ds: str = cand.repo_id) -> None:
                if ev.evaluation % 25 == 0 or ev.evaluation == max_evaluations:
                    emit(
                        {
                            "type": "fit",
                            "dataset": _ds,
                            "evaluation": ev.evaluation,
                            "best_score": _json_float(ev.best_score),
                            "elapsed_s": ev.elapsed_s,
                        }
                    )

            result = calibrate(
                train,
                eval_,
                simulate=simulate,
                start=SimParams.default(),
                max_evaluations=max_evaluations,
                seed=seed,
                on_progress=on_fit,
            )
            name = lock_name(cand.repo_id)
            lock = RobotLock.create(ds.repo_id, ds.robot_type, train, result, name=name)
            lock.save(lock_dir / f"{name}{LOCK_SUFFIX}")
            entry = RobotEntry(
                dataset=ds.repo_id,
                robot_type=ds.robot_type,
                n_episodes=len(episodes),
                fps=float(ds.fps),
                n_train=len(train),
                n_eval=len(eval_),
                baseline_eval=float(result.baseline_eval_score),
                calibrated_eval=float(result.eval_score),
                improvement=float(result.improvement),
                params=result.params.to_dict(),
                lock=name,
                per_joint_before=dict(result.per_joint_before),
                per_joint_after=dict(result.per_joint_after),
                n_evaluations=int(result.n_evaluations),
                elapsed_s=time.monotonic() - t_robot,
                frames_per_episode=float(np.mean([len(e) for e in episodes])) if episodes else 0.0,
                data_bytes=cand.data_bytes,
                downloads=cand.downloads,
            )
            report.robots.append(entry)
            emit({"type": "robot", "dataset": cand.repo_id, "entry": entry.to_dict()})
        except Exception as e:  # noqa: BLE001 - one bad dataset must not end the scan
            msg = f"{type(e).__name__}: {e}"
            report.failures.append({"dataset": cand.repo_id, "error": msg})
            emit({"type": "failed", "dataset": cand.repo_id, "error": msg})
        report.elapsed_s = time.monotonic() - t0
        report.save(report_path)

    report.elapsed_s = time.monotonic() - t0
    report.save(report_path)
    emit({"type": "done", "summary": report.summary(), "path": str(report_path)})
    return report


__all__ = [
    "ALLOWED_FPS",
    "FLEET_LOCK_DIR",
    "FLEET_REPORT_NAME",
    "JOINTS",
    "MAX_DATA_BYTES",
    "MIN_EPISODES",
    "PINNED",
    "Candidate",
    "FleetReport",
    "RobotEntry",
    "accepted",
    "check_info",
    "discover",
    "fetch_data_bytes",
    "fetch_info",
    "hub_search",
    "load_with_retries",
    "lock_name",
    "plan_split",
    "scan",
]
