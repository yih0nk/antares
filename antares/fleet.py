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
from antares.data import Dataset, Episode
from antares.lock import LOCK_DIR, LOCK_SUFFIX, slug
from antares.validate import ALLOWED_FPS, JOINTS, merged_name_match, sim_name_match, validate_dataset_info

FLEET_LOCK_DIR: Path = LOCK_DIR / "fleet"
FLEET_REPORT_NAME = "fleet.json"

#: Hub search strings and tag filters tried in turn; hits are merged.
SEARCH_TERMS: tuple[str, ...] = ("so100", "so101", "so-100", "so-101", "so_100", "so_101", "so_arm")
TAG_FILTERS: tuple[str, ...] = ("so100", "so101")

#: Always inspected first and exempt from the one-dataset-per-owner rule.
PINNED: tuple[str, ...] = ("lerobot/svla_so101_pickplace", "lerobot/svla_so100_pickplace")

#: The fleet wants a proper held-out split, so it asks for more episodes than an upload.
MIN_EPISODES = 15
MAX_DATA_BYTES = 200 * 2**20  # tabular part only; videos are never downloaded
MAX_DATA_FILES = 200  # v2 datasets store one parquet per episode; thousands of tiny downloads are too slow
#: A recording the *uncalibrated* sim already tracks this closely (held-out
#: RMSE, degrees) is not a physical arm: no real SO-100/101 in the fleet
#: comes within 1.2 deg of the MJCF defaults. Such logs are most likely
#: recorded in a simulator under a name that does not say so.
MIN_BASELINE_DEG = 0.5


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
    data_files: int = 0
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


def check_info(
    repo_id: str,
    info: dict[str, Any],
    data_bytes: int | None = None,
    downloads: int = 0,
    data_files: int | None = None,
) -> Candidate:
    """Decide from ``meta/info.json`` whether a dataset can join the fleet.

    The dataset checks are :func:`antares.validate.validate_dataset_info`
    with the fleet's stricter episode minimum; on top of that the fleet
    rejects what the validator only warns about (a name that looks like a
    simulator or a merge, a missing codebase version), because an unattended
    scan cannot ask. ``data_bytes``/``data_files`` are the total size and
    count of the ``data/**/*.parquet`` files when known; ``None`` skips
    those checks (the caller fills them in afterwards).
    """
    verdict = validate_dataset_info(info, repo_id, min_episodes=MIN_EPISODES, recommended_episodes=MIN_EPISODES)
    robot_type = verdict.robot_type
    fps = verdict.fps
    n_episodes = verdict.n_episodes
    n_frames = verdict.n_frames
    version = verdict.codebase_version

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
            data_files=int(data_files or 0),
            downloads=downloads,
            codebase_version=version,
        )

    sim = sim_name_match(repo_id)
    if sim:
        return reject(f"looks like a simulation dataset (name contains {sim!r})")
    merged = merged_name_match(repo_id)
    if merged:
        return reject(f"looks like a merge of several datasets (name contains {merged!r})")
    if not version:
        return reject("codebase_version missing (need the v2/v3 parquet layout)")
    if not verdict.ok:
        return reject(verdict.reasons[0])
    if data_bytes is not None:
        if data_bytes <= 0:
            return reject("no data/*.parquet files")
        if data_bytes > MAX_DATA_BYTES:
            return reject(f"tabular data {data_bytes / 2**20:.0f} MB > {MAX_DATA_BYTES // 2**20} MB")
    if data_files is not None and data_files > MAX_DATA_FILES:
        return reject(f"{data_files} parquet files > {MAX_DATA_FILES} (one file per episode; too slow to fetch)")
    return Candidate(
        repo_id=repo_id,
        accepted=True,
        robot_type=robot_type,
        fps=fps,
        n_episodes=n_episodes,
        n_frames=n_frames,
        data_bytes=int(data_bytes or 0),
        data_files=int(data_files or 0),
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


def fetch_data_files(repo_id: str, api: Any = None) -> tuple[int, int]:
    """``(total bytes, file count)`` of ``data/**/*.parquet`` from the repo's file metadata."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    info = api.dataset_info(repo_id, files_metadata=True)
    total, count = 0, 0
    for s in info.siblings or []:
        name = s.rfilename
        if name.startswith("data/") and name.endswith(".parquet"):
            total += int(s.size or 0)
            count += 1
    return total, count


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
    fetch_size: Callable[[str], tuple[int, int]] | None = None,
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
    get_size = fetch_size or (lambda rid: fetch_data_files(rid, api))

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
                        size, n_files = get_size(repo_id)
                    except Exception as e:  # noqa: BLE001
                        cand = Candidate(repo_id, False, f"could not list files ({type(e).__name__})", downloads=downloads)
                    else:
                        cand = check_info(repo_id, info, size, downloads, n_files)
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
    cross_n: int = 6,
    cross_fn: Callable[..., dict[str, Any]] | None = None,
) -> FleetReport:
    """Calibrate every accepted candidate in turn and build the fleet report.

    Each dataset is loaded (tabular files only, with retries on network
    errors), split with :func:`antares.calibrate.split_episodes` at
    ``seed``, calibrated from the default parameters, and saved as a lock
    ``<lock_dir>/<owner>__<name>.lock.json``. Any exception from one dataset
    is recorded under ``failures`` and the scan continues. Candidates not
    reached before ``time_budget_s`` runs out are listed under ``skipped``.
    A dataset the default parameters already track to under
    :data:`MIN_BASELINE_DEG` is moved to ``rejected`` and gets no lock. ``on_progress`` receives dict events
    (``start``, ``fit``, ``robot``, ``rejected``, ``failed``, ``skipped``,
    ``done``). The report is saved to
    ``report_path`` (default ``<lock_dir>/fleet.json``) after every robot.

    Afterwards :func:`finalize` adds the fleet statistics, outliers and a
    partial cross matrix over up to ``cross_n`` robots (0 disables it).
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
            if result.baseline_eval_score < MIN_BASELINE_DEG:
                reason = (
                    f"the default sim already tracks it to {result.baseline_eval_score:.2f} deg "
                    f"(< {MIN_BASELINE_DEG}); likely recorded in a simulator, not a physical arm"
                )
                report.rejected.append({"dataset": cand.repo_id, "reason": reason})
                emit({"type": "rejected", "dataset": cand.repo_id, "reason": reason})
                report.elapsed_s = time.monotonic() - t0
                report.save(report_path)
                continue
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

    try:
        finalize(report, lock_dir=lock_dir, cross_n=cross_n, load=load, cross_fn=cross_fn, retries=retries)
    except Exception as e:  # noqa: BLE001 - keep the per-robot results even if the summary fails
        report.failures.append({"dataset": "<cross matrix>", "error": f"{type(e).__name__}: {e}"})
        emit({"type": "failed", "dataset": "<cross matrix>", "error": f"{type(e).__name__}: {e}"})
    report.elapsed_s = time.monotonic() - t0
    report.save(report_path)
    emit({"type": "done", "summary": report.summary(), "path": str(report_path)})
    return report


# --------------------------------------------------------------------------
# Fleet statistics
# --------------------------------------------------------------------------

#: Parameters searched on a log scale; their spread is measured in log10.
LOG_PARAMS: tuple[str, ...] = ("kp", "kv", "damping", "armature", "frictionloss")
#: Per-robot scores that are summarised alongside the parameters.
SCORE_FIELDS: tuple[str, ...] = ("baseline_eval", "calibrated_eval", "improvement")
STAT_PARAMS: tuple[str, ...] = LOG_PARAMS + ("latency_frames", "latency_ms") + SCORE_FIELDS
_LOG_SCALE = frozenset(LOG_PARAMS) | {"baseline_eval", "calibrated_eval"}
OUTLIER_Z = 2.5


def _value_of(r: RobotEntry, param: str) -> float | None:
    if param == "latency_ms":
        v = r.latency_ms
    elif param in SCORE_FIELDS:
        v = getattr(r, param)
    else:
        v = r.params.get(param)
    return None if v is None else float(v)


def _param_values(robots: Sequence[RobotEntry], param: str) -> list[tuple[str, float]]:
    out = []
    for r in robots:
        v = _value_of(r, param)
        if v is None:
            continue
        if math.isfinite(v) and (param not in _LOG_SCALE or v > 0):
            out.append((r.dataset, v))
    return out


def robust_z(values: np.ndarray) -> np.ndarray:
    """``(x - median) / (1.4826 * MAD)``.

    When the MAD is zero (most robots share one value, typical for the
    integer latency) the scale falls back to the IQR, then to the standard
    deviation; with no spread at all every z is zero.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale <= 0:
        q25, q75 = np.percentile(x, [25, 75])
        scale = (q75 - q25) / 1.349
    if scale <= 0:
        scale = float(x.std())
    if scale <= 0:
        return np.zeros_like(x)
    return (x - med) / scale


def fleet_stats(robots: Sequence[RobotEntry], params: Sequence[str] = STAT_PARAMS) -> dict[str, dict[str, Any]]:
    """Per-parameter median, quartiles, IQR and a robust z-score per robot.

    Log-scale parameters are analysed in log10 (so ``z`` is in decades of
    MAD) but the quantiles are reported in natural units.
    """
    stats: dict[str, dict[str, Any]] = {}
    for param in params:
        pairs = _param_values(robots, param)
        if not pairs:
            continue
        names = [n for n, _ in pairs]
        x = np.array([v for _, v in pairs], dtype=np.float64)
        log = param in _LOG_SCALE
        q25, med, q75 = np.percentile(x, [25, 50, 75])
        z = robust_z(np.log10(x) if log else x)
        stats[param] = {
            "scale": "log" if log else "linear",
            "n": int(x.size),
            "median": _json_float(med),
            "q25": _json_float(q25),
            "q75": _json_float(q75),
            "iqr": _json_float(q75 - q25),
            "min": _json_float(x.min()),
            "max": _json_float(x.max()),
            "z": {n: _json_float(v) for n, v in zip(names, z)},
        }
    return stats


def find_outliers(
    robots: Sequence[RobotEntry],
    stats: dict[str, dict[str, Any]],
    threshold: float = OUTLIER_Z,
    params: Sequence[str] = LOG_PARAMS + ("latency_frames",) + SCORE_FIELDS,
) -> list[dict[str, Any]]:
    """Robots whose fitted value (or score) sits more than ``threshold``
    robust sigmas from the fleet median, most extreme first."""
    by_name = {r.dataset: r for r in robots}
    out = []
    for param in params:
        st = stats.get(param)
        if not st:
            continue
        med = st["median"]
        for name, z in st["z"].items():
            if z is None or abs(z) < threshold:
                continue
            r = by_name.get(name)
            value = _value_of(r, param) if r is not None else None
            if value is None or med is None:
                continue
            if param == "latency_frames":
                ms = f" ({1000 * value / r.fps:.0f} ms)" if r.fps else ""
                note = f"latency {int(value)} frames{ms} vs fleet median {med:g}"
            elif param == "improvement":
                note = f"{value:.0%} of the held-out gap closed vs fleet median {med:.0%}"
            elif param in SCORE_FIELDS:
                note = f"{param.replace('_', ' ')} {value:.3g} deg vs fleet median {med:.3g} deg"
            else:
                ratio = value / med if med else float("inf")
                note = f"{param} {value:.3g} is {ratio:.2g}x the fleet median {med:.3g}"
            out.append({"dataset": name, "param": param, "z": round(float(z), 2), "value": value, "median": med, "note": note})
    out.sort(key=lambda o: -abs(o["z"]))
    return out


# --------------------------------------------------------------------------
# Partial cross matrix
# --------------------------------------------------------------------------


def pick_cross_robots(
    robots: Sequence[RobotEntry],
    n: int = 6,
    prefer: Sequence[str] = PINNED,
    n_train: int = 20,
    n_eval: int = 10,
    exclude: Iterable[str] = (),
) -> list[RobotEntry]:
    """Up to ``n`` robots for the cross matrix: the preferred (demo) ones
    first, then others in scan order. Only robots with enough episodes for
    the full ``n_train + n_eval`` split qualify, so every row uses the same
    held-out split the lock was scored on. ``exclude`` names robots to skip
    (the scan passes its score outliers, so one arm the sim cannot explain
    does not dominate the matrix)."""
    skip = set(exclude)
    ok = [
        r
        for r in robots
        if r.n_episodes >= n_train + n_eval and math.isfinite(r.calibrated_eval) and r.dataset not in skip
    ]
    chosen = [r for p in prefer for r in ok if r.dataset == p]
    for r in ok:
        if len(chosen) >= n:
            break
        if r not in chosen:
            chosen.append(r)
    return chosen[:n]


def cross_summary(matrix: dict[str, Any]) -> dict[str, Any]:
    """Mean of each robot on its own lock (diagonal) vs on the others' locks
    (off-diagonal) vs the default parameters (last column)."""
    scores = np.asarray(matrix["scores"], dtype=np.float64)
    k = len(matrix["rows"])
    if scores.shape != (k, k + 1) or k == 0:
        raise ValueError("cross matrix must be k rows by k locks + default")
    locks = scores[:, :k]
    default = scores[:, k]
    off_mask = ~np.eye(k, dtype=bool)
    diag = np.diag(locks)
    off = locks[off_mask]
    ratio = lambda a: _json_float(a / diag.mean()) if diag.mean() > 0 else None  # noqa: E731
    safe_diag = np.where(diag > 0, diag, np.nan)
    row_off = np.array([locks[i, off_mask[i]].mean() for i in range(k)]) if k > 1 else np.full(k, np.nan)
    return {
        "n": k,
        "mean_diagonal": _json_float(diag.mean()),
        "mean_off_diagonal": _json_float(off.mean()) if off.size else None,
        "mean_default": _json_float(default.mean()),
        "off_over_diagonal": ratio(off.mean()) if off.size else None,
        "default_over_diagonal": ratio(default.mean()),
        # per-robot ratios, then the median: robust to one robot with a huge gap
        "median_row_off_over_diagonal": _json_float(np.nanmedian(row_off / safe_diag)) if off.size else None,
        "median_row_default_over_diagonal": _json_float(np.nanmedian(default / safe_diag)),
        # share of (robot, other lock) pairs where the other robot's lock still beats the default
        "other_lock_beats_default": _json_float(np.mean((locks < default[:, None])[off_mask])) if off.size else None,
        # robots whose own lock is the best of all locks on their eval split
        "own_lock_is_best": int(np.sum(np.argmin(locks, axis=1) == np.arange(k))),
    }


def default_cross_fn(
    datasets: Sequence[Dataset],
    locks: Sequence[str],
    n_train: int,
    n_eval: int,
    seed: int,
    lock_dir: Path,
) -> dict[str, Any]:
    """Score with :func:`antares.server.cross_matrix` (the shared MuJoCo arm)."""
    from antares import server

    for ds in datasets:
        server.put_dataset(ds)
    return server.cross_matrix(
        [ds.repo_id for ds in datasets], list(locks), n_train=n_train, n_eval=n_eval, seed=seed, lock_dir=lock_dir
    )


def fleet_cross(
    robots: Sequence[RobotEntry],
    *,
    lock_dir: Path | str = FLEET_LOCK_DIR,
    n: int = 6,
    prefer: Sequence[str] = PINNED,
    n_train: int = 20,
    n_eval: int = 10,
    seed: int = 0,
    load: Callable[[str], Dataset] | None = None,
    cross_fn: Callable[..., dict[str, Any]] | None = None,
    retries: int = 3,
    exclude: Iterable[str] = (),
) -> dict[str, Any] | None:
    """Score each chosen robot's lock on every chosen robot's eval split."""
    chosen = pick_cross_robots(robots, n=n, prefer=prefer, n_train=n_train, n_eval=n_eval, exclude=exclude)
    if len(chosen) < 2:
        return None
    if load is None:
        from antares.data import load_dataset

        load = load_dataset
    datasets = [load_with_retries(r.dataset, load, retries=retries) for r in chosen]
    matrix = (cross_fn or default_cross_fn)(datasets, [r.lock for r in chosen], n_train, n_eval, seed, Path(lock_dir))
    out = dict(matrix)
    out["locks"] = [r.lock for r in chosen]
    out["summary"] = cross_summary(matrix)
    return out


def finalize(
    report: FleetReport,
    *,
    lock_dir: Path | str = FLEET_LOCK_DIR,
    cross_n: int = 6,
    cross_prefer: Sequence[str] = PINNED,
    load: Callable[[str], Dataset] | None = None,
    cross_fn: Callable[..., dict[str, Any]] | None = None,
    retries: int = 3,
) -> FleetReport:
    """Fill ``stats``, ``outliers`` and (``cross_n`` > 1) ``cross`` in place.

    Robots flagged as outliers on their scores (an arm the simulator cannot
    explain) are kept out of the cross matrix unless they are preferred."""
    report.stats = fleet_stats(report.robots)
    report.outliers = find_outliers(report.robots, report.stats)
    if cross_n > 1 and report.robots:
        s = report.settings
        score_outliers = {o["dataset"] for o in report.outliers if o["param"] in SCORE_FIELDS}
        report.cross = fleet_cross(
            report.robots,
            lock_dir=lock_dir,
            n=cross_n,
            prefer=cross_prefer,
            n_train=int(s.get("n_train", 20)),
            n_eval=int(s.get("n_eval", 10)),
            seed=int(s.get("seed", 0)),
            load=load,
            cross_fn=cross_fn,
            retries=retries,
            exclude=score_outliers - set(cross_prefer),
        )
    return report


__all__ = [
    "ALLOWED_FPS",
    "FLEET_LOCK_DIR",
    "FLEET_REPORT_NAME",
    "JOINTS",
    "MAX_DATA_BYTES",
    "MAX_DATA_FILES",
    "MIN_BASELINE_DEG",
    "MIN_EPISODES",
    "PINNED",
    "Candidate",
    "FleetReport",
    "RobotEntry",
    "accepted",
    "check_info",
    "discover",
    "fetch_data_files",
    "fetch_info",
    "hub_search",
    "load_with_retries",
    "lock_name",
    "plan_split",
    "scan",
    "LOG_PARAMS",
    "SCORE_FIELDS",
    "STAT_PARAMS",
    "OUTLIER_Z",
    "robust_z",
    "fleet_stats",
    "find_outliers",
    "pick_cross_robots",
    "cross_summary",
    "fleet_cross",
    "finalize",
]
