"""HTTP API behind the web UI. Run with ``antares serve``.

Dataset ids contain ``/`` so they are always passed as query parameters.
Heavy modules (``antares.sim``, ``antares.gap``, ``antares.calibrate``,
``antares.lock``) are imported lazily inside the functions that need them,
so the app imports quickly and tests can stub them through ``sys.modules``.

The pure helpers (``replay_episode``, ``run_calibration``, ``run_drift``,
``run_dr``, ``run_check``, ``cross_matrix``, lock / drift-scan / DR-range
file I/O) are shared with the CLI.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import antares
from antares.data import Dataset, Episode

# Datasets the demo offers out of the box.
DEMO_DATASETS = [
    "lerobot/svla_so101_pickplace",
    "lerobot/svla_so100_pickplace",
]

_ROOT = Path(__file__).resolve().parent.parent
WEB_DIST = Path(os.environ.get("ANTARES_WEB_DIST", _ROOT / "web" / "dist"))

# Where RobotLock JSON files live (one ``<name>.lock.json`` per lock, the same
# layout ``antares.lock.list_locks`` reads). Tests monkeypatch LOCK_DIR.
LOCK_DIR: Path = Path(os.environ.get("ANTARES_LOCK_DIR", "locks"))
LOCK_SUFFIX = ".lock.json"
try:  # prefer the lock module's defaults when it exists
    from antares import lock as _lock_mod  # type: ignore

    LOCK_SUFFIX = getattr(_lock_mod, "LOCK_SUFFIX", LOCK_SUFFIX)
    if "ANTARES_LOCK_DIR" not in os.environ:
        LOCK_DIR = Path(getattr(_lock_mod, "LOCK_DIR", LOCK_DIR))
    del _lock_mod
except Exception:  # module not written yet
    pass

# Domain-randomization ranges live next to the locks as ``<name>.dr.json``.
DR_SUFFIX = ".dr.json"

# Override for the STL directory (tests). None means MODEL_PATH.parent / "assets".
MESH_DIR: Path | None = None

ROUND = 4


# --------------------------------------------------------------------------
# Dataset cache
# --------------------------------------------------------------------------

_DATASETS: dict[str, Dataset] = {}
_DATASETS_LOCK = threading.Lock()


def get_dataset(repo_id: str) -> Dataset:
    """Return a cached Dataset, loading it (slow first time) if needed.

    ``repo_id`` is a Hub dataset id, or the path of a LeRobot dataset
    directory on disk (a fresh local recording), which is read in place.
    """
    with _DATASETS_LOCK:
        ds = _DATASETS.get(repo_id)
        if ds is not None:
            return ds
    from antares.data import load_dataset, read_local

    try:
        if os.path.isdir(repo_id):
            ds = read_local(repo_id, repo_id=repo_id)
        else:
            ds = load_dataset(repo_id)
    except Exception as e:  # noqa: BLE001 - surface loader errors to the client
        raise HTTPException(status_code=502, detail=f"could not load {repo_id}: {e}")
    with _DATASETS_LOCK:
        _DATASETS.setdefault(repo_id, ds)
        return _DATASETS[repo_id]


def dataset_summary(repo_id: str) -> dict:
    with _DATASETS_LOCK:
        ds = _DATASETS.get(repo_id)
    out: dict[str, Any] = {"id": repo_id, "loaded": ds is not None}
    if ds is not None:
        out.update(
            robot_type=ds.robot_type,
            fps=ds.fps,
            n_episodes=len(ds.episodes),
            joints=list(ds.joint_names),
        )
    return out


def find_episode(ds: Dataset, index: int) -> Episode:
    for ep in ds.episodes:
        if ep.index == index:
            return ep
    raise HTTPException(status_code=404, detail=f"episode {index} not in {ds.repo_id}")


# --------------------------------------------------------------------------
# Sim access: one Arm per process, guarded by a lock (Arm mutates state)
# --------------------------------------------------------------------------

_arm = None
_ARM_LOCK = threading.Lock()


def get_arm():
    global _arm
    if _arm is None:
        from antares.sim import Arm

        _arm = Arm()
    return _arm


def default_params():
    from antares.sim import SimParams

    return SimParams.default()


def params_from_dict(d: dict):
    from antares.sim import SimParams

    return SimParams.from_dict(d)


def mesh_dir() -> Path:
    if MESH_DIR is not None:
        return Path(MESH_DIR)
    from antares.sim import MODEL_PATH

    return Path(MODEL_PATH).parent / "assets"


def _rounded(a: np.ndarray) -> list:
    return np.round(np.asarray(a, dtype=np.float64), ROUND).tolist()


def replay_episode(ds: Dataset, episode: int, params=None) -> dict:
    """Real FK + sim replay for one episode. Angles in degrees, poses (T,nbody,7)."""
    from antares.gap import tracking_error
    from antares.sim import JOINTS

    ep = find_episode(ds, episode)
    if params is None:
        params = default_params()
    arm = get_arm()
    with _ARM_LOCK:
        arm.apply(params)
        real_q = np.asarray(arm.real_q(ep))
        sim_q = np.asarray(arm.replay(ep))
        real_poses = np.asarray(arm.fk(real_q))
        sim_poses = np.asarray(arm.fk(sim_q))
        body_names = list(arm.body_names)
    gap = tracking_error(real_q, sim_q, joint_names=JOINTS)
    return {
        "dataset": ds.repo_id,
        "episode": ep.index,
        "fps": ep.fps,
        "joints": list(JOINTS),
        "body_names": body_names,
        "params": params.to_dict(),
        "real_q": _rounded(np.degrees(real_q)),
        "sim_q": _rounded(np.degrees(sim_q)),
        "real_poses": _rounded(real_poses),
        "sim_poses": _rounded(sim_poses),
        "gap": gap.to_dict(),
    }


def score_episodes(params, episodes) -> Any:
    """Aggregate GapReport of ``params`` over ``episodes`` using the shared Arm."""
    from antares.gap import aggregate, tracking_error
    from antares.sim import JOINTS

    arm = get_arm()
    reports = []
    with _ARM_LOCK:
        arm.apply(params)
        for ep in episodes:
            real_q = np.asarray(arm.real_q(ep))
            sim_q = np.asarray(arm.replay(ep))
            reports.append(tracking_error(real_q, sim_q, joint_names=JOINTS))
    return aggregate(reports)


# --------------------------------------------------------------------------
# Locks (JSON files in LOCK_DIR, one per name)
# --------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _lock_path(name: str) -> Path:
    if not _NAME_RE.match(name) or ".." in name:
        raise HTTPException(status_code=400, detail="invalid lock name")
    return Path(LOCK_DIR) / f"{name}{LOCK_SUFFIX}"


def list_lock_dicts() -> list[dict]:
    d = Path(LOCK_DIR)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob(f"*{LOCK_SUFFIX}")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda l: str(l.get("created_at", "")), reverse=True)
    return out


def load_lock_dict(name: str) -> dict:
    p = _lock_path(name)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"no lock named {name!r}")
    return json.loads(p.read_text())


def save_lock(lock) -> Path:
    """Write a RobotLock to ``LOCK_DIR/<name>.lock.json``."""
    return lock.save(_lock_path(str(lock.name)))


def load_lock(name: str):
    """``load_lock_dict`` parsed into a RobotLock."""
    from antares.lock import RobotLock

    try:
        return RobotLock.from_dict(load_lock_dict(name))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"lock {name!r} is malformed: {e}")


def export_lock_xml(name: str) -> bytes:
    """The calibrated MJCF for ``name`` as bytes (meshdir = package assets)."""
    from antares.export import export_mjcf

    lock = load_lock(name)
    with tempfile.TemporaryDirectory() as tmp:
        return export_mjcf(lock, Path(tmp) / f"{lock.name}.xml", bundle_assets=False).read_bytes()


def export_lock_zip(name: str) -> bytes:
    """``<name>/`` with the MJCF, sidecar, scene.xml and STL meshes, zipped."""
    from antares.export import export_zip

    return export_zip(load_lock(name))


def lock_summary(lock: dict) -> dict:
    res = lock.get("result") or {}
    return {
        "name": lock.get("name"),
        "dataset": lock.get("dataset"),
        "robot_type": lock.get("robot_type"),
        "created_at": lock.get("created_at"),
        "params": lock.get("params"),
        "scores": {
            k: res.get(k)
            for k in (
                "train_score",
                "eval_score",
                "baseline_train_score",
                "baseline_eval_score",
            )
        },
        "n_evaluations": res.get("n_evaluations"),
        "elapsed_s": res.get("elapsed_s"),
        "space": res.get("space", "shared"),
    }


def build_lock(ds: Dataset, train: list, result, name: str | None):
    """Make a RobotLock for a finished calibration (digest covers the train split)."""
    from antares.lock import RobotLock

    return RobotLock.create(
        dataset=ds.repo_id,
        robot_type=ds.robot_type,
        train_episodes=train,
        result=result,
        name=name,
    )


# --------------------------------------------------------------------------
# Drift scans (``LOCK_DIR/<lock>.drift.json``, one per lock)
# --------------------------------------------------------------------------

DRIFT_SUFFIX = ".drift.json"


def _drift_path(name: str) -> Path:
    if not _NAME_RE.match(name) or ".." in name:
        raise HTTPException(status_code=400, detail="invalid lock name")
    return Path(LOCK_DIR) / f"{name}{DRIFT_SUFFIX}"


def load_drift_dict(name: str) -> dict:
    p = _drift_path(name)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"no drift scan for lock {name!r}")
    return json.loads(p.read_text())


def list_drift_dicts() -> list[dict]:
    d = Path(LOCK_DIR)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob(f"*{DRIFT_SUFFIX}")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda s: str(s.get("created_at", "")), reverse=True)
    return out


def drift_summary(scan: dict) -> dict:
    return {
        "lock": scan.get("lock"),
        "dataset": scan.get("dataset"),
        "window": scan.get("window"),
        "step": scan.get("step"),
        "n_windows": len(scan.get("windows") or []),
        "drifting": [k for k, v in (scan.get("drifting") or {}).items() if v],
        "verdict": scan.get("verdict"),
        "created_at": scan.get("created_at"),
    }


def run_drift(
    ds: Dataset,
    lock_name: str,
    *,
    window: int = 10,
    step: int = 5,
    max_evaluations: int = 60,
    seed: int = 0,
    on_progress: Callable[[Any], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[Any, Path]:
    """Sliding-window refits of ``lock_name`` over ``ds``; saves and returns (DriftScan, path)."""
    from antares.drift import drift_scan

    lock = load_lock(lock_name)
    path = _drift_path(lock_name)
    try:
        scan = drift_scan(
            ds.episodes,
            lock,
            window=window,
            step=step,
            max_evaluations=max_evaluations,
            seed=seed,
            on_progress=on_progress,
            should_stop=should_stop,
        )
    except ValueError as e:  # e.g. fewer episodes than one window
        raise HTTPException(status_code=400, detail=str(e))
    scan.save(path)
    return scan, path


def _drift_worker(job: Job, req: "DriftRequest") -> None:
    try:
        ds = get_dataset(req.dataset)

        def on_progress(ev) -> None:
            job.push({"type": "window", **ev.to_dict()})

        scan, _ = run_drift(
            ds,
            req.lock,
            window=req.window,
            step=req.step,
            max_evaluations=req.max_evaluations,
            seed=req.seed,
            on_progress=on_progress,
            should_stop=job.stop.is_set,
        )
        scan_d = scan.to_dict()
        job.push({"type": "done", "scan": scan_d})
        job.finish("done", result=scan_d)
    except Exception as e:  # noqa: BLE001 - report to the client
        job_fail(job, e)


# --------------------------------------------------------------------------
# Domain-randomization ranges (JSON files next to the locks)
# --------------------------------------------------------------------------


def _dr_path(name: str) -> Path:
    if not _NAME_RE.match(name) or ".." in name:
        raise HTTPException(status_code=400, detail="invalid ranges name")
    return Path(LOCK_DIR) / f"{name}{DR_SUFFIX}"


def list_dr_dicts() -> list[dict]:
    d = Path(LOCK_DIR)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob(f"*{DR_SUFFIX}")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
    return out


def load_dr_dict(name: str) -> dict:
    p = _dr_path(name)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"no ranges named {name!r}")
    return json.loads(p.read_text())


def dr_summary(ranges: dict) -> dict:
    """Listing row: provenance plus lo/hi per parameter, without the samples."""
    keys = ("name", "dataset", "lock", "n_fits", "n_train", "max_evaluations", "seed", "created_at")
    out = {k: ranges.get(k) for k in keys}
    out["ranges"] = {
        p: {k: r.get(k) for k in ("nominal", "lo", "hi", "scale")}
        for p, r in (ranges.get("ranges") or {}).items()
    }
    return out


# --------------------------------------------------------------------------
# Background jobs (calibration, DR ranges, drift scans) streamed over SSE
# --------------------------------------------------------------------------


@dataclass
class Job:
    """A background fit streamed to the UI; ``kind`` is ``calibrate`` or ``drift``."""

    id: str
    dataset: str
    kind: str = "calibrate"  # calibrate | dr | drift
    status: str = "running"  # running | done | error
    events: list[dict] = field(default_factory=list)
    cond: threading.Condition = field(default_factory=threading.Condition)
    stop: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    lock: dict | None = None
    error: str | None = None

    def push(self, event: dict) -> None:
        with self.cond:
            self.events.append(event)
            self.cond.notify_all()

    def finish(self, status: str, **extra) -> None:
        with self.cond:
            self.status = status
            for k, v in extra.items():
                setattr(self, k, v)
            self.cond.notify_all()

    def stream(self, keepalive_s: float = 15.0) -> Iterator[dict | None]:
        """Yield buffered events from the start, then live ones; None = keepalive."""
        i = 0
        while True:
            with self.cond:
                while i >= len(self.events) and self.status == "running":
                    if not self.cond.wait(timeout=keepalive_s):
                        break
                batch = self.events[i:]
                finished = self.status != "running"
            if not batch and not finished:
                yield None
                continue
            for ev in batch:
                yield ev
            i += len(batch)
            if finished and i >= len(self.events):
                return


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def get_job(job_id: str) -> Job:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job


def start_job(kind: str, dataset: str, worker: Callable[..., None], *args: Any) -> Job:
    """Register a Job and run ``worker(job, *args)`` on a daemon thread."""
    job = Job(id=uuid.uuid4().hex[:12], dataset=dataset, kind=kind)
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=worker, args=(job, *args), daemon=True).start()
    return job


def job_status_dict(job: Job) -> dict:
    with job.cond:
        out: dict[str, Any] = {
            "job_id": job.id,
            "kind": job.kind,
            "dataset": job.dataset,
            "status": job.status,
            "events_count": len(job.events),
            "cancelled": job.stop.is_set(),
        }
        if job.result is not None:
            out["result"] = job.result
        if job.lock is not None:
            out["lock"] = job.lock
        if job.error is not None:
            out["error"] = job.error
    return out


def job_fail(job: Job, e: Exception) -> None:
    msg = e.detail if isinstance(e, HTTPException) else f"{type(e).__name__}: {e}"
    job.push({"type": "error", "message": str(msg)})
    job.finish("error", error=str(msg))


def run_calibration(
    ds: Dataset,
    *,
    n_train: int = 20,
    n_eval: int = 10,
    max_evaluations: int = 150,
    seed: int = 0,
    name: str | None = None,
    per_joint: bool = False,
    per_joint_max_evaluations: int = 600,
    per_joint_sigma0: float | None = None,
    on_progress: Callable[[Any], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[Any, dict, Path]:
    """Calibrate on ``ds``, save a lock, return (CalibrationResult, lock dict, path).

    ``per_joint`` adds the per-joint refinement stage after the shared fit,
    with its own evaluation budget and (optionally) CMA-ES step size.
    """
    from antares.calibrate import calibrate, split_episodes

    if n_train < 1:
        raise HTTPException(status_code=400, detail="n_train must be >= 1")
    if name is not None:
        _lock_path(name)  # validate the name before spending minutes fitting
    train, eval_ = split_episodes(ds.episodes, n_train, n_eval, seed=seed)
    extra: dict[str, Any] = {}
    if per_joint:
        extra = {"per_joint": True, "per_joint_max_evaluations": int(per_joint_max_evaluations)}
        if per_joint_sigma0 is not None:
            extra["per_joint_sigma0"] = float(per_joint_sigma0)
    result = calibrate(
        train,
        eval_,
        start=default_params(),
        max_evaluations=max_evaluations,
        seed=seed,
        on_progress=on_progress,
        should_stop=should_stop,
        **extra,
    )
    lock = build_lock(ds, train, result, name)
    path = save_lock(lock)
    return result, lock.to_dict(), path


def _job_worker(job: Job, req: "CalibrateRequest") -> None:
    try:
        ds = get_dataset(req.dataset)

        def on_progress(ev) -> None:
            job.push({"type": "progress", **ev.to_dict()})

        result, lock, _ = run_calibration(
            ds,
            n_train=req.n_train,
            n_eval=req.n_eval,
            max_evaluations=req.max_evaluations,
            seed=req.seed,
            name=req.name,
            per_joint=req.per_joint,
            per_joint_max_evaluations=req.per_joint_max_evaluations,
            on_progress=on_progress,
            should_stop=job.stop.is_set,
        )
        result_d = result.to_dict()
        job.push({"type": "done", "lock": lock, "result": result_d})
        job.finish("done", result=result_d, lock=lock)
    except Exception as e:  # noqa: BLE001 - report to the client
        job_fail(job, e)


def run_dr(
    ds: Dataset,
    lock_name: str,
    *,
    n_fits: int = 8,
    n_train: int = 12,
    max_evaluations: int = 60,
    seed: int = 0,
    name: str | None = None,
    on_progress: Callable[[Any], None] | None = None,
    on_fit: Callable[[int, Any], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[Any, dict, Path]:
    """Derive DR ranges for ``lock_name`` from ``ds``; save and return
    ``(DRRanges, ranges dict, path)``.

    The bootstrap pool is every episode the lock did not hold out, so the
    lock's eval split stays clean for validating the ranges afterwards.
    """
    from antares.dr import DRRanges, bootstrap_fits

    if n_fits < 1:
        raise HTTPException(status_code=400, detail="n_fits must be >= 1")
    if n_train < 1:
        raise HTTPException(status_code=400, detail="n_train must be >= 1")
    name = name or lock_name
    _dr_path(name)  # validate the name before spending time fitting
    lock = load_lock_dict(lock_name)
    params = params_from_dict(lock["params"])
    held_out = set((lock.get("result") or {}).get("eval_episodes") or [])
    pool = [ep for ep in ds.episodes if ep.index not in held_out]
    if not pool:
        raise HTTPException(status_code=400, detail="lock holds out every episode of the dataset")

    results = bootstrap_fits(
        pool,
        params,
        n_fits=n_fits,
        n_train=n_train,
        max_evaluations=max_evaluations,
        seed=seed,
        on_progress=on_progress,
        should_stop=should_stop,
        on_fit=on_fit,
    )
    if not results:
        raise HTTPException(status_code=409, detail="cancelled before the first fit finished")
    ranges = DRRanges.from_fits(results, params, dataset=ds.repo_id, lock=lock_name, name=name, seed=seed)
    path = ranges.save(_dr_path(name))
    return ranges, ranges.to_dict(), path


def _dr_job_worker(job: Job, req: "DRRequest") -> None:
    try:
        ds = get_dataset(req.dataset)

        def on_progress(ev) -> None:
            job.push({"type": "progress", **ev.to_dict()})

        def on_fit(i: int, result) -> None:
            job.push(
                {
                    "type": "fit",
                    "fit": i,
                    "n_fits": req.n_fits,
                    "params": result.params.to_dict(),
                    "train_score": result.to_dict()["train_score"],
                    "train_episodes": [int(e) for e in result.train_episodes],
                }
            )

        _, ranges, _ = run_dr(
            ds,
            req.lock or _slug(req.dataset),
            n_fits=req.n_fits,
            n_train=req.n_train,
            max_evaluations=req.max_evaluations,
            seed=req.seed,
            name=req.name,
            on_progress=on_progress,
            on_fit=on_fit,
            should_stop=job.stop.is_set,
        )
        job.push({"type": "done", "ranges": ranges})
        job.finish("done", result=ranges)
    except Exception as e:  # noqa: BLE001 - report to the client
        msg = e.detail if isinstance(e, HTTPException) else f"{type(e).__name__}: {e}"
        job.push({"type": "error", "message": str(msg)})
        job.finish("error", error=str(msg))


def _slug(repo_id: str) -> str:
    from antares.lock import slug

    return slug(repo_id)


def cross_matrix(
    datasets: list[str],
    locks: list[str],
    *,
    n_train: int = 20,
    n_eval: int = 10,
    seed: int = 0,
) -> dict:
    """Score every lock (and default params) on every dataset's eval split."""
    from antares.calibrate import split_episodes

    if n_train < 1:
        raise HTTPException(status_code=400, detail="n_train must be >= 1")
    cols = list(locks) + ["default"]
    col_params = [params_from_dict(load_lock_dict(n)["params"]) for n in locks]
    col_params.append(default_params())

    scores: list[list[float]] = []
    per_joint: list[list[dict]] = []
    for repo_id in datasets:
        ds = get_dataset(repo_id)
        _, eval_ = split_episodes(ds.episodes, n_train, n_eval, seed=seed)
        if len(eval_) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"{repo_id} has {len(ds.episodes)} episodes; "
                f"n_train={n_train} + n_eval={n_eval} leaves no eval split",
            )
        row_s, row_j = [], []
        for p in col_params:
            rep = score_episodes(p, eval_)
            row_s.append(round(float(rep.overall_deg), ROUND))
            row_j.append({k: round(float(v), ROUND) for k, v in rep.per_joint_deg.items()})
        scores.append(row_s)
        per_joint.append(row_j)
    return {"rows": list(datasets), "cols": cols, "scores": scores, "per_joint": per_joint}


# --------------------------------------------------------------------------
# Drift check (pre-deploy gate)
# --------------------------------------------------------------------------


def run_check(
    ds: Dataset,
    lock: str | dict,
    *,
    threshold_deg: float | None = None,
    ratio: float = 1.5,
    last: int | None = None,
    min_episodes: int = 5,
):
    """Score ``lock`` (a name in LOCK_DIR, or a lock dict) on ``ds`` with the
    shared Arm and return an :class:`antares.check.CheckResult`."""
    from antares.check import check

    lock_d = lock if isinstance(lock, dict) else load_lock_dict(lock)
    arm = get_arm()

    def simulate(params, ep):
        arm.apply(params)
        return np.asarray(arm.real_q(ep)), np.asarray(arm.replay(ep))

    try:
        with _ARM_LOCK:
            return check(
                ds.episodes,
                lock_d,
                threshold_deg=threshold_deg,
                ratio=ratio,
                min_episodes=min_episodes,
                last=last,
                simulate=simulate,
                dataset=ds.repo_id,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(title="antares", version=antares.__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": antares.__version__}


@app.get("/api/datasets")
def datasets() -> list[dict]:
    with _DATASETS_LOCK:
        ids = list(DEMO_DATASETS) + [k for k in _DATASETS if k not in DEMO_DATASETS]
    return [dataset_summary(i) for i in ids]


@app.post("/api/datasets/load")
def datasets_load(dataset: str = Query(...)) -> dict:
    get_dataset(dataset)
    return dataset_summary(dataset)


@app.get("/api/episodes")
def episodes(dataset: str = Query(...)) -> list[dict]:
    ds = get_dataset(dataset)
    return [
        {"index": e.index, "frames": len(e), "duration_s": round(e.duration, 3)}
        for e in ds.episodes
    ]


@app.get("/api/model")
def model() -> dict:
    from antares.sim import model_geometry

    return model_geometry()


_STL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.stl$", re.IGNORECASE)


@app.get("/api/model/mesh/{file}")
def model_mesh(file: str):
    if "/" in file or "\\" in file or ".." in file or not _STL_RE.match(file):
        raise HTTPException(status_code=400, detail="mesh must be a plain *.stl basename")
    path = mesh_dir() / file
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no mesh {file}")
    return FileResponse(path, media_type="model/stl", filename=file)


@app.get("/api/replay")
def replay(
    dataset: str = Query(...),
    episode: int = Query(...),
    lock: str | None = None,
    baseline: int = 0,
) -> dict:
    ds = get_dataset(dataset)
    params = None
    if lock and not baseline:
        params = params_from_dict(load_lock_dict(lock)["params"])
    out = replay_episode(ds, episode, params)
    out["lock"] = lock if (lock and not baseline) else None
    return out


class ReplayRequest(BaseModel):
    dataset: str
    episode: int
    # A value is a scalar or a per-joint list of 6, as SimParams.from_dict accepts.
    params: dict[str, float | list[float]]


@app.post("/api/replay")
def replay_with_params(req: ReplayRequest) -> dict:
    """Replay with explicit params, e.g. the best fit so far of a running job."""
    ds = get_dataset(req.dataset)
    try:
        params = params_from_dict(req.params)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"bad params: {e}")
    out = replay_episode(ds, req.episode, params)
    out["lock"] = None
    return out


class CalibrateRequest(BaseModel):
    dataset: str
    n_train: int = Field(20, ge=1)
    n_eval: int = Field(10, ge=0)
    max_evaluations: int = Field(150, ge=1)
    seed: int = 0
    name: str | None = None
    # Per-joint refinement after the shared fit; off by default so the live
    # demo stays at ~150 evaluations.
    per_joint: bool = False
    per_joint_max_evaluations: int = Field(600, ge=1)


@app.post("/api/calibrate")
def calibrate_start(req: CalibrateRequest) -> dict:
    if req.name is not None:
        _lock_path(req.name)  # validate early
    job = start_job("calibrate", req.dataset, _job_worker, req)
    return {"job_id": job.id}


# Job routes are shared by every background fit; the ``/api/calibrate/...``
# spellings are kept for the frontend.
@app.get("/api/jobs/{job_id}")
@app.get("/api/calibrate/{job_id}")
def job_status(job_id: str) -> dict:
    return job_status_dict(get_job(job_id))


@app.post("/api/jobs/{job_id}/cancel")
@app.post("/api/calibrate/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    job = get_job(job_id)
    job.stop.set()
    return {"job_id": job.id, "cancelled": True, "status": job.status}


@app.get("/api/jobs/{job_id}/events")
@app.get("/api/calibrate/{job_id}/events")
def job_events(job_id: str):
    job = get_job(job_id)

    def gen() -> Iterator[bytes]:
        for ev in job.stream():
            if ev is None:
                yield b": keepalive\n\n"
            else:
                yield f"data: {json.dumps(ev)}\n\n".encode()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class DRRequest(BaseModel):
    dataset: str
    lock: str | None = None  # default: the lock named after the dataset
    n_fits: int = Field(8, ge=1)
    n_train: int = Field(12, ge=1)
    max_evaluations: int = Field(60, ge=1)
    seed: int = 0
    name: str | None = None  # default: the lock name


@app.post("/api/dr")
def dr_start(req: DRRequest) -> dict:
    lock_name = req.lock or _slug(req.dataset)
    load_lock_dict(lock_name)  # 404 now rather than as a job error
    _dr_path(req.name or lock_name)
    job = Job(id=uuid.uuid4().hex[:12], dataset=req.dataset, kind="dr")
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=_dr_job_worker, args=(job, req), daemon=True).start()
    return {
        "job_id": job.id,
        "status_url": f"/api/jobs/{job.id}",
        "events_url": f"/api/jobs/{job.id}/events",
    }


@app.get("/api/dr")
def dr_list() -> list[dict]:
    return [dr_summary(r) for r in list_dr_dicts()]


@app.get("/api/dr/{name}")
def dr_get(name: str) -> dict:
    return load_dr_dict(name)


@app.get("/api/locks")
def locks() -> list[dict]:
    return [lock_summary(l) for l in list_lock_dicts()]


@app.get("/api/locks/{name}")
def lock_get(name: str) -> dict:
    return load_lock_dict(name)


def _attachment(filename: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


@app.get("/api/locks/{name}/export")
def lock_export(name: str) -> Response:
    """Calibrated MJCF as a download. ``meshdir`` is the absolute package
    assets dir; use ``export.zip`` for a self-contained bundle."""
    return Response(
        export_lock_xml(name),
        media_type="application/xml",
        headers=_attachment(f"{name}.xml"),
    )


@app.get("/api/locks/{name}/export.zip")
def lock_export_zip(name: str) -> Response:
    """Zip of ``<name>/<name>.xml``, ``<name>.antares.json``, ``scene.xml``
    and ``assets/*.stl``."""
    return Response(
        export_lock_zip(name),
        media_type="application/zip",
        headers=_attachment(f"{name}.zip"),
    )


@app.delete("/api/locks/{name}")
def lock_delete(name: str) -> dict:
    p = _lock_path(name)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"no lock named {name!r}")
    p.unlink()
    return {"deleted": name}


class DriftRequest(BaseModel):
    dataset: str
    lock: str
    window: int = Field(10, ge=1)
    step: int = Field(5, ge=1)
    max_evaluations: int = Field(60, ge=1)
    seed: int = 0


@app.post("/api/drift")
def drift_start(req: DriftRequest) -> dict:
    load_lock_dict(req.lock)  # 400/404 before a thread is spent
    job = start_job("drift", req.dataset, _drift_worker, req)
    return {"job_id": job.id}


@app.get("/api/drift")
def drift_list() -> list[dict]:
    return [drift_summary(s) for s in list_drift_dicts()]


@app.get("/api/drift/{name}")
def drift_get(name: str) -> dict:
    return load_drift_dict(name)


def _csv(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


@app.get("/api/cross")
def cross(
    datasets: str = Query(..., description="comma-separated dataset ids"),
    locks: str = Query("", description="comma-separated lock names"),
    n_train: int = 20,
    n_eval: int = 10,
    seed: int = 0,
) -> dict:
    ds_ids = _csv(datasets)
    if not ds_ids:
        raise HTTPException(status_code=400, detail="datasets is required")
    return cross_matrix(ds_ids, _csv(locks), n_train=n_train, n_eval=n_eval, seed=seed)


class CheckRequest(BaseModel):
    dataset: str
    lock: str
    threshold_deg: float | None = Field(None, ge=0)
    ratio: float = Field(1.5, gt=0)
    last: int | None = Field(None, ge=1)
    min_episodes: int = Field(5, ge=1)


@app.post("/api/check")
def check_drift(req: CheckRequest) -> dict:
    """Has the robot drifted from its lock? Always 200; read ``verdict``."""
    _lock_path(req.lock)  # validate the name before loading a dataset
    ds = get_dataset(req.dataset)
    result = run_check(
        ds,
        req.lock,
        threshold_deg=req.threshold_deg,
        ratio=req.ratio,
        last=req.last,
        min_episodes=req.min_episodes,
    )
    return result.to_dict()


# Built frontend, mounted after the API routes so `antares serve` alone demos.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
