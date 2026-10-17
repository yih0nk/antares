"""HTTP API behind the web UI. Run with ``antares serve``.

Dataset ids contain ``/`` so they are always passed as query parameters.
Heavy modules (``antares.sim``, ``antares.gap``, ``antares.calibrate``,
``antares.lock``) are imported lazily inside the functions that need them,
so the app imports quickly and tests can stub them through ``sys.modules``.

The pure helpers (``replay_episode``, ``run_calibration``, ``cross_matrix``,
lock file I/O) are shared with the CLI.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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

# Override for the STL directory (tests). None means MODEL_PATH.parent / "assets".
MESH_DIR: Path | None = None

ROUND = 4


# --------------------------------------------------------------------------
# Dataset cache
# --------------------------------------------------------------------------

_DATASETS: dict[str, Dataset] = {}
_DATASETS_LOCK = threading.Lock()


def get_dataset(repo_id: str) -> Dataset:
    """Return a cached Dataset, loading it (slow first time) if needed."""
    with _DATASETS_LOCK:
        ds = _DATASETS.get(repo_id)
        if ds is not None:
            return ds
    from antares.data import load_dataset

    try:
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
# Calibration jobs
# --------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    dataset: str
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


def run_calibration(
    ds: Dataset,
    *,
    n_train: int = 20,
    n_eval: int = 10,
    max_evaluations: int = 150,
    seed: int = 0,
    name: str | None = None,
    on_progress: Callable[[Any], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[Any, dict, Path]:
    """Calibrate on ``ds``, save a lock, return (CalibrationResult, lock dict, path)."""
    from antares.calibrate import calibrate, split_episodes

    if n_train < 1:
        raise HTTPException(status_code=400, detail="n_train must be >= 1")
    if name is not None:
        _lock_path(name)  # validate the name before spending minutes fitting
    train, eval_ = split_episodes(ds.episodes, n_train, n_eval, seed=seed)
    result = calibrate(
        train,
        eval_,
        start=default_params(),
        max_evaluations=max_evaluations,
        seed=seed,
        on_progress=on_progress,
        should_stop=should_stop,
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
            on_progress=on_progress,
            should_stop=job.stop.is_set,
        )
        result_d = result.to_dict()
        job.push({"type": "done", "lock": lock, "result": result_d})
        job.finish("done", result=result_d, lock=lock)
    except Exception as e:  # noqa: BLE001 - report to the client
        msg = e.detail if isinstance(e, HTTPException) else f"{type(e).__name__}: {e}"
        job.push({"type": "error", "message": str(msg)})
        job.finish("error", error=str(msg))


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
# Lag compensation (sim-only validation)
# --------------------------------------------------------------------------

COMPENSATE_METHODS = ("shift", "ilc")
SIM_ONLY_NOTE = (
    "Tracking is judged in the calibrated MuJoCo sim against the real measured "
    "trajectory; no hardware was driven with these commands."
)


def _check_method(method: str) -> str:
    if method not in COMPENSATE_METHODS:
        raise HTTPException(
            status_code=400, detail=f"method must be one of {list(COMPENSATE_METHODS)}, got {method!r}"
        )
    return method


def compensate_episode_dict(
    ds: Dataset, episode: int, params=None, method: str = "ilc", lock: str | None = None
) -> dict:
    """Raw vs shift vs ILC commands for one episode, judged in the calibrated sim.

    The desired trajectory is the episode's real measured trajectory. Angles
    are degrees; ``ctrl_comp``/``q_sim_comp`` follow ``method``. Poses of the
    compensated ghost are included so the viewer can overlay it.
    """
    from antares.compensate import compensate_episode, max_joint_velocity
    from antares.sim import JOINTS

    _check_method(method)
    ep = find_episode(ds, episode)
    if params is None:
        params = default_params()
    arm = get_arm()
    with _ARM_LOCK:
        vmax = max_joint_velocity(ds.episodes, arm)
        rep = compensate_episode(arm, ep, params, method=method, real_vmax_deg_s=vmax)
        comp_poses = np.asarray(arm.fk(rep.q_sim_comp))
        body_names = list(arm.body_names)
    return {
        "dataset": ds.repo_id,
        "episode": ep.index,
        "fps": rep.fps,
        "joints": list(JOINTS),
        "lock": lock,
        "method": method,
        "params": params.to_dict(),
        "validation": "sim-only",
        "note": SIM_ONLY_NOTE,
        "q_des": _rounded(np.degrees(rep.q_des)),
        "ctrl_raw": _rounded(np.degrees(rep.ctrl_raw)),
        "ctrl_comp": _rounded(np.degrees(rep.ctrl_comp)),
        "q_sim_raw": _rounded(np.degrees(rep.q_sim_raw)),
        "q_sim_comp": _rounded(np.degrees(rep.q_sim_comp)),
        "body_names": body_names,
        "sim_comp_poses": _rounded(comp_poses),
        "rms": {k: round(float(v), ROUND) for k, v in rep.rms.items()},
        "gap": {"raw": rep.gap_raw.to_dict(), "shift": rep.gap_shift.to_dict(), "ilc": rep.gap_ilc.to_dict()},
        "ilc_iterations": int(rep.ilc_iterations),
        "ilc_history_deg": [round(float(h), ROUND) for h in rep.ilc_history_deg],
        "feasibility": rep.feasibility,
    }


def compensate_heldout(ds: Dataset, lock_dict: dict, method: str = "ilc") -> dict:
    """Aggregate raw/shift/ILC tracking over the lock's held-out episodes (sim-only)."""
    from antares.compensate import compensate_episodes, held_out_episodes

    _check_method(method)
    eval_idx = (lock_dict.get("result") or {}).get("eval_episodes") or []
    if not eval_idx:
        raise HTTPException(status_code=400, detail="lock has no held-out episodes to evaluate on")
    try:
        episodes = held_out_episodes(ds, eval_idx)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    params = params_from_dict(lock_dict["params"])
    arm = get_arm()
    with _ARM_LOCK:
        summary = compensate_episodes(arm, episodes, params, method=method)
    out = summary.to_dict()
    out.update(dataset=ds.repo_id, lock=lock_dict.get("name"), method=method, params=params.to_dict(),
               validation="sim-only", note=SIM_ONLY_NOTE)
    return out


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


@app.get("/api/compensate")
def compensate(
    dataset: str = Query(...),
    episode: int = Query(...),
    lock: str | None = None,
    method: str = Query("ilc", description="which compensated commands to return: shift | ilc"),
) -> dict:
    """Commands that make the calibrated sim track the episode's real trajectory.

    Sim-only validation: ``rms`` compares each command set's sim rollout with
    the real measured trajectory. Without ``lock`` the default parameters
    (latency 0) are used and ``shift`` degenerates to the raw commands.
    """
    _check_method(method)
    ds = get_dataset(dataset)
    params = params_from_dict(load_lock_dict(lock)["params"]) if lock else None
    return compensate_episode_dict(ds, episode, params, method=method, lock=lock or None)


class CalibrateRequest(BaseModel):
    dataset: str
    n_train: int = Field(20, ge=1)
    n_eval: int = Field(10, ge=0)
    max_evaluations: int = Field(150, ge=1)
    seed: int = 0
    name: str | None = None


@app.post("/api/calibrate")
def calibrate_start(req: CalibrateRequest) -> dict:
    if req.name is not None:
        _lock_path(req.name)  # validate early
    job = Job(id=uuid.uuid4().hex[:12], dataset=req.dataset)
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=_job_worker, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


@app.get("/api/calibrate/{job_id}")
def calibrate_status(job_id: str) -> dict:
    job = get_job(job_id)
    with job.cond:
        out: dict[str, Any] = {
            "job_id": job.id,
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


@app.post("/api/calibrate/{job_id}/cancel")
def calibrate_cancel(job_id: str) -> dict:
    job = get_job(job_id)
    job.stop.set()
    return {"job_id": job.id, "cancelled": True, "status": job.status}


@app.get("/api/calibrate/{job_id}/events")
def calibrate_events(job_id: str):
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


@app.get("/api/locks")
def locks() -> list[dict]:
    return [lock_summary(l) for l in list_lock_dicts()]


@app.get("/api/locks/{name}")
def lock_get(name: str) -> dict:
    return load_lock_dict(name)


@app.delete("/api/locks/{name}")
def lock_delete(name: str) -> dict:
    p = _lock_path(name)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"no lock named {name!r}")
    p.unlink()
    return {"deleted": name}


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


# Built frontend, mounted after the API routes so `antares serve` alone demos.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
