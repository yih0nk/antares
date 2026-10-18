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

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import antares
from antares.data import Dataset, Episode
from antares.robots import DEFAULT_ROBOT, get_robot, model_for_dataset, model_for_family
from antares.validate import Verdict, merge_verdicts, robot_family, validate_dataset, validate_dataset_info

# Datasets the demo offers out of the box.
DEMO_DATASETS = [
    "lerobot/svla_so101_pickplace",
    "lerobot/svla_so100_pickplace",
]

_ROOT = Path(__file__).resolve().parent.parent
WEB_DIST = Path(os.environ.get("ANTARES_WEB_DIST", _ROOT / "web" / "dist"))

# Uploaded datasets are extracted under here, one ``<slug>-<shorthash>/``
# directory each, and addressed as ``upload/<slug>``.
UPLOAD_DIR: Path = Path(os.environ.get("ANTARES_UPLOAD_DIR", "uploads"))
UPLOAD_PREFIX = "upload/"
UPLOAD_MARKER = ".antares-upload.json"
MAX_UPLOAD_BYTES = 200 * 2**20

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


# Fleet scan output: ``<FLEET_DIR>/fleet.json`` plus one lock per robot.
FLEET_DIR: Path = Path(os.environ.get("ANTARES_FLEET_DIR", Path(LOCK_DIR) / "fleet"))
FLEET_REPORT_NAME = "fleet.json"

# Override for the STL directory (tests). None means MODEL_PATH.parent / "assets".
MESH_DIR: Path | None = None

ROUND = 4


# --------------------------------------------------------------------------
# Dataset cache and validation
# --------------------------------------------------------------------------

_DATASETS: dict[str, Dataset] = {}
# Per loaded dataset: ``source`` (demo | hub | upload | local), the validation
# ``warnings`` and, for uploads, the extracted ``path``.
_DATASET_META: dict[str, dict[str, Any]] = {}
_DATASETS_LOCK = threading.Lock()


def dataset_source(repo_id: str) -> str:
    if repo_id in DEMO_DATASETS:
        return "demo"
    if repo_id.startswith(UPLOAD_PREFIX):
        return "upload"
    if os.path.isdir(repo_id):
        return "local"
    return "hub"


def rejected(verdict: Verdict, what: str) -> HTTPException:
    """400 whose ``detail`` carries both a sentence and the full verdict."""
    return HTTPException(
        status_code=400,
        detail={"message": f"{what} cannot be replayed: " + "; ".join(verdict.reasons), "verdict": verdict.to_dict()},
    )


def fetch_hub_info(repo_id: str) -> dict:
    """Only ``meta/info.json`` from the Hub (cached by huggingface_hub)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
    return json.loads(Path(path).read_text())


def _upload_dir_for(repo_id: str) -> Path | None:
    """The extracted directory behind ``upload/<name>``, from the registry or
    by scanning ``UPLOAD_DIR`` for its marker file (survives a restart)."""
    with _DATASETS_LOCK:
        meta = _DATASET_META.get(repo_id)
    if meta and meta.get("path") and Path(meta["path"]).is_dir():
        return Path(meta["path"])
    root = Path(UPLOAD_DIR)
    if not root.is_dir():
        return None
    for marker in sorted(root.glob(f"*/{UPLOAD_MARKER}")):
        try:
            if json.loads(marker.read_text()).get("id") == repo_id:
                return marker.parent
        except (OSError, ValueError):
            continue
    return None


def _local_info(root: Path) -> dict:
    p = Path(root) / "meta" / "info.json"
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"{root} has no meta/info.json; not a LeRobot dataset directory")
    try:
        return json.loads(p.read_text())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{p} is not valid JSON: {e}")


def min_episodes_for(repo_id: str) -> int:
    """A directory on disk is the user's own recording (``antares check``
    enforces its own minimum), so only what cannot be replayed is refused;
    Hub datasets and uploads need :data:`antares.validate.MIN_EPISODES`."""
    from antares.validate import MIN_EPISODES

    return 1 if dataset_source(repo_id) == "local" else MIN_EPISODES


def validate_id(repo_id: str) -> Verdict:
    """Pre-check a dataset id without loading its data (info.json only).

    Hub ids download one small file; local directories and uploads read it
    from disk. 404 when there is no such dataset, 502 when the Hub cannot
    be reached.
    """
    n_min = min_episodes_for(repo_id)
    if repo_id.startswith(UPLOAD_PREFIX):
        root = _upload_dir_for(repo_id)
        if root is None:
            raise HTTPException(status_code=404, detail=f"no upload named {repo_id!r}")
        return validate_dataset_info(_local_info(root), repo_id, min_episodes=n_min)
    if os.path.isdir(repo_id):
        return validate_dataset_info(_local_info(Path(repo_id)), repo_id, min_episodes=n_min)
    try:
        info = fetch_hub_info(repo_id)
    except Exception as e:  # noqa: BLE001 - a Hub error, in any shape
        name = type(e).__name__
        if "NotFound" in name or "GatedRepo" in name or "Revision" in name:
            raise HTTPException(status_code=404, detail=f"{repo_id}: no such dataset on the Hub, or no meta/info.json ({name})")
        raise HTTPException(status_code=502, detail=f"could not fetch {repo_id} meta/info.json: {name}: {e}")
    return validate_dataset_info(info, repo_id)


def get_dataset(repo_id: str) -> Dataset:
    """Return a cached Dataset, loading it (slow first time) if needed.

    ``repo_id`` is a Hub dataset id, the path of a LeRobot dataset directory
    on disk (a fresh local recording, read in place) or ``upload/<name>``
    for a dataset posted to ``/api/datasets/upload``. The dataset is
    validated before (``meta/info.json``) and after loading; a dataset the
    sim cannot replay is a 400 whose detail carries the reasons.
    """
    with _DATASETS_LOCK:
        ds = _DATASETS.get(repo_id)
        if ds is not None:
            return ds
    from antares.data import load_dataset, read_local

    pre = validate_id(repo_id)
    if not pre.ok:
        raise rejected(pre, repo_id)
    try:
        if repo_id.startswith(UPLOAD_PREFIX):
            ds = read_local(_upload_dir_for(repo_id), repo_id=repo_id)
        elif os.path.isdir(repo_id):
            ds = read_local(repo_id, repo_id=repo_id)
        else:
            ds = load_dataset(repo_id)
    except Exception as e:  # noqa: BLE001 - surface loader errors to the client
        raise HTTPException(status_code=502, detail=f"could not load {repo_id}: {e}")
    verdict = merge_verdicts(pre, validate_dataset(ds, min_episodes=min_episodes_for(repo_id)))
    if not verdict.ok:
        raise rejected(verdict, repo_id)
    return put_dataset(ds, warnings=verdict.warnings)


def put_dataset(ds: Dataset, *, source: str | None = None, warnings: list[str] | None = None, path: Path | None = None) -> Dataset:
    """Register an already loaded dataset so ``get_dataset`` finds it."""
    with _DATASETS_LOCK:
        _DATASETS.setdefault(ds.repo_id, ds)
        meta = _DATASET_META.setdefault(ds.repo_id, {})
        meta["source"] = source or dataset_source(ds.repo_id)
        if warnings is not None:
            meta["warnings"] = list(warnings)
        if path is not None:
            meta["path"] = str(path)
        return _DATASETS[ds.repo_id]


def dataset_robot(ds: Dataset) -> str:
    """Registry name of the model that simulates ``ds`` (by its robot_type)."""
    return model_for_dataset(ds.robot_type)


def dataset_summary(repo_id: str) -> dict:
    with _DATASETS_LOCK:
        ds = _DATASETS.get(repo_id)
        meta = dict(_DATASET_META.get(repo_id) or {})
    family = robot_family(ds.robot_type) if ds is not None else robot_family(repo_id)
    out: dict[str, Any] = {
        "id": repo_id,
        "loaded": ds is not None,
        "source": meta.get("source") or dataset_source(repo_id),
        "warnings": list(meta.get("warnings") or []),
        "robot_family": family,
        "robot": model_for_dataset(ds.robot_type) if ds is not None else model_for_dataset(repo_id),
    }
    if ds is not None:
        out.update(
            robot_type=ds.robot_type,
            fps=ds.fps,
            n_episodes=len(ds.episodes),
            joints=list(ds.joint_names),
        )
    return out


def forget_dataset(repo_id: str) -> dict:
    """Drop ``repo_id`` from the in-process cache; an upload's files go too."""
    with _DATASETS_LOCK:
        ds = _DATASETS.pop(repo_id, None)
        meta = _DATASET_META.pop(repo_id, None) or {}
    with _BASELINES_LOCK:
        for key in [k for k in _BASELINES if k[0] == repo_id]:
            del _BASELINES[key]
    removed_dir = None
    if repo_id.startswith(UPLOAD_PREFIX):
        root = Path(meta["path"]) if meta.get("path") else _upload_dir_for(repo_id)
        if root is None and ds is None:
            raise HTTPException(status_code=404, detail=f"no upload named {repo_id!r}")
        if root is not None and root.is_dir() and _inside(root, Path(UPLOAD_DIR)):
            shutil.rmtree(root)
            removed_dir = str(root)
    elif ds is None:
        raise HTTPException(status_code=404, detail=f"{repo_id} is not loaded")
    return {"deleted": repo_id, "removed_dir": removed_dir}


# --------------------------------------------------------------------------
# Uploads: a zip of a LeRobot dataset directory, or one parquet + info.json
# --------------------------------------------------------------------------


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_member(name: str) -> Path | None:
    """A zip member name as a relative, traversal-free path, or None to skip."""
    name = name.replace("\\", "/")
    if not name or name.endswith("/"):
        return None
    p = Path(name)
    if p.is_absolute() or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise HTTPException(status_code=400, detail=f"zip member {name!r} has an absolute path")
    if any(part in ("..", "") for part in p.parts) or any(part.startswith("..") and part != ".." for part in p.parts if "/" in part):
        raise HTTPException(status_code=400, detail=f"zip member {name!r} escapes the archive")
    return p


def _upload_name(name: str | None, filename: str | None) -> str:
    from antares.lock import slug

    base = (name or "").strip() or Path(filename or "dataset").name
    base = re.sub(r"\.(zip|parquet)$", "", base, flags=re.IGNORECASE)
    return slug(base)


def _spool_upload(upload: UploadFile) -> tuple[Path, str, int]:
    """Stream the body to a temp file with a running SHA-256; 413 past the cap."""
    h = hashlib.sha256()
    size = 0
    tmp = tempfile.NamedTemporaryFile(prefix="antares-upload-", delete=False)
    try:
        while True:
            chunk = upload.file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES // 2**20} MB")
            h.update(chunk)
            tmp.write(chunk)
        tmp.close()
    except BaseException:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise
    return Path(tmp.name), h.hexdigest(), size


def extract_dataset_zip(zip_path: Path, dest: Path) -> int:
    """Copy ``meta/**`` and ``data/**/*.parquet`` out of a LeRobot dataset zip.

    The dataset may sit at the archive root or inside one top-level folder
    (``<name>/meta/info.json``). Videos and anything else are ignored.
    Member paths are checked for traversal and the uncompressed size is
    capped. Returns the number of files written.
    """
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="not a zip file")
    with zf:
        members = []
        for zi in zf.infolist():
            rel = _safe_member(zi.filename)
            if rel is not None:
                members.append((zi, rel))
        roots = [rel.parent.parent for _, rel in members if rel.parts[-2:] == ("meta", "info.json")]
        if not roots:
            raise HTTPException(status_code=400, detail="zip has no meta/info.json (expected a LeRobot dataset directory)")
        root = sorted(roots, key=lambda r: len(r.parts))[0]
        total = 0
        wanted: list[tuple[zipfile.ZipInfo, Path]] = []
        for zi, rel in members:
            try:
                inner = rel.relative_to(root)
            except ValueError:
                continue
            parts = inner.parts
            if parts[0] == "meta" or (parts[0] == "data" and inner.suffix.lower() == ".parquet"):
                total += int(zi.file_size)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"uncompressed data exceeds {MAX_UPLOAD_BYTES // 2**20} MB")
                wanted.append((zi, inner))
        if not any(inner.parts[0] == "data" for _, inner in wanted):
            raise HTTPException(status_code=400, detail="zip has no data/**/*.parquet files")
        dest.mkdir(parents=True, exist_ok=True)
        for zi, inner in wanted:
            out = dest / inner
            if not _inside(out, dest):
                raise HTTPException(status_code=400, detail=f"zip member {zi.filename!r} escapes the archive")
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zi) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return len(wanted)


def _is_parquet(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(4)
        if head != b"PAR1":
            return False
        f.seek(-4, os.SEEK_END)
        return f.read(4) == b"PAR1"


def register_upload(
    body: Path,
    *,
    filename: str | None,
    name: str | None = None,
    info_json: bytes | None = None,
    sha256: str | None = None,
) -> dict:
    """Extract/validate/load an uploaded dataset and register it as ``upload/<name>``.

    ``body`` is the spooled upload: a zip of a LeRobot dataset directory, or
    a single parquet file in which case ``info_json`` must hold the
    ``meta/info.json`` text. On any failure the extracted directory is
    removed and an HTTPException (400/409/413) is raised.
    """
    from antares.data import read_local

    slug_name = _upload_name(name, filename)
    repo_id = f"{UPLOAD_PREFIX}{slug_name}"
    digest = sha256 or hashlib.sha256(body.read_bytes()).hexdigest()
    dest = Path(UPLOAD_DIR) / f"{slug_name}-{digest[:8]}"

    with _DATASETS_LOCK:
        existing = _DATASET_META.get(repo_id)
    if existing is not None:
        if existing.get("sha256") == digest:
            return {**dataset_summary(repo_id), "verdict": existing.get("verdict")}
        raise HTTPException(
            status_code=409,
            detail=f"{repo_id} already exists with different data; delete it first or upload under another name",
        )

    if dest.exists():
        shutil.rmtree(dest)
    try:
        if zipfile.is_zipfile(body):
            extract_dataset_zip(body, dest)
        elif _is_parquet(body):
            if not info_json:
                raise HTTPException(status_code=400, detail="a single parquet upload needs an info.json field (meta/info.json)")
            try:
                json.loads(info_json)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"info.json is not valid JSON: {e}")
            (dest / "meta").mkdir(parents=True, exist_ok=True)
            (dest / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
            (dest / "meta" / "info.json").write_bytes(info_json)
            shutil.copyfile(body, dest / "data" / "chunk-000" / "file-000.parquet")
        else:
            raise HTTPException(status_code=400, detail="upload must be a .zip of a LeRobot dataset directory or a .parquet file")

        pre = validate_dataset_info(_local_info(dest), repo_id)
        if not pre.ok:
            raise rejected(pre, repo_id)
        try:
            ds = read_local(dest, repo_id=repo_id)
        except Exception as e:  # noqa: BLE001 - a bad parquet is the client's problem
            raise HTTPException(status_code=400, detail=f"could not read the uploaded data: {e}")
        verdict = merge_verdicts(pre, validate_dataset(ds))
        if not verdict.ok:
            raise rejected(verdict, repo_id)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    marker = {
        "id": repo_id,
        "name": slug_name,
        "sha256": digest,
        "filename": filename,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    (dest / UPLOAD_MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    put_dataset(ds, source="upload", warnings=verdict.warnings, path=dest)
    with _DATASETS_LOCK:
        _DATASET_META[repo_id].update(sha256=digest, verdict=verdict.to_dict())
    return {**dataset_summary(repo_id), "verdict": verdict.to_dict()}


def find_episode(ds: Dataset, index: int) -> Episode:
    for ep in ds.episodes:
        if ep.index == index:
            return ep
    raise HTTPException(status_code=404, detail=f"episode {index} not in {ds.repo_id}")


# --------------------------------------------------------------------------
# Sim access: one Arm per robot model, each behind its own lock (Arm mutates
# state). Tests inject fakes into ``_ARMS``.
# --------------------------------------------------------------------------

_ARMS: dict[str, Any] = {}
_ARM_LOCKS: dict[str, threading.Lock] = {}
_ARMS_GUARD = threading.Lock()


def check_robot(robot: str | None) -> str:
    """Registry name for ``robot`` (default model when None); 400 if unknown."""
    try:
        return get_robot(robot).name
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def get_arm(robot: str | None = None):
    """The process-wide Arm for ``robot`` (built on first use)."""
    name = check_robot(robot)
    with _ARMS_GUARD:
        arm = _ARMS.get(name)
    if arm is None:
        from antares.sim import Arm

        arm = Arm(robot=name)
        with _ARMS_GUARD:
            arm = _ARMS.setdefault(name, arm)
    return arm


def arm_lock(robot: str | None = None) -> threading.Lock:
    """The lock serializing use of ``get_arm(robot)``."""
    name = check_robot(robot)
    with _ARMS_GUARD:
        return _ARM_LOCKS.setdefault(name, threading.Lock())


def simulate_for(robot: str | None):
    """A ``simulate`` callable on a private Arm of ``robot`` for the fitters
    (calibrate/drift/dr build their own by default; None keeps that default,
    which is the SO-101 model)."""
    name = check_robot(robot)
    if name == DEFAULT_ROBOT:
        return None
    from antares.calibrate import default_simulate

    return default_simulate(robot=name)


def default_params():
    from antares.sim import SimParams

    return SimParams.default()


def params_from_dict(d: dict):
    from antares.sim import SimParams

    return SimParams.from_dict(d)


def command_map_from_lock(lock: dict):
    """The lock's CommandMap, or None for a lock fitted without one."""
    d = lock.get("command_map")
    if d is None:
        return None
    from antares.sim import CommandMap

    return CommandMap.from_dict(d)


@contextmanager
def _command_map(arm, cmap):
    """Use ``cmap`` on the shared ``arm`` for the block, then restore identity
    so callers that never set one keep the plain mapping."""
    if cmap is None:
        yield
        return
    from antares.sim import CommandMap

    arm.command_map = cmap
    try:
        yield
    finally:
        arm.command_map = CommandMap.identity()


def lock_simulate(lock: dict):
    """A SimulateFn holding the lock's command map fixed on the lock's robot
    model, or None (the callee's default: identity map, SO-101) when the lock
    has neither. For refits warm-started from a lock: the dynamics move, the
    sensor calibration and the model do not."""
    cmap = command_map_from_lock(lock)
    robot = check_robot(lock.get("robot"))
    if cmap is None and robot == DEFAULT_ROBOT:
        return None
    from antares.calibrate import default_simulate

    return default_simulate(cmap, robot=robot)


def mesh_dir(robot: str | None = None) -> Path:
    if MESH_DIR is not None:
        return Path(MESH_DIR)
    return Path(get_robot(check_robot(robot)).mesh_dir)


def _rounded(a: np.ndarray) -> list:
    return np.round(np.asarray(a, dtype=np.float64), ROUND).tolist()


def replay_episode(ds: Dataset, episode: int, params=None, command_map=None, robot: str | None = None) -> dict:
    """Real FK + sim replay for one episode. Angles in degrees, poses (T,nbody,7).

    ``command_map`` (a CommandMap, or None for identity) corrects the commands.
    ``robot`` overrides the model (default: the one for the dataset's family).
    """
    from antares.gap import tracking_error
    from antares.sim import JOINTS

    ep = find_episode(ds, episode)
    if params is None:
        params = default_params()
    robot = check_robot(robot or dataset_robot(ds))
    arm = get_arm(robot)
    with arm_lock(robot), _command_map(arm, command_map):
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
        "robot": robot,
        "joints": list(JOINTS),
        "body_names": body_names,
        "params": params.to_dict(),
        "command_map": None if command_map is None else command_map.to_dict(),
        "real_q": _rounded(np.degrees(real_q)),
        "sim_q": _rounded(np.degrees(sim_q)),
        "real_poses": _rounded(real_poses),
        "sim_poses": _rounded(sim_poses),
        "gap": gap.to_dict(),
    }


def score_episodes(params, episodes, command_map=None, robot: str | None = None) -> Any:
    """Aggregate GapReport of ``params`` (and ``command_map``) over ``episodes``
    using the shared Arm of ``robot``."""
    from antares.gap import aggregate, tracking_error
    from antares.sim import JOINTS

    arm = get_arm(robot)
    reports = []
    with arm_lock(robot), _command_map(arm, command_map):
        arm.apply(params)
        for ep in episodes:
            real_q = np.asarray(arm.real_q(ep))
            sim_q = np.asarray(arm.replay(ep))
            reports.append(tracking_error(real_q, sim_q, joint_names=JOINTS))
    return aggregate(reports)


# --------------------------------------------------------------------------
# Residual / contact events
# --------------------------------------------------------------------------

# Frames at the start of a replay that the detector ignores: the sim starts
# at rest at the real pose and needs a few frames to catch up with the
# command stream.
WARMUP_FRAMES = 6

_BASELINES: dict[tuple[str, str], Any] = {}
_BASELINES_LOCK = threading.Lock()


def baseline_episodes(ds: Dataset, lock: dict | None) -> list[Episode]:
    """Held-out episodes for the residual baseline: the lock's eval split,
    else the default eval split, else (tiny datasets) every episode."""
    from antares.calibrate import split_episodes

    wanted: list[int] = []
    if lock is not None:
        wanted = [int(i) for i in (lock.get("result") or {}).get("eval_episodes", [])]
    by_index = {ep.index: ep for ep in ds.episodes}
    eps = [by_index[i] for i in wanted if i in by_index]
    if not eps:
        _, eps = split_episodes(ds.episodes, 20, 10, seed=0)
    return eps or list(ds.episodes)


def get_baseline(ds: Dataset, lock_name: str | None):
    """Cached ``(ResidualBaseline, GripperThresholds, params)`` for a dataset + lock."""
    from antares.events import gripper_thresholds, residual_baseline

    key = (ds.repo_id, lock_name or "")
    with _BASELINES_LOCK:
        hit = _BASELINES.get(key)
    if hit is not None:
        return hit
    lock = load_lock_dict(lock_name) if lock_name else None
    params = params_from_dict(lock["params"]) if lock else default_params()
    g = ds.joint_names.index("gripper") if "gripper" in ds.joint_names else -1
    thr = gripper_thresholds(np.concatenate([ep.action[:, g] for ep in ds.episodes]))
    eps = baseline_episodes(ds, lock)
    cmap = command_map_from_lock(lock) if lock else None
    robot = dataset_robot(ds)
    arm = get_arm(robot)
    with arm_lock(robot), _command_map(arm, cmap):
        arm.apply(params)
        bl = residual_baseline(eps, arm, thr)
    out = (bl, thr, params, cmap)
    with _BASELINES_LOCK:
        _BASELINES.setdefault(key, out)
        return _BASELINES[key]


def _episode_residual(ds: Dataset, ep: Episode, lock_name: str | None, z: float, min_frames: int):
    from antares.events import detect_events, residuals, validate_episode

    bl, thr, params, cmap = get_baseline(ds, lock_name)
    robot = dataset_robot(ds)
    arm = get_arm(robot)
    with arm_lock(robot), _command_map(arm, cmap):
        arm.apply(params)
        real_q = np.asarray(arm.real_q(ep))
        sim_q = np.asarray(arm.replay(ep))
    res = residuals(real_q, sim_q)
    events = detect_events(res, bl, z=z, min_frames=min_frames, warmup=WARMUP_FRAMES)
    g = ds.joint_names.index("gripper")
    cmd = ep.action[:, g]
    val = validate_episode(res, events, cmd, thr, episode=ep.index)
    return bl, thr, params, res, events, cmd, val


def residual_episode(
    ds: Dataset, episode: int, lock_name: str | None = None, z: float = 4.0, min_frames: int = 5
) -> dict:
    """Residual, robust z, detected events and the gripper proxy label for one episode."""
    from antares.events import SCORED_JOINTS, gripper_transitions
    from antares.sim import JOINTS

    ep = find_episode(ds, episode)
    bl, thr, params, res, events, cmd, val = _episode_residual(ds, ep, lock_name, z, min_frames)
    grasps, releases = gripper_transitions(cmd, thr)
    return {
        "dataset": ds.repo_id,
        "episode": ep.index,
        "fps": ep.fps,
        "lock": lock_name,
        "params": params.to_dict(),
        "joints": list(JOINTS),
        "scored_joints": list(SCORED_JOINTS),
        "z_threshold": float(z),
        "min_frames": int(min_frames),
        "warmup_frames": WARMUP_FRAMES,
        "residual": _rounded(res),
        "z": _rounded(bl.z(res)),
        "events": [ev.to_dict() for ev in events],
        "gripper_cmd": _rounded(cmd),
        "gripper_thresholds": thr.to_dict(),
        "grasps": [int(i) for i in grasps],
        "releases": [int(i) for i in releases],
        "grasp_windows": [[int(s), int(e)] for s, e in val.grasp_windows],
        "baseline": bl.to_dict(),
        "validation": val.to_dict(),
    }


def events_summary(ds: Dataset, lock_name: str | None = None, z: float = 4.0, min_frames: int = 5) -> dict:
    """Detect events in every episode and score them against the gripper proxy label."""
    from antares.events import summarize_validation

    bl, thr, params, _ = get_baseline(ds, lock_name)
    held = set(bl.episodes)
    rows, vals = [], []
    for ep in ds.episodes:
        _, _, _, _, events, _, val = _episode_residual(ds, ep, lock_name, z, min_frames)
        vals.append(val)
        rows.append(
            {
                "episode": ep.index,
                "held_out": ep.index in held,
                "n_frames": len(ep),
                "events": [ev.to_dict() for ev in events],
                **{k: v for k, v in val.to_dict().items() if k not in ("episode", "n_frames", "grasp_windows")},
            }
        )
    return {
        "dataset": ds.repo_id,
        "lock": lock_name,
        "params": params.to_dict(),
        "z_threshold": float(z),
        "min_frames": int(min_frames),
        "warmup_frames": WARMUP_FRAMES,
        "gripper_thresholds": thr.to_dict(),
        "baseline": bl.to_dict(),
        "episodes": rows,
        "summary": summarize_validation(vals),
        "summary_held_out": summarize_validation([v for v in vals if v.episode in held]),
    }


# --------------------------------------------------------------------------
# Locks (JSON files in LOCK_DIR, one per name)
# --------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _lock_path(name: str, lock_dir: Path | str | None = None) -> Path:
    if not _NAME_RE.match(name) or ".." in name:
        raise HTTPException(status_code=400, detail="invalid lock name")
    return Path(lock_dir if lock_dir is not None else LOCK_DIR) / f"{name}{LOCK_SUFFIX}"


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


def load_lock_dict(name: str, lock_dir: Path | str | None = None) -> dict:
    p = _lock_path(name, lock_dir)
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
        "robot": lock.get("robot") or DEFAULT_ROBOT,
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


def build_lock(ds: Dataset, train: list, result, name: str | None, robot: str | None = None, **sensor):
    """Make a RobotLock for a finished calibration (digest covers the train split).

    ``robot`` is the registry name the fit ran on (default: the dataset's
    family). ``sensor`` carries ``command_map``/``dynamics_only`` for a
    sensor fit.
    """
    from antares.lock import RobotLock

    robot = check_robot(robot or dataset_robot(ds))
    return RobotLock.create(
        dataset=ds.repo_id,
        robot_type=ds.robot_type,
        train_episodes=train,
        result=result,
        name=name,
        robot=robot,
        **sensor,
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
    sim = simulate_for(dataset_robot(ds))
    try:
        scan = drift_scan(
            ds.episodes,
            lock,
            simulate=lock_simulate(lock.to_dict()),
            window=window,
            step=step,
            max_evaluations=max_evaluations,
            seed=seed,
            on_progress=on_progress,
            should_stop=should_stop,
            **({"simulate": sim} if sim is not None else {}),
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
    """A background fit streamed to the UI; see ``kind``."""

    id: str
    dataset: str
    kind: str = "calibrate"  # calibrate | dr | drift | diagnose
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
    robot: str | None = None,
    on_progress: Callable[[Any], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    fit_sensor: bool = False,
    sensor_mode: str = "sim",
    n_seeds: int = 1,
) -> tuple[Any, dict, Path]:
    """Calibrate on ``ds``, save a lock, return (CalibrationResult, lock dict, path).

    ``per_joint`` adds the per-joint refinement stage after the shared fit,
    with its own evaluation budget and (optionally) CMA-ES step size.
    ``n_seeds`` above 1 runs the shared stage from that many CMA-ES seeds
    and keeps the best held-out score (every dynamics fit, with
    ``fit_sensor``). ``robot`` overrides the simulator model (default: the
    dataset's family).

    With ``fit_sensor`` the arm's command map is fitted alongside the
    dynamics (see :mod:`antares.sensor`) and stored in the lock, together
    with the dynamics-only fit for comparison. ``sensor_mode`` is ``"sim"``
    (fit through the sim, alternating) or ``"kinematic"`` (pre-fit from
    still frames, then frozen).
    """
    from antares.calibrate import calibrate, split_episodes

    if n_train < 1:
        raise HTTPException(status_code=400, detail="n_train must be >= 1")
    if name is not None:
        _lock_path(name)  # validate the name before spending minutes fitting
    robot = check_robot(robot or dataset_robot(ds))
    train, eval_ = split_episodes(ds.episodes, n_train, n_eval, seed=seed)
    extra: dict[str, Any] = {}
    if per_joint:
        extra = {"per_joint": True, "per_joint_max_evaluations": int(per_joint_max_evaluations)}
        if per_joint_sigma0 is not None:
            extra["per_joint_sigma0"] = float(per_joint_sigma0)
    if n_seeds > 1:
        extra["n_seeds"] = int(n_seeds)
    if fit_sensor:
        from antares.sensor import calibrate_with_sensor

        if robot != DEFAULT_ROBOT:
            # The sensor fitter builds its own SO-101 arm by default; give it
            # one of this model instead.
            from antares.sensor import sensor_simulate

            extra["simulate"] = sensor_simulate(robot)
        out = calibrate_with_sensor(
            train,
            eval_,
            start=default_params(),
            max_evaluations=max_evaluations,
            seed=seed,
            on_progress=on_progress,
            should_stop=should_stop,
            mode=sensor_mode,
            **extra,
        )
        lock = build_lock(
            ds, train, out.result, name,
            robot=robot,
            command_map={**out.command_map.to_dict(), "mode": sensor_mode},
            dynamics_only=out.dynamics_only.to_dict(),
        )
        return out.result, lock.to_dict(), save_lock(lock)
    sim = simulate_for(robot)
    if sim is not None:
        extra["simulate"] = sim
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
    lock = build_lock(ds, train, result, name, robot=robot)
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
            robot=req.robot,
            on_progress=on_progress,
            should_stop=job.stop.is_set,
            fit_sensor=req.fit_sensor,
            sensor_mode=req.sensor_mode,
            n_seeds=req.n_seeds,
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

    sim = simulate_for(dataset_robot(ds))
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
        simulate=lock_simulate(lock),
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
    lock_dir: Path | str | None = None,
) -> dict:
    """Score every lock (and default params) on every dataset's eval split.

    Locks are read from ``lock_dir`` (default ``LOCK_DIR``); the fleet scan
    passes its own ``locks/fleet`` directory.
    """
    from antares.calibrate import split_episodes

    if n_train < 1:
        raise HTTPException(status_code=400, detail="n_train must be >= 1")
    cols = list(locks) + ["default"]
    lock_dicts = [load_lock_dict(n, lock_dir) for n in locks]
    col_params = [(params_from_dict(d["params"]), command_map_from_lock(d)) for d in lock_dicts]
    col_params.append((default_params(), None))

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
        for p, cmap in col_params:
            rep = score_episodes(p, eval_, cmap, robot=dataset_robot(ds))
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
    robot = dataset_robot(ds)
    arm = get_arm(robot)

    def simulate(params, ep):
        arm.apply(params)
        return np.asarray(arm.real_q(ep)), np.asarray(arm.replay(ep))

    try:
        with arm_lock(robot), _command_map(arm, command_map_from_lock(lock_d)):
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


def run_diagnose(
    ds: Dataset,
    lock: str | dict,
    *,
    last: int | None = None,
    max_evaluations: int = 240,
    seed: int = 0,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[int, float], None] | None = None,
):
    """Name what changed since ``lock`` was fitted, from the same episodes
    :func:`run_check` scores; returns an :class:`antares.diagnose.Diagnosis`.

    The refit runs on its own MuJoCo arm (with the lock's command map held
    fixed and the real trajectories cached), not under the shared arm's
    lock, so the other endpoints keep answering while it runs.
    """
    from antares.calibrate import default_simulate
    from antares.check import select_episodes
    from antares.diagnose import diagnose

    lock_d = lock if isinstance(lock, dict) else load_lock_dict(lock)
    simulate = lock_simulate(lock_d) or default_simulate()
    try:
        eps = select_episodes(ds.episodes, last)
        return diagnose(
            eps,
            lock_d,
            simulate=simulate,
            max_evaluations=max_evaluations,
            seed=seed,
            dataset=ds.repo_id,
            should_stop=should_stop,
            on_progress=on_progress,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _diagnose_worker(job: Job, req: "DiagnoseRequest") -> None:
    try:
        ds = get_dataset(req.dataset)

        def on_progress(evaluation: int, best_score: float) -> None:
            job.push({"type": "fit", "evaluation": evaluation, "best_score": best_score})

        d = run_diagnose(
            ds,
            req.lock,
            last=req.last,
            max_evaluations=req.max_evaluations,
            seed=req.seed,
            should_stop=job.stop.is_set,
            on_progress=on_progress,
        )
        d_dict = d.to_dict()
        job.push({"type": "done", "diagnosis": d_dict})
        job.finish("done", result=d_dict)
    except Exception as e:  # noqa: BLE001 - report to the client
        job_fail(job, e)


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
    cmap = command_map_from_lock(load_lock_dict(lock)) if lock else None
    robot = dataset_robot(ds)
    arm = get_arm(robot)
    with arm_lock(robot), _command_map(arm, cmap):
        vmax = max_joint_velocity(ds.episodes, arm)
        rep = compensate_episode(arm, ep, params, method=method, real_vmax_deg_s=vmax)
        comp_poses = np.asarray(arm.fk(rep.q_sim_comp))
        body_names = list(arm.body_names)
    return {
        "dataset": ds.repo_id,
        "episode": ep.index,
        "fps": rep.fps,
        "robot": robot,
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
    robot = dataset_robot(ds)
    arm = get_arm(robot)
    with arm_lock(robot), _command_map(arm, command_map_from_lock(lock_dict)):
        summary = compensate_episodes(arm, episodes, params, method=method)
    out = summary.to_dict()
    out.update(dataset=ds.repo_id, robot=robot, lock=lock_dict.get("name"), method=method, params=params.to_dict(),
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
    """The pinned demo datasets first, then everything loaded in this process
    (Hub ids, uploads, local directories), each with ``source``, ``warnings``,
    ``robot_family`` and the sim ``robot`` used for it."""
    with _DATASETS_LOCK:
        ids = list(DEMO_DATASETS) + [k for k in _DATASETS if k not in DEMO_DATASETS]
    return [dataset_summary(i) for i in ids]


@app.post("/api/datasets/load")
def datasets_load(dataset: str = Query(...)) -> dict:
    """Load any Hub id / upload / local directory. 400 with the verdict's
    reasons when the dataset cannot be replayed; warnings ride along in the
    summary when it can."""
    get_dataset(dataset)
    return dataset_summary(dataset)


@app.get("/api/datasets/validate")
def datasets_validate(dataset: str = Query(...)) -> dict:
    """Cheap pre-check (meta/info.json only, no data download): the Verdict."""
    with _DATASETS_LOCK:
        loaded = dataset in _DATASETS
    verdict = validate_id(dataset)
    return {
        "dataset": dataset,
        "loaded": loaded,
        "source": dataset_source(dataset),
        "robot": model_for_family(verdict.robot_family),
        **verdict.to_dict(),
    }


@app.post("/api/datasets/upload")
def datasets_upload(
    file: UploadFile = File(..., description="a .zip of a LeRobot dataset directory, or one .parquet"),
    info: UploadFile | None = File(None, description="meta/info.json, required with a bare parquet"),
    name: str | None = Form(None, description="dataset name; default: the file name"),
) -> dict:
    """Bring your own dataset. Registers it as ``upload/<name>`` and returns
    its summary plus the validation ``verdict``. 413 past 200 MB, 400 when
    the archive is not a LeRobot dataset or fails validation, 409 when the
    name is taken by different data."""
    body, digest, _ = _spool_upload(file)
    try:
        info_bytes = info.file.read() if info is not None else None
        return register_upload(body, filename=file.filename, name=name, info_json=info_bytes, sha256=digest)
    finally:
        body.unlink(missing_ok=True)


@app.delete("/api/datasets")
def datasets_delete(dataset: str = Query(...)) -> dict:
    """Forget a loaded dataset; an upload's extracted files are removed too."""
    return forget_dataset(dataset)


@app.get("/api/robots")
def robots_list() -> list[dict]:
    from antares.robots import REGISTRY

    return [r.to_dict() for r in REGISTRY.values()]


@app.get("/api/episodes")
def episodes(dataset: str = Query(...)) -> list[dict]:
    ds = get_dataset(dataset)
    return [
        {"index": e.index, "frames": len(e), "duration_s": round(e.duration, 3)}
        for e in ds.episodes
    ]


@app.get("/api/model")
def model(robot: str | None = Query(None, description="registry name, default so101")) -> dict:
    from antares.sim import model_geometry

    name = check_robot(robot)
    out = model_geometry(getattr(get_arm(name), "model", None))
    out["robot"] = name
    return out


_STL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.stl$", re.IGNORECASE)


@app.get("/api/model/mesh/{file}")
def model_mesh(file: str, robot: str | None = Query(None)):
    if "/" in file or "\\" in file or ".." in file or not _STL_RE.match(file):
        raise HTTPException(status_code=400, detail="mesh must be a plain *.stl basename")
    path = mesh_dir(robot) / file
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no mesh {file}")
    return FileResponse(path, media_type="model/stl", filename=file)


@app.get("/api/replay")
def replay(
    dataset: str = Query(...),
    episode: int = Query(...),
    lock: str | None = None,
    baseline: int = 0,
    robot: str | None = Query(None, description="override the sim model (default: the dataset's family)"),
) -> dict:
    ds = get_dataset(dataset)
    params = cmap = None
    if lock and not baseline:
        lock_d = load_lock_dict(lock)
        params = params_from_dict(lock_d["params"])
        cmap = command_map_from_lock(lock_d)
    out = replay_episode(ds, episode, params, cmap, robot=robot)
    out["lock"] = lock if (lock and not baseline) else None
    return out


class ReplayRequest(BaseModel):
    dataset: str
    episode: int
    # A value is a scalar or a per-joint list of 6, as SimParams.from_dict accepts.
    params: dict[str, float | list[float]]
    # Optional {"gain": [6], "bias": [6]}, as a lock stores it (extra keys such
    # as its "mode" are ignored); omitted means identity.
    command_map: dict[str, Any] | None = None
    # Sim model override (registry name); default: the dataset's family.
    robot: str | None = None


@app.post("/api/replay")
def replay_with_params(req: ReplayRequest) -> dict:
    """Replay with explicit params, e.g. the best fit so far of a running job."""
    ds = get_dataset(req.dataset)
    try:
        params = params_from_dict(req.params)
        cmap = command_map_from_lock({"command_map": req.command_map})
    except (TypeError, ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"bad params: {e}")
    if cmap is not None and cmap.gain.shape != (len(ds.joint_names),):
        raise HTTPException(
            status_code=400,
            detail=f"bad params: command_map needs {len(ds.joint_names)} gains and biases, got {cmap.gain.shape}",
        )
    out = replay_episode(ds, req.episode, params, cmap, robot=req.robot)
    out["lock"] = None
    return out


@app.get("/api/residual")
def residual(
    dataset: str = Query(...),
    episode: int = Query(...),
    lock: str | None = None,
    z: float = Query(4.0, gt=0),
    min_frames: int = Query(5, ge=1),
) -> dict:
    ds = get_dataset(dataset)
    return residual_episode(ds, episode, lock or None, z=z, min_frames=min_frames)


@app.get("/api/events")
def events(
    dataset: str = Query(...),
    lock: str | None = None,
    z: float = Query(4.0, gt=0),
    min_frames: int = Query(5, ge=1),
) -> dict:
    ds = get_dataset(dataset)
    return events_summary(ds, lock or None, z=z, min_frames=min_frames)


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
    # Best of N CMA-ES seeds on the shared stage (held-out score decides);
    # 1 keeps the demo fast, 3 is recommended for a lock worth keeping.
    n_seeds: int = Field(1, ge=1, le=10)
    name: str | None = None
    # Per-joint refinement after the shared fit; off by default so the live
    # demo stays at ~150 evaluations.
    per_joint: bool = False
    per_joint_max_evaluations: int = Field(600, ge=1)
    fit_sensor: bool = False
    sensor_mode: Literal["sim", "kinematic"] = "sim"
    # Sim model override (registry name); default: the dataset's family.
    robot: str | None = None


@app.post("/api/calibrate")
def calibrate_start(req: CalibrateRequest) -> dict:
    if req.name is not None:
        _lock_path(req.name)  # validate early
    check_robot(req.robot)
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


def load_fleet_dict() -> dict:
    """Contents of ``FLEET_DIR/fleet.json`` (the last fleet scan)."""
    p = Path(FLEET_DIR) / FLEET_REPORT_NAME
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"no fleet report at {p}; run `antares fleet scan`")
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"unreadable fleet report: {e}")


@app.get("/api/fleet")
def fleet_report() -> dict:
    return load_fleet_dict()


@app.get("/api/fleet/cross")
def fleet_cross() -> dict:
    cross = load_fleet_dict().get("cross")
    if not cross:
        raise HTTPException(status_code=404, detail="the fleet report has no cross matrix")
    return cross


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
    explain: bool = False
    explain_evaluations: int = Field(240, ge=10, le=2000)
    notify_url: str | None = None
    notify_when: Literal["drifted", "failure", "always"] = "drifted"
    notify_format: Literal["auto", "slack", "json"] = "auto"
    record: bool = True


@app.post("/api/check")
def check_drift(req: CheckRequest) -> dict:
    """Has the robot drifted from its lock? Always 200; read ``verdict``.

    With ``explain`` a drifted result also carries ``diagnosis``: the
    parameter changes a refit found and which of them explain the drift.

    With ``notify_url`` the verdict is posted to that webhook (Slack blocks
    for ``hooks.slack.com``, a generic JSON envelope otherwise) when
    ``notify_when`` says so; ``notified`` is True/False, or None when no
    URL was given, and ``notify_error`` says why a POST failed.

    Unless ``record`` is false the result (insufficient data included) is
    appended to ``LOCK_DIR/<lock>.history.jsonl``; see ``GET /api/history``.
    """
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
    out = result.to_dict()
    if req.record:
        from antares import history

        try:
            history.append(result, LOCK_DIR, name=req.lock)
        except OSError as e:  # the verdict stands; say why it was not recorded
            out["history_error"] = str(e)
    if req.explain and result.verdict == "drifted":
        try:
            out["diagnosis"] = run_diagnose(
                ds, req.lock, last=req.last, max_evaluations=req.explain_evaluations
            ).to_dict()
        except HTTPException as e:  # the verdict stands; say why it has no diagnosis
            out["diagnosis"] = None
            out["diagnosis_error"] = str(e.detail)
    out["notified"] = None
    out["notify_error"] = None
    if req.notify_url:
        from antares import notify

        try:
            out["notified"] = notify.notify(
                result, req.notify_url, when=req.notify_when, fmt=req.notify_format
            )
        except Exception as e:  # noqa: BLE001 - the verdict stands; say why nobody was told
            out["notified"] = False
            out["notify_error"] = str(e)
    return out


@app.get("/api/history")
def history_list() -> dict:
    """Every lock in the lock dir with its check trend and latest recorded check."""
    from antares import history

    return {"locks": history.summary_all(LOCK_DIR)}


@app.get("/api/history/{lock}")
def history_get(lock: str) -> dict:
    """The checks recorded for ``lock`` (oldest first) and their trend.

    404 when the lock does not exist; a lock with no checks yet has
    ``entries: []`` and a trend with ``n`` 0.
    """
    from antares import history

    if not _lock_path(lock).is_file():  # 400 on a bad name
        raise HTTPException(status_code=404, detail=f"no lock named {lock!r}")
    return history.report(lock, LOCK_DIR)


class DiagnoseRequest(BaseModel):
    dataset: str
    lock: str
    last: int | None = Field(None, ge=1)
    max_evaluations: int = Field(240, ge=10, le=2000)
    seed: int = 0


@app.post("/api/diagnose")
def diagnose_start(req: DiagnoseRequest) -> dict:
    """Start a cancellable diagnosis job (``POST /api/jobs/{id}/cancel``);
    its ``done`` event and ``result`` carry the diagnosis."""
    load_lock_dict(req.lock)  # 400/404 before a thread is spent
    job = start_job("diagnose", req.dataset, _diagnose_worker, req)
    return {
        "job_id": job.id,
        "status_url": f"/api/jobs/{job.id}",
        "events_url": f"/api/jobs/{job.id}/events",
    }


# Built frontend, mounted after the API routes so `antares serve` alone demos.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
