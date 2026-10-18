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
import base64
import importlib
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
    from antares.hub import download

    return json.loads(download(repo_id, "meta/info.json").read_text())


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
    """Registry name of the model that simulates ``ds``: the catalog's (or a
    ``--robot`` assignment's) pick for it, else its robot_type's family."""
    from antares.catalog import robot_for_dataset

    return robot_for_dataset(ds.repo_id) or model_for_dataset(ds.robot_type)


def _scoring(robot: str | None) -> dict[str, Any]:
    """``joint_names``/``exclude`` for scoring on ``robot``'s model."""
    from antares.calibrate import scoring_names

    return scoring_names(get_robot(check_robot(robot)))


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
        "robot": dataset_robot(ds) if ds is not None else model_for_dataset(repo_id),
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


def default_params(robot: str | None = None):
    """STS3215 defaults on the SO arms; other robots start from their own
    MJCF values when their profile says ``default_params="model"``."""
    from antares.sim import SimParams

    if robot is None or get_robot(check_robot(robot)).default_params != "model":
        return SimParams.default()
    return get_arm(robot).default_params(from_model=True)


def _gripper_column(ds: Dataset) -> int:
    """Index of the gripper whose command marks grasps (the first gripper
    on a two-arm rig; the last column if the robot names none)."""
    r = get_robot(dataset_robot(ds))
    for name in r.grippers:
        if name in ds.joint_names:
            return ds.joint_names.index(name)
    # Validation holds the columns to the profile's order, so a dataset
    # whose columns go by an alias (motor_5, ..._joint_6) is read by index.
    if r.grippers and len(ds.joint_names) == r.n_joints:
        return r.gripper_indices[0]
    return -1


def start_params(servo: str = "position", robot: str | None = None):
    """Where a fit starts: the robot's defaults (:func:`default_params`), or
    BAM's values for the STS3215 servo model."""
    from antares.sim import SimParams

    if servo not in ("position", "sts3215"):
        raise HTTPException(status_code=400, detail="servo must be one of position, sts3215")
    return SimParams.sts3215() if servo == "sts3215" else default_params(robot)


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


def unit_map_from_lock(lock: dict):
    """The UnitMap a lock's ``calibration`` block (the follower's LeRobot
    calibration, :mod:`antares.lerobot_calib`) was fitted with, or None for
    a lock on the model-derived map."""
    rec = lock.get("calibration")
    if not rec:
        return None
    from antares.lerobot_calib import unit_map_from_record

    model = getattr(get_arm(check_robot(lock.get("robot"))), "model", None)
    return unit_map_from_record(rec, model)


@contextmanager
def _command_map(arm, cmap, unit_map=None):
    """Use ``cmap`` (and ``unit_map``, a lock's calibration-derived map) on
    the shared ``arm`` for the block, then restore identity and the arm's
    own map so callers that never set one keep the plain mapping."""
    if cmap is None and unit_map is None:
        yield
        return
    from antares.sim import CommandMap

    prev_unit_map = getattr(arm, "unit_map", None)
    if cmap is not None:
        arm.command_map = cmap
    if unit_map is not None:
        arm.unit_map = unit_map
    try:
        yield
    finally:
        arm.command_map = CommandMap.identity(getattr(arm, "n_joints", None) or 6)
        if unit_map is not None:
            arm.unit_map = prev_unit_map


def lock_simulate(lock: dict):
    """A SimulateFn holding the lock's command map fixed on the lock's robot
    model, or None (the callee's default: identity map, SO-101) when the lock
    has neither. For refits warm-started from a lock: the dynamics move, the
    sensor calibration and the model do not."""
    cmap = command_map_from_lock(lock)
    unit_map = unit_map_from_lock(lock)
    robot = check_robot(lock.get("robot"))
    if cmap is None and unit_map is None and robot == DEFAULT_ROBOT:
        return None
    from antares.calibrate import default_simulate

    return default_simulate(cmap, robot=robot, **({"unit_map": unit_map} if unit_map is not None else {}))


def mesh_dir(robot: str | None = None) -> Path:
    if MESH_DIR is not None:
        return Path(MESH_DIR)
    return Path(get_robot(check_robot(robot)).mesh_dir)


def _rounded(a: np.ndarray) -> list:
    return np.round(np.asarray(a, dtype=np.float64), ROUND).tolist()


def replay_episode(
    ds: Dataset, episode: int, params=None, command_map=None, robot: str | None = None, unit_map=None
) -> dict:
    """Real FK + sim replay for one episode. Angles in degrees, poses (T,nbody,7).

    ``command_map`` (a CommandMap, or None for identity) corrects the commands.
    ``robot`` overrides the model (default: the one for the dataset's family).
    ``unit_map`` (a UnitMap, or None for the model's) is a lock's
    calibration-derived map (``unit_map_from_lock``).
    """
    from antares.gap import tracking_error

    ep = find_episode(ds, episode)
    robot = check_robot(robot or dataset_robot(ds))
    if params is None:
        params = default_params(robot)
    arm = get_arm(robot)
    with arm_lock(robot), _command_map(arm, command_map, unit_map):
        arm.apply(params)
        real_q = np.asarray(arm.real_q(ep))
        sim_q = np.asarray(arm.replay(ep))
        real_poses = np.asarray(arm.fk(real_q))
        sim_poses = np.asarray(arm.fk(sim_q))
        body_names = list(arm.body_names)
    gap = tracking_error(real_q, sim_q, **_scoring(robot))
    return {
        "dataset": ds.repo_id,
        "episode": ep.index,
        "fps": ep.fps,
        "robot": robot,
        "joints": list(get_robot(robot).joints),
        "body_names": body_names,
        "params": params.to_dict(),
        "command_map": None if command_map is None else command_map.to_dict(),
        "unit_map": None if unit_map is None else unit_map.to_dict(),
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

    arm = get_arm(robot)
    names = _scoring(robot)
    reports = []
    with arm_lock(robot), _command_map(arm, command_map):
        arm.apply(params)
        for ep in episodes:
            real_q = np.asarray(arm.real_q(ep))
            sim_q = np.asarray(arm.replay(ep))
            reports.append(tracking_error(real_q, sim_q, **names))
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
    params = params_from_dict(lock["params"]) if lock else default_params(dataset_robot(ds))
    g = _gripper_column(ds)
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
    r = get_robot(robot)
    scored = r.scored_joints
    events = detect_events(res, bl, z=z, min_frames=min_frames, warmup=WARMUP_FRAMES, joints=scored)
    cmd = ep.action[:, _gripper_column(ds)]
    val = validate_episode(res, events, cmd, thr, episode=ep.index, joints=scored, joint_names=r.joints)
    return bl, thr, params, res, events, cmd, val


def residual_episode(
    ds: Dataset, episode: int, lock_name: str | None = None, z: float = 4.0, min_frames: int = 5
) -> dict:
    """Residual, robust z, detected events and the gripper proxy label for one episode."""
    from antares.events import gripper_transitions

    ep = find_episode(ds, episode)
    bl, thr, params, res, events, cmd, val = _episode_residual(ds, ep, lock_name, z, min_frames)
    grasps, releases = gripper_transitions(cmd, thr)
    r = get_robot(dataset_robot(ds))
    return {
        "dataset": ds.repo_id,
        "episode": ep.index,
        "fps": ep.fps,
        "lock": lock_name,
        "params": params.to_dict(),
        "joints": list(r.joints),
        "scored_joints": list(r.scored_joints),
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
    name = resolve_lock_ref(name, lock_dir)  # @<robot> -> its active lock
    p = _lock_path(name, lock_dir)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"no lock named {name!r}")
    return json.loads(p.read_text())


# Robot registry (``LOCK_DIR/registry.json``): one id per physical robot, a
# lineage of its locks and the active one. ``@<robot>`` stands for the
# active lock wherever a lock name is accepted; every read goes through
# ``resolve_lock_ref`` so an unregistered lock name passes through unchanged.


def registry_path(lock_dir: Path | str | None = None) -> Path:
    from antares.registry import registry_path as _rp

    return _rp(lock_dir if lock_dir is not None else LOCK_DIR)


def load_registry(lock_dir: Path | str | None = None):
    """The registry in ``lock_dir`` (default LOCK_DIR); a missing file is empty."""
    from antares.registry import Registry, RegistryError

    try:
        return Registry.load(registry_path(lock_dir))
    except RegistryError as e:
        raise HTTPException(status_code=500, detail=str(e))


def resolve_lock_ref(ref: str, lock_dir: Path | str | None = None) -> str:
    """A lock name (or path) passes through; ``@<robot>`` becomes the robot's
    active lock name, 404 when the robot is unknown or has no active lock."""
    from antares.registry import RegistryError, UnknownRobot, is_robot_ref

    if not is_robot_ref(ref):
        return ref
    try:
        return load_registry(lock_dir).resolve(ref)
    except UnknownRobot as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))


def register_lock(
    robot_id: str,
    lock: Any,
    *,
    activate: bool | None = None,
    note: str | None = None,
    max_regression: float | None = None,
) -> dict:
    """Register ``lock`` (a name, dict or RobotLock) for ``robot_id`` in
    LOCK_DIR's registry, save, and return the ``RegistrationResult`` dict."""
    from antares.registry import DEFAULT_MAX_REGRESSION, RegistryError

    reg = load_registry()
    try:
        res = reg.register(
            robot_id,
            lock,
            activate=activate,
            note=note,
            max_regression=DEFAULT_MAX_REGRESSION if max_regression is None else max_regression,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    reg.save()
    return res.to_dict()


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


def export_lock_zip(name: str, targets: list[str] | tuple[str, ...] = (), fps: float | None = None) -> bytes:
    """``<name>/`` with the MJCF, sidecar, scene.xml and STL meshes, zipped,
    plus each extra export target's files (``antares.export_targets.TARGETS``)."""
    from antares.export import export_zip

    return export_zip(load_lock(name), targets=targets, fps=fps, lock_dir=LOCK_DIR)


def export_lock_target(name: str, target: str, fps: float | None = None):
    """One non-MJCF export target rendered for ``name`` (a ``TargetExport``);
    DR ranges are read from ``LOCK_DIR/<name>.dr.json`` when present."""
    from antares.export_targets import TargetExportError, render

    try:
        return render(load_lock(name), target, fps=fps, lock_dir=LOCK_DIR)
    except TargetExportError as e:
        raise HTTPException(status_code=400, detail=str(e))


def parse_export_targets(targets: str | None) -> list[str]:
    """``?targets=`` of the zip route: ``all``, a comma list of targets, or empty."""
    from antares.export_targets import TARGETS

    if not targets:
        return []
    if targets == "all":
        return list(TARGETS)
    wanted = [t.strip() for t in targets.split(",") if t.strip()]
    unknown = [t for t in wanted if t not in TARGETS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown export targets {unknown}; known: {list(TARGETS)} or all")
    return wanted


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


def save_tuned_lock(
    ds: Dataset,
    params,
    name: str,
    *,
    from_lock: str | None = None,
    n_train: int = 20,
    n_eval: int = 10,
    seed: int = 0,
) -> dict:
    """Save hand-picked ``params`` (the viewer's tuning panel) as lock ``name``.

    Scored like a calibration on the same train/eval split, against the
    robot's default parameters, so the lock's numbers compare with fitted
    ones. Tuned from a saved lock (``from_lock``), the new lock keeps that
    lock's model, sensor command map, LeRobot calibration and camera/scene
    fits: only the physics changed. Refuses to overwrite an existing lock.
    """
    import math
    import os
    import time

    from antares.calibrate import CalibrationResult, _safe_score, default_simulate, split_episodes

    path = _lock_path(name)
    if path.exists():
        raise HTTPException(status_code=409, detail=f"a lock named {name!r} already exists")
    src = load_lock_dict(from_lock) if from_lock else None
    # The lock is for this dataset's robot; a source lock fitted for another
    # model, or with a command map of another width, cannot lend its setup.
    robot = check_robot(dataset_robot(ds))
    if src is not None:
        src_robot = check_robot(src.get("robot") or robot)
        if src_robot != robot:
            raise HTTPException(
                status_code=400,
                detail=f"lock {from_lock!r} was fitted for robot {src_robot!r}; {ds.repo_id} simulates on {robot!r}",
            )
        cmap = command_map_from_lock(src)
        if cmap is not None and cmap.gain.shape != (len(ds.joint_names),):
            raise HTTPException(
                status_code=400,
                detail=f"lock {from_lock!r} has a command map for {cmap.gain.shape[0]} joints; "
                f"{ds.repo_id} has {len(ds.joint_names)}",
            )
    sim = (lock_simulate(src) if src else None) or default_simulate(robot=robot)
    try:
        train, eval_ = split_episodes(ds.episodes, n_train, n_eval, seed=seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    t0 = time.perf_counter()
    # The same start a calibration of this lock would use (BAM's values for
    # a servo-model lock), so the saved scores compare with fitted ones.
    start = start_params(str(getattr(params, "servo", None) or "position"), robot)

    def scored(p, episodes):
        report, value = _safe_score(p, episodes, sim)
        if report is None or not math.isfinite(value) or not math.isfinite(float(report.overall_deg)):
            raise HTTPException(
                status_code=400,
                detail="these parameters make the simulation unstable; move the sliders back toward the lock",
            )
        return report

    before_train, after_train = scored(start, train), scored(params, train)
    if eval_:
        before_eval, after_eval = scored(start, eval_), scored(params, eval_)
    else:  # no held-out episodes: the training reports stand in, as calibrate() does
        before_eval, after_eval = before_train, after_train
    result = CalibrationResult(
        params=params,
        baseline=start,
        train_score=float(after_train.overall_deg),
        eval_score=float(after_eval.overall_deg),
        baseline_train_score=float(before_train.overall_deg),
        baseline_eval_score=float(before_eval.overall_deg),
        per_joint_before={k: float(v) for k, v in before_eval.per_joint_deg.items()},
        per_joint_after={k: float(v) for k, v in after_eval.per_joint_deg.items()},
        history=[],
        n_evaluations=0,
        elapsed_s=time.perf_counter() - t0,
        train_episodes=[ep.index for ep in train],
        eval_episodes=[ep.index for ep in eval_],
        space="per_joint" if params.has_per_joint else "shared",
        # No search ran: the one "manual" stage records the hand-set values.
        stages=[{"name": "manual", "n_evaluations": 0, "train_score": float(after_train.overall_deg),
                 "params": params.to_dict()}],
    )
    # Only the fixed setup carries over: the sensor map and the LeRobot
    # calibration. The source's dynamics-only fit describes a search that
    # never ran for these values.
    carried = {k: src[k] for k in ("command_map", "calibration") if src and src.get(k) is not None}
    lock = build_lock(ds, train, result, name, robot=robot, **carried)
    if src:
        lock = lock.replace(cameras=dict(src.get("cameras") or {}), scene=src.get("scene"), objects=src.get("objects"))
    # Refuse to overwrite: the file is claimed exclusively between the check
    # above (before the scoring passes) and the write.
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise HTTPException(status_code=409, detail=f"a lock named {name!r} was saved meanwhile")
    try:
        save_lock(lock)
        if from_lock:
            _copy_fit_sidecars(from_lock, name)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return lock.to_dict()


def _copy_fit_sidecars(from_lock: str, name: str) -> None:
    """The scene and object fits live next to the lock in ``<lock>.scene.json``
    / ``<lock>.objects.json`` and every reader looks them up by lock name, so
    a lock tuned from another gets its own copies (naming the new lock)."""
    import json as _json

    from antares.objects import objects_path
    from antares.scene import scene_path

    for path_of in (scene_path, objects_path):
        src_path = path_of(from_lock, LOCK_DIR)
        if not src_path.is_file():
            continue
        try:
            d = _json.loads(src_path.read_text())
        except ValueError:
            continue
        if isinstance(d, dict) and "lock" in d:
            d["lock"] = name
        path_of(name, LOCK_DIR).write_text(_json.dumps(d, indent=2) + "\n")


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

    lock_name = resolve_lock_ref(lock_name)
    lock = load_lock(lock_name)
    path = _drift_path(lock_name)
    # The lock's own sim (its robot, sensor and unit maps), else the dataset's robot.
    sim = lock_simulate(lock.to_dict()) or simulate_for(dataset_robot(ds))
    try:
        scan = drift_scan(
            ds.episodes,
            lock,
            simulate=sim,
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
        if getattr(job, "registration", None) is not None:  # calibrate with robot_id
            out["registration"] = job.registration
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
    fit_gravity: bool = False,
    fit_deadband: bool = False,
    robot: str | None = None,
    on_progress: Callable[[Any], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    fit_sensor: bool = False,
    sensor_mode: str = "sim",
    n_seeds: int = 1,
    factors: list[str] | None = None,
    per_joint_factors: list[str] | None = None,
    calibration_path: str | Path | None = None,
    robot_id: str | None = None,
    robot_type: str = "so101_follower",
    servo: str = "position",
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

    ``factors``/``per_joint_factors`` choose which registered factors the
    shared and per-joint stages fit (see :mod:`antares.factors`); ``None``
    keeps the built-in lists.
    ``fit_gravity`` adds ``gravity_scale`` to the shared stage and
    ``fit_deadband`` the servo ``deadband`` to both stages. ``servo``
    ``"sts3215"`` starts from BAM's STS3215 servo model
    (:meth:`antares.sim.SimParams.sts3215`) and fits its per-arm terms
    instead of the position actuator's gains.
    ``calibration_path`` (a LeRobot calibration file) or ``robot_id`` (the
    ``--robot.id`` LeRobot saved it under, with ``robot_type``) replaces the
    model-derived unit map with the follower's own tick-to-angle map for
    the replay and records it in the lock's ``calibration`` block
    (:mod:`antares.lerobot_calib`). With ``fit_sensor`` the map fit starts
    from that calibration's prior.
    """
    from antares.calibrate import calibrate, split_episodes

    if n_train < 1:
        raise HTTPException(status_code=400, detail="n_train must be >= 1")
    if name is not None:
        _lock_path(name)  # validate the name before spending minutes fitting
    robot = check_robot(robot or dataset_robot(ds))
    unit_map = cal_record = cal_prior = None
    # An explicit file must exist; a bare robot id is looked up opportunistically.
    calib = resolve_calibration(calibration_path, robot_id, robot_type, strict=calibration_path is not None)
    if calib is not None:
        from antares.lerobot_calib import calibration_record, command_map_prior, unit_map_from_calibration

        calib_d, cal_path = calib
        model = get_arm(robot).model
        unit_map = unit_map_from_calibration(calib_d, model)
        cal_prior = command_map_prior(calib_d, model)
        cal_record = calibration_record(calib_d, model, path=cal_path, robot_id=robot_id, robot_type=robot_type)
    start = start_params(servo, robot)
    train, eval_ = split_episodes(ds.episodes, n_train, n_eval, seed=seed)
    extra: dict[str, Any] = {}
    if per_joint:
        extra = {"per_joint": True, "per_joint_max_evaluations": int(per_joint_max_evaluations)}
        if per_joint_sigma0 is not None:
            extra["per_joint_sigma0"] = float(per_joint_sigma0)
    if fit_gravity:
        extra["fit_gravity"] = True
    if fit_deadband:
        extra["fit_deadband"] = True
    if n_seeds > 1:
        extra["n_seeds"] = int(n_seeds)
    try:
        if factors is not None or per_joint_factors is not None:
            # +/- edits apply to what this fit searches by default (the servo
            # model's fields on a servo start).
            from antares.calibrate import default_factor_names

            servo_fit = start.uses_servo_model
        if factors is not None:
            extra["fit_factors"] = _factor_names(factors, "shared", default_factor_names(servo_fit)[0])
        if per_joint_factors is not None:
            extra["per_joint_factors"] = _factor_names(
                per_joint_factors, "per_joint", default_factor_names(servo_fit, extra.get("fit_factors"))[1]
            )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e).strip("'\"")) from e
    if fit_sensor:
        from antares.sensor import calibrate_with_sensor

        if robot != DEFAULT_ROBOT or unit_map is not None:
            # The sensor fitter builds its own SO-101 arm by default; give it
            # one of this model (and of this arm's unit map) instead.
            from antares.sensor import sensor_simulate

            extra["simulate"] = sensor_simulate(robot, unit_map=unit_map)
        if cal_prior is not None:
            extra["prior"] = cal_prior
        out = calibrate_with_sensor(
            train,
            eval_,
            start=start,
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
            **({"calibration": cal_record} if cal_record is not None else {}),
        )
        return out.result, lock.to_dict(), save_lock(lock)
    sim = simulate_for(robot)
    if unit_map is not None:
        from antares.calibrate import default_simulate

        sim = default_simulate(robot=robot, unit_map=unit_map)
    if sim is not None:
        extra["simulate"] = sim
    result = calibrate(
        train,
        eval_,
        start=start,
        max_evaluations=max_evaluations,
        seed=seed,
        on_progress=on_progress,
        should_stop=should_stop,
        **extra,
    )
    lock = build_lock(ds, train, result, name, robot=robot,
                      **({"calibration": cal_record} if cal_record is not None else {}))
    path = save_lock(lock)
    return result, lock.to_dict(), path


def _factor_names(selection: list[str], stage: str, defaults=None) -> list[str] | None:
    """Resolve a factor selection (names, or ``+x``/``-y`` edits of
    ``defaults``, what the fit searches without one) into the list calibrate
    takes (``None``: defaults)."""
    from antares import factors

    if not selection:
        raise ValueError(f"{stage} factors: an empty list leaves nothing to fit; omit it for the defaults")
    out = factors.parse_selection(",".join(selection), stage, defaults=defaults)
    return None if out is None else list(out)
def resolve_calibration(
    calibration_path, robot_id: str | None, robot_type: str = "so101_follower", *, strict: bool = True
):
    """``(calibration, path)`` from an explicit LeRobot calibration file or from
    the robot id LeRobot saved it under; None when neither is given. 404 when
    the id has no file, 400 when the file does not parse.

    With ``strict=False`` an id without a file is not an error: None comes
    back and the caller proceeds on the model's unit map. ``calibrate`` uses
    that, because ``robot_id`` also names the robot in the registry and a
    registered robot need not have a LeRobot calibration on this machine."""
    if calibration_path is None and not robot_id:
        return None
    from antares.lerobot_calib import calibration_dirs, find_calibration, load_calibration

    path = Path(calibration_path) if calibration_path is not None else None
    if path is None:
        try:
            path = find_calibration(robot_id, robot_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if path is None and not strict:
            return None
        if path is None:
            looked = ", ".join(str(d) for d in calibration_dirs(robot_type))
            raise HTTPException(
                status_code=404,
                detail=f"no LeRobot calibration named {robot_id!r} for {robot_type} (looked in {looked})",
            )
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no calibration file {path}")
    try:
        return load_calibration(path), path
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{path}: {e}")


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
            factors=req.factors,
            per_joint_factors=req.per_joint_factors,
            fit_gravity=req.fit_gravity,
            fit_deadband=req.fit_deadband,
            servo=req.servo,
            calibration_path=req.calibration_path,
            robot_id=req.robot_id,
            robot_type=req.robot_type,
        )
        result_d = result.to_dict()
        extra: dict[str, Any] = {}
        if req.robot_id:
            extra["registration"] = register_lock(req.robot_id, lock, activate=req.activate)
        job.push({"type": "done", "lock": lock, "result": result_d, **extra})
        job.finish("done", result=result_d, lock=lock, **extra)
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
    lock_name = resolve_lock_ref(lock_name)
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
    from antares.calibrate import describe_simulate
    from antares.check import check

    lock_d = lock if isinstance(lock, dict) else load_lock_dict(lock)
    robot = dataset_robot(ds)
    arm = get_arm(robot)

    def simulate(params, ep):
        arm.apply(params)
        return np.asarray(arm.real_q(ep)), np.asarray(arm.replay(ep))

    # Tagged with the robot so scoring uses its joints, not the SO arms' six.
    describe_simulate(simulate, arm)
    try:
        with arm_lock(robot), _command_map(arm, command_map_from_lock(lock_d), unit_map_from_lock(lock_d)):
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
    max_evaluations: int = 300,
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

    _check_method(method)
    ep = find_episode(ds, episode)
    robot = dataset_robot(ds)
    if params is None:
        params = default_params(robot)
    cmap = command_map_from_lock(load_lock_dict(lock)) if lock else None
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
        "joints": list(get_robot(robot).joints),
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


@app.get("/api/datasets/calibration-check")
def datasets_calibration_check(
    dataset: str = Query(...),
    calibration_path: str | None = Query(None, description="a LeRobot calibration file"),
    robot_id: str | None = Query(None, description="the --robot.id LeRobot saved the calibration under"),
    robot_type: str = Query("so101_follower"),
    robot: str | None = Query(None, description="override the sim model (default: the dataset's family)"),
) -> dict:
    """The model's assumed unit map against this arm's LeRobot calibration,
    joint by joint (degrees at -100/0/+100, flagged over 2 deg). 400 when
    neither ``calibration_path`` nor ``robot_id`` is given."""
    from antares.lerobot_calib import calibration_check, file_sha256

    calib = resolve_calibration(calibration_path, robot_id, robot_type)
    if calib is None:
        raise HTTPException(status_code=400, detail="give calibration_path or robot_id")
    calib_d, path = calib
    ds = get_dataset(dataset)
    robot = check_robot(robot or dataset_robot(ds))
    out = calibration_check(calib_d, get_arm(robot).model)
    out.update(dataset=ds.repo_id, robot=robot, source=str(path), sha256=file_sha256(path),
               robot_id=robot_id, robot_type=robot_type)
    return out


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


@app.get("/api/factors")
def factors_list() -> dict:
    """Every registered factor (what it explains, bounds, data it needs,
    whether the fleet showed it measurable) and the effects not modelled."""
    from antares import factors

    return factors.describe()


@app.get("/api/datasets/catalog")
def datasets_catalog() -> list[dict]:
    """Public datasets of robots other than the SO arms, with the model each
    replays on."""
    from antares.catalog import entries

    return entries()


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
    r = get_robot(name)
    arm = get_arm(name)
    if r.menagerie is None:
        out = model_geometry(getattr(arm, "model", None))
    else:
        from antares.sim import mesh_files

        out = model_geometry(arm.model, mesh_files(arm.model_path))
    out["robot"] = name
    return out


_STL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.stl$", re.IGNORECASE)
_MESH_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*\.(stl|obj)$", re.IGNORECASE)
_MESH_TYPES = {".stl": "model/stl", ".obj": "model/obj"}


@app.get("/api/model/mesh/{file:path}")
def model_mesh(file: str, robot: str | None = Query(None)):
    """A mesh of the robot's model: a plain ``*.stl`` name in the SO arms'
    assets, or a path under a fetched model's directory (``*.stl``/``*.obj``)."""
    fetched = robot is not None and get_robot(check_robot(robot)).menagerie is not None
    if not fetched:
        if "/" in file or "\\" in file or ".." in file or not _STL_RE.match(file):
            raise HTTPException(status_code=400, detail="mesh must be a plain *.stl basename")
        path = mesh_dir(robot) / file
    else:
        if ".." in file.split("/") or "\\" in file or not _MESH_PATH_RE.match(file):
            raise HTTPException(status_code=400, detail="mesh must be a relative *.stl or *.obj path")
        root = get_robot(check_robot(robot)).model_path.parent.resolve()
        path = (root / file).resolve()
        if not path.is_relative_to(root):
            raise HTTPException(status_code=400, detail="mesh must stay inside the model directory")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no mesh {file}")
    return FileResponse(path, media_type=_MESH_TYPES[path.suffix.lower()], filename=path.name)


@app.get("/api/replay")
def replay(
    dataset: str = Query(...),
    episode: int = Query(...),
    lock: str | None = None,
    baseline: int = 0,
    robot: str | None = Query(None, description="override the sim model (default: the dataset's family)"),
) -> dict:
    ds = get_dataset(dataset)
    params = cmap = unit_map = None
    if lock and not baseline:
        lock_d = load_lock_dict(lock)
        params = params_from_dict(lock_d["params"])
        cmap = command_map_from_lock(lock_d)
        unit_map = unit_map_from_lock(lock_d)
    out = replay_episode(ds, episode, params, cmap, robot=robot, unit_map=unit_map)
    out["lock"] = lock if (lock and not baseline) else None
    return out


class ReplayRequest(BaseModel):
    dataset: str
    episode: int
    # A value is a scalar or a per-joint list of 6, as SimParams.from_dict
    # accepts; ``servo`` (the actuator model's name, written by a servo-model
    # fit's to_dict) is a string. Anything else is rejected by params_from_dict.
    params: dict[str, float | list[float] | str]
    # Optional {"gain": [6], "bias": [6]}, as a lock stores it (extra keys such
    # as its "mode" are ignored); omitted means identity.
    command_map: dict[str, Any] | None = None
    # A saved lock whose command map and calibration unit map the replay
    # keeps (e.g. params tuned by hand from that lock); an explicit
    # ``command_map`` still wins.
    lock: str | None = None
    # Sim model override (registry name); default: the dataset's family.
    robot: str | None = None
    # Factors to fit (names, or "+x"/"-y" edits of the defaults); see
    # GET /api/factors. None keeps the built-in lists.
    factors: list[str] | None = None
    per_joint_factors: list[str] | None = None


@app.post("/api/replay")
def replay_with_params(req: ReplayRequest) -> dict:
    """Replay with explicit params, e.g. the best fit so far of a running job."""
    ds = get_dataset(req.dataset)
    lock_d = load_lock_dict(req.lock) if req.lock else None
    unit_map = unit_map_from_lock(lock_d) if lock_d else None
    try:
        params = params_from_dict(req.params)
        if req.command_map is None and lock_d is not None:
            cmap = command_map_from_lock(lock_d)
        else:
            cmap = command_map_from_lock({"command_map": req.command_map})
    except (TypeError, ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"bad params: {e}")
    if cmap is not None and cmap.gain.shape != (len(ds.joint_names),):
        raise HTTPException(
            status_code=400,
            detail=f"bad params: command_map needs {len(ds.joint_names)} gains and biases, got {cmap.gain.shape}",
        )
    out = replay_episode(ds, req.episode, params, cmap, robot=req.robot, unit_map=unit_map)
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
    # Optional terms: the arm's gravity load (shared stage) and the servo
    # deadband (both stages); see run_calibration.
    fit_gravity: bool = False
    fit_deadband: bool = False
    # Actuator model: the position actuator, or BAM's STS3215 servo model.
    servo: Literal["position", "sts3215"] = "position"
    # Sim model override (registry name); default: the dataset's family.
    robot: str | None = None
    # Factors to fit (names, or "+x"/"-y" edits of the defaults); see
    # GET /api/factors. None keeps the built-in lists.
    factors: list[str] | None = None
    per_joint_factors: list[str] | None = None
    # The physical robot's id (LeRobot's --robot.id, e.g. "arm_a"). Two effects:
    # the follower's LeRobot calibration saved under it is used as the unit map
    # when such a file exists, and the saved lock is registered for it in
    # locks/registry.json and activated by the registry's rule.
    robot_id: str | None = None
    # A calibration file given explicitly (takes precedence over the lookup).
    calibration_path: str | None = None
    robot_type: str = "so101_follower"
    # With robot_id: None applies the registry rule, True/False forces it.
    activate: bool | None = None


@app.post("/api/calibrate")
def calibrate_start(req: CalibrateRequest) -> dict:
    if req.name is not None:
        if _lock_path(req.name).exists():  # validates the name too
            raise HTTPException(
                status_code=409,
                detail=f"a lock named {req.name!r} already exists; pick another name or delete it first",
            )
    if req.robot_id is not None:
        from antares.registry import RegistryError, validate_robot_id

        try:
            validate_robot_id(req.robot_id)
        except RegistryError as e:
            raise HTTPException(status_code=400, detail=str(e))
    check_robot(req.robot)
    try:  # a bad or empty factor selection is a usage error, not a failed job
        if req.factors is not None:
            _factor_names(req.factors, "shared")
        if req.per_joint_factors is not None:
            _factor_names(req.per_joint_factors, "per_joint")
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e).strip("'\""))
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
    lock_name = resolve_lock_ref(req.lock or _slug(req.dataset))
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
def lock_export(
    name: str,
    format: Literal["mjcf", "isaaclab", "maniskill", "mjlab", "json"] = Query("mjcf"),
    fps: float | None = Query(None, gt=0, description="dataset fps for latency seconds"),
) -> Response:
    """Calibrated MJCF as a download (``format=mjcf``, the default; ``meshdir``
    is the absolute package assets dir; use ``export.zip`` for a self-contained
    bundle), or the main file of another export target: the Isaac Lab /
    ManiSkill / mjlab Python config or the neutral ``.params.json``."""
    name = resolve_lock_ref(name)
    if format == "mjcf":
        return Response(
            export_lock_xml(name),
            media_type="application/xml",
            headers=_attachment(f"{name}.xml"),
        )
    from antares.export_targets import media_type

    export = export_lock_target(name, format, fps=fps)
    return Response(
        export.files[export.main],
        media_type=media_type(export.main),
        headers=_attachment(export.main),
    )


@app.get("/api/locks/{name}/export.zip")
def lock_export_zip(
    name: str,
    targets: str | None = Query(None, description="all, or a comma list of isaaclab,maniskill,mjlab,json"),
    fps: float | None = Query(None, gt=0),
) -> Response:
    """Zip of ``<name>/<name>.xml``, ``<name>.antares.json``, ``scene.xml``
    and ``assets/*.stl``; with ``?targets=all`` also every other export
    target's files."""
    name = resolve_lock_ref(name)
    return Response(
        export_lock_zip(name, targets=parse_export_targets(targets), fps=fps),
        media_type="application/zip",
        headers=_attachment(f"{name}.zip"),
    )


class SaveLockRequest(BaseModel):
    dataset: str
    name: str
    # As POST /api/replay takes them: scalars or per-joint lists, plus "servo".
    params: dict[str, float | list[float] | str]
    # The saved lock the values were tuned from; its sensor map, calibration
    # and camera fits carry over.
    from_lock: str | None = None
    n_train: int = Field(20, ge=1)
    n_eval: int = Field(10, ge=0)
    seed: int = 0


@app.post("/api/locks")
def lock_save(req: SaveLockRequest) -> dict:
    """Save hand-tuned parameters as a new lock; returns its summary."""
    ds = get_dataset(req.dataset)
    try:
        params = params_from_dict(req.params)
    except (TypeError, ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"bad params: {e}")
    return lock_summary(save_tuned_lock(
        ds, params, req.name, from_lock=req.from_lock, n_train=req.n_train, n_eval=req.n_eval, seed=req.seed
    ))


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


# --------------------------------------------------------------------------
# Fleet dashboard: one row per lock in LOCK_DIR, with its check history
# --------------------------------------------------------------------------

# Joints that count for "worst joint": the gripper is a different mechanism
# and is excluded from the tracking score too.
_DASHBOARD_BODY_JOINTS_EXCLUDE = ("gripper",)


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _worst_body_joint(per_joint: dict | None) -> tuple[str | None, float | None]:
    """``(joint, deg)`` with the largest error among the body joints."""
    worst: tuple[str, float] | None = None
    for k, v in (per_joint or {}).items():
        if k in _DASHBOARD_BODY_JOINTS_EXCLUDE:
            continue
        f = _finite(v)
        if f is None:
            continue
        if worst is None or f > worst[1]:
            worst = (str(k), f)
    return worst if worst is not None else (None, None)


def _drift_verdicts() -> dict[str, str | None]:
    """lock name -> verdict for every ``<lock>.drift.json`` in LOCK_DIR.

    The saved scan carries its verdict; when it does not, the drift module
    (imported lazily, tolerated missing) recomputes it from the dict.
    """
    d = Path(LOCK_DIR)
    if not d.is_dir():
        return {}
    out: dict[str, str | None] = {}
    for p in sorted(d.glob(f"*{DRIFT_SUFFIX}")):
        name = p.name[: -len(DRIFT_SUFFIX)]
        verdict: str | None = None
        try:
            scan = json.loads(p.read_text())
        except (OSError, ValueError):
            out[name] = None
            continue
        if isinstance(scan.get("verdict"), str):
            verdict = scan["verdict"]
        else:
            try:
                from antares.drift import DriftScan

                verdict = str(DriftScan.from_dict(scan).verdict())
            except Exception:
                verdict = None
        out[name] = verdict
    return out


def _history_index() -> dict[str, dict]:
    """lock name -> ``{"latest": entry | None, "trend": dict | None}``.

    ``antares.history`` (the check-history store) is optional. When it is
    importable it is asked for the ``GET /api/history`` shape
    (``{"locks": [{"lock", "trend", "latest"}, ...]}``) through the first
    callable it exposes under one of the names below; anything else degrades
    to "never checked".
    """
    try:
        # import_module (not ``from antares import history``) so a stubbed or
        # missing ``sys.modules["antares.history"]`` is honored.
        _history = importlib.import_module("antares.history")
    except ImportError:
        return {}
    for fn_name in ("summary_all", "history_index", "dashboard_index", "summary", "index", "list_history", "load_all"):
        fn = getattr(_history, fn_name, None)
        if not callable(fn):
            continue
        try:
            try:
                data = fn(Path(LOCK_DIR))
            except TypeError:
                data = fn()
        except Exception:
            continue
        rows = data.get("locks") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            continue
        out: dict[str, dict] = {}
        for row in rows:
            if isinstance(row, dict) and row.get("lock"):
                out[str(row["lock"])] = {"latest": row.get("latest"), "trend": row.get("trend")}
        return out
    return {}


def _fleet_index() -> tuple[dict | None, dict[str, dict]]:
    """``(fleet summary or None, lock name -> fleet robot entry)``."""
    try:
        report = load_fleet_dict()
    except HTTPException:
        return None, {}
    robots = [r for r in (report.get("robots") or []) if isinstance(r, dict)]
    by_lock = {str(r["lock"]): r for r in robots if r.get("lock")}
    summary = dict(report.get("summary") or {})
    summary.setdefault("n_robots", len(robots))
    summary["created_at"] = report.get("created_at")
    summary["locks"] = sorted(by_lock)
    return summary, by_lock


def _dataset_fps(dataset: str | None, fleet_robot: dict | None) -> float | None:
    if fleet_robot is not None and _finite(fleet_robot.get("fps")) is not None:
        return _finite(fleet_robot.get("fps"))
    if dataset:
        with _DATASETS_LOCK:
            ds = _DATASETS.get(dataset)
        if ds is not None:
            return _finite(getattr(ds, "fps", None))
    return None


def _registry_index() -> dict[str, dict]:
    """lock name -> ``{"robot_id", "is_active", "version", "n_versions"}`` from
    the registry; empty (never an error) when there is none or it is unreadable."""
    try:
        return load_registry().index_by_lock()
    except Exception:  # noqa: BLE001 - a broken registry must not take the dashboard down
        return {}


def dashboard_robot(
    lock: dict,
    *,
    drift: dict[str, str | None],
    history: dict[str, dict],
    fleet_by_lock: dict[str, dict],
    registry: dict[str, dict] | None = None,
) -> dict:
    """One dashboard row for a lock dict."""
    name = str(lock.get("name") or "")
    res = lock.get("result") or {}
    params = lock.get("params") or res.get("params") or {}
    before = _finite(res.get("baseline_eval_score"))
    after = _finite(res.get("eval_score"))
    improvement = _finite(res.get("improvement"))
    if improvement is None and before is not None and after is not None and before > 0:
        improvement = 1.0 - after / before
    worst_joint, worst_deg = _worst_body_joint(res.get("per_joint_after"))
    seeds = lock.get("seeds") if isinstance(lock.get("seeds"), dict) else None
    pinned = seeds.get("pinned") if seeds else None
    hist = history.get(name) or {}
    fleet_robot = fleet_by_lock.get(name)
    reg = (registry or {}).get(name) or {}
    return {
        "name": name,
        # registry: which physical robot this lock belongs to, and where it
        # sits in that robot's lineage (all null/None when unregistered)
        "robot_id": reg.get("robot_id"),
        "is_active": reg.get("is_active"),
        "version": reg.get("version"),
        "n_versions": reg.get("n_versions"),
        "dataset": lock.get("dataset"),
        "robot_type": lock.get("robot_type"),
        "robot": lock.get("robot") or DEFAULT_ROBOT,
        "created_at": lock.get("created_at"),
        "space": res.get("space", "shared"),
        "baseline_eval_score": before,
        "eval_score": after,
        "improvement": improvement,
        "per_joint_after": res.get("per_joint_after"),
        "worst_joint": worst_joint,
        "worst_joint_deg": worst_deg,
        "latency_frames": params.get("latency_frames"),
        "fps": _dataset_fps(lock.get("dataset"), fleet_robot),
        "drift": {"exists": name in drift, "verdict": drift.get(name)},
        "in_fleet": fleet_robot is not None,
        "pinned": list(pinned) if isinstance(pinned, (list, tuple)) else None,
        "history": {"latest": hist.get("latest"), "trend": hist.get("trend")},
    }


def dashboard() -> dict:
    """Every lock in LOCK_DIR as a dashboard row plus the fleet summary."""
    drift = _drift_verdicts()
    history = _history_index()
    fleet, fleet_by_lock = _fleet_index()
    registry = _registry_index()
    robots = [
        dashboard_robot(l, drift=drift, history=history, fleet_by_lock=fleet_by_lock, registry=registry)
        for l in list_lock_dicts()
        if l.get("name")
    ]
    return {
        "robots": robots,
        "fleet": fleet,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


@app.get("/api/dashboard")
def dashboard_get() -> dict:
    return dashboard()


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
    explain_evaluations: int = Field(300, ge=20, le=2000)
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
    lock_name = resolve_lock_ref(req.lock)  # @<robot> -> its active lock
    _lock_path(lock_name)  # validate the name before loading a dataset
    ds = get_dataset(req.dataset)
    result = run_check(
        ds,
        lock_name,
        threshold_deg=req.threshold_deg,
        ratio=req.ratio,
        last=req.last,
        min_episodes=req.min_episodes,
    )
    out = result.to_dict()
    if req.record:
        from antares import history

        try:
            history.append(result, LOCK_DIR, name=lock_name)
        except OSError as e:  # the verdict stands; say why it was not recorded
            out["history_error"] = str(e)
    if req.explain and result.verdict == "drifted":
        try:
            out["diagnosis"] = run_diagnose(
                ds, lock_name, last=req.last, max_evaluations=req.explain_evaluations
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

    lock = resolve_lock_ref(lock)
    if not _lock_path(lock).is_file():  # 400 on a bad name
        raise HTTPException(status_code=404, detail=f"no lock named {lock!r}")
    return history.report(lock, LOCK_DIR)


# --------------------------------------------------------------------------
# Robot registry routes (see antares.registry)
# --------------------------------------------------------------------------


def registry_robot(reg, robot_id: str) -> dict:
    """``GET /api/registry/{robot_id}``: the summary plus the lineage with pinned flags."""
    from antares.registry import UnknownRobot, param_drift

    try:
        out = reg.summary(robot_id)
        lineage = reg.lineage(robot_id)
    except UnknownRobot as e:
        raise HTTPException(status_code=404, detail=str(e))
    out["lineage"] = lineage
    out["param_drift"] = param_drift(lineage)
    return out


@app.get("/api/registry")
def registry_list() -> dict:
    """Every registered robot: active lock, version, held-out score, lineage entries."""
    reg = load_registry()
    return {
        "version": reg.version,
        "path": str(reg.path),
        "robots": [reg.summary(rid) for rid in reg.robots()],
    }


@app.get("/api/registry/{robot_id}")
def registry_get(robot_id: str) -> dict:
    return registry_robot(load_registry(), robot_id)


class RegisterRequest(BaseModel):
    lock: str
    activate: bool | None = None
    note: str | None = None
    max_regression: float | None = Field(None, ge=0)


@app.post("/api/registry/{robot_id}/register")
def registry_register(robot_id: str, req: RegisterRequest) -> dict:
    """Add a saved lock to the robot's lineage; the activation rule decides
    whether it becomes active (see ``registration.reason``)."""
    registration = register_lock(
        robot_id, req.lock, activate=req.activate, note=req.note, max_regression=req.max_regression
    )
    out = registry_robot(load_registry(), robot_id)
    out["registration"] = registration
    return out


class ActivateRequest(BaseModel):
    lock: str


@app.post("/api/registry/{robot_id}/activate")
def registry_activate(robot_id: str, req: ActivateRequest) -> dict:
    from antares.registry import RegistryError, UnknownRobot

    reg = load_registry()
    try:
        previous = reg.activate(robot_id, req.lock)
    except UnknownRobot as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    reg.save()
    out = registry_robot(reg, robot_id)
    out["previous_active"] = previous
    return out


@app.post("/api/registry/{robot_id}/rollback")
def registry_rollback(robot_id: str) -> dict:
    from antares.registry import RegistryError, UnknownRobot

    reg = load_registry()
    try:
        previous, _ = reg.rollback(robot_id)
    except UnknownRobot as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RegistryError as e:
        raise HTTPException(status_code=409, detail=str(e))
    reg.save()
    out = registry_robot(reg, robot_id)
    out["previous_active"] = previous
    return out


@app.delete("/api/registry/{robot_id}")
def registry_delete(robot_id: str) -> dict:
    """Forget the robot; its lock files stay."""
    from antares.registry import RegistryError, UnknownRobot

    reg = load_registry()
    try:
        entry = reg.remove(robot_id)
    except UnknownRobot as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    reg.save()
    return {"removed": robot_id, "locks": entry.names}


class DiagnoseRequest(BaseModel):
    dataset: str
    lock: str
    last: int | None = Field(None, ge=1)
    max_evaluations: int = Field(300, ge=20, le=2000)
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


# ==========================================================================
# Camera fits (antares.camera / antares.video / antares.project)
#
# Fit a dataset's fixed camera from its videos and the logged joint angles,
# list the fits saved for a lock, and draw the sim arm's outline over a real
# frame for the UI. Kept together here, after the other routes, so the
# section merges cleanly.
# ==========================================================================

CAMERA_SUFFIX = ".camera.json"

_cam_arms: dict[str, tuple[Any, Any]] = {}
_CAM_LOCK = threading.Lock()  # the camera Arms and hull caches are shared across requests
_BACKGROUNDS: dict[tuple[str, str], Any] = {}
_BACKGROUNDS_LOCK = threading.Lock()
BACKGROUND_EPISODES = 6
BACKGROUND_FRAMES = 60


def _video_error(e: Exception) -> HTTPException:
    """Map a video error to a status the client can act on: 501 only when
    decoding is impossible here (PyAV missing), 404 for an episode, file or
    frames the dataset does not have, 400 for a camera it does not have, and
    500 (with the message) when a file could not be decoded."""
    from antares import video

    if isinstance(e, video.VideoUnsupported):
        status = 501
    elif isinstance(e, video.VideoNotFound):
        status = 404
    elif isinstance(e, video.VideoUnavailable) and not isinstance(e, video.VideoDecodeError):
        status = 400
    else:
        status = 500
    return HTTPException(status_code=status, detail=str(e))


def _camera_arm(robot: str | None = None):
    """An Arm (and its hulls) of ``robot``'s model for silhouette work, separate
    from the replay Arm. One per model: the unit map decides where the hulls
    land, so a degrees dataset must not be drawn with the normalized map."""
    from antares.robots import DEFAULT_ROBOT

    name = check_robot(robot) if robot else DEFAULT_ROBOT
    with _CAM_LOCK:
        if name not in _cam_arms:
            from antares.project import arm_hulls
            from antares.sim import Arm

            arm = Arm(robot=name)
            _cam_arms[name] = (arm, arm_hulls(arm.model))
        return _cam_arms[name]


def camera_background(ds: Dataset, camera: str):
    """The per-camera background model, built once per process from frames
    spread over a few episodes (a few seconds of decoding the first time)."""
    key = (ds.repo_id, camera)
    with _BACKGROUNDS_LOCK:
        bgm = _BACKGROUNDS.get(key)
    if bgm is not None:
        return bgm
    from antares import video
    from antares.camera import spread_frames
    from antares.silhouette import BackgroundModel

    picks = spread_frames(ds.episodes, BACKGROUND_EPISODES, BACKGROUND_FRAMES)
    by_ep: dict[int, list[int]] = {}
    for ep, f in picks:
        by_ep.setdefault(ep, []).append(f)
    frames = np.concatenate([video.frames(ds.repo_id, camera, ep, idx, ds) for ep, idx in by_ep.items()])
    bgm = BackgroundModel.fit(frames)
    with _BACKGROUNDS_LOCK:
        _BACKGROUNDS.setdefault(key, bgm)
        return _BACKGROUNDS[key]


def list_camera_infos(repo_id: str) -> list[dict]:
    from antares import video

    try:
        return [c.to_dict() for c in video.list_cameras(repo_id)]
    except Exception as e:  # noqa: BLE001 - loader errors go to the client
        raise HTTPException(status_code=502, detail=f"could not read {repo_id}'s cameras: {e}")


def load_camera_fit_dicts(lock_name: str) -> dict[str, dict]:
    """Camera fits for a lock from ``<lock>.camera.json`` (falls back to the lock's ``cameras``)."""
    from antares.camera import load_camera_fits

    lock_name = resolve_lock_ref(lock_name)
    _lock_path(lock_name)  # validates the name
    fits = {name: f.to_dict() for name, f in load_camera_fits(lock_name, LOCK_DIR).items()}
    if not fits:
        p = _lock_path(lock_name)
        if p.is_file():
            fits = dict(json.loads(p.read_text()).get("cameras") or {})
    return fits


def run_camera_fit(
    ds: Dataset,
    camera: str,
    lock_name: str,
    *,
    n_frames: int = 40,
    n_episodes: int = 4,
    n_starts: int = 8,
    seed: int = 0,
    on_progress: Callable[[Any], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_second_pass: Callable[[str], None] | None = None,
):
    """Fit one camera and save it next to the lock. Returns ``(fit, path)``.

    When another fixed camera of the dataset with the same image size is
    already fitted (most likely the same webcam model), a second pass holds
    the field of view near that camera's, and the pass with the better
    held-out overlap is kept: from silhouettes alone a narrower view from
    further away lines up the base but leaves the gripper off.
    ``on_second_pass`` hears the sibling camera's name when that pass starts.
    """
    from antares.camera import fit_camera, save_camera_fit
    from antares.sim import Arm

    lock_name = resolve_lock_ref(lock_name)
    _lock_path(lock_name)
    # A fit runs for a minute; it gets its own Arm so overlay requests
    # (which share ``_camera_arm``) are not blocked behind it.
    arm = Arm(robot=dataset_robot(ds))
    common = dict(
        n_frames=n_frames,
        n_episodes=n_episodes,
        n_starts=n_starts,
        seed=seed,
        arm=arm,
        on_progress=on_progress,
        should_stop=should_stop,
    )
    fit = fit_camera(ds, camera, **common)
    prior = sibling_fovy(ds.repo_id, camera, (fit.width, fit.height))
    if prior is not None and not fit.cancelled:
        fovy, sibling = prior
        if on_second_pass is not None:
            on_second_pass(sibling)
        banded = fit_camera(ds, camera, fovy_range=(fovy - SIBLING_FOVY_BAND, fovy + SIBLING_FOVY_BAND), **common)
        if not banded.cancelled and _heldout(banded.to_dict()) > _heldout(fit.to_dict()):
            fit = banded
    path = save_camera_fit(fit, lock_name, LOCK_DIR)
    return fit, path


# Half-width (degrees) of the field-of-view band a sibling camera's fit sets.
SIBLING_FOVY_BAND = 2.0


def sibling_fovy(dataset: str, camera: str, size: tuple[int, int] | None = None) -> tuple[float, str] | None:
    """``(fovy, camera)`` of the best-fitted other camera of ``dataset`` (over
    all its locks), of image ``size`` when given; None when there is none."""
    from antares.video import camera_name

    name = camera_name(camera)
    others = [
        (n, f)
        for n, f in dataset_camera_fit_dicts(dataset).items()
        if n != name and f.get("fovy") and (size is None or (int(f.get("width") or 0), int(f.get("height") or 0)) == size)
    ]
    if not others:
        return None
    n, f = max(others, key=lambda nf: _heldout(nf[1]))
    return float(f["fovy"]), n


def _camera_job_worker(job: Job, req: "CameraFitRequest") -> None:
    try:
        ds = get_dataset(req.dataset)

        def on_progress(ev) -> None:
            job.push({"type": "progress", **ev.to_dict()})

        def on_second_pass(sibling: str) -> None:
            job.push({"type": "pass", "pass": 2, "sibling": sibling})

        fit, path = run_camera_fit(
            ds,
            req.camera,
            req.lock,
            n_frames=req.n_frames,
            n_episodes=req.n_episodes,
            n_starts=req.n_starts,
            seed=req.seed,
            on_progress=on_progress,
            should_stop=job.stop.is_set,
            on_second_pass=on_second_pass,
        )
        summary = {**fit.summary(), "lock": req.lock, "path": str(path)}
        job.push({"type": "done", "fit": summary})
        job.finish("done", result=summary)
    except Exception as e:  # noqa: BLE001 - report to the client
        job_fail(job, e)


def camera_overlay(ds: Dataset, camera: str, episode: int, frame: int, lock_name: str | None, mask: bool = True) -> dict:
    """The real frame with the sim arm's outlines for the lock's camera fit."""
    from antares import video
    from antares.project import Camera, iou, project_polygons, rasterize
    from antares.silhouette import mask_polygons

    ep = find_episode(ds, episode)
    if not 0 <= frame < len(ep):
        raise HTTPException(status_code=404, detail=f"frame {frame} not in episode {episode} ({len(ep)} frames)")
    fit = None
    if lock_name:
        fits = load_camera_fit_dicts(lock_name)
        fit = fits.get(video.camera_name(camera))
    try:
        rgb = video.frames(ds.repo_id, camera, ep.index, [frame], ds)[0]
    except Exception as e:  # noqa: BLE001
        raise _video_error(e)
    out: dict[str, Any] = {
        "dataset": ds.repo_id,
        "camera": video.camera_name(camera),
        "episode": ep.index,
        "frame": frame,
        "lock": lock_name,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "image": "data:image/jpeg;base64," + base64.b64encode(video.frame_jpeg(rgb)).decode(),
        "sim_polygons": [],
        "fit": None,
        "iou": None,
    }
    real_mask = None
    if mask:
        bgm = camera_background(ds, camera)
        real_mask = bgm.mask(rgb)
        out["real_mask_polygon"] = mask_polygons(real_mask)
    if fit is not None:
        arm, hulls = _camera_arm(dataset_robot(ds))
        cam = Camera.from_dict(fit)
        with _CAM_LOCK:
            pts = hulls.world_points(arm.fk(arm.unit_map.to_rad(ep.state[frame])))
        polys = project_polygons(hulls, pts, cam)
        out["sim_polygons"] = [np.round(p, 1).tolist() for p in polys if len(p) >= 3]
        out["sim_bodies"] = [g.body_name for g, p in zip(hulls.geoms, polys) if len(p) >= 3]
        out["fit"] = {k: fit.get(k) for k in ("fit_iou", "heldout_iou", "fovy", "pos", "quat", "created_at")}
        if real_mask is not None:
            out["iou"] = iou(rasterize(polys, cam.width, cam.height), real_mask)
    return out


class CameraFitRequest(BaseModel):
    dataset: str
    camera: str
    lock: str
    n_frames: int = Field(40, ge=4)
    n_episodes: int = Field(4, ge=1)
    n_starts: int = Field(8, ge=1)
    seed: int = 0


@app.get("/api/camera/cameras")
def camera_cameras(dataset: str = Query(...)) -> list[dict]:
    """The dataset's video streams, with ``fixed`` false for wrist cameras."""
    return list_camera_infos(dataset)


@app.post("/api/camera/fit")
def camera_fit_start(req: CameraFitRequest) -> dict:
    _lock_path(resolve_lock_ref(req.lock))  # validate early
    job = start_job("camera", req.dataset, _camera_job_worker, req)
    return {"job_id": job.id}


@app.get("/api/camera")
def camera_fits(lock: str = Query(...)) -> dict:
    fits = load_camera_fit_dicts(lock)
    return {"lock": lock, "cameras": {name: {k: v for k, v in f.items() if k not in ("fit_ious", "heldout_ious", "appearance")} for name, f in fits.items()}}


def dataset_camera_fit_dicts(dataset: str) -> dict[str, dict]:
    """The best saved fit of each of ``dataset``'s cameras over all its locks.

    A camera's pose belongs to the recording, not to the dynamics a lock
    fitted, so a fit saved under one lock lines up the video for any lock
    of the same dataset (the per-joint, sensor or a freshly calibrated one).
    Each fit carries the ``lock`` it was saved under.
    """
    best: dict[str, dict] = {}
    for lock in list_lock_dicts():
        if lock.get("dataset") != dataset or not lock.get("name"):
            continue
        try:
            fits = load_camera_fit_dicts(str(lock["name"]))
        except HTTPException:
            continue
        for name, fit in fits.items():
            if fit.get("dataset") not in (None, "", dataset) or fit.get("cancelled"):
                continue
            if name not in best or _heldout(fit) > _heldout(best[name]):
                best[name] = {**_fit_summary(fit), "lock": str(lock["name"])}
    return best


def _heldout(fit: dict) -> float:
    try:
        v = float(fit.get("heldout_iou"))
    except (TypeError, ValueError):
        return -1.0
    return v if np.isfinite(v) else -1.0


def _fit_summary(fit: dict) -> dict:
    return {k: v for k, v in fit.items() if k not in ("fit_ious", "heldout_ious", "appearance")}


@app.get("/api/camera/dataset")
def camera_fits_for_dataset(dataset: str = Query(...)) -> dict:
    """Best saved camera fit per camera of a dataset, whichever lock holds it."""
    return {"dataset": dataset, "cameras": dataset_camera_fit_dicts(dataset)}


@app.get("/api/camera/overlay")
def camera_overlay_get(
    dataset: str = Query(...),
    camera: str = Query(...),
    episode: int = Query(...),
    frame: int = Query(0, ge=0),
    lock: str | None = Query(None),
    mask: bool = Query(True),
) -> dict:
    ds = get_dataset(dataset)
    try:
        return camera_overlay(ds, camera, episode, frame, lock, mask=mask)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - decoding / segmentation errors
        raise _video_error(e)


@app.get("/api/camera/clip")
def camera_clip_get(
    dataset: str = Query(...),
    camera: str = Query(...),
    episode: int = Query(...),
    format: Literal["mp4", "webm"] = Query("mp4"),
) -> FileResponse:
    """One episode of a camera as a clip starting at the episode's frame 0,
    for the viewer's video underlay (encoded once, then cached)."""
    from antares import video

    ds = get_dataset(dataset)
    try:
        path = video.episode_clip(ds.repo_id, camera, episode, ds, fmt=format)
    except Exception as e:  # noqa: BLE001 - missing PyAV / camera / decoding errors
        raise _video_error(e)
    return FileResponse(path, media_type=f"video/{format}")


# ==========================================================================
# Scene (antares.scene)
#
# The static scene recovered from the logs: table height from the fingertips
# at grasp instants, workspace, grasp/release points, and per fitted camera
# the clean background with the table plane checked against it. The sim view
# composites what that camera would see in the sim, without OpenGL.
# ==========================================================================


def load_scene_dict(lock_name: str) -> dict | None:
    """``<lock>.scene.json`` for a lock, or None when no scene was fitted."""
    from antares.scene import load_scene_fit

    _lock_path(lock_name)  # validates the name
    fit = load_scene_fit(lock_name, LOCK_DIR)
    return fit.to_dict() if fit is not None else None


def run_scene_fit(ds: Dataset, lock_name: str, cameras: list[str] | None = None):
    """Fit the scene for ``lock_name`` from ``ds`` and save it (JSON, backdrops,
    the lock's ``scene`` field). Returns ``(fit, path)``.

    The arm carries the lock's parameters and robot; the cameras are the
    lock's fitted ones (or ``cameras``, a subset by name), each with the
    background model the overlay already uses.
    """
    from antares.project import Camera
    from antares.scene import fit_scene, save_scene_fit
    from antares.sim import Arm

    lock = load_lock(lock_name)
    arm = Arm(lock.params, robot=lock.robot)
    fits = load_camera_fit_dicts(lock_name)
    names = [c for c in (cameras or sorted(fits)) if c in fits]
    cams = {name: Camera.from_dict(fits[name]) for name in names}
    backgrounds = {}
    for name in names:
        try:
            backgrounds[name] = camera_background(ds, name).bg
        except Exception as e:  # noqa: BLE001 - a camera without video still leaves the table fit
            raise _video_error(e)
    fit = fit_scene(ds, arm, lock_name, cams, backgrounds)
    path = save_scene_fit(fit, lock_name, LOCK_DIR, backgrounds)
    return fit, path


def scene_view(ds: Dataset, camera: str, episode: int, frame: int, lock_name: str) -> dict:
    """Real frame | sim view of the same instant from the same camera."""
    from antares import video
    from antares.project import Camera, iou, rasterize
    from antares.scene import load_backdrop, load_scene_fit, sim_view

    ep = find_episode(ds, episode)
    if not 0 <= frame < len(ep):
        raise HTTPException(status_code=404, detail=f"frame {frame} not in episode {episode} ({len(ep)} frames)")
    name = video.camera_name(camera)
    fits = load_camera_fit_dicts(lock_name)
    fit = fits.get(name)
    if fit is None:
        raise HTTPException(status_code=404, detail=f"lock {lock_name!r} has no camera fit for {name!r}")
    scene = load_scene_fit(lock_name, LOCK_DIR)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"lock {lock_name!r} has no scene fit; run scene-fit first")
    backdrop = scene.backdrops.get(name)
    if backdrop is None:
        raise HTTPException(status_code=404, detail=f"the scene fit of {lock_name!r} has no backdrop for {name!r}")
    try:
        rgb = video.frames(ds.repo_id, camera, ep.index, [frame], ds)[0]
        bg = load_backdrop(backdrop, LOCK_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise _video_error(e)
    cam = Camera.from_dict(fit)
    arm, hulls = _camera_arm(dataset_robot(ds))
    with _CAM_LOCK:
        poses = arm.fk(arm.unit_map.to_rad(ep.state[frame]))
    dark = int((fit.get("appearance") or {}).get("sign", -1)) < 0
    table_poly = np.asarray(backdrop.plane_polygon, dtype=np.float64)
    image, polys = sim_view(bg, cam, hulls, poses, table_poly, dark_arm=dark)
    real_mask = camera_background(ds, camera).mask(rgb)
    sim_mask = rasterize(polys, cam.width, cam.height)
    return {
        "dataset": ds.repo_id,
        "camera": name,
        "episode": ep.index,
        "frame": frame,
        "lock": lock_name,
        "width": int(cam.width),
        "height": int(cam.height),
        "real_image": "data:image/jpeg;base64," + base64.b64encode(video.frame_jpeg(rgb)).decode(),
        "sim_image": "data:image/jpeg;base64," + base64.b64encode(video.frame_jpeg(image)).decode(),
        "iou": iou(sim_mask, real_mask),
        "table_polygon": [list(p) for p in backdrop.plane_polygon],
        "table_z": scene.table.z,
        "n_geoms": len(polys),
    }


class SceneFitRequest(BaseModel):
    dataset: str
    lock: str
    cameras: list[str] | None = None


@app.post("/api/scene/fit")
def scene_fit_post(req: SceneFitRequest) -> dict:
    """Fit and save the scene for a lock (synchronous: FK over the grasp
    instants plus the cameras' background models, a few seconds)."""
    _lock_path(req.lock)
    ds = get_dataset(req.dataset)
    try:
        fit, path = run_scene_fit(ds, req.lock, req.cameras)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {**fit.to_dict(), "path": str(path)}


@app.get("/api/scene")
def scene_get(lock: str = Query(...)) -> dict:
    return {"lock": lock, "scene": load_scene_dict(lock)}


@app.get("/api/scene/view")
def scene_view_get(
    dataset: str = Query(...),
    camera: str = Query(...),
    episode: int = Query(...),
    frame: int = Query(0, ge=0),
    lock: str = Query(...),
) -> dict:
    ds = get_dataset(dataset)
    try:
        return scene_view(ds, camera, episode, frame, lock)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - decoding / segmentation errors
        raise _video_error(e)


# ==========================================================================
# Policy (antares.policy, antares.shadow, antares.closedloop)
#
# A state-based chunked policy trained on the lock's training episodes, then
# judged on its held-out ones in the default sim and in the calibrated one:
# open-loop "shadow" (the policy's commands on the real observations, rolled
# through each sim) and closed-loop (the policy drives each sim). The gap
# between the two sims' verdicts is how much the stock sim misjudges the
# policy. Everything is sim-only; torch is the ``[policy]`` extra.
# ==========================================================================

POLICY_MODES = ("shadow", "closedloop")
#: Keywords ``robots`` accepts besides saved lock names.
POLICY_ROBOT_KEYS = ("default", "lock", "perjoint")
POLICY_RESULT_SUFFIX = ".policy.{mode}.json"


def _policy_error(e: Exception) -> HTTPException:
    if isinstance(e, HTTPException):
        return e
    if isinstance(e, ImportError):
        return HTTPException(status_code=501, detail=str(e))
    if isinstance(e, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    return HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


def _check_policy_mode(mode: str) -> str:
    if mode not in POLICY_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {list(POLICY_MODES)}, got {mode!r}")
    return mode


def policy_results_path(lock_name: str, mode: str) -> Path:
    _lock_path(lock_name)  # validates the name
    return Path(LOCK_DIR) / f"{lock_name}{POLICY_RESULT_SUFFIX.format(mode=_check_policy_mode(mode))}"


def list_policy_dicts(lock: str | None = None) -> list[dict]:
    """Saved policies' sidecars (``locks/*.policy.json``), newest first."""
    from antares.policy import list_policies

    if lock is not None:
        _lock_path(lock)
    return list_policies(LOCK_DIR, lock)


def load_policy_results(lock_name: str, mode: str) -> dict | None:
    p = policy_results_path(lock_name, mode)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


def policy_split(ds: Dataset, lock_dict: dict) -> tuple[list[Episode], list[Episode]]:
    """The lock's train / held-out episodes of ``ds`` (400 when the lock has no split)."""
    from antares.compensate import held_out_episodes

    result = lock_dict.get("result") or {}
    train_idx, eval_idx = result.get("train_episodes") or [], result.get("eval_episodes") or []
    if not train_idx or not eval_idx:
        raise HTTPException(status_code=400, detail="lock records no train/held-out split to train and evaluate on")
    try:
        return held_out_episodes(ds, train_idx), held_out_episodes(ds, eval_idx)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def grasp_context_for(ds: Dataset, lock_dict: dict | None = None):
    """Gripper thresholds, empty-jaw level and fingertip FK for ``ds`` on the lock's model."""
    from antares.policy import GraspContext

    robot = check_robot((lock_dict or {}).get("robot") or dataset_robot(ds))
    return GraspContext.for_dataset(ds.episodes, robot)


def run_policy_training(
    ds: Dataset,
    lock_dict: dict,
    epochs: int = 200,
    name: str | None = None,
    seed: int = 0,
    on_progress: Callable[[Any], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Train the chunked policy on the lock's training split, early-stop on its
    held-out split, save it as ``locks/<name>.policy.pt`` (default: the lock's
    name). Returns the policy's metadata (metrics included) plus the path."""
    from antares.policy import save_policy, train_policy

    train, eval = policy_split(ds, lock_dict)
    name = name or str(lock_dict["name"])
    _lock_path(name)  # same naming rules as locks
    ctx = grasp_context_for(ds, lock_dict)
    res = train_policy(
        train,
        eval,
        epochs=epochs,
        seed=seed,
        ctx=ctx,
        on_progress=on_progress,
        should_stop=should_stop,
        meta={"dataset": ds.repo_id, "lock": str(lock_dict["name"]), "robot": check_robot(lock_dict.get("robot"))},
    )
    path = save_policy(res.policy, name, LOCK_DIR)
    return {**res.policy.meta, "path": str(path)}


def _policy_train_worker(job: Job, req: "PolicyTrainRequest") -> None:
    try:
        ds = get_dataset(req.dataset)
        lock_dict = load_lock_dict(req.lock)
        summary = run_policy_training(
            ds, lock_dict, epochs=req.epochs, name=req.name, seed=req.seed,
            on_progress=lambda p: job.push(p.to_dict()), should_stop=job.stop.is_set,
        )
        job.push({"type": "done", "policy": summary})
        job.finish("done", result=summary)
    except Exception as e:  # noqa: BLE001 - reported through the job
        job_fail(job, _policy_error(e))


def resolve_policy_variants(lock_dict: dict, robots: Sequence[str] | None = None) -> tuple[list, list[str]]:
    """The sims to judge the policy in. ``robots`` entries are ``default``,
    ``lock`` (this lock), ``perjoint`` (``<lock>_perjoint`` when saved) or any
    saved lock's name. Returns ``(variants, missing)``; a ``perjoint`` with
    no such lock is reported in ``missing`` rather than failing."""
    from antares.shadow import default_variant, variant_from_lock

    robots = list(robots) if robots else list(POLICY_ROBOT_KEYS)
    robot = check_robot(lock_dict.get("robot"))
    variants, missing, seen = [], [], set()
    for key in robots:
        if key in seen:
            continue
        seen.add(key)
        if key == "default":
            variants.append(default_variant(robot))
        elif key == "lock":
            variants.append(variant_from_lock(lock_dict, "lock"))
        elif key == "perjoint":
            name = f"{lock_dict['name']}_perjoint"
            if lock_dict.get("result", {}).get("space") == "per_joint":
                continue  # the lock itself is per-joint; "lock" already covers it
            if not _lock_path(name).is_file():
                missing.append(name)
                continue
            variants.append(variant_from_lock(load_lock_dict(name), "perjoint"))
        else:
            variants.append(variant_from_lock(load_lock_dict(key), key))
    if not variants:
        raise HTTPException(status_code=400, detail="no sim variants to evaluate in")
    return variants, missing


def _policy_eval_episodes(ds: Dataset, lock_dict: dict, policy_meta: dict | None, episode: int | None = None) -> list[Episode]:
    """The held-out episodes the policy never trained on: the lock's eval
    split, narrowed to the policy's own if it records one; or one episode."""
    from antares.compensate import held_out_episodes

    _, eval = policy_split(ds, lock_dict)
    pol_eval = set((policy_meta or {}).get("eval_episodes") or [])
    if pol_eval:
        eval = [e for e in eval if e.index in pol_eval] or eval
    if episode is not None:
        if episode not in {e.index for e in eval}:
            raise HTTPException(status_code=400, detail=f"episode {episode} is not a held-out episode of {lock_dict['name']}")
        return held_out_episodes(ds, [episode])
    return eval


def run_policy_evaluation(
    ds: Dataset,
    lock_dict: dict,
    policy_name: str | None,
    mode: str,
    robots: Sequence[str] | None = None,
    on_progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    save: bool = True,
) -> dict:
    """Shadow or closed-loop evaluation of a saved policy on the lock's held-out
    episodes in every requested sim; saved as ``locks/<lock>.policy.<mode>.json``.
    Per-episode tables, no trajectories (``policy_viewer`` renders those)."""
    from antares.closedloop import closed_loop_episodes
    from antares.lock import now_iso
    from antares.policy import load_policy
    from antares.shadow import shadow_episodes

    _check_policy_mode(mode)
    policy_name = policy_name or str(lock_dict["name"])
    _lock_path(policy_name)
    try:
        policy = load_policy(policy_name, LOCK_DIR)
    except (ImportError, FileNotFoundError) as e:
        raise _policy_error(e)
    episodes = _policy_eval_episodes(ds, lock_dict, policy.meta)
    variants, missing = resolve_policy_variants(lock_dict, robots)
    ctx = grasp_context_for(ds, lock_dict)
    if mode == "shadow":
        summary = shadow_episodes(
            policy, episodes, variants, ctx,
            on_progress=lambda i, n, se: on_progress and on_progress(
                {"type": "progress", "mode": mode, "i": i, "n": n, "episode": se.episode, "row": se.to_dict()}
            ),
            should_stop=should_stop,
        )
    else:
        summary = closed_loop_episodes(
            policy, episodes, variants, ctx,
            on_progress=lambda i, n, batch: on_progress and on_progress(
                {"type": "progress", "mode": mode, "i": i, "n": n, "episode": batch[0].episode,
                 "rows": [r.to_dict() for r in batch]}
            ),
            should_stop=should_stop,
        )
    out = summary.to_dict(include_arrays=False)
    out.update(
        dataset=ds.repo_id,
        lock=str(lock_dict["name"]),
        policy=policy_name,
        policy_meta={k: policy.meta.get(k) for k in ("k", "h", "features", "created_at", "eval_episodes", "metrics")},
        robots=list(robots) if robots else list(POLICY_ROBOT_KEYS),
        missing_locks=missing,
        episodes_evaluated=[e.index for e in episodes],
        created_at=now_iso(),
    )
    if save:
        p = policy_results_path(str(lock_dict["name"]), mode)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        out["path"] = str(p)
    return out


def _policy_eval_worker(job: Job, req: "PolicyEvaluateRequest") -> None:
    try:
        ds = get_dataset(req.dataset)
        lock_dict = load_lock_dict(req.lock)
        out = run_policy_evaluation(
            ds, lock_dict, req.policy, req.mode, req.robots, on_progress=job.push, should_stop=job.stop.is_set,
        )
        job.push({"type": "done", "mode": req.mode, "summary": out["summary"], "lock": out["lock"], "policy": out["policy"]})
        job.finish("done", result=out)
    except Exception as e:  # noqa: BLE001
        job_fail(job, _policy_error(e))


def policy_viewer(
    ds: Dataset, lock_dict: dict, policy_name: str | None, mode: str, episode: int, robots: Sequence[str] | None = None
) -> dict:
    """One held-out episode under the policy in every sim, with body poses
    ``(T,nbody,7)`` per sim for the viewer's ghosts (like ``/api/replay``).
    Recomputed on request (a fraction of a second), so result files stay small."""
    from antares.closedloop import closed_loop_episodes
    from antares.policy import load_policy
    from antares.shadow import shadow_episodes

    _check_policy_mode(mode)
    policy_name = policy_name or str(lock_dict["name"])
    _lock_path(policy_name)
    try:
        policy = load_policy(policy_name, LOCK_DIR)
    except (ImportError, FileNotFoundError) as e:
        raise _policy_error(e)
    eps = _policy_eval_episodes(ds, lock_dict, policy.meta, episode)
    variants, missing = resolve_policy_variants(lock_dict, robots)
    ctx = grasp_context_for(ds, lock_dict)
    arm = variants[0].arm()
    sims: dict[str, dict] = {}
    try:
        if mode == "shadow":
            s = shadow_episodes(policy, eps, variants, ctx)
            if not s.episodes:
                raise HTTPException(status_code=400, detail=f"episode {episode} has no grasp point for the policy")
            se = s.episodes[0]
            real_q = se.real_q[se.baseline]
            rows = {n: (se.sim_q[n], {"sim_vs_real_deg": float(se.gaps[n].overall_deg),
                                      **({"divergence_deg": float(se.divergence[n].overall_deg)} if n in se.divergence else {})})
                    for n in se.sim_q}
            table = se.to_dict()
            goal = se.goal
        else:
            s = closed_loop_episodes(policy, eps, variants, ctx)
            if not s.runs:
                raise HTTPException(status_code=400, detail=f"episode {episode} has no grasp point for the policy")
            real_q = arm.real_q(eps[0])
            rows = {r.variant: (r.sim_q, {k: v for k, v in r.to_dict().items() if k not in ("variant", "episode")}) for r in s.runs}
            table = {"runs": [r.to_dict() for r in s.runs]}
            goal = None if s.runs[0].real_grasp_xyz is None else np.asarray(s.runs[0].real_grasp_xyz)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    for name, (q, metrics) in rows.items():
        sims[name] = {"q": _rounded(np.degrees(q)), "poses": _rounded(arm.fk(q)), **metrics}
    return {
        "dataset": ds.repo_id,
        "lock": str(lock_dict["name"]),
        "policy": policy_name,
        "mode": mode,
        "episode": int(eps[0].index),
        "fps": float(eps[0].fps),
        "robot": arm.robot_name,
        "joints": list(arm.robot.canonical_joints),
        "body_names": list(arm.body_names),
        "goal": None if goal is None else [float(v) for v in goal],
        "real_q": _rounded(np.degrees(real_q)),
        "real_poses": _rounded(arm.fk(real_q)),
        "sims": sims,
        "variants": [v.to_dict() for v in variants],
        "missing_locks": missing,
        "table": table,
        "validation": "sim-only",
    }


class PolicyTrainRequest(BaseModel):
    dataset: str
    lock: str
    epochs: int = Field(200, ge=1, le=5000)
    seed: int = 0
    # Saved as locks/<name>.policy.pt; default: the lock's name.
    name: str | None = None


class PolicyEvaluateRequest(BaseModel):
    dataset: str
    lock: str
    policy: str | None = None  # default: the policy named after the lock
    mode: Literal["shadow", "closedloop"] = "shadow"
    # "default", "lock", "perjoint" (= <lock>_perjoint when saved) or saved lock names.
    robots: list[str] | None = None


@app.post("/api/policy/train")
def policy_train_start(req: PolicyTrainRequest) -> dict:
    load_lock_dict(req.lock)  # 404 early
    if req.name is not None:
        _lock_path(req.name)
    job = start_job("policy_train", req.dataset, _policy_train_worker, req)
    return {"job_id": job.id}


@app.get("/api/policy")
def policy_list(lock: str | None = Query(None)) -> dict:
    return {"lock": lock, "policies": list_policy_dicts(lock)}


@app.post("/api/policy/evaluate")
def policy_evaluate_start(req: PolicyEvaluateRequest) -> dict:
    lock_dict = load_lock_dict(req.lock)
    resolve_policy_variants(lock_dict, req.robots)  # unknown lock names fail now, not in the job
    job = start_job("policy_eval", req.dataset, _policy_eval_worker, req)
    return {"job_id": job.id}


@app.get("/api/policy/results")
def policy_results(lock: str = Query(...), mode: str | None = Query(None)) -> dict:
    modes = [_check_policy_mode(mode)] if mode else list(POLICY_MODES)
    return {"lock": lock, **{m: load_policy_results(lock, m) for m in modes}}


@app.get("/api/policy/viewer")
def policy_viewer_get(
    dataset: str = Query(...),
    lock: str = Query(...),
    episode: int = Query(...),
    mode: str = Query("shadow"),
    policy: str | None = Query(None),
    robots: str | None = Query(None, description="comma-separated: default,lock,perjoint or lock names"),
) -> dict:
    ds = get_dataset(dataset)
    lock_dict = load_lock_dict(lock)
    return policy_viewer(ds, lock_dict, policy, _check_policy_mode(mode), episode, _csv(robots) or None)


# Objects (antares.objects / antares.payload)
#
# The manipulated object found in the video, tracked through every episode
# (vision on the table plane, FK while held), its start/end pose
# distribution against the task's FK picks, and the payload mass estimated
# from the residual while it is carried. The fit is a job (progress per
# episode); the per-frame track of one episode feeds the overlay and the
# 3D viewer.
# ==========================================================================

_TRACKS: dict[tuple[str, str, int], Any] = {}
_TRACKS_LOCK = threading.Lock()


def load_objects_dict(lock_name: str) -> dict | None:
    """``<lock>.objects.json`` for a lock, or None when no objects fit was run."""
    from antares.objects import load_object_fit

    _lock_path(lock_name)  # validates the name
    fit = load_object_fit(lock_name, LOCK_DIR)
    return fit.to_dict() if fit is not None else None


def _objects_camera(lock_name: str, camera: str | None) -> tuple[str, dict]:
    """The camera to track in: the one asked for, else the lock's best-fitted one."""
    fits = load_camera_fit_dicts(lock_name)
    if not fits:
        raise HTTPException(status_code=400, detail=f"lock {lock_name!r} has no camera fit; run camera-fit first")
    if camera is not None:
        name = camera.removeprefix("observation.images.")
        if name not in fits:
            raise HTTPException(status_code=404, detail=f"lock {lock_name!r} has no camera fit for {name!r}; fitted: {sorted(fits)}")
        return name, fits
    best = max(fits, key=lambda n: float(fits[n].get("fit_iou") or 0.0))
    return best, fits


def run_objects_fit(
    ds: Dataset,
    lock_name: str,
    camera: str | None = None,
    *,
    episodes: list[int] | None = None,
    payload: bool = True,
    on_progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
):
    """Track the object in ``ds`` for ``lock_name`` and save the fit (JSON,
    tracks ``.npz``, the lock's ``objects`` field). Returns ``(fit, tracks, path)``.

    Needs the lock's camera fit and scene fit (the table plane). With two
    fitted cameras the cross-camera triangulation check runs too; with
    ``payload`` the residual-based mass estimate is attached.
    """
    from antares.objects import cross_camera_check, fit_objects, save_object_fit
    from antares.payload import estimate_payload
    from antares.project import Camera
    from antares.scene import load_scene_fit
    from antares.sim import Arm

    lock_name = resolve_lock_ref(lock_name)
    lock = load_lock(lock_name)
    scene = load_scene_fit(lock_name, LOCK_DIR)
    if scene is None:
        raise HTTPException(status_code=400, detail=f"lock {lock_name!r} has no scene fit (the table plane); run scene-fit first")
    name, fits = _objects_camera(lock_name, camera)
    cam = Camera.from_dict(fits[name])
    try:
        bgm = camera_background(ds, name)
    except Exception as e:  # noqa: BLE001
        raise _video_error(e)
    arm = Arm(lock.params, robot=lock.robot)
    bd = scene.backdrops.get(name)
    fit, tracks = fit_objects(
        ds, arm, lock_name, name, cam, bgm, scene.table.z,
        object_half_height=scene.table.object_half_height,
        table_rgb=bd.table_rgb if bd is not None else None,
        episodes=episodes,
        on_progress=on_progress,
        should_stop=should_stop,
    )
    others = [n for n in fits if n != name]
    if others and not (should_stop and should_stop()):
        try:
            cams = {n: Camera.from_dict(fits[n]) for n in (name, others[0])}
            bgs = {n: camera_background(ds, n) for n in cams}
            fit.cross_camera = cross_camera_check(ds, arm, cams, bgs, scene.table.z + scene.table.object_half_height, name)
        except Exception as e:  # noqa: BLE001 - the check is optional
            fit.cross_camera = {"n": 0, "note": f"{type(e).__name__}: {e}"}
        if on_progress is not None:
            on_progress({"stage": "cross", "cross_camera": fit.cross_camera})
    if payload and not (should_stop and should_stop()):
        picks = {t.episode: [(g, r) for g, r in t.picks if r < len(t.frames)] for t in tracks}
        est = estimate_payload(ds, arm, lock_name, picks=picks, episodes=[t.episode for t in tracks], on_progress=on_progress, should_stop=should_stop)
        fit.payload = est.to_dict()
    path = save_object_fit(fit, lock_name, LOCK_DIR, tracks)
    with _TRACKS_LOCK:
        for t in tracks:
            _TRACKS[(ds.repo_id, lock_name, t.episode)] = t
    return fit, tracks, path


def _objects_job_worker(job: Job, req: "ObjectsFitRequest") -> None:
    try:
        ds = get_dataset(req.dataset)

        def on_progress(ev: dict) -> None:
            job.push({"type": "progress", **ev})

        fit, tracks, path = run_objects_fit(
            ds, req.lock, req.camera, episodes=req.episodes, payload=req.payload,
            on_progress=on_progress, should_stop=job.stop.is_set,
        )
        summary = {**{k: v for k, v in fit.to_dict().items() if k != "tracks"}, "path": str(path)}
        job.push({"type": "done", "fit": summary})
        job.finish("done", result=summary)
    except Exception as e:  # noqa: BLE001 - report to the client
        job_fail(job, e)


def objects_track(ds: Dataset, episode: int, lock_name: str) -> dict:
    """The per-frame object track of one episode: from the saved ``.npz``,
    else tracked now (a few seconds) and kept in memory."""
    from antares.objects import load_object_fit, load_track

    lock_name = resolve_lock_ref(lock_name)
    ep = find_episode(ds, episode)
    fit = load_object_fit(lock_name, LOCK_DIR)
    if fit is None:
        raise HTTPException(status_code=404, detail=f"lock {lock_name!r} has no objects fit; run objects-fit first")
    key = (ds.repo_id, lock_name, ep.index)
    with _TRACKS_LOCK:
        track = _TRACKS.get(key)
    if track is None:
        track = load_track(fit, ep.index, LOCK_DIR)
    if track is None:
        _, tracks, _ = run_objects_fit(ds, lock_name, fit.camera, episodes=[ep.index], payload=False)
        track = next((t for t in tracks if t.episode == ep.index), None)
        if track is None:
            raise HTTPException(status_code=404, detail=f"episode {episode} could not be tracked")
    with _TRACKS_LOCK:
        _TRACKS[key] = track
    return {
        "dataset": ds.repo_id,
        "lock": lock_name,
        "camera": fit.camera,
        "rgb": list(fit.prior.rgb),
        "footprint_m": None if fit.footprint is None else list(fit.footprint),
        "half_height": fit.object_half_height,
        **track.to_dict(),
    }


class ObjectsFitRequest(BaseModel):
    dataset: str
    lock: str
    camera: str | None = None
    episodes: list[int] | None = None
    payload: bool = True


@app.post("/api/objects/fit")
def objects_fit_start(req: ObjectsFitRequest) -> dict:
    _lock_path(resolve_lock_ref(req.lock))  # validate early
    job = start_job("objects", req.dataset, _objects_job_worker, req)
    return {"job_id": job.id}


@app.get("/api/objects")
def objects_get(lock: str = Query(...)) -> dict:
    d = load_objects_dict(lock)
    if d is not None:
        d = {k: v for k, v in d.items() if k != "tracks"} | {"tracks": [t for t in d["tracks"]]}
    return {"lock": lock, "objects": d}


@app.get("/api/objects/track")
def objects_track_get(dataset: str = Query(...), episode: int = Query(...), lock: str = Query(...)) -> dict:
    ds = get_dataset(dataset)
    try:
        return objects_track(ds, episode, lock)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - decoding / tracking errors
        raise _video_error(e)


# Built frontend, mounted after the API routes so `antares serve` alone demos.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
