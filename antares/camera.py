"""Fit a dataset's fixed camera to the robot's own logs.

For a handful of frames spread over several episodes we know the joint
angles, so :mod:`antares.project` can draw where the sim arm would be for
any candidate camera, and :mod:`antares.silhouette` says where the real arm
is. CMA-ES then searches the seven camera parameters (position, rotation
as a rotation vector, vertical field of view) for the pose that maximises
the mean intersection-over-union of the two silhouettes.

The objective has many local optima (a camera on the wrong side of the arm
still overlaps it), so the search is multi-start: a ring of candidate poses
around the arm plus one looking straight down, each run briefly at low
resolution, and the best is refined at full resolution. Frames not used in
the fit are scored afterwards as a held-out check.

The result, a :class:`CameraFit`, is saved next to the lock as
``locks/<lock>.camera.json`` (one entry per camera) and copied into the
lock's ``cameras`` field so an export carries ``<camera>`` elements.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from antares import __version__
from antares.data import Dataset, Episode
from antares.lock import now_iso
from antares.project import ArmHulls, Camera, arm_hulls, mat_to_rotvec, project_polygons, rasterize, rotvec_to_mat
from antares.silhouette import BackgroundModel, masks

CAMERA_SUFFIX = ".camera.json"

#: Vertical field of view bounds (deg); the rest of the search box is defined
#: with :class:`Space` below.
FOVY_LO, FOVY_HI = 20.0, 110.0


# --------------------------------------------------------------------------
# Frame selection
# --------------------------------------------------------------------------


def spread_frames(
    episodes: Sequence[Episode], n_episodes: int, n_frames: int, margin: float = 0.08, phase: float = 0.0
) -> list[tuple[int, int]]:
    """``(episode_index, frame_index)`` pairs: ``n_episodes`` episodes evenly
    spread over the sequence, ``n_frames`` spread evenly over them, each
    episode sampled between ``margin`` and ``1 - margin`` of its length
    (the arm sits at rest at both ends). ``phase`` in [0, 1) shifts the
    episode choice so a second call picks different episodes."""
    if not episodes:
        return []
    n_episodes = max(1, min(n_episodes, len(episodes)))
    pos = (np.arange(n_episodes) + phase) / n_episodes * len(episodes)
    chosen = [episodes[min(len(episodes) - 1, int(p))] for p in pos]
    per = [n_frames // n_episodes + (1 if i < n_frames % n_episodes else 0) for i in range(n_episodes)]
    out: list[tuple[int, int]] = []
    for ep, k in zip(chosen, per):
        if k <= 0:
            continue
        T = len(ep)
        lo, hi = int(margin * T), max(int(margin * T), int((1 - margin) * T) - 1)
        idx = np.linspace(lo, hi, k).round().astype(int)
        out.extend((ep.index, int(i)) for i in dict.fromkeys(idx.tolist()))
    return out


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------


DIST_LO, DIST_HI = 0.15, 3.0  # m, log-scaled
TARGET_RANGE = 0.5  # m, target point box around the arm's workspace centre
EL_LO, EL_HI = math.radians(-15.0), math.radians(95.0)
ROLL_LIMIT = math.pi / 2


def look_at_params(cam: Camera, centre: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    """``(target, distance, azimuth, elevation, roll)`` of a camera.

    The target is the point on the optical axis at the depth of ``centre``
    (so the decomposition is unique); azimuth/elevation are the direction
    from the target to the camera; roll is the rotation about the optical
    axis relative to the world-up ``look_at`` frame.
    """
    forward = -cam.R[:, 2]
    depth = max(DIST_LO, float(np.dot(np.asarray(centre) - cam.pos, forward)))
    target = cam.pos + depth * forward
    v = cam.pos - target
    az = math.atan2(v[1], v[0]) if math.hypot(v[0], v[1]) > 1e-9 else 0.0
    el = math.atan2(v[2], math.hypot(v[0], v[1]))
    base = Camera.look_at(cam.pos, target, cam.fovy, cam.width, cam.height, up=_up_for(el, az))
    rel = base.R.T @ cam.R
    roll = math.atan2(rel[1, 0], rel[0, 0])
    return target, depth, az, el, roll


def _up_for(el: float, az: float) -> tuple[float, float, float]:
    """World up, except near the zenith where the look-at frame would degenerate."""
    if abs(el) < math.radians(80):
        return (0.0, 0.0, 1.0)
    return (math.cos(az + math.pi), math.sin(az + math.pi), 0.0)


def camera_from_look_at(
    target: np.ndarray, distance: float, az: float, el: float, roll: float, fovy: float, width: int, height: int
) -> Camera:
    pos = np.asarray(target) + distance * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    cam = Camera.look_at(pos, target, fovy, width, height, up=_up_for(el, az))
    return cam.with_rotation(cam.R @ rotvec_to_mat(np.array([0.0, 0.0, roll])))


@dataclass(frozen=True)
class Space:
    """Normalized [0,1]^7 look-at coordinates around a reference camera.

    Searching position and rotation vector directly is badly conditioned:
    moving the camera without re-aiming it loses the arm from the image.
    Here a candidate is the point it looks at (3), the distance (log), the
    azimuth and elevation of the viewing direction, the roll about the
    optical axis, and the field of view. Azimuth and roll are relative to
    the reference so the start sits mid-box.
    """

    ref: Camera  # start camera: image size and the reference azimuth/roll
    centre: np.ndarray  # (3,) the arm's workspace centre the target box is around

    n_dims: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", np.asarray(self.centre, dtype=np.float64).reshape(3))

    @property
    def _ref_params(self) -> tuple[np.ndarray, float, float, float, float]:
        return look_at_params(self.ref, self.centre)

    def encode(self, cam: Camera) -> np.ndarray:
        target, dist, az, el, roll = look_at_params(cam, self.centre)
        _, _, az0, _, roll0 = self._ref_params
        x = np.empty(8)
        x[:3] = (target - self.centre + TARGET_RANGE) / (2 * TARGET_RANGE)
        x[3] = (math.log(dist) - math.log(DIST_LO)) / (math.log(DIST_HI) - math.log(DIST_LO))
        x[4] = (_wrap(az - az0) + math.pi) / (2 * math.pi)
        x[5] = (el - EL_LO) / (EL_HI - EL_LO)
        x[6] = (_wrap(roll - roll0) + ROLL_LIMIT) / (2 * ROLL_LIMIT)
        x[7] = (cam.fovy - FOVY_LO) / (FOVY_HI - FOVY_LO)
        return np.clip(x, 0.0, 1.0)

    def decode(self, x: np.ndarray) -> Camera:
        x = np.clip(np.asarray(x, dtype=np.float64).reshape(8), 0.0, 1.0)
        _, _, az0, _, roll0 = self._ref_params
        target = self.centre - TARGET_RANGE + x[:3] * 2 * TARGET_RANGE
        dist = math.exp(math.log(DIST_LO) + x[3] * (math.log(DIST_HI) - math.log(DIST_LO)))
        az = az0 - math.pi + x[4] * 2 * math.pi
        el = EL_LO + x[5] * (EL_HI - EL_LO)
        roll = roll0 - ROLL_LIMIT + x[6] * 2 * ROLL_LIMIT
        fovy = FOVY_LO + x[7] * (FOVY_HI - FOVY_LO)
        return camera_from_look_at(target, dist, az, el, roll, fovy, self.ref.width, self.ref.height)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def ring_starts(
    width: int,
    height: int,
    n: int = 8,
    radius: float = 0.7,
    height_m: float = 0.45,
    target: Sequence[float] = (0.0, 0.0, 0.12),
    fovy: float = 60.0,
    top_down: bool = True,
) -> list[Camera]:
    """``n`` cameras on a ring around the arm looking at its base, plus one
    straight above it (the SO-100 demo dataset's top camera)."""
    starts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        pos = [radius * math.cos(a), radius * math.sin(a), height_m]
        starts.append(Camera.look_at(pos, target, fovy, width, height))
    if top_down:
        starts.append(Camera.look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], fovy, width, height, up=(1.0, 0.0, 0.0)))
    return starts


def sample_cameras(
    rng: np.random.Generator,
    n: int,
    width: int,
    height: int,
    target: Sequence[float] = (0.0, 0.0, 0.12),
    radius: tuple[float, float] = (0.3, 1.4),
    elevation_deg: tuple[float, float] = (0.0, 90.0),
    fovy: tuple[float, float] = (35.0, 95.0),
    roll_deg: float = 12.0,
    target_jitter: float = 0.08,
) -> list[Camera]:
    """``n`` random cameras on a hemisphere over the arm, each looking near
    ``target`` with a little roll: the prior for a camera someone clamped to
    the table or a stand and pointed at the robot."""
    out = []
    target = np.asarray(target, dtype=np.float64)
    for _ in range(n):
        r = rng.uniform(*radius)
        el = math.radians(rng.uniform(*elevation_deg))
        az = rng.uniform(0.0, 2 * math.pi)
        pos = target + r * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        look = target + rng.normal(0.0, target_jitter, 3)
        up = (0.0, 0.0, 1.0) if el < math.radians(80) else (math.cos(az + math.pi), math.sin(az + math.pi), 0.0)
        cam = Camera.look_at(pos, look, rng.uniform(*fovy), width, height, up=up)
        roll = math.radians(rng.normal(0.0, roll_deg))
        R = cam.R @ rotvec_to_mat(np.array([0.0, 0.0, roll]))
        out.append(cam.with_rotation(R))
    return out


def pick_distinct(cams: Sequence[Camera], scores: Sequence[float], n: int, min_dist: float = 0.15) -> list[int]:
    """Indices of the ``n`` best cameras that are at least ``min_dist`` m apart."""
    order = np.argsort(scores)[::-1]
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= n:
            break
        if all(np.linalg.norm(cams[i].pos - cams[j].pos) >= min_dist for j in chosen):
            chosen.append(int(i))
    return chosen


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------


class SilhouetteObjective:
    """Mean IoU of sim and real silhouettes over a set of posed frames.

    ``points`` are the hull vertices in the world for each frame (from
    :meth:`ArmHulls.world_points`); ``real`` the real masks at full size.
    ``scale`` < 1 evaluates on downscaled masks for speed.
    """

    def __init__(self, hulls: ArmHulls, points: Sequence[np.ndarray], real: np.ndarray, scale: float = 1.0):
        import cv2

        self.hulls = hulls
        self.points = list(points)
        self.scale = float(scale)
        H, W = real.shape[1:3]
        self.w, self.h = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
        if scale != 1.0:
            self.real = np.stack(
                [cv2.resize(m.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_AREA) > 0.5 for m in real]
            )
        else:
            self.real = real.astype(bool)
        self.real_area = self.real.reshape(len(real), -1).sum(axis=1)

    def ious(self, cam: Camera) -> np.ndarray:
        out = np.empty(len(self.points))
        for i, pts in enumerate(self.points):
            polys = project_polygons(self.hulls, pts, cam)
            sim = rasterize(polys, self.w, self.h, self.scale).astype(bool)
            inter = np.count_nonzero(sim & self.real[i])
            union = np.count_nonzero(sim) + self.real_area[i] - inter
            out[i] = inter / union if union else 0.0
        return out

    def __call__(self, cam: Camera) -> float:
        return float(self.ious(cam).mean())


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressEvent:
    stage: str  # "start" | "refine" | "heldout"
    start: int  # index of the start pose being run (refine: the chosen one)
    n_starts: int
    evaluation: int  # evaluations so far over all stages
    stage_evaluation: int
    candidate_iou: float
    stage_best_iou: float
    best_iou: float
    elapsed_s: float
    best_camera: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "stage": self.stage,
            "start": self.start,
            "n_starts": self.n_starts,
            "evaluation": self.evaluation,
            "stage_evaluation": self.stage_evaluation,
            "candidate_iou": self.candidate_iou,
            "stage_best_iou": self.stage_best_iou,
            "best_iou": self.best_iou,
            "elapsed_s": self.elapsed_s,
        }
        if self.best_camera is not None:
            d["best_camera"] = self.best_camera
        return d


@dataclass(frozen=True)
class CameraFit:
    camera: str  # short name ("up")
    key: str  # dataset feature key ("observation.images.up")
    dataset: str
    pos: tuple[float, float, float]
    rotvec: tuple[float, float, float]
    quat: tuple[float, float, float, float]  # MuJoCo wxyz
    fovy: float
    width: int
    height: int
    fit_iou: float
    heldout_iou: float
    fit_frames: tuple[tuple[int, int], ...]  # (episode, frame)
    heldout_frames: tuple[tuple[int, int], ...]
    fit_ious: tuple[float, ...]
    heldout_ious: tuple[float, ...]
    n_starts: int
    best_start: int
    start_ious: tuple[float, ...]  # low-res IoU each start reached
    n_evaluations: int
    elapsed_s: float
    cancelled: bool = False
    created_at: str = ""
    antares_version: str = __version__
    appearance: dict[str, Any] = field(default_factory=dict)

    def to_camera(self) -> Camera:
        return Camera(pos=self.pos, rotvec=self.rotvec, fovy=self.fovy, width=self.width, height=self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "key": self.key,
            "dataset": self.dataset,
            "pos": list(self.pos),
            "rotvec": list(self.rotvec),
            "quat": list(self.quat),
            "fovy": self.fovy,
            "width": self.width,
            "height": self.height,
            "fit_iou": self.fit_iou,
            "heldout_iou": self.heldout_iou,
            "fit_frames": [list(f) for f in self.fit_frames],
            "heldout_frames": [list(f) for f in self.heldout_frames],
            "fit_ious": list(self.fit_ious),
            "heldout_ious": list(self.heldout_ious),
            "n_starts": self.n_starts,
            "best_start": self.best_start,
            "start_ious": list(self.start_ious),
            "n_evaluations": self.n_evaluations,
            "elapsed_s": self.elapsed_s,
            "cancelled": self.cancelled,
            "created_at": self.created_at,
            "antares_version": self.antares_version,
            "appearance": dict(self.appearance),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraFit:
        return cls(
            camera=str(d["camera"]),
            key=str(d.get("key", f"observation.images.{d['camera']}")),
            dataset=str(d.get("dataset", "")),
            pos=tuple(float(v) for v in d["pos"]),
            rotvec=tuple(float(v) for v in d["rotvec"]),
            quat=tuple(float(v) for v in d["quat"]),
            fovy=float(d["fovy"]),
            width=int(d["width"]),
            height=int(d["height"]),
            fit_iou=float(d.get("fit_iou", float("nan"))),
            heldout_iou=float(d.get("heldout_iou", float("nan"))),
            fit_frames=tuple((int(e), int(f)) for e, f in d.get("fit_frames", [])),
            heldout_frames=tuple((int(e), int(f)) for e, f in d.get("heldout_frames", [])),
            fit_ious=tuple(float(v) for v in d.get("fit_ious", [])),
            heldout_ious=tuple(float(v) for v in d.get("heldout_ious", [])),
            n_starts=int(d.get("n_starts", 0)),
            best_start=int(d.get("best_start", -1)),
            start_ious=tuple(float(v) for v in d.get("start_ious", [])),
            n_evaluations=int(d.get("n_evaluations", 0)),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            cancelled=bool(d.get("cancelled", False)),
            created_at=str(d.get("created_at", "")),
            antares_version=str(d.get("antares_version", "")),
            appearance=dict(d.get("appearance") or {}),
        )

    def summary(self) -> dict[str, Any]:
        """The fields the UI lists (no per-frame arrays)."""
        d = self.to_dict()
        for k in ("fit_ious", "heldout_ious", "appearance"):
            d.pop(k, None)
        return d

    def lock_entry(self) -> dict[str, Any]:
        """The compact record kept in the lock's ``cameras`` field: the camera
        itself and its scores. The full fit (frames, per-frame IoUs) stays in
        ``<lock>.camera.json``."""
        return {
            "camera": self.camera,
            "key": self.key,
            "dataset": self.dataset,
            "pos": list(self.pos),
            "rotvec": list(self.rotvec),
            "quat": list(self.quat),
            "fovy": self.fovy,
            "width": self.width,
            "height": self.height,
            "fit_iou": self.fit_iou,
            "heldout_iou": self.heldout_iou,
            "n_fit_frames": len(self.fit_frames),
            "n_heldout_frames": len(self.heldout_frames),
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------


def _run_cma(
    objective: Callable[[Camera], float],
    space: Space,
    x0: np.ndarray,
    sigma0: float,
    max_evaluations: int,
    seed: int,
    emit: Callable[[Camera, float, float], bool],
) -> tuple[Camera, float, int, bool]:
    """Maximise ``objective`` from ``x0``. Returns (best camera, best value, evaluations, cancelled).

    ``emit(candidate, value, stage_best)`` is called per generation and
    returns True to stop early.
    """
    import cma

    opts = {
        "bounds": [0.0, 1.0],
        "seed": int(np.random.default_rng(seed).integers(1, 2**31 - 1)),
        "verbose": -9,
        "verb_log": 0,
        "maxfevals": max_evaluations,
        "tolfun": 1e-4,
        "tolx": 1e-4,
    }
    es = cma.CMAEvolutionStrategy(np.clip(x0, 1e-3, 1 - 1e-3), sigma0, opts)
    best_cam = space.decode(x0)
    best_val = objective(best_cam)
    n_eval = 1
    cancelled = False
    while n_eval < max_evaluations and not es.stop():
        solutions = es.ask()
        values = []
        for x in solutions:
            cam = space.decode(x)
            v = objective(cam)
            n_eval += 1
            values.append(-v)
            if v > best_val:
                best_val, best_cam = v, cam
        es.tell(solutions, values)
        if emit(space.decode(solutions[int(np.argmin(values))]), -min(values), best_val):
            cancelled = True
            break
    return best_cam, best_val, n_eval, cancelled


def prepare_frames(
    dataset: Dataset,
    camera: str,
    fit_frames: Sequence[tuple[int, int]],
    heldout_frames: Sequence[tuple[int, int]],
    background_model: BackgroundModel | None = None,
) -> tuple[np.ndarray, np.ndarray, BackgroundModel, np.ndarray, np.ndarray]:
    """Decode the frames and segment the arm. Returns
    ``(fit_masks, heldout_masks, background_model, fit_rgb, heldout_rgb)``."""
    from antares import video

    by_ep: dict[int, list[int]] = {}
    for ep, f in list(fit_frames) + list(heldout_frames):
        by_ep.setdefault(ep, []).append(f)
    rgb: dict[tuple[int, int], np.ndarray] = {}
    for ep, idx in by_ep.items():
        uniq = sorted(set(idx))
        arr = video.frames(dataset.repo_id, camera, ep, uniq, dataset)
        for i, a in zip(uniq, arr):
            rgb[(ep, i)] = a
    fit_rgb = np.stack([rgb[tuple(k)] for k in fit_frames])
    held_rgb = np.stack([rgb[tuple(k)] for k in heldout_frames]) if len(heldout_frames) else fit_rgb[:0]
    if background_model is None:
        background_model = BackgroundModel.fit(np.concatenate([fit_rgb, held_rgb]) if len(held_rgb) else fit_rgb)
    fit_masks = masks(fit_rgb, background_model)
    held_masks = masks(held_rgb, background_model) if len(held_rgb) else np.zeros((0,) + fit_masks.shape[1:], np.uint8)
    return fit_masks, held_masks, background_model, fit_rgb, held_rgb


def posed_points(arm: Any, hulls: ArmHulls, dataset: Dataset, frames: Sequence[tuple[int, int]]) -> list[np.ndarray]:
    """Hull vertices in the world for each (episode, frame) from the logged joint state."""
    eps = {ep.index: ep for ep in dataset.episodes}
    out = []
    for ep_i, f in frames:
        q = arm.unit_map.to_rad(eps[ep_i].state[f])
        out.append(hulls.world_points(arm.fk(q)))
    return out


def fit_camera(
    dataset: Dataset,
    camera: str,
    frames_idx: Sequence[tuple[int, int]] | None = None,
    init: Camera | None = None,
    *,
    n_frames: int = 40,
    n_episodes: int = 4,
    n_heldout: int = 12,
    n_starts: int = 8,
    n_samples: int = 3000,
    start_evaluations: int = 250,
    refine_evaluations: int = 450,
    low_res: float = 0.25,
    seed: int = 0,
    arm: Any | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> CameraFit:
    """Fit ``camera`` of ``dataset`` from its own frames and joint logs.

    Stages: ``n_samples`` random cameras on a hemisphere over the arm are
    scored at 1/8 resolution; the ``n_starts`` best distinct ones each get a
    short CMA-ES run at ``low_res``; the best of those is refined at full
    resolution; the result is scored on held-out frames.

    ``frames_idx`` overrides the automatic ``(episode, frame)`` selection for
    the fit (held-out frames are still chosen automatically from other
    episodes). ``init`` adds one start at a given camera (for example the
    previous fit) ahead of the sampled ones.
    """
    from antares import video
    from antares.sim import Arm

    t0 = time.perf_counter()
    cam_info = video.find_camera(dataset.repo_id, camera)
    if not cam_info.fixed:
        raise ValueError(f"{cam_info.key} is a wrist camera; only fixed cameras can be fitted")
    W, H = cam_info.width, cam_info.height

    fit_frames = list(frames_idx) if frames_idx is not None else spread_frames(dataset.episodes, n_episodes, n_frames)
    used_eps = {e for e, _ in fit_frames}
    others = [ep for ep in dataset.episodes if ep.index not in used_eps] or list(dataset.episodes)
    heldout_frames = spread_frames(others, min(n_episodes, len(others)), n_heldout, phase=0.5)
    if not fit_frames:
        raise ValueError("no frames to fit")

    fit_masks, held_masks, bgm, _, _ = prepare_frames(dataset, camera, fit_frames, heldout_frames)

    arm = arm or Arm()
    hulls = arm_hulls(arm.model)
    pts_fit = posed_points(arm, hulls, dataset, fit_frames)
    pts_held = posed_points(arm, hulls, dataset, heldout_frames)

    # Downscaled objectives for the global stages: never narrower than 80 px
    # for the sample scan and 160 px for the starts, whatever the video size.
    # The scan only has to find the basin, so it looks at every third frame.
    tiny = SilhouetteObjective(hulls, pts_fit[::3], fit_masks[::3], scale=min(1.0, max(0.125, 80.0 / W)))
    low = SilhouetteObjective(hulls, pts_fit, fit_masks, scale=min(1.0, max(low_res, 160.0 / W)))
    full = SilhouetteObjective(hulls, pts_fit, fit_masks, scale=1.0)
    stop = should_stop or (lambda: False)
    evaluations = 0

    # Stage 0: where is the camera, roughly? Random cameras aimed at the
    # arm's workspace centre, scored at 1/8 resolution.
    rng = np.random.default_rng(seed)
    centre = np.mean([p.mean(axis=0) for p in pts_fit], axis=0)
    samples = sample_cameras(rng, n_samples, W, H, target=centre)
    sample_scores = np.array([tiny(c) for c in samples])
    evaluations += len(samples)
    starts = [samples[i] for i in pick_distinct(samples, sample_scores, n_starts)]
    if init is not None:
        starts.insert(0, Camera(init.pos, init.rotvec, init.fovy, W, H))
    n_total = len(starts)
    if on_progress is not None:
        on_progress(
            ProgressEvent(
                stage="sample",
                start=-1,
                n_starts=n_total,
                evaluation=evaluations,
                stage_evaluation=len(samples),
                candidate_iou=float(sample_scores.mean()),
                stage_best_iou=float(sample_scores.max()),
                best_iou=float(sample_scores.max()),
                elapsed_s=time.perf_counter() - t0,
                best_camera=samples[int(np.argmax(sample_scores))].to_dict(),
            )
        )

    best_global: tuple[float, Camera, int] = (-1.0, starts[0], 0)
    start_ious: list[float] = []
    cancelled = False

    def emitter(stage: str, k: int, stage_offset: list[int]):
        def emit(cand: Camera, value: float, stage_best: float) -> bool:
            nonlocal evaluations
            if on_progress is not None:
                best_val = max(best_global[0], stage_best)
                best_cam = best_global[1] if best_global[0] >= stage_best else None
                on_progress(
                    ProgressEvent(
                        stage=stage,
                        start=k,
                        n_starts=n_total,
                        evaluation=evaluations + stage_offset[0],
                        stage_evaluation=stage_offset[0],
                        candidate_iou=float(value),
                        stage_best_iou=float(stage_best),
                        best_iou=float(best_val),
                        elapsed_s=time.perf_counter() - t0,
                        best_camera=best_cam.to_dict() if best_cam is not None else None,
                    )
                )
            return stop()

        return emit

    # Stage 1: every start, briefly, at low resolution.
    for k, start in enumerate(starts):
        if stop():
            cancelled = True
            break
        space = Space(ref=start, centre=centre)
        counter = [0]

        def counting(cam: Camera, _obj=low, _c=counter) -> float:
            _c[0] += 1
            return _obj(cam)

        cam_k, val_k, n_k, cancelled = _run_cma(
            counting, space, space.encode(start), 0.08, start_evaluations, seed + k, emitter("start", k, counter)
        )
        evaluations += n_k
        start_ious.append(float(val_k))
        if val_k > best_global[0]:
            best_global = (float(val_k), cam_k, k)
        if cancelled:
            break

    # Stage 2: refine the best start at full resolution.
    best_val, best_cam, best_k = best_global
    if not cancelled and not stop():
        space = Space(ref=best_cam, centre=centre)
        counter = [0]

        def counting_full(cam: Camera, _c=counter) -> float:
            _c[0] += 1
            return full(cam)

        cam_r, val_r, n_r, cancelled = _run_cma(
            counting_full, space, space.encode(best_cam), 0.04, refine_evaluations, seed + 1000,
            emitter("refine", best_k, counter),
        )
        evaluations += n_r
        if val_r >= full(best_cam):
            best_cam = cam_r
    else:
        cancelled = True

    fit_ious = full.ious(best_cam)
    held_obj = SilhouetteObjective(hulls, pts_held, held_masks, scale=1.0) if len(pts_held) else None
    held_ious = held_obj.ious(best_cam) if held_obj is not None else np.zeros(0)
    elapsed = time.perf_counter() - t0
    fit = CameraFit(
        camera=cam_info.name,
        key=cam_info.key,
        dataset=dataset.repo_id,
        pos=tuple(float(v) for v in best_cam.pos),
        rotvec=tuple(float(v) for v in best_cam.rotvec),
        quat=tuple(float(v) for v in best_cam.quat),
        fovy=float(best_cam.fovy),
        width=W,
        height=H,
        fit_iou=float(fit_ious.mean()),
        heldout_iou=float(held_ious.mean()) if len(held_ious) else float("nan"),
        fit_frames=tuple((int(e), int(f)) for e, f in fit_frames),
        heldout_frames=tuple((int(e), int(f)) for e, f in heldout_frames),
        fit_ious=tuple(float(v) for v in fit_ious),
        heldout_ious=tuple(float(v) for v in held_ious),
        n_starts=n_total,
        best_start=int(best_k),
        start_ious=tuple(start_ious),
        n_evaluations=int(evaluations),
        elapsed_s=float(elapsed),
        cancelled=bool(cancelled),
        created_at=now_iso(),
        appearance={k: v for k, v in bgm.appearance.to_dict().items() if k != "p_arm"},
    )
    if on_progress is not None:
        on_progress(
            ProgressEvent(
                stage="heldout",
                start=int(best_k),
                n_starts=n_total,
                evaluation=evaluations,
                stage_evaluation=0,
                candidate_iou=fit.heldout_iou,
                stage_best_iou=fit.fit_iou,
                best_iou=fit.fit_iou,
                elapsed_s=elapsed,
                best_camera=best_cam.to_dict(),
            )
        )
    return fit


# --------------------------------------------------------------------------
# Files next to the locks
# --------------------------------------------------------------------------


def camera_path(lock_name: str, lock_dir: Path | str = "locks") -> Path:
    return Path(lock_dir) / f"{lock_name}{CAMERA_SUFFIX}"


def load_camera_fits(lock_name: str, lock_dir: Path | str = "locks") -> dict[str, CameraFit]:
    """Camera fits saved for a lock, by camera name (empty if none)."""
    p = camera_path(lock_name, lock_dir)
    if not p.is_file():
        return {}
    d = json.loads(p.read_text())
    return {name: CameraFit.from_dict(v) for name, v in (d.get("cameras") or {}).items()}


def save_camera_fit(fit: CameraFit, lock_name: str, lock_dir: Path | str = "locks") -> Path:
    """Merge ``fit`` into ``<lock>.camera.json`` and into the lock's ``cameras`` field (if the lock exists)."""
    from antares.lock import LOCK_SUFFIX, RobotLock

    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    p = camera_path(lock_name, lock_dir)
    existing = json.loads(p.read_text()) if p.is_file() else {}
    cams = dict(existing.get("cameras") or {})
    cams[fit.camera] = fit.to_dict()
    p.write_text(json.dumps({"lock": lock_name, "cameras": cams}, indent=2, sort_keys=True) + "\n")

    lock_file = lock_dir / f"{lock_name}{LOCK_SUFFIX}"
    if lock_file.is_file():
        lock = RobotLock.load(lock_file)
        merged = dict(lock.cameras)
        merged[fit.camera] = fit.lock_entry()
        lock.replace(cameras=merged).save(lock_file)
    return p


__all__ = [
    "CAMERA_SUFFIX",
    "CameraFit",
    "ProgressEvent",
    "SilhouetteObjective",
    "Space",
    "camera_path",
    "fit_camera",
    "load_camera_fits",
    "pick_distinct",
    "posed_points",
    "prepare_frames",
    "ring_starts",
    "sample_cameras",
    "save_camera_fit",
    "spread_frames",
]
