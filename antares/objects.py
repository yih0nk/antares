"""The manipulated object: found in the video, placed on the table, tracked.

Phase 1 (:mod:`antares.camera`) fitted the fixed cameras and Phase 2
(:mod:`antares.scene`) the table plane. Anything that sits on the table and
shows up in a camera can then be located in 3D by intersecting its pixel
ray with the plane. A pick-and-place log moves one object into a
container; this module finds it, follows it and compares what it sees with
what the arm's own kinematics say.

Per frame:

* **Foreground.** The camera's :class:`~antares.silhouette.BackgroundModel`
  gives the moving pixels (signed background difference). The sim arm's
  silhouette at the logged joint angles (:mod:`antares.project`), dilated to
  cover hull slack and the real gripper reaching lower than the sim's, is
  subtracted. What is left is what the arm moved: the object, its shadow,
  bits of the arm the hulls miss.
* **Colour prior.** The object sits still on the table before the grasp,
  so the biggest foreground blob of the first frames of each episode is
  the object. Its Lab colour (median over episodes) is the prior; pixels
  far from it (shadows, the SO-101's red cables are close but small, the
  SO-100's white arm parts) are dropped, and a blob must be within a size
  band of the prior's area.
* **Tracking.** While the gripper command is closed on something (the
  picks from :mod:`antares.task`), the object is between the jaws and
  mostly hidden: its position is the fingertip from FK (``source =
  "held"``). Otherwise the blob nearest the last known position (after a
  release: nearest the fingertip where it let go) is the object (``source
  = "vision"``); when none is near enough it is ``"hidden"`` and the last
  position is carried forward.
* **Pixel -> 3D.** The ray through the blob centroid is intersected with
  the plane ``z = table + object_half_height`` (the scene's assumption).
  The blob's pixels back-projected onto that plane give the footprint.

The built-in validation: if the camera, the table and the tracking are
right, the gripper closes where the object was seen. Per episode the
distance between the vision start pose and the FK fingertip at the grasp
is reported, and likewise the vision end pose against the release point.
With two fitted cameras the object is also triangulated from both rays
and compared with the single-camera plane intersection.

The result, an :class:`ObjectFit`, is saved as ``locks/<lock>.objects.json``
with the per-frame tracks in ``locks/objects/<lock>_<camera>_tracks.npz``,
and a compact ``objects`` entry goes into the lock.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from antares import __version__
from antares.data import Dataset, Episode
from antares.lock import now_iso
from antares.project import ArmHulls, Camera, arm_hulls, project_polygons, rasterize
from antares.scene import OBJECT_HALF_HEIGHT, fingertips
from antares.silhouette import BackgroundModel, _clean, _dilate, to_lab

OBJECTS_SUFFIX = ".objects.json"
TRACKS_DIRNAME = "objects"

#: Pixels the sim arm's silhouette is grown by before it is subtracted:
#: hull slack plus the real gripper reaching a little lower than the sim's.
ARM_DILATE_PX = 14
#: Frames at the start of an episode the colour prior is learned from.
PRIOR_FRAMES: tuple[int, ...] = (0, 4, 8)
#: A blob counts as the object when its area is within this band of the
#: prior's area (the object shrinks with distance and under partial occlusion).
AREA_BAND = (0.15, 4.0)
MIN_BLOB_PX = 40
#: Tolerance of the per-pixel colour test, in units of the prior's spread.
COLOUR_TOL = 3.0
#: Floors on that spread (Lab units): lighting changes L a lot, a/b a little.
COLOUR_FLOOR = (14.0, 7.0, 7.0)
#: Search radius for the next blob, as a multiple of the prior's side length;
#: wider right after a release, when the object drops or slides from the jaws.
GATE_SIDES = 2.5
GATE_MIN_PX = 40.0
RELEASE_GATE_SIDES = 4.0
RELEASE_GATE_MIN_PX = 60.0
#: Frames before a grasp where the gripper already covers the object; left
#: out of the start pose.
OCCLUSION_PAD = 12
SOURCE_CODES = ("vision", "held", "hidden")


# --------------------------------------------------------------------------
# Geometry: rays and the plane
# --------------------------------------------------------------------------


def pixel_rays(camera: Camera, uv: np.ndarray) -> np.ndarray:
    """Unit world-frame directions (N,3) of the rays through pixels ``uv`` (N,2)."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    f = camera.focal
    d = np.stack([(uv[:, 0] - camera.width / 2.0) / f, -(uv[:, 1] - camera.height / 2.0) / f, -np.ones(len(uv))], axis=1)
    d = d @ camera.R.T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def ray_plane(camera: Camera, uv: np.ndarray, z: float) -> np.ndarray:
    """World points (N,3) where the rays through pixels ``uv`` hit the horizontal plane ``z``.

    Rays that run away from the plane (or parallel to it) give NaN.
    """
    d = pixel_rays(camera, uv)
    dz = d[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z - camera.pos[2]) / dz
    s = np.where((np.abs(dz) > 1e-9) & (s > 0), s, np.nan)
    return camera.pos[None, :] + s[:, None] * d


def triangulate(cam_a: Camera, uv_a: Sequence[float], cam_b: Camera, uv_b: Sequence[float]) -> tuple[np.ndarray, float]:
    """Closest point between the two pixel rays and the gap between them (m)."""
    da = pixel_rays(cam_a, np.asarray(uv_a))[0]
    db = pixel_rays(cam_b, np.asarray(uv_b))[0]
    pa, pb = cam_a.pos, cam_b.pos
    w = pa - pb
    a, b, c = float(da @ da), float(da @ db), float(db @ db)
    d, e = float(da @ w), float(db @ w)
    den = a * c - b * b
    if abs(den) < 1e-12:
        return np.full(3, np.nan), float("nan")
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    qa, qb = pa + s * da, pb + t * db
    return 0.5 * (qa + qb), float(np.linalg.norm(qa - qb))


def footprint_on_plane(mask: np.ndarray, camera: Camera, z: float) -> tuple[float, float, float] | None:
    """Oriented extent ``(long, short, angle_deg)`` in metres of a blob's
    pixels back-projected onto the plane ``z``; None for a tiny blob."""
    import cv2

    ys, xs = np.nonzero(mask)
    if xs.size < 6:
        return None
    pts = ray_plane(camera, np.stack([xs + 0.5, ys + 0.5], axis=1), z)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 6:
        return None
    (_, (w, h), ang) = cv2.minAreaRect(pts[:, :2].astype(np.float32))
    long_, short = (w, h) if w >= h else (h, w)
    return float(long_), float(short), float(ang)


# --------------------------------------------------------------------------
# Colour prior and blobs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColourPrior:
    """What the object looks like: Lab colour (median and spread of its
    pixels at rest) and its pixel area at the start of an episode."""

    lab: tuple[float, float, float]
    spread: tuple[float, float, float]
    area_px: float
    rgb: tuple[int, int, int]
    n_samples: int = 0

    def match(self, lab: np.ndarray, tol: float = COLOUR_TOL) -> np.ndarray:
        """Per-pixel: within ``tol`` spreads of the prior on every Lab axis."""
        lo = np.asarray(self.lab) - tol * np.asarray(self.spread)
        hi = np.asarray(self.lab) + tol * np.asarray(self.spread)
        return np.all((lab >= lo) & (lab <= hi), axis=-1)

    def distance(self, lab: Sequence[float]) -> float:
        return float(np.linalg.norm((np.asarray(lab) - np.asarray(self.lab)) / np.asarray(self.spread)))

    @property
    def side_px(self) -> float:
        return math.sqrt(max(1.0, self.area_px))

    def to_dict(self) -> dict[str, Any]:
        return {
            "lab": list(self.lab),
            "spread": list(self.spread),
            "area_px": self.area_px,
            "rgb": list(self.rgb),
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ColourPrior:
        return cls(
            lab=tuple(float(v) for v in d["lab"]),
            spread=tuple(float(v) for v in d["spread"]),
            area_px=float(d["area_px"]),
            rgb=tuple(int(v) for v in d.get("rgb", (128, 128, 128))),
            n_samples=int(d.get("n_samples", 0)),
        )


@dataclass(frozen=True)
class Blob:
    area: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[float, float]
    label: int

    def to_dict(self) -> dict[str, Any]:
        return {"area": self.area, "bbox": list(self.bbox), "centroid": list(self.centroid)}


def sim_arm_mask(arm: Any, hulls: ArmHulls, camera: Camera, q: np.ndarray, dilate_px: int = ARM_DILATE_PX) -> np.ndarray:
    """The sim arm's silhouette at joint angles ``q`` (rad), dilated."""
    pts = hulls.world_points(arm.fk(q))
    mask = rasterize(project_polygons(hulls, pts, camera), camera.width, camera.height)
    return _dilate(mask, dilate_px)


def foreground(lab: np.ndarray, bg_lab: np.ndarray, bgm: BackgroundModel, arm_mask: np.ndarray) -> np.ndarray:
    """Moving pixels (cleaned) minus the dilated sim arm, (H,W) uint8."""
    mv = _clean(bgm.appearance.moving(lab, bg_lab), 3, 5)
    return (mv.astype(bool) & ~arm_mask.astype(bool)).astype(np.uint8)


def find_blobs(mask: np.ndarray, min_area: int = MIN_BLOB_PX) -> tuple[list[Blob], np.ndarray]:
    """Connected components of ``mask`` at least ``min_area`` px, largest first, and the label image."""
    import cv2

    n, labels, stats, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        out.append(Blob(area=a, bbox=(x, y, w, h), centroid=(float(cen[i, 0]), float(cen[i, 1])), label=i))
    out.sort(key=lambda b: -b.area)
    return out, labels


def learn_prior(
    samples: Sequence[tuple[np.ndarray, np.ndarray]],
    bgm: BackgroundModel,
    min_area: int = 120,
    reject_sigma: float = 4.0,
) -> ColourPrior:
    """The object's colour from ``samples`` of ``(frame RGB, dilated arm mask)``
    taken at the start of episodes, where the object sits still on the table.

    Per sample the largest foreground blob that does not touch the arm
    region is taken, its pixels' Lab median recorded; the prior is the
    median over samples, with samples whose colour is far from it
    (clutter that happened to move) dropped once. The spread is the
    pooled per-pixel MAD, floored (:data:`COLOUR_FLOOR`).
    """
    bg_lab = to_lab(bgm.bg)
    meds, pix, areas, rgbs = [], [], [], []
    for rgb, arm in samples:
        lab = to_lab(rgb)
        fg = foreground(lab, bg_lab, bgm, arm)
        blobs, labels = find_blobs(fg, min_area)
        touch = _dilate(arm, 4).astype(bool)
        for b in blobs:
            m = labels == b.label
            if np.count_nonzero(m & touch) > 0.1 * b.area:
                continue
            meds.append(np.median(lab[m], axis=0))
            pix.append(lab[m])
            areas.append(b.area)
            rgbs.append(np.median(rgb[m].reshape(-1, 3), axis=0))
            break
    if not meds:
        raise ValueError("no object found at the start of any episode: nothing but the arm moves in these frames")
    meds_a = np.array(meds)
    centre = np.median(meds_a, axis=0)
    mad = np.maximum(1.4826 * np.median(np.abs(meds_a - centre), axis=0), COLOUR_FLOOR)
    keep = np.all(np.abs(meds_a - centre) <= reject_sigma * mad, axis=1)
    if keep.sum() >= 1:
        meds_a = meds_a[keep]
        pix = [p for p, k in zip(pix, keep) if k]
        areas = [a for a, k in zip(areas, keep) if k]
        rgbs = [r for r, k in zip(rgbs, keep) if k]
    centre = np.median(meds_a, axis=0)
    allpix = np.concatenate(pix)
    spread = np.maximum(1.4826 * np.median(np.abs(allpix - centre), axis=0), COLOUR_FLOOR)
    rgb_c = np.median(np.array(rgbs), axis=0)
    return ColourPrior(
        lab=tuple(float(v) for v in centre),
        spread=tuple(float(v) for v in spread),
        area_px=float(np.median(areas)),
        rgb=tuple(int(round(float(v))) for v in rgb_c),
        n_samples=int(len(areas)),
    )


def object_blobs(lab: np.ndarray, fg: np.ndarray, prior: ColourPrior) -> tuple[list[Blob], np.ndarray]:
    """Foreground blobs of the object's colour within the area band of the prior."""
    m = (fg.astype(bool) & prior.match(lab)).astype(np.uint8)
    m = _clean(m, 0, 5)
    blobs, labels = find_blobs(m, max(MIN_BLOB_PX, int(AREA_BAND[0] * prior.area_px)))
    hi = AREA_BAND[1] * prior.area_px
    return [b for b in blobs if b.area <= hi], labels


# --------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------


@dataclass
class ObjectTrack:
    """Where the object was, frame by frame, in one episode and one camera.

    ``xyz`` is in the arm's base frame (m); ``source`` per frame is
    ``vision`` (seen on the table plane), ``held`` (between the jaws:
    fingertip from FK) or ``hidden`` (not seen: last position carried
    forward); ``visible`` is ``source == vision``. ``centroid``/``bbox`` are
    pixels, NaN where nothing was seen. ``footprint`` is the object's
    oriented extent on the table (long, short, m) from the start frames.
    """

    episode: int
    camera: str
    frames: np.ndarray  # (T,)
    xyz: np.ndarray  # (T,3)
    visible: np.ndarray  # (T,) bool
    source: np.ndarray  # (T,) int8 index into SOURCE_CODES
    centroid: np.ndarray  # (T,2) px
    bbox: np.ndarray  # (T,4) px
    area: np.ndarray  # (T,) px
    picks: list[tuple[int, int]]  # held windows [grasp, release] (release = T when never let go)
    start_pose: np.ndarray | None
    end_pose: np.ndarray | None
    end_source: str  # "vision" | "fk" | "none"
    footprint: tuple[float, float] | None
    fk_grasp: np.ndarray | None  # fingertip at the first pick
    fk_release: np.ndarray | None  # fingertip at the last release
    vision_vs_fk: float  # m, start pose vs fk_grasp (NaN without either)
    end_vs_fk: float  # m, end pose vs fk_release
    n_visible: int = 0
    note: str = ""

    def source_names(self) -> list[str]:
        return [SOURCE_CODES[int(s)] for s in self.source]

    def summary(self) -> dict[str, Any]:
        f = lambda v: None if v is None else [round(float(x), 4) for x in v]  # noqa: E731
        return {
            "episode": int(self.episode),
            "camera": self.camera,
            "n_frames": int(len(self.frames)),
            "n_visible": int(self.n_visible),
            "picks": [[int(g), int(r)] for g, r in self.picks],
            "start_pose": f(self.start_pose),
            "end_pose": f(self.end_pose),
            "end_source": self.end_source,
            "footprint_m": None if self.footprint is None else [round(float(v), 4) for v in self.footprint],
            "fk_grasp": f(self.fk_grasp),
            "fk_release": f(self.fk_release),
            "vision_vs_fk_m": None if not math.isfinite(self.vision_vs_fk) else round(float(self.vision_vs_fk), 4),
            "end_vs_fk_m": None if not math.isfinite(self.end_vs_fk) else round(float(self.end_vs_fk), 4),
            "note": self.note,
        }

    def to_dict(self) -> dict[str, Any]:
        """Everything, per frame, for the API (floats rounded)."""
        d = self.summary()
        d.update(
            {
                "frames": self.frames.tolist(),
                "xyz": np.round(self.xyz, 4).tolist(),
                "visible": self.visible.tolist(),
                "source": self.source_names(),
                "centroid": [None if not np.isfinite(c).all() else [round(float(c[0]), 1), round(float(c[1]), 1)] for c in self.centroid],
                "bbox": [None if not np.isfinite(b).all() else [int(v) for v in b] for b in self.bbox],
            }
        )
        return d


def held_windows(episode: Episode, thresholds: Any, tip: Any, unit_map: Any, empty: Any) -> list[tuple[int, int]]:
    """``[grasp, release]`` frame windows of the grasps that held something
    (:func:`antares.task.find_grasps`); a pick never let go runs to the end."""
    from antares.task import find_grasps

    out = []
    for g in find_grasps(episode, thresholds, tip, unit_map, empty):
        if not g.held:
            continue
        r = g.release_frame if g.release_frame is not None else len(episode)
        out.append((int(g.frame), int(r)))
    return out


def iter_frames(repo_id: str, camera: str, episode: int, n_frames: int, dataset: Dataset | None, chunk: int = 96) -> Iterator[tuple[int, np.ndarray]]:
    """Decode an episode in chunks (never cached: whole episodes are too big for the frame cache)."""
    from antares import video

    for start in range(0, n_frames, chunk):
        idx = list(range(start, min(n_frames, start + chunk)))
        arr = video.frames(repo_id, camera, episode, idx, dataset, cache=False)
        for i, f in zip(idx, arr):
            yield i, f


def track_episode(
    episode: Episode,
    frames: Iterator[tuple[int, np.ndarray]] | Sequence[np.ndarray],
    camera_name: str,
    camera: Camera,
    bgm: BackgroundModel,
    arm: Any,
    hulls: ArmHulls,
    prior: ColourPrior,
    picks: Sequence[tuple[int, int]],
    plane_z: float,
    *,
    dilate_px: int = ARM_DILATE_PX,
) -> ObjectTrack:
    """Track the object through one episode; see the module docstring.

    ``frames`` yields ``(index, RGB)`` in order (or is a sequence of RGB
    frames); ``picks`` are the held windows; ``plane_z`` is the height the
    object's centre sits at (table + half height).
    """
    T = len(episode)
    q = arm.unit_map.to_rad(episode.state)
    tips = fingertips(arm, arm.fk(q))
    bg_lab = to_lab(bgm.bg)
    xyz = np.full((T, 3), np.nan)
    visible = np.zeros(T, dtype=bool)
    source = np.full(T, SOURCE_CODES.index("hidden"), dtype=np.int8)
    centroid = np.full((T, 2), np.nan)
    bbox = np.full((T, 4), np.nan)
    area = np.zeros(T, dtype=np.int32)
    held = np.zeros(T, dtype=bool)
    for g, r in picks:
        held[max(0, g) : min(T, r + 1)] = True
    gate_track = max(GATE_MIN_PX, GATE_SIDES * prior.side_px)
    gate_release = max(RELEASE_GATE_MIN_PX, RELEASE_GATE_SIDES * prior.side_px)
    gate = gate_track
    expect: np.ndarray | None = None  # pixel position to look near
    last_xyz: np.ndarray | None = None
    first_pick = picks[0][0] if picks else T
    footprints: list[tuple[float, float]] = []
    it = frames if not isinstance(frames, (list, tuple, np.ndarray)) else enumerate(frames)
    for t, rgb in it:
        if t >= T:
            break
        if held[t]:
            xyz[t] = tips[t]
            source[t] = SOURCE_CODES.index("held")
            last_xyz = tips[t]
            # After a release the object is near where the jaws opened.
            uv, depth = camera.project(tips[t][None])
            expect = uv[0] if depth[0] > 0 else None
            gate = gate_release
            continue
        lab = to_lab(rgb)
        arm_mask = sim_arm_mask(arm, hulls, camera, q[t], dilate_px)
        fg = foreground(lab, bg_lab, bgm, arm_mask)
        blobs, labels = object_blobs(lab, fg, prior)
        pick: Blob | None = None
        if blobs:
            if expect is None:
                pick = blobs[0]  # the biggest object-coloured blob starts the track
            else:
                near = [(math.hypot(b.centroid[0] - expect[0], b.centroid[1] - expect[1]), b) for b in blobs]
                d, b = min(near, key=lambda x: x[0])
                if d <= gate:
                    pick = b
        if pick is not None:
            c = np.asarray(pick.centroid)
            p = ray_plane(camera, c[None], plane_z)[0]
            if np.isfinite(p).all():
                xyz[t] = p
                visible[t] = True
                source[t] = SOURCE_CODES.index("vision")
                centroid[t] = c
                bbox[t] = pick.bbox
                area[t] = pick.area
                expect = c
                gate = gate_track
                last_xyz = p
                if t < first_pick - OCCLUSION_PAD and t % 4 == 0 and len(footprints) < 8:
                    fp = footprint_on_plane(labels == pick.label, camera, plane_z)
                    if fp is not None:
                        footprints.append((fp[0], fp[1]))
                continue
        if last_xyz is not None:
            xyz[t] = last_xyz
    # Fill frames before the object was first seen with the first sighting.
    seen = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    if seen.size:
        xyz[: seen[0]] = xyz[seen[0]]
    note = ""
    pre = visible.copy()
    pre[max(0, first_pick - OCCLUSION_PAD) :] = False
    start_pose = np.median(xyz[pre], axis=0) if pre.any() else None
    if start_pose is None and visible[:first_pick].any():
        start_pose = np.median(xyz[visible[:first_pick]], axis=0)
    fk_grasp = tips[picks[0][0]] if picks else None
    last_release = picks[-1][1] if picks else None
    fk_release = tips[min(T - 1, last_release)] if last_release is not None and last_release < T else None
    post = visible.copy()
    if last_release is not None:
        post[: min(T, last_release + 1)] = False
    else:
        post[:] = False
    if post.any():
        idx = np.flatnonzero(post)
        tail = idx[-max(1, len(idx) // 3) :]  # once it has come to rest
        end_pose, end_source = np.median(xyz[tail], axis=0), "vision"
    elif fk_release is not None:
        end_pose, end_source = fk_release.copy(), "fk"
        note = "object not seen after the release; end pose is the fingertip"
    else:
        end_pose, end_source = None, "none"
    if not picks:
        note = "no held grasp in this episode"
    vis_fk = float(np.linalg.norm(start_pose - fk_grasp)) if start_pose is not None and fk_grasp is not None else float("nan")
    end_fk = float(np.linalg.norm(end_pose - fk_release)) if end_source == "vision" and fk_release is not None else float("nan")
    fp = tuple(float(v) for v in np.median(np.array(footprints), axis=0)) if footprints else None
    return ObjectTrack(
        episode=int(episode.index),
        camera=camera_name,
        frames=np.arange(T),
        xyz=xyz,
        visible=visible,
        source=source,
        centroid=centroid,
        bbox=bbox,
        area=area,
        picks=[(int(g), int(r)) for g, r in picks],
        start_pose=start_pose,
        end_pose=end_pose,
        end_source=end_source,
        footprint=fp,
        fk_grasp=fk_grasp,
        fk_release=fk_release,
        vision_vs_fk=vis_fk,
        end_vs_fk=end_fk,
        n_visible=int(visible.sum()),
        note=note,
    )


# --------------------------------------------------------------------------
# The fit over a dataset
# --------------------------------------------------------------------------


def _cloud(points: Sequence[np.ndarray]) -> dict[str, Any] | None:
    pts = np.array([p for p in points if p is not None], dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return None
    return {
        "n": int(len(pts)),
        "mean": pts.mean(axis=0).tolist(),
        "std": pts.std(axis=0).tolist(),
        "min": pts.min(axis=0).tolist(),
        "max": pts.max(axis=0).tolist(),
        "points": np.round(pts, 4).tolist(),
    }


def _quartiles(values: Sequence[float]) -> dict[str, float] | None:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "max": float(v.max())}


@dataclass
class ObjectFit:
    lock: str
    dataset: str
    camera: str
    robot: str
    table_z: float
    object_half_height: float
    prior: ColourPrior
    description: str
    tracks: list[dict[str, Any]]  # ObjectTrack.summary() per episode
    start: dict[str, Any] | None  # cloud of vision start poses
    end: dict[str, Any] | None
    footprint_m: dict[str, Any] | None  # {"long": quartiles, "short": quartiles}
    vision_vs_fk_m: dict[str, float] | None
    end_vs_fk_m: dict[str, float] | None
    #: Mean of (vision start pose - FK fingertip at the grasp), base frame (m):
    #: a systematic offset between where the camera puts the object and where
    #: the kinematics put the jaws; ``vision_vs_fk_centred_m`` is the spread
    #: left once it is removed.
    vision_vs_fk_offset_m: list[float] | None = None
    vision_vs_fk_centred_m: dict[str, float] | None = None
    cross_camera: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    n_episodes: int = 0
    n_tracked: int = 0
    elapsed_s: float = 0.0
    created_at: str = ""
    antares_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return {
            "lock": self.lock,
            "dataset": self.dataset,
            "camera": self.camera,
            "robot": self.robot,
            "table_z": self.table_z,
            "object_half_height": self.object_half_height,
            "prior": self.prior.to_dict(),
            "description": self.description,
            "tracks": list(self.tracks),
            "start": self.start,
            "end": self.end,
            "footprint_m": self.footprint_m,
            "vision_vs_fk_m": self.vision_vs_fk_m,
            "end_vs_fk_m": self.end_vs_fk_m,
            "vision_vs_fk_offset_m": self.vision_vs_fk_offset_m,
            "vision_vs_fk_centred_m": self.vision_vs_fk_centred_m,
            "cross_camera": self.cross_camera,
            "payload": self.payload,
            "n_episodes": self.n_episodes,
            "n_tracked": self.n_tracked,
            "elapsed_s": self.elapsed_s,
            "created_at": self.created_at,
            "antares_version": self.antares_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ObjectFit:
        return cls(
            lock=str(d.get("lock", "")),
            dataset=str(d.get("dataset", "")),
            camera=str(d.get("camera", "")),
            robot=str(d.get("robot", "")),
            table_z=float(d.get("table_z", 0.0)),
            object_half_height=float(d.get("object_half_height", OBJECT_HALF_HEIGHT)),
            prior=ColourPrior.from_dict(d["prior"]),
            description=str(d.get("description", "")),
            tracks=list(d.get("tracks") or []),
            start=d.get("start"),
            end=d.get("end"),
            footprint_m=d.get("footprint_m"),
            vision_vs_fk_m=d.get("vision_vs_fk_m"),
            end_vs_fk_m=d.get("end_vs_fk_m"),
            vision_vs_fk_offset_m=d.get("vision_vs_fk_offset_m"),
            vision_vs_fk_centred_m=d.get("vision_vs_fk_centred_m"),
            cross_camera=d.get("cross_camera"),
            payload=d.get("payload"),
            n_episodes=int(d.get("n_episodes", 0)),
            n_tracked=int(d.get("n_tracked", 0)),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            created_at=str(d.get("created_at", "")),
            antares_version=str(d.get("antares_version", "")),
        )

    # -- what the scene needs ---------------------------------------------------

    @property
    def footprint(self) -> tuple[float, float] | None:
        """``(long, short)`` in metres (medians), or None."""
        if not self.footprint_m or not self.footprint_m.get("long") or not self.footprint_m.get("short"):
            return None
        return float(self.footprint_m["long"]["median"]), float(self.footprint_m["short"]["median"])

    @property
    def mass_kg(self) -> float | None:
        """The payload estimate when it is a measurement (the verdict is
        ``consistent``: above the noise, both load joints agree), else None."""
        p = self.payload or {}
        m = p.get("mass_kg")
        if not isinstance(m, (int, float)) or not math.isfinite(m) or m <= 0:
            return None
        return float(m) if str(p.get("verdict", "")).startswith("consistent") else None

    @property
    def measured_mass_kg(self) -> float | None:
        """The raw median mass estimate, whatever the verdict (for reports)."""
        m = (self.payload or {}).get("mass_kg")
        return float(m) if isinstance(m, (int, float)) and math.isfinite(m) else None

    @property
    def start_points(self) -> np.ndarray:
        return np.asarray((self.start or {}).get("points", []), dtype=np.float64).reshape(-1, 3)

    @property
    def end_points(self) -> np.ndarray:
        return np.asarray((self.end or {}).get("points", []), dtype=np.float64).reshape(-1, 3)

    def sample_start(self, n: int, rng: np.random.Generator | None = None, jitter_m: float = 0.01) -> np.ndarray:
        """``n`` start positions (N,3): a logged vision start pose each plus Gaussian jitter."""
        pts = self.start_points
        if len(pts) == 0:
            raise ValueError("no vision start poses to sample from")
        rng = rng if rng is not None else np.random.default_rng()
        idx = rng.integers(len(pts), size=int(n))
        return pts[idx] + rng.normal(0.0, jitter_m, (int(n), 3))

    def lock_entry(self) -> dict[str, Any]:
        """The compact record kept in the lock's ``objects`` field."""
        fp = self.footprint
        p = self.payload or {}
        return {
            "camera": self.camera,
            "description": self.description,
            "rgb": list(self.prior.rgb),
            "footprint_m": None if fp is None else [round(fp[0], 4), round(fp[1], 4)],
            "half_height": self.object_half_height,
            "start_mean": None if not self.start else [round(v, 4) for v in self.start["mean"]],
            "start_std": None if not self.start else [round(v, 4) for v in self.start["std"]],
            "end_mean": None if not self.end else [round(v, 4) for v in self.end["mean"]],
            "end_std": None if not self.end else [round(v, 4) for v in self.end["std"]],
            "vision_vs_fk_m": None if not self.vision_vs_fk_m else round(self.vision_vs_fk_m["median"], 4),
            "mass_kg": self.mass_kg,
            "measured_mass_kg": self.measured_mass_kg,
            "mass_iqr_kg": p.get("mass_iqr_kg"),
            "payload_verdict": p.get("verdict"),
            "n_episodes": self.n_episodes,
            "n_tracked": self.n_tracked,
            "created_at": self.created_at,
        }


def describe(prior: ColourPrior, footprint: tuple[float, float] | None, table_rgb: Sequence[int] | None) -> str:
    """One line: colour name, size, on what."""
    L, a, b = prior.lab
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a)) % 360
    if chroma < 12:
        colour = "white" if L > 70 else "black" if L < 30 else "grey"
    elif hue < 48 or hue >= 335:
        colour = "red" if L < 85 else "pink"
    elif hue < 95:
        colour = "orange" if hue < 70 else "yellow"
    elif hue < 160:
        colour = "green"
    elif hue < 260:
        colour = "blue"
    else:
        colour = "purple"
    size = f"about {footprint[0] * 100:.1f} x {footprint[1] * 100:.1f} cm" if footprint else "size unknown"
    on = ""
    if table_rgb is not None:
        tl = float(np.mean(table_rgb))
        on = " on a light table" if tl > 150 else " on a dark table" if tl < 80 else " on the table"
    return f"a {colour} object, {size}{on}"


def fit_objects(
    dataset: Dataset,
    arm: Any,
    lock_name: str,
    camera_name: str,
    camera: Camera,
    bgm: BackgroundModel,
    table_z: float,
    *,
    object_half_height: float = OBJECT_HALF_HEIGHT,
    table_rgb: Sequence[int] | None = None,
    episodes: Sequence[int] | None = None,
    hulls: ArmHulls | None = None,
    frame_source: Callable[[Episode], Iterator[tuple[int, np.ndarray]]] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[ObjectFit, list[ObjectTrack]]:
    """Find and track the object in every episode of ``dataset`` seen by ``camera``.

    ``arm`` carries the lock's unit map and robot; ``bgm`` is the camera's
    background model; ``table_z`` the scene's table height. ``frame_source``
    (tests) replaces the video decoder. Returns the fit and the tracks.
    """
    from antares.task import Tip, dataset_thresholds, empty_level

    t0 = time.perf_counter()
    hulls = hulls or arm_hulls(arm.model)
    eps = [ep for ep in dataset.episodes if episodes is None or ep.index in set(episodes)]
    if not eps:
        raise ValueError("no episodes to track")
    source = frame_source or (lambda ep: iter_frames(dataset.repo_id, camera_name, ep.index, len(ep), dataset))
    plane_z = table_z + object_half_height
    thr = dataset_thresholds(list(dataset.episodes))
    empty = empty_level(list(dataset.episodes), thr)
    tip = Tip(arm.model)

    # The colour prior from the first frames, where the object is at rest.
    samples = []
    for ep in eps[: min(len(eps), 12)]:
        wanted = [f for f in PRIOR_FRAMES if f < len(ep)]
        q = arm.unit_map.to_rad(ep.state)
        got = {}
        for i, rgb in source(ep):
            if i in wanted:
                got[i] = rgb
            if i >= max(wanted):
                break
        for i in wanted:
            if i in got:
                samples.append((got[i], sim_arm_mask(arm, hulls, camera, q[i])))
    prior = learn_prior(samples, bgm)
    if on_progress is not None:
        on_progress({"stage": "prior", "episode": None, "i": 0, "n": len(eps), "prior": prior.to_dict(), "elapsed_s": time.perf_counter() - t0})

    tracks: list[ObjectTrack] = []
    stop = should_stop or (lambda: False)
    for i, ep in enumerate(eps):
        if stop():
            break
        picks = held_windows(ep, thr, tip, arm.unit_map, empty)
        tr = track_episode(ep, source(ep), camera_name, camera, bgm, arm, hulls, prior, picks, plane_z)
        tracks.append(tr)
        if on_progress is not None:
            on_progress({"stage": "track", "episode": ep.index, "i": i + 1, "n": len(eps), "track": tr.summary(), "elapsed_s": time.perf_counter() - t0})

    starts = [t.start_pose for t in tracks if t.start_pose is not None]
    ends = [t.end_pose for t in tracks if t.end_pose is not None and t.end_source == "vision"]
    longs = [t.footprint[0] for t in tracks if t.footprint is not None]
    shorts = [t.footprint[1] for t in tracks if t.footprint is not None]
    fp_q = {"long": _quartiles(longs), "short": _quartiles(shorts)} if longs else None
    fp = (fp_q["long"]["median"], fp_q["short"]["median"]) if fp_q and fp_q["long"] and fp_q["short"] else None
    diffs = np.array([t.start_pose - t.fk_grasp for t in tracks if t.start_pose is not None and t.fk_grasp is not None]).reshape(-1, 3)
    offset = diffs.mean(axis=0).tolist() if len(diffs) else None
    centred = _quartiles(np.linalg.norm(diffs - diffs.mean(axis=0), axis=1).tolist()) if len(diffs) else None
    fit = ObjectFit(
        lock=lock_name,
        dataset=dataset.repo_id,
        camera=camera_name,
        robot=str(getattr(arm, "robot_name", "")),
        table_z=float(table_z),
        object_half_height=float(object_half_height),
        prior=prior,
        description=describe(prior, fp, table_rgb),
        tracks=[t.summary() for t in tracks],
        start=_cloud(starts),
        end=_cloud(ends),
        footprint_m=fp_q,
        vision_vs_fk_m=_quartiles([t.vision_vs_fk for t in tracks]),
        end_vs_fk_m=_quartiles([t.end_vs_fk for t in tracks]),
        vision_vs_fk_offset_m=offset,
        vision_vs_fk_centred_m=centred,
        n_episodes=len(eps),
        n_tracked=sum(1 for t in tracks if t.start_pose is not None),
        elapsed_s=time.perf_counter() - t0,
        created_at=now_iso(),
    )
    return fit, tracks


# --------------------------------------------------------------------------
# Cross-camera check
# --------------------------------------------------------------------------


def cross_camera_check(
    dataset: Dataset,
    arm: Any,
    cameras: Mapping[str, Camera],
    backgrounds: Mapping[str, BackgroundModel],
    plane_z: float,
    primary: str,
    *,
    frames: Sequence[int] = (0, 6, 12),
    max_episodes: int = 12,
    hulls: ArmHulls | None = None,
    frame_source: Callable[[Episode, str, Sequence[int]], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Triangulate the object at rest from two cameras and compare with the
    plane intersection of ``primary``.

    For the first ``frames`` of up to ``max_episodes`` episodes the object is
    detected in both cameras (each with its own colour prior); the closest
    point between the two rays is compared with the primary camera's
    plane point. Reports the median 3D distance, the median height of the
    triangulated point above the plane, and the ray gap.
    """
    from antares import video

    hulls = hulls or arm_hulls(arm.model)
    names = [primary] + [n for n in cameras if n != primary]
    if len(names) < 2:
        return {"n": 0, "note": "needs two fitted cameras"}
    other = names[1]
    get = frame_source or (lambda ep, cam, idx: video.frames(dataset.repo_id, cam, ep.index, list(idx), dataset, cache=False))
    eps = list(dataset.episodes)[:max_episodes]
    rgb: dict[str, dict[tuple[int, int], np.ndarray]] = {n: {} for n in (primary, other)}
    for ep in eps:
        idx = [f for f in frames if f < len(ep)]
        for n in (primary, other):
            for f, a in zip(idx, get(ep, n, idx)):
                rgb[n][(ep.index, f)] = a
    priors = {}
    for n in (primary, other):
        samples = []
        for (e, f), a in rgb[n].items():
            ep = next(x for x in eps if x.index == e)
            samples.append((a, sim_arm_mask(arm, hulls, cameras[n], arm.unit_map.to_rad(ep.state[f]))))
        try:
            priors[n] = learn_prior(samples, backgrounds[n])
        except ValueError as e:
            return {"n": 0, "note": f"{n}: {e}", "primary": primary, "other": other}
    dists, heights, gaps, plane_other = [], [], [], []
    for ep in eps:
        for f in frames:
            if (ep.index, f) not in rgb[primary] or (ep.index, f) not in rgb[other]:
                continue
            q = arm.unit_map.to_rad(ep.state[f])
            cents = {}
            for n in (primary, other):
                cam = cameras[n]
                lab = to_lab(rgb[n][(ep.index, f)])
                fg = foreground(lab, to_lab(backgrounds[n].bg), backgrounds[n], sim_arm_mask(arm, hulls, cam, q))
                blobs, _ = object_blobs(lab, fg, priors[n])
                if blobs:
                    cents[n] = np.asarray(blobs[0].centroid)
            if len(cents) < 2:
                continue
            p_plane = ray_plane(cameras[primary], cents[primary][None], plane_z)[0]
            p_other = ray_plane(cameras[other], cents[other][None], plane_z)[0]
            p_tri, gap = triangulate(cameras[primary], cents[primary], cameras[other], cents[other])
            if not (np.isfinite(p_plane).all() and np.isfinite(p_tri).all()):
                continue
            dists.append(float(np.linalg.norm(p_tri - p_plane)))
            heights.append(float(p_tri[2] - plane_z))
            gaps.append(gap)
            plane_other.append(float(np.linalg.norm(p_other[:2] - p_plane[:2])) if np.isfinite(p_other).all() else float("nan"))
    if not dists:
        return {"n": 0, "note": "object not seen in both cameras", "primary": primary, "other": other}
    return {
        "n": len(dists),
        "primary": primary,
        "other": other,
        "triangulated_vs_plane_m": _quartiles(dists),
        "triangulated_height_above_plane_m": _quartiles(heights),
        "ray_gap_m": _quartiles(gaps),
        "other_plane_vs_primary_plane_xy_m": _quartiles(plane_other),
    }


# --------------------------------------------------------------------------
# Files next to the locks
# --------------------------------------------------------------------------


def objects_path(lock_name: str, lock_dir: Path | str = "locks") -> Path:
    return Path(lock_dir) / f"{lock_name}{OBJECTS_SUFFIX}"


def tracks_path(lock_name: str, camera: str, lock_dir: Path | str = "locks") -> Path:
    return Path(lock_dir) / TRACKS_DIRNAME / f"{lock_name}_{camera}_tracks.npz"


def load_object_fit(lock_name: str, lock_dir: Path | str = "locks") -> ObjectFit | None:
    p = objects_path(lock_name, lock_dir)
    if not p.is_file():
        return None
    return ObjectFit.from_dict(json.loads(p.read_text()))


def save_tracks(tracks: Sequence[ObjectTrack], lock_name: str, camera: str, lock_dir: Path | str = "locks") -> Path:
    p = tracks_path(lock_name, camera, lock_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for t in tracks:
        k = f"ep{t.episode}"
        arrays[f"{k}_xyz"] = t.xyz.astype(np.float32)
        arrays[f"{k}_source"] = t.source.astype(np.int8)
        arrays[f"{k}_centroid"] = t.centroid.astype(np.float32)
        arrays[f"{k}_bbox"] = t.bbox.astype(np.float32)
    np.savez_compressed(p, **arrays)
    return p


def load_track(fit: ObjectFit, episode: int, lock_dir: Path | str = "locks") -> ObjectTrack | None:
    """One episode's per-frame track from the saved ``.npz`` (None if absent)."""
    p = tracks_path(fit.lock, fit.camera, lock_dir)
    if not p.is_file():
        return None
    with np.load(p) as z:
        k = f"ep{episode}"
        if f"{k}_xyz" not in z:
            return None
        xyz = z[f"{k}_xyz"].astype(np.float64)
        source = z[f"{k}_source"].astype(np.int8)
        centroid = z[f"{k}_centroid"].astype(np.float64)
        bbox = z[f"{k}_bbox"].astype(np.float64)
    s = next((t for t in fit.tracks if int(t["episode"]) == int(episode)), {})
    vis = source == SOURCE_CODES.index("vision")
    v = lambda key: None if s.get(key) is None else np.asarray(s[key], dtype=np.float64)  # noqa: E731
    return ObjectTrack(
        episode=int(episode),
        camera=fit.camera,
        frames=np.arange(len(xyz)),
        xyz=xyz,
        visible=vis,
        source=source,
        centroid=centroid,
        bbox=bbox,
        area=np.zeros(len(xyz), dtype=np.int32),
        picks=[(int(g), int(r)) for g, r in s.get("picks", [])],
        start_pose=v("start_pose"),
        end_pose=v("end_pose"),
        end_source=str(s.get("end_source", "none")),
        footprint=None if s.get("footprint_m") is None else tuple(float(x) for x in s["footprint_m"]),
        fk_grasp=v("fk_grasp"),
        fk_release=v("fk_release"),
        vision_vs_fk=float(s.get("vision_vs_fk_m") if s.get("vision_vs_fk_m") is not None else "nan"),
        end_vs_fk=float(s.get("end_vs_fk_m") if s.get("end_vs_fk_m") is not None else "nan"),
        n_visible=int(vis.sum()),
        note=str(s.get("note", "")),
    )


def save_object_fit(
    fit: ObjectFit, lock_name: str, lock_dir: Path | str = "locks", tracks: Sequence[ObjectTrack] | None = None
) -> Path:
    """Write ``<lock>.objects.json`` (+ the tracks ``.npz``) and the lock's ``objects`` field (if the lock exists)."""
    from antares.lock import LOCK_SUFFIX, RobotLock

    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    if tracks:
        save_tracks(tracks, lock_name, fit.camera, lock_dir)
    path = objects_path(lock_name, lock_dir)
    path.write_text(json.dumps(fit.to_dict(), indent=2, sort_keys=True) + "\n")
    lock_file = lock_dir / f"{lock_name}{LOCK_SUFFIX}"
    if lock_file.is_file():
        lock = RobotLock.load(lock_file)
        lock.replace(objects=fit.lock_entry()).save(lock_file)
    return path


__all__ = [
    "ARM_DILATE_PX",
    "Blob",
    "ColourPrior",
    "OBJECTS_SUFFIX",
    "ObjectFit",
    "ObjectTrack",
    "SOURCE_CODES",
    "TRACKS_DIRNAME",
    "cross_camera_check",
    "describe",
    "find_blobs",
    "fit_objects",
    "footprint_on_plane",
    "foreground",
    "held_windows",
    "iter_frames",
    "learn_prior",
    "load_object_fit",
    "load_track",
    "object_blobs",
    "objects_path",
    "pixel_rays",
    "ray_plane",
    "save_object_fit",
    "save_tracks",
    "sim_arm_mask",
    "track_episode",
    "tracks_path",
    "triangulate",
]
