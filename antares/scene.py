"""The static scene around the arm, recovered from the same logs.

Phase 1 (:mod:`antares.camera`) fitted the dataset's fixed cameras to the
arm. The rest of what those cameras see is static too, and the logs say
where it is:

* **Table.** The gripper command closing is a grasp (:mod:`antares.events`),
  and at a grasp the fingertips are on the object, which sits on the table.
  ``Arm.fk`` at every grasp instant gives the fingertip height in the base
  frame; the lower cluster of those heights, minus an assumed object
  half-height, is the table. The lowest fingertip over every frame of every
  episode is the cross-check: the arm never goes below the table.
* **Plane normal.** The SO-101 base is bolted to the table, so the plane is
  taken horizontal; a plane is still fitted through the grasp points and its
  tilt reported, and used instead only when the points clearly disagree.
* **Workspace.** The bounding box and convex hull of the fingertip
  positions over every episode, plus the grasp and release points: where
  objects were picked and where they were put down.
* **Backdrops.** Each fitted camera's background model (the per-pixel median
  over frames, so the moving arm is gone) is the clean image of everything
  static. The estimated table plane is projected into the camera and
  compared with the dominant large colour region of that image, which is
  the honest check that the geometric and the visual table agree.

The result is a :class:`SceneFit` saved as ``locks/<lock>.scene.json`` with
the backdrops as ``locks/scene/<lock>_<camera>_backdrop.jpg``, a compact
entry in the lock's ``scene`` field, and :mod:`antares.export` turns it
into a table geom and textured backdrop quads in ``scene.xml``.
:func:`sim_view` composites what a policy camera would see in the sim, with
no OpenGL: backdrop, faint table, and the arm's hulls filled far to near.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from antares import __version__
from antares.data import Dataset
from antares.events import GRIPPER_INDEX, gripper_thresholds, gripper_transitions
from antares.project import ArmHulls, Camera, arm_hulls, project_polygons, quat_to_mat

SCENE_SUFFIX = ".scene.json"
BACKDROP_DIRNAME = "scene"

#: The site between the jaws of the Menagerie SO-101 (the tool centre point).
TIP_SITE = "gripperframe"
#: Bodies whose hulls are the jaws; their lowest point is the second height estimate.
JAW_BODIES: tuple[str, ...] = ("gripper", "moving_jaw_so101_v1")

#: Assumed half-height of a grasped object (m): the fingertips close around
#: its middle, so the table is this far below them. A guess, stated in the
#: result rather than hidden.
OBJECT_HALF_HEIGHT = 0.015
#: Grasps whose fingertip height is within this of the lowest decile are
#: "on the table"; higher ones are re-grasps in the air or at a container.
ON_TABLE_BAND = 0.015
#: Above this fitted tilt (with grasp points spread enough to tell) the
#: horizontal assumption is dropped and the fitted plane used.
TILT_TRUST_DEG = 5.0
TILT_MIN_SPREAD = 0.10  # m of grasp-point spread in x and y needed to judge tilt
#: Margin around the fingertip workspace for the table geom and the plane check.
WORKSPACE_MARGIN = 0.10
#: Backdrop quads sit this far along the camera's optical axis in the export.
BACKDROP_DISTANCE = 3.0
NEAR_PLANE = 0.05


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Fingertips
# --------------------------------------------------------------------------


def tip_frame(model: mujoco.MjModel) -> tuple[int, np.ndarray]:
    """``(body id, body-frame offset)`` of the fingertip point: the
    ``gripperframe`` site, or the gripper body origin if the model has none."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
    if sid >= 0:
        return int(model.site_bodyid[sid]), np.asarray(model.site_pos[sid], dtype=np.float64).copy()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    if bid < 0:
        bid = model.nbody - 1
    return int(bid), np.zeros(3)


def fingertips(arm: Any, body_poses: np.ndarray) -> np.ndarray:
    """Fingertip world positions from ``Arm.fk`` body poses, (nbody,7) -> (3,) or (T,nbody,7) -> (T,3)."""
    body, offset = tip_frame(arm.model)
    poses = np.asarray(body_poses, dtype=np.float64)
    if poses.ndim == 2:
        p = poses[body]
        return p[:3] + quat_to_mat(p[3:]) @ offset
    out = np.empty((poses.shape[0], 3))
    for t in range(poses.shape[0]):
        p = poses[t, body]
        out[t] = p[:3] + quat_to_mat(p[3:]) @ offset
    return out


def jaw_low_z(hulls: ArmHulls, body_poses: np.ndarray, bodies: Sequence[str] = JAW_BODIES) -> float:
    """Lowest hull vertex of the jaw bodies for one posed arm (world z)."""
    pts = hulls.world_points(body_poses)
    lows = [pts[s:e, 2].min() for g, (s, e) in zip(hulls.geoms, hulls.slices) if g.body_name in bodies and e > s]
    return float(min(lows)) if lows else float(pts[:, 2].min())


# --------------------------------------------------------------------------
# Table plane
# --------------------------------------------------------------------------


def _stats(z: np.ndarray) -> dict[str, float]:
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size == 0:
        return {}
    pct = np.percentile(z, [5, 10, 25, 50, 75, 90, 95])
    return {
        "n": int(z.size),
        "min": float(z.min()),
        "p5": float(pct[0]),
        "p10": float(pct[1]),
        "p25": float(pct[2]),
        "median": float(pct[3]),
        "p75": float(pct[4]),
        "p90": float(pct[5]),
        "p95": float(pct[6]),
        "max": float(z.max()),
        "mean": float(z.mean()),
        "std": float(z.std()),
    }


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane through ``points`` (N,3): ``(unit normal with n_z >= 0, tilt from vertical in deg)``.

    Fewer than three points, or points on a line, give the horizontal plane.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0
    c = p - p.mean(axis=0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    if s[1] < 1e-9:  # collinear
        return np.array([0.0, 0.0, 1.0]), 0.0
    n = vt[2]
    if n[2] < 0:
        n = -n
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))
    return n, tilt


@dataclass(frozen=True)
class TablePlane:
    """Where the table is, and the evidence.

    ``z`` is the plane height at the arm's base (world ``x = y = 0``);
    ``normal`` is ``(0, 0, 1)`` unless ``horizontal`` is False. ``z_grasp``
    is the median fingertip height of the on-table grasp cluster, so
    ``z = z_grasp - object_half_height``. ``fingertip_z`` and ``jaw_low_z``
    are the statistics of the fingertip (tool centre point) height and the
    jaws' lowest hull vertex at every grasp instant; ``episode_min_z`` those
    of the lowest fingertip per episode (the arm never goes below the table,
    so their spread says how repeatable the floor of the motion is).
    """

    z: float
    normal: tuple[float, float, float]
    tilt_deg: float  # of the plane fitted through the on-table grasp points
    horizontal: bool  # True: normal forced to +z (the base is bolted to the table)
    n_grasps: int
    n_on_table: int
    z_grasp: float
    object_half_height: float
    fingertip_z: dict[str, float]
    jaw_low_z: dict[str, float]
    episode_min_z: dict[str, float]
    min_fingertip_z: float  # over every frame of every episode

    def to_dict(self) -> dict[str, Any]:
        return {
            "z": self.z,
            "normal": list(self.normal),
            "tilt_deg": self.tilt_deg,
            "horizontal": self.horizontal,
            "n_grasps": self.n_grasps,
            "n_on_table": self.n_on_table,
            "z_grasp": self.z_grasp,
            "object_half_height": self.object_half_height,
            "fingertip_z": dict(self.fingertip_z),
            "jaw_low_z": dict(self.jaw_low_z),
            "episode_min_z": dict(self.episode_min_z),
            "min_fingertip_z": self.min_fingertip_z,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TablePlane:
        return cls(
            z=float(d["z"]),
            normal=tuple(float(v) for v in d.get("normal", (0.0, 0.0, 1.0))),
            tilt_deg=float(d.get("tilt_deg", 0.0)),
            horizontal=bool(d.get("horizontal", True)),
            n_grasps=int(d.get("n_grasps", 0)),
            n_on_table=int(d.get("n_on_table", 0)),
            z_grasp=float(d.get("z_grasp", d["z"])),
            object_half_height=float(d.get("object_half_height", OBJECT_HALF_HEIGHT)),
            fingertip_z={k: float(v) for k, v in (d.get("fingertip_z") or {}).items()},
            jaw_low_z={k: float(v) for k, v in (d.get("jaw_low_z") or {}).items()},
            episode_min_z={k: float(v) for k, v in (d.get("episode_min_z") or {}).items()},
            min_fingertip_z=float(d.get("min_fingertip_z", d["z"])),
        )

    def height_at(self, x: float, y: float) -> float:
        """Plane height at world ``(x, y)``."""
        nx, ny, nz = self.normal
        return self.z - (nx * x + ny * y) / max(1e-9, nz)


def fit_table(
    grasp_tips: np.ndarray,
    min_fingertip_z: float,
    episode_min_z: Sequence[float] = (),
    jaw_lows: Sequence[float] = (),
    object_half_height: float = OBJECT_HALF_HEIGHT,
    band: float = ON_TABLE_BAND,
) -> TablePlane:
    """Table plane from fingertip positions at the grasp instants (N,3).

    The on-table cluster is every grasp within ``band`` of the lowest decile
    of fingertip height; its median minus ``object_half_height`` is the
    table. The plane through the cluster's points gives the tilt; it
    replaces the horizontal assumption only when it exceeds
    :data:`TILT_TRUST_DEG` and the points span at least
    :data:`TILT_MIN_SPREAD` in both x and y (a plane through points on a
    line says nothing about tilt).
    """
    tips = np.asarray(grasp_tips, dtype=np.float64).reshape(-1, 3)
    if tips.shape[0] == 0:
        raise ValueError("no grasp instants: the gripper command never closes")
    z = tips[:, 2]
    ref = float(np.percentile(z, 10))
    on = z <= ref + band
    cluster = tips[on]
    z_grasp = float(np.median(cluster[:, 2]))
    normal, tilt = fit_plane(cluster)
    spread = cluster.max(axis=0) - cluster.min(axis=0) if len(cluster) else np.zeros(3)
    horizontal = not (tilt > TILT_TRUST_DEG and spread[0] >= TILT_MIN_SPREAD and spread[1] >= TILT_MIN_SPREAD)
    if horizontal:
        normal = np.array([0.0, 0.0, 1.0])
        z_table = z_grasp - object_half_height
    else:
        # Height of the fitted plane at the base, shifted down by the object.
        c = cluster.mean(axis=0)
        z_table = float(c[2] + (normal[0] * c[0] + normal[1] * c[1]) / normal[2]) - object_half_height
    ep = np.asarray(list(episode_min_z), dtype=np.float64)
    ep_stats = _stats(ep)
    if ep.size:
        ep_stats["spread_mm"] = float((ep.max() - ep.min()) * 1000.0)
    return TablePlane(
        z=float(z_table),
        normal=tuple(float(v) for v in normal),
        tilt_deg=float(tilt),
        horizontal=bool(horizontal),
        n_grasps=int(tips.shape[0]),
        n_on_table=int(on.sum()),
        z_grasp=z_grasp,
        object_half_height=float(object_half_height),
        fingertip_z=_stats(z),
        jaw_low_z=_stats(np.asarray(list(jaw_lows), dtype=np.float64)),
        episode_min_z=ep_stats,
        min_fingertip_z=float(min_fingertip_z),
    )


# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Workspace:
    """Bounding box and xy convex hull of the fingertip positions, base frame (m)."""

    min: tuple[float, float, float]
    max: tuple[float, float, float]
    hull_xy: tuple[tuple[float, float], ...]
    n_frames: int

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(float(b - a) for a, b in zip(self.min, self.max))

    def to_dict(self) -> dict[str, Any]:
        return {
            "min": list(self.min),
            "max": list(self.max),
            "size": list(self.size),
            "hull_xy": [list(p) for p in self.hull_xy],
            "n_frames": self.n_frames,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Workspace:
        return cls(
            min=tuple(float(v) for v in d["min"]),
            max=tuple(float(v) for v in d["max"]),
            hull_xy=tuple((float(p[0]), float(p[1])) for p in d.get("hull_xy", [])),
            n_frames=int(d.get("n_frames", 0)),
        )


def hull_xy(points: np.ndarray) -> np.ndarray:
    """Convex hull of (N,2+) points in xy, counter-clockwise (K,2); the points themselves if fewer than three."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, points.shape[-1])[:, :2]
    if p.shape[0] < 3:
        return p.copy()
    from scipy.spatial import ConvexHull, QhullError

    try:
        h = ConvexHull(p)
    except QhullError:  # collinear
        order = np.lexsort((p[:, 1], p[:, 0]))
        return p[[order[0], order[-1]]]
    return p[h.vertices]


def fit_workspace(tips: np.ndarray) -> Workspace:
    t = np.asarray(tips, dtype=np.float64).reshape(-1, 3)
    if t.shape[0] == 0:
        raise ValueError("no fingertip positions")
    return Workspace(
        min=tuple(float(v) for v in t.min(axis=0)),
        max=tuple(float(v) for v in t.max(axis=0)),
        hull_xy=tuple((float(x), float(y)) for x, y in hull_xy(t)),
        n_frames=int(t.shape[0]),
    )


# --------------------------------------------------------------------------
# Projection of the table plane
# --------------------------------------------------------------------------


def table_quad(table: TablePlane, workspace: Workspace, margin: float = WORKSPACE_MARGIN) -> np.ndarray:
    """Corners (4,3) of the table rectangle: the workspace's xy box plus ``margin``, on the plane."""
    x0, y0 = workspace.min[0] - margin, workspace.min[1] - margin
    x1, y1 = workspace.max[0] + margin, workspace.max[1] + margin
    return np.array([[x, y, table.height_at(x, y)] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])


def _clip_near(points_cam: np.ndarray, near: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon in camera space against depth >= near."""
    out: list[np.ndarray] = []
    n = len(points_cam)
    for i in range(n):
        a, b = points_cam[i], points_cam[(i + 1) % n]
        da, db = -a[2] - near, -b[2] - near
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return np.array(out) if out else np.zeros((0, 3))


def project_quad(corners: np.ndarray, camera: Camera, near: float = NEAR_PLANE) -> np.ndarray:
    """A convex planar polygon in the world (N,3) -> its image polygon (K,2) in
    pixels, clipped to the near plane first so corners behind the camera
    (a table that runs under it) are handled rather than dropped."""
    p = (np.asarray(corners, dtype=np.float64).reshape(-1, 3) - camera.pos) @ camera.R
    p = _clip_near(p, near)
    if len(p) < 3:
        return np.zeros((0, 2))
    depth = -p[:, 2]
    f = camera.focal
    u = camera.width / 2.0 + f * p[:, 0] / depth
    v = camera.height / 2.0 - f * p[:, 1] / depth
    # Far-off points would overflow the rasterizer's fixed point.
    lim = 50.0 * max(camera.width, camera.height)
    return np.stack([np.clip(u, -lim, lim), np.clip(v, -lim, lim)], axis=1)


def polygon_mask(poly: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if len(poly) >= 3:
        pts = np.round(np.asarray(poly, dtype=np.float64) * 16).astype(np.int32)
        cv2.fillConvexPoly(mask, pts.reshape(-1, 1, 2), 1, lineType=cv2.LINE_8, shift=4)
    return mask


# --------------------------------------------------------------------------
# Backdrops: the visual table
# --------------------------------------------------------------------------


def table_region(bg_rgb: np.ndarray, k: int = 4, scale: float = 0.25) -> tuple[np.ndarray, tuple[int, int, int]]:
    """The dominant large colour region of a background image: ``(mask (H,W) uint8, its mean RGB)``.

    k-means on Lab at low resolution, the most populous cluster, then its
    largest connected component after a morphological close. For the demo
    cells that is the white table (SO-101) or the black one (SO-100).
    """
    import cv2

    from antares.silhouette import to_lab

    bg = np.asarray(bg_rgb)
    H, W = bg.shape[:2]
    w, h = max(8, int(round(W * scale))), max(8, int(round(H * scale)))
    small = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
    lab = to_lab(small).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(lab, min(k, lab.shape[0]), None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)
    main = int(np.argmax(np.bincount(labels.ravel(), minlength=k)))
    m = (labels == main).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (cc == biggest).astype(np.uint8)
    mask = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    colour = bg[mask.astype(bool)].reshape(-1, 3).mean(axis=0) if mask.any() else bg.reshape(-1, 3).mean(axis=0)
    return mask, tuple(int(round(float(c))) for c in colour)


@dataclass(frozen=True)
class Backdrop:
    """One camera's clean background and how the estimated table sits in it.

    ``path`` is relative to the lock directory. ``plane_polygon`` is the
    table rectangle (:func:`table_quad`) projected into the camera, in
    pixels. ``plane_in_region`` is the share of that polygon that lands on
    the dominant colour region, ``region_in_plane`` the share of the region
    the polygon covers (low when the real table is larger than the
    workspace rectangle, which it usually is), ``iou`` their overlap. The
    quad for the export sits ``distance`` metres along the optical axis
    with half-size ``half_width`` x ``half_height``, which fills the view.
    """

    camera: str
    path: str
    width: int
    height: int
    distance: float
    half_width: float
    half_height: float
    plane_polygon: tuple[tuple[float, float], ...]
    plane_frac: float  # of the image the projected table covers
    region_frac: float  # of the image the dominant colour region covers
    plane_in_region: float
    region_in_plane: float
    iou: float
    table_rgb: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "distance": self.distance,
            "half_width": self.half_width,
            "half_height": self.half_height,
            "plane_polygon": [list(p) for p in self.plane_polygon],
            "plane_frac": self.plane_frac,
            "region_frac": self.region_frac,
            "plane_in_region": self.plane_in_region,
            "region_in_plane": self.region_in_plane,
            "iou": self.iou,
            "table_rgb": list(self.table_rgb),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Backdrop:
        return cls(
            camera=str(d["camera"]),
            path=str(d["path"]),
            width=int(d["width"]),
            height=int(d["height"]),
            distance=float(d.get("distance", BACKDROP_DISTANCE)),
            half_width=float(d["half_width"]),
            half_height=float(d["half_height"]),
            plane_polygon=tuple((float(p[0]), float(p[1])) for p in d.get("plane_polygon", [])),
            plane_frac=float(d.get("plane_frac", 0.0)),
            region_frac=float(d.get("region_frac", 0.0)),
            plane_in_region=float(d.get("plane_in_region", 0.0)),
            region_in_plane=float(d.get("region_in_plane", 0.0)),
            iou=float(d.get("iou", 0.0)),
            table_rgb=tuple(int(v) for v in d.get("table_rgb", (128, 128, 128))),
        )


def backdrop_quad(camera: Camera, distance: float = BACKDROP_DISTANCE) -> tuple[float, float]:
    """Half-width and half-height of a quad ``distance`` along the optical axis that fills the camera's view."""
    hh = distance * math.tan(math.radians(camera.fovy) / 2.0)
    return hh * camera.width / camera.height, hh


def backdrop_path(lock_name: str, camera: str) -> str:
    """``scene/<lock>_<camera>_backdrop.jpg``, relative to the lock directory."""
    return f"{BACKDROP_DIRNAME}/{lock_name}_{camera}_backdrop.jpg"


def check_backdrop(
    camera_name: str,
    camera: Camera,
    bg_rgb: np.ndarray,
    table: TablePlane,
    workspace: Workspace,
    path: str,
    margin: float = WORKSPACE_MARGIN,
    distance: float = BACKDROP_DISTANCE,
) -> Backdrop:
    """Project the table rectangle into ``camera`` and compare it with the
    dominant colour region of the background image."""
    H, W = bg_rgb.shape[:2]
    cam = Camera(camera.pos, camera.rotvec, camera.fovy, W, H)
    poly = project_quad(table_quad(table, workspace, margin), cam)
    plane = polygon_mask(poly, W, H).astype(bool)
    region, colour = table_region(bg_rgb)
    region = region.astype(bool)
    inter = int(np.count_nonzero(plane & region))
    union = int(np.count_nonzero(plane | region))
    hw, hh = backdrop_quad(cam, distance)
    return Backdrop(
        camera=camera_name,
        path=path,
        width=W,
        height=H,
        distance=float(distance),
        half_width=float(hw),
        half_height=float(hh),
        plane_polygon=tuple((float(round(u, 1)), float(round(v, 1))) for u, v in poly),
        plane_frac=float(plane.mean()),
        region_frac=float(region.mean()),
        plane_in_region=float(inter / max(1, plane.sum())),
        region_in_plane=float(inter / max(1, region.sum())),
        iou=float(inter / union) if union else 0.0,
        table_rgb=colour,
    )


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneFit:
    lock: str
    dataset: str
    robot: str
    table: TablePlane
    workspace: Workspace
    grasp_points: tuple[tuple[float, float, float], ...]
    release_points: tuple[tuple[float, float, float], ...]
    backdrops: dict[str, Backdrop] = field(default_factory=dict)
    n_episodes: int = 0
    n_frames: int = 0
    created_at: str = ""
    antares_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return {
            "lock": self.lock,
            "dataset": self.dataset,
            "robot": self.robot,
            "table": self.table.to_dict(),
            "workspace": self.workspace.to_dict(),
            "grasp_points": [list(p) for p in self.grasp_points],
            "release_points": [list(p) for p in self.release_points],
            "backdrops": {k: v.to_dict() for k, v in sorted(self.backdrops.items())},
            "n_episodes": self.n_episodes,
            "n_frames": self.n_frames,
            "created_at": self.created_at,
            "antares_version": self.antares_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SceneFit:
        return cls(
            lock=str(d.get("lock", "")),
            dataset=str(d.get("dataset", "")),
            robot=str(d.get("robot", "")),
            table=TablePlane.from_dict(d["table"]),
            workspace=Workspace.from_dict(d["workspace"]),
            grasp_points=tuple(tuple(float(v) for v in p) for p in d.get("grasp_points", [])),
            release_points=tuple(tuple(float(v) for v in p) for p in d.get("release_points", [])),
            backdrops={k: Backdrop.from_dict(v) for k, v in (d.get("backdrops") or {}).items()},
            n_episodes=int(d.get("n_episodes", 0)),
            n_frames=int(d.get("n_frames", 0)),
            created_at=str(d.get("created_at", "")),
            antares_version=str(d.get("antares_version", "")),
        )

    def lock_entry(self) -> dict[str, Any]:
        """The compact record kept in the lock's ``scene`` field."""
        return {
            "table_z": self.table.z,
            "normal": list(self.table.normal),
            "tilt_deg": self.table.tilt_deg,
            "horizontal": self.table.horizontal,
            "object_half_height": self.table.object_half_height,
            "workspace_min": list(self.workspace.min),
            "workspace_max": list(self.workspace.max),
            "n_grasps": len(self.grasp_points),
            "n_releases": len(self.release_points),
            "backdrops": {k: v.path for k, v in sorted(self.backdrops.items())},
            "created_at": self.created_at,
        }


def grasp_instants(dataset: Dataset) -> tuple[Any, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Gripper thresholds fitted over the dataset and, per episode, the
    ``(grasp frames, release frames)`` of its command."""
    thr = gripper_thresholds(np.concatenate([ep.action[:, GRIPPER_INDEX] for ep in dataset.episodes]))
    return thr, {ep.index: gripper_transitions(ep.action[:, GRIPPER_INDEX], thr) for ep in dataset.episodes}


def fit_scene(
    dataset: Dataset,
    arm: Any,
    lock_name: str,
    cameras: Mapping[str, Camera] | None = None,
    backgrounds: Mapping[str, np.ndarray] | None = None,
    *,
    object_half_height: float = OBJECT_HALF_HEIGHT,
    margin: float = WORKSPACE_MARGIN,
    hulls: ArmHulls | None = None,
) -> SceneFit:
    """Recover the table, workspace and grasp/release points of ``dataset``
    with ``arm`` (an :class:`antares.sim.Arm` carrying the lock's
    parameters and unit map), and check the table against each camera's
    background image (``backgrounds``: camera name -> RGB uint8).
    """
    if not dataset.episodes:
        raise ValueError("dataset has no episodes")
    hulls = hulls or arm_hulls(arm.model)
    _, transitions = grasp_instants(dataset)
    tips_all: list[np.ndarray] = []
    grasps: list[np.ndarray] = []
    releases: list[np.ndarray] = []
    jaw_lows: list[float] = []
    ep_min: list[float] = []
    for ep in dataset.episodes:
        poses = arm.fk(arm.unit_map.to_rad(ep.state))
        tips = fingertips(arm, poses)
        tips_all.append(tips)
        ep_min.append(float(tips[:, 2].min()))
        g, r = transitions[ep.index]
        grasps.extend(tips[g])
        releases.extend(tips[r])
        jaw_lows.extend(jaw_low_z(hulls, poses[t]) for t in g)
    tips_cat = np.concatenate(tips_all)
    grasp_pts = np.array(grasps).reshape(-1, 3)
    release_pts = np.array(releases).reshape(-1, 3)
    table = fit_table(
        grasp_pts, float(tips_cat[:, 2].min()), ep_min, jaw_lows, object_half_height=object_half_height
    )
    workspace = fit_workspace(tips_cat)
    backdrops: dict[str, Backdrop] = {}
    for name, cam in sorted((cameras or {}).items()):
        bg = (backgrounds or {}).get(name)
        if bg is None:
            continue
        backdrops[name] = check_backdrop(name, cam, bg, table, workspace, backdrop_path(lock_name, name), margin)
    return SceneFit(
        lock=lock_name,
        dataset=dataset.repo_id,
        robot=str(getattr(arm, "robot_name", "")),
        table=table,
        workspace=workspace,
        grasp_points=tuple(tuple(float(v) for v in p) for p in grasp_pts),
        release_points=tuple(tuple(float(v) for v in p) for p in release_pts),
        backdrops=backdrops,
        n_episodes=len(dataset.episodes),
        n_frames=int(tips_cat.shape[0]),
        created_at=_now_iso(),
    )


# --------------------------------------------------------------------------
# Sim view: what the policy camera would see in the sim, without OpenGL
# --------------------------------------------------------------------------

TABLE_TINT = (91, 140, 255)


def sim_view(
    backdrop: np.ndarray,
    camera: Camera,
    hulls: ArmHulls,
    body_poses: np.ndarray,
    table_polygon: np.ndarray | None = None,
    dark_arm: bool = True,
    table_alpha: float = 0.14,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Composite ``(RGB uint8 image, the drawn hull polygons far to near)``.

    The backdrop (resized to the camera if needed) gets the table polygon
    as a faint tint, then every visible hull filled with flat shading:
    geoms sorted far to near by the camera-space depth of their vertices'
    centroid, so nearer parts paint over farther ones, each shaded by its
    depth (nearer is lighter) with a per-body step so neighbouring parts
    read as separate, and outlined. ``dark_arm`` picks a dark grey ramp
    (SO-101's black prints) or a light warm one.
    """
    import cv2

    W, H = camera.width, camera.height
    img = np.asarray(backdrop)
    if img.shape[0] != H or img.shape[1] != W:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = np.ascontiguousarray(img.astype(np.float32))
    if table_polygon is not None and len(table_polygon) >= 3:
        m = polygon_mask(table_polygon, W, H).astype(bool)
        img[m] = (1 - table_alpha) * img[m] + table_alpha * np.asarray(TABLE_TINT, dtype=np.float32)
    out = img.astype(np.uint8)
    if table_polygon is not None and len(table_polygon) >= 3:
        pts = np.round(np.asarray(table_polygon) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, TABLE_TINT, 1, lineType=cv2.LINE_AA, shift=4)

    pts_w = hulls.world_points(body_poses)
    polys = project_polygons(hulls, pts_w, camera)
    cam_pts = (pts_w - camera.pos) @ camera.R
    depths = np.array([-cam_pts[s:e, 2].mean() if e > s else np.inf for s, e in hulls.slices])
    visible = [i for i, p in enumerate(polys) if len(p) >= 3]
    if not visible:
        return out, []
    order = sorted(visible, key=lambda i: -depths[i])
    dmin, dmax = depths[visible].min(), depths[visible].max()
    span = max(1e-6, dmax - dmin)
    drawn: list[np.ndarray] = []
    for i in order:
        t = (dmax - depths[i]) / span  # 0 far .. 1 near
        step = 6.0 * (hulls.geoms[i].body % 3 - 1)
        if dark_arm:
            L = 42.0 + 70.0 * t + step
            fill = (L * 0.96, L * 0.98, L * 1.08)
            line = (min(255.0, L + 60.0),) * 3
        else:
            L = 150.0 + 80.0 * t + step
            fill = (L * 1.05, L * 0.92, L * 0.78)
            line = (max(0.0, L - 80.0),) * 3
        poly = np.round(np.asarray(polys[i]) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(out, poly, tuple(int(min(255, max(0, c))) for c in fill), lineType=cv2.LINE_AA, shift=4)
        cv2.polylines(out, [poly], True, tuple(int(c) for c in line), 1, lineType=cv2.LINE_AA, shift=4)
        drawn.append(polys[i])
    return out, drawn


# --------------------------------------------------------------------------
# Files next to the locks
# --------------------------------------------------------------------------


def scene_path(lock_name: str, lock_dir: Path | str = "locks") -> Path:
    return Path(lock_dir) / f"{lock_name}{SCENE_SUFFIX}"


def load_scene_fit(lock_name: str, lock_dir: Path | str = "locks") -> SceneFit | None:
    p = scene_path(lock_name, lock_dir)
    if not p.is_file():
        return None
    return SceneFit.from_dict(json.loads(p.read_text()))


def load_backdrop(backdrop: Backdrop, lock_dir: Path | str = "locks") -> np.ndarray:
    """The saved backdrop as RGB uint8 (H,W,3)."""
    import cv2

    p = Path(lock_dir) / backdrop.path
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"backdrop {p} is missing or unreadable")
    return np.ascontiguousarray(img[:, :, ::-1])


def save_scene_fit(
    fit: SceneFit, lock_name: str, lock_dir: Path | str = "locks", backgrounds: Mapping[str, np.ndarray] | None = None
) -> Path:
    """Write ``<lock>.scene.json``, the backdrop JPEGs (``backgrounds``:
    camera -> RGB) and the lock's ``scene`` field (if the lock exists)."""
    import cv2

    from antares.lock import LOCK_SUFFIX, RobotLock

    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    for name, bd in fit.backdrops.items():
        bg = (backgrounds or {}).get(name)
        if bg is None:
            continue
        p = lock_dir / bd.path
        p.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(p), np.ascontiguousarray(np.asarray(bg)[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError(f"could not write backdrop {p}")
    path = scene_path(lock_name, lock_dir)
    path.write_text(json.dumps(fit.to_dict(), indent=2, sort_keys=True) + "\n")
    lock_file = lock_dir / f"{lock_name}{LOCK_SUFFIX}"
    if lock_file.is_file():
        lock = RobotLock.load(lock_file)
        lock.replace(scene=fit.lock_entry()).save(lock_file)
    return path


__all__ = [
    "BACKDROP_DIRNAME",
    "BACKDROP_DISTANCE",
    "Backdrop",
    "OBJECT_HALF_HEIGHT",
    "SCENE_SUFFIX",
    "SceneFit",
    "TablePlane",
    "WORKSPACE_MARGIN",
    "Workspace",
    "backdrop_path",
    "backdrop_quad",
    "check_backdrop",
    "fingertips",
    "fit_plane",
    "fit_scene",
    "fit_table",
    "fit_workspace",
    "grasp_instants",
    "hull_xy",
    "jaw_low_z",
    "load_backdrop",
    "load_scene_fit",
    "polygon_mask",
    "project_quad",
    "save_scene_fit",
    "scene_path",
    "sim_view",
    "table_quad",
    "table_region",
    "tip_frame",
]
