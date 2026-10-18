"""Sim arm silhouette without OpenGL: convex hulls, a pinhole camera, polygon fill.

Each visual mesh geom of the SO-101 is reduced once to its convex hull
(``scipy.spatial.ConvexHull`` over the compiled ``mesh_vert``), expressed
in its body's frame with the compiled ``geom_pos``/``geom_quat``. Posing the
arm is then :meth:`antares.sim.Arm.fk` body poses applied to those vertices;
the camera projects them; and because the projection of a convex solid is
convex, filling the 2D convex hull of each geom's projected vertices with
``cv2.fillConvexPoly`` is an exact silhouette of the hull. The union over
geoms over-approximates the true arm by the hull slack, which is small for
these parts (servos, printed brackets).

Camera convention is MuJoCo's: the camera looks down its ``-z`` axis with
``y`` up; ``fovy`` is the vertical field of view in degrees. Pixel ``(0,0)``
is the top-left corner, ``u`` to the right and ``v`` down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import mujoco
import numpy as np

# --------------------------------------------------------------------------
# Rotations (numpy only; MuJoCo quaternions are [w, x, y, z])
# --------------------------------------------------------------------------


def rotvec_to_mat(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64).reshape(4))
    return m.reshape(3, 3)


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    q = mat_to_quat(R)
    if q[0] < 0:
        q = -q
    w = min(1.0, float(q[0]))
    s = math.sqrt(max(0.0, 1.0 - w * w))
    angle = 2.0 * math.acos(w)
    if s < 1e-9:
        return np.zeros(3)
    return q[1:] / s * angle


def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    return mat_to_quat(rotvec_to_mat(r))


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    return mat_to_rotvec(quat_to_mat(q))


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    """Pinhole camera in MuJoCo's convention (looks down -z, y up)."""

    pos: np.ndarray  # (3,) world
    rotvec: np.ndarray  # (3,) world-from-camera rotation
    fovy: float  # degrees, vertical
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", np.asarray(self.pos, dtype=np.float64).reshape(3))
        object.__setattr__(self, "rotvec", np.asarray(self.rotvec, dtype=np.float64).reshape(3))
        object.__setattr__(self, "fovy", float(self.fovy))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))

    @property
    def R(self) -> np.ndarray:
        """World-from-camera rotation; columns are the camera axes in the world."""
        return rotvec_to_mat(self.rotvec)

    @property
    def quat(self) -> np.ndarray:
        """MuJoCo ``[w,x,y,z]`` quaternion for ``<camera quat=...>``."""
        return mat_to_quat(self.R)

    @property
    def focal(self) -> float:
        """Focal length in pixels (from the vertical fov and image height)."""
        return (self.height / 2.0) / math.tan(math.radians(self.fovy) / 2.0)

    @property
    def fovx(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.focal))

    @classmethod
    def look_at(
        cls, pos: Sequence[float], target: Sequence[float], fovy: float, width: int, height: int,
        up: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> Camera:
        pos = np.asarray(pos, dtype=np.float64)
        z = pos - np.asarray(target, dtype=np.float64)  # camera z points backward
        z /= max(1e-12, np.linalg.norm(z))
        upv = np.asarray(up, dtype=np.float64)
        x = np.cross(upv, z)
        if np.linalg.norm(x) < 1e-6:  # looking straight along up: pick any x
            x = np.cross(np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0]), z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=1)
        return cls(pos=pos, rotvec=mat_to_rotvec(R), fovy=fovy, width=width, height=height)

    def with_rotation(self, R: np.ndarray) -> Camera:
        return replace(self, rotvec=mat_to_rotvec(R))

    def scaled(self, scale: float) -> Camera:
        """Same camera at a different image size (fov unchanged)."""
        return replace(self, width=max(1, int(round(self.width * scale))), height=max(1, int(round(self.height * scale))))

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) -> pixel coordinates (N,2) and depth (N,) along the view axis.

        Points at or behind the camera get a non-positive depth; their pixel
        coordinates are meaningless and callers must drop them.
        """
        p = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.pos) @ self.R  # = R^T (p - c) row-wise
        depth = -p[:, 2]
        safe = np.where(depth > 1e-9, depth, 1e-9)
        f = self.focal
        u = self.width / 2.0 + f * p[:, 0] / safe
        v = self.height / 2.0 - f * p[:, 1] / safe
        return np.stack([u, v], axis=1), depth

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos": self.pos.tolist(),
            "rotvec": self.rotvec.tolist(),
            "quat": self.quat.tolist(),
            "fovy": self.fovy,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Camera:
        rotvec = d.get("rotvec")
        if rotvec is None:
            rotvec = quat_to_rotvec(np.asarray(d["quat"], dtype=np.float64))
        return cls(pos=d["pos"], rotvec=rotvec, fovy=d["fovy"], width=d["width"], height=d["height"])


# --------------------------------------------------------------------------
# Hulls
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GeomHull:
    geom: int
    body: int
    body_name: str
    name: str  # mesh name or geom type
    verts: np.ndarray  # (V,3) hull vertices in the body frame


@dataclass(frozen=True)
class ArmHulls:
    """Convex hulls of the arm's visual geoms, in body frames, plus the stacked layout."""

    geoms: tuple[GeomHull, ...]
    body_of_point: np.ndarray  # (P,) body id per stacked vertex
    slices: tuple[tuple[int, int], ...]  # per geom: [start, end) into the stacked arrays
    points_body: np.ndarray  # (P,3) stacked body-frame vertices

    @property
    def n_points(self) -> int:
        return int(self.points_body.shape[0])

    def world_points(self, body_poses: np.ndarray) -> np.ndarray:
        """Body poses (nbody,7) [x,y,z,qw,qx,qy,qz] -> stacked world vertices (P,3)."""
        body_poses = np.asarray(body_poses, dtype=np.float64)
        out = np.empty_like(self.points_body)
        for g, (s, e) in zip(self.geoms, self.slices):
            pose = body_poses[g.body]
            out[s:e] = self.points_body[s:e] @ quat_to_mat(pose[3:]).T + pose[:3]
        return out


def _box_corners(size: np.ndarray) -> np.ndarray:
    s = np.asarray(size, dtype=np.float64)[:3]
    return np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])


def arm_hulls(
    model: mujoco.MjModel,
    group: int = 2,
    exclude_bodies: Sequence[str] = ("camera_mount",),
    max_verts: int = 160,
) -> ArmHulls:
    """Hull every visual (``group``) mesh and box geom; bodies in ``exclude_bodies`` are skipped.

    Hulls with more than ``max_verts`` vertices are thinned by keeping every
    k-th vertex (the parts here hull to well under that; the cap only bounds
    the cost of pathological meshes).
    """
    from scipy.spatial import ConvexHull

    hulls: list[GeomHull] = []
    hull_cache: dict[int, np.ndarray] = {}
    excluded = set(exclude_bodies)
    for g in range(model.ngeom):
        if model.geom_group[g] != group:
            continue
        body = int(model.geom_bodyid[g])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or str(body)
        if body_name in excluded:
            continue
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[g])
            if mesh not in hull_cache:
                a = int(model.mesh_vertadr[mesh])
                n = int(model.mesh_vertnum[mesh])
                verts = np.asarray(model.mesh_vert[a : a + n], dtype=np.float64)
                try:
                    hv = verts[np.unique(ConvexHull(verts).vertices)]
                except Exception:  # degenerate (flat) mesh: fall back to all vertices
                    hv = verts
                if len(hv) > max_verts:
                    hv = hv[:: int(math.ceil(len(hv) / max_verts))]
                hull_cache[mesh] = hv
            local = hull_cache[mesh]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh) or f"mesh{mesh}"
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            local = _box_corners(model.geom_size[g])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"box{g}"
        else:
            continue
        R = quat_to_mat(model.geom_quat[g])
        verts_body = local @ R.T + model.geom_pos[g]
        hulls.append(GeomHull(geom=g, body=body, body_name=body_name, name=name, verts=verts_body))
    slices, start = [], 0
    for h in hulls:
        slices.append((start, start + len(h.verts)))
        start += len(h.verts)
    points = np.concatenate([h.verts for h in hulls]) if hulls else np.zeros((0, 3))
    body_of_point = np.concatenate([np.full(len(h.verts), h.body) for h in hulls]) if hulls else np.zeros(0, int)
    return ArmHulls(geoms=tuple(hulls), body_of_point=body_of_point, slices=tuple(slices), points_body=points)


# --------------------------------------------------------------------------
# Silhouette
# --------------------------------------------------------------------------


def project_polygons(
    hulls: ArmHulls, points_world: np.ndarray, camera: Camera, min_depth: float = 0.02
) -> list[np.ndarray]:
    """Per geom, the 2D convex outline (K,2) float pixels, or an empty array if
    the geom is behind the camera (fewer than three vertices in front)."""
    import cv2

    uv, depth = camera.project(points_world)
    polys: list[np.ndarray] = []
    for s, e in hulls.slices:
        ok = depth[s:e] > min_depth
        if ok.sum() < 3:
            polys.append(np.zeros((0, 2)))
            continue
        pts = uv[s:e][ok].astype(np.float32)
        hull = cv2.convexHull(pts)  # (K,1,2)
        polys.append(hull.reshape(-1, 2).astype(np.float64))
    return polys


def rasterize(polys: Sequence[np.ndarray], width: int, height: int, scale: float = 1.0) -> np.ndarray:
    """Fill the union of convex polygons into a (height,width) uint8 mask.

    ``scale`` multiplies the polygon coordinates (polygons projected at full
    resolution can be drawn into a smaller mask).
    """
    import cv2

    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    shift = 4  # fixed-point sub-pixel precision for fillConvexPoly
    for poly in polys:
        if len(poly) < 3:
            continue
        pts = np.round(np.asarray(poly, dtype=np.float64) * scale * (1 << shift)).astype(np.int32)
        cv2.fillConvexPoly(mask, pts.reshape(-1, 1, 2), 1, lineType=cv2.LINE_8, shift=shift)
    return mask


def silhouette(hulls: ArmHulls, points_world: np.ndarray, camera: Camera) -> tuple[np.ndarray, list[np.ndarray]]:
    """Mask (H,W) uint8 of the posed arm seen by ``camera`` and its per-geom outlines."""
    polys = project_polygons(hulls, points_world, camera)
    return rasterize(polys, camera.width, camera.height), polys


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b) / union)


__all__ = [
    "ArmHulls",
    "Camera",
    "GeomHull",
    "arm_hulls",
    "iou",
    "mat_to_quat",
    "mat_to_rotvec",
    "project_polygons",
    "quat_to_mat",
    "quat_to_rotvec",
    "rasterize",
    "rotvec_to_mat",
    "rotvec_to_quat",
    "silhouette",
]
