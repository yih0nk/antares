"""What an image policy's cameras would see in the sim.

An ACT or SmolVLA checkpoint trained on a LeRobot dataset takes the
dataset's camera frames as well as the joint state. In closed loop those
frames have to come from the sim, so this module renders them through the
lock's fitted cameras (:mod:`antares.camera`, a pinhole in MuJoCo's own
convention, so the fit goes straight into ``<camera pos quat fovy>``) over
the scene's backdrops (:mod:`antares.scene`, the per-pixel median of the
real video with the moving arm gone):

* the arm is MuJoCo's offscreen render of the robot's own meshes at the
  sim's joint angles, recoloured to :data:`ARM_RGBA` (the SO-101's black
  prints) and pasted over the backdrop through MuJoCo's segmentation
  render, so only the arm's pixels change;
* or, the composite route (:class:`VideoArm`), the real arm's pixels cut
  out of the recorded frame whose joint state is nearest the sim's;
* the task object is a mocap box of the fitted footprint and colour
  (:mod:`antares.objects`), drawn wherever the closed loop's
  :class:`~antares.task.KinematicObject` says it is;
* everything else is the backdrop, or, with ``backgrounds``, any other
  image of that camera: the real video frame of the same instant is the
  *underlay* diagnostic (the real arm and object show through wherever the
  sim arm does not cover them, so it leaks the real episode and is not an
  evaluation).

The render is not the real video: flat MuJoCo lighting, no shadows, the
object's yaw unknown (axis-aligned), and the backdrop keeps a faint smear
where the arm spent most of its time. How much that matters to a policy is
measured, not assumed (``experiments/lerobot_closedloop.py`` scores a
checkpoint's actions on real against rendered frames of the same states).

Needs an OpenGL context for :class:`mujoco.Renderer` (CGL on macOS; set
``MUJOCO_GL=egl`` or ``osmesa`` on a headless Linux box).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from antares.lock import LOCK_DIR
from antares.project import Camera
from antares.robots import get_robot

#: The SO-101's black PLA prints and servos.
ARM_RGBA = (0.07, 0.07, 0.075, 1.0)
#: Share of the object's fitted colour shown unlit (see :class:`SimCameras`).
OBJECT_EMISSION = 0.0
OBJECT_BODY = "antares_render_object"
#: Geom group of the object, so it can be rendered alone.
OBJECT_GROUP = 5
IMAGE_PREFIX = "observation.images."


class SimCameras:
    """Offscreen renders of the robot (and one box object) through fitted cameras.

    ``cameras`` maps a camera name (``up``, ``side``) to its
    :class:`~antares.project.Camera`; ``backdrops`` the same names to RGB
    uint8 images of that camera's size. ``object_size`` is the box's full
    extent (m, x y z in the box's own frame) and ``object_rgb`` its colour
    (0..255); None draws no object. ``object_emission`` (0..1) is how
    much of that colour the box shows regardless of MuJoCo's lighting (the
    fitted colour is already what the camera saw, lit).
    """

    def __init__(
        self,
        cameras: Mapping[str, Camera],
        backdrops: Mapping[str, np.ndarray],
        robot: str | None = None,
        object_size: Sequence[float] | None = None,
        object_rgb: Sequence[float] = (196, 99, 94),
        arm_rgba: Sequence[float] = ARM_RGBA,
        object_emission: float = OBJECT_EMISSION,
    ):
        missing = [n for n in cameras if n not in backdrops]
        if missing:
            raise ValueError(f"no backdrop for camera(s) {missing}")
        self.cameras = dict(cameras)
        self.backdrops = {n: np.ascontiguousarray(np.asarray(backdrops[n], dtype=np.uint8)) for n in cameras}
        for n, cam in self.cameras.items():
            if self.backdrops[n].shape[:2] != (cam.height, cam.width):
                raise ValueError(f"backdrop {n} is {self.backdrops[n].shape[:2]}, camera is {(cam.height, cam.width)}")
        spec = mujoco.MjSpec.from_file(str(get_robot(robot).model_path))
        for n, cam in self.cameras.items():
            spec.worldbody.add_camera(name=n, pos=list(cam.pos), quat=list(cam.quat), fovy=float(cam.fovy))
        self.has_object = object_size is not None
        self.object_size = None if object_size is None else tuple(float(s) for s in object_size)
        if self.has_object:
            body = spec.worldbody.add_body(name=OBJECT_BODY, mocap=True)
            rgba = [float(c) / 255.0 for c in object_rgb] + [1.0]
            mat = spec.add_material(name=OBJECT_BODY, rgba=rgba, emission=float(object_emission))
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5 * s for s in self.object_size], rgba=rgba,
                contype=0, conaffinity=0, material=mat.name, group=OBJECT_GROUP,
            )
        spec.visual.global_.offwidth = max(c.width for c in self.cameras.values())
        spec.visual.global_.offheight = max(c.height for c in self.cameras.values())
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        arm = np.asarray(arm_rgba, dtype=np.float32)
        obj_mat = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, OBJECT_BODY)
        for i in range(self.model.nmat):
            if i != obj_mat:
                self.model.mat_rgba[i] = arm
        obj_geoms: list[int] = []
        if self.has_object:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
            obj_geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid]
        self._object_geoms = np.asarray(obj_geoms, dtype=np.int32)
        for g in range(self.model.ngeom):
            if g not in obj_geoms and self.model.geom_matid[g] < 0:
                self.model.geom_rgba[g] = arm
        self.n_arm_q = self.model.nq
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    @classmethod
    def from_lock(
        cls,
        lock: Mapping[str, Any],
        lock_dir: Path | str = LOCK_DIR,
        cameras: Sequence[str] | None = None,
        object_height: float | None = None,
        **kwargs: Any,
    ) -> SimCameras:
        """The lock's fitted cameras over its scene backdrops, with its fitted
        object (footprint, colour; ``object_height`` defaults to the
        footprint's short side, the fits do not measure it). ``kwargs`` go
        to the constructor (``arm_rgba``, ``object_emission``)."""
        from antares.objects import load_object_fit
        from antares.scene import load_backdrop, load_scene_fit

        name = str(lock["name"])
        fits = lock.get("cameras") or {}
        names = list(cameras) if cameras is not None else sorted(fits)
        if not names or any(n not in fits for n in names):
            raise ValueError(f"lock {name!r} has camera fits for {sorted(fits)}, need {names}")
        scene = load_scene_fit(name, lock_dir)
        if scene is None:
            raise FileNotFoundError(f"lock {name!r} has no scene fit (locks/{name}.scene.json)")
        backdrops = {n: load_backdrop(scene.backdrops[n], lock_dir) for n in names}
        size, rgb = None, (196, 99, 94)
        obj = load_object_fit(name, lock_dir)
        if obj is not None and obj.footprint_m:
            long, short = float(obj.footprint_m["long"]["median"]), float(obj.footprint_m["short"]["median"])
            size = (long, short, float(object_height) if object_height is not None else short)
            rgb = tuple(obj.prior.rgb)
        cams = {n: Camera.from_dict(fits[n]) for n in names}
        return cls(cams, backdrops, robot=lock.get("robot"), object_size=size, object_rgb=rgb, **kwargs)

    def _renderer(self, cam: Camera) -> mujoco.Renderer:
        key = (cam.height, cam.width)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, cam.height, cam.width)
        return self._renderers[key]

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def __enter__(self) -> SimCameras:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def layers(
        self,
        q_rad: np.ndarray,
        object_xyz: Sequence[float] | None = None,
        cameras: Sequence[str] | None = None,
        object_only: bool = False,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Camera name -> ``(rgb, arm mask, object mask)`` of the raw MuJoCo
        render (the masks from its segmentation render, (H,W) bool).
        ``object_only`` hides the arm (nothing occludes the object)."""
        q = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.n_arm_q:
            raise ValueError(f"q_rad must have {self.n_arm_q} joints, got {q.shape[0]}")
        d = self.data
        d.qpos[:] = q
        if self.has_object:
            d.mocap_pos[0] = object_xyz if object_xyz is not None else (0.0, 0.0, -10.0)
        mujoco.mj_forward(self.model, d)
        opt = mujoco.MjvOption()  # groups 0-2 are visible by default; the object's is not
        if object_only:
            opt.geomgroup[:] = 0
        opt.geomgroup[OBJECT_GROUP] = 1
        out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for n in cameras if cameras is not None else self.cameras:
            r = self._renderer(self.cameras[n])
            r.update_scene(d, camera=n, scene_option=opt)
            rgb = r.render()
            r.enable_segmentation_rendering()
            try:
                r.update_scene(d, camera=n, scene_option=opt)
                seg = r.render()
            finally:
                r.disable_segmentation_rendering()
            geom = (seg[..., 0] >= 0) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            obj = geom & np.isin(seg[..., 0], self._object_geoms)
            out[n] = (rgb, geom & ~obj, obj)
        return out

    def render(
        self,
        q_rad: np.ndarray,
        object_xyz: Sequence[float] | None = None,
        backgrounds: Mapping[str, np.ndarray] | None = None,
        cameras: Sequence[str] | None = None,
        arm: VideoArm | None = None,
    ) -> dict[str, np.ndarray]:
        """Camera name -> RGB uint8 (H,W,3) of the arm at ``q_rad`` (and the
        object centred at ``object_xyz``; hidden when None) over each
        camera's backdrop, or over ``backgrounds[name]`` when given. With
        ``arm`` the arm's pixels are the real ones of the nearest recorded
        pose (:class:`VideoArm`), pasted over the rendered object."""
        names = list(cameras) if cameras is not None else list(self.cameras)
        lay = self.layers(q_rad, object_xyz, names)
        real = arm.pixels(self, q_rad, names) if arm is not None else None
        out: dict[str, np.ndarray] = {}
        for n in names:
            cam = self.cameras[n]
            rgb, arm_m, obj_m = lay[n]
            bg = backgrounds.get(n) if backgrounds is not None else None
            img = np.array(self.backdrops[n] if bg is None else bg, dtype=np.uint8, copy=True)
            if img.shape[:2] != (cam.height, cam.width):
                raise ValueError(f"background {n} is {img.shape[:2]}, camera is {(cam.height, cam.width)}")
            if real is None:
                m = arm_m | obj_m
                img[m] = rgb[m]
            else:
                img[obj_m] = rgb[obj_m]
                px, m = real[n]
                img[m] = px[m]
            out[n] = img
        return out

    def policy_inputs(
        self,
        cameras: Sequence[str] | None = None,
        background: Callable[[Any, str], np.ndarray | None] | None = None,
        object_offset: Sequence[float] = (0.0, 0.0, 0.0),
        prefix: str = IMAGE_PREFIX,
        on_render: Callable[[Any, dict[str, np.ndarray]], None] | None = None,
        arm: VideoArm | None = None,
    ) -> Callable[[Any], dict[str, Any]]:
        """``extra_inputs`` for :class:`antares.policy.LeRobotPolicy`: the
        render at the frame's ``ctx.q_rad`` and ``ctx.object_xyz`` (plus
        ``object_offset``: where the box's centre sits relative to the
        point the loop tracks) as ``{prefix + name: (3,H,W) float32 in
        [0,1]}``, LeRobot's image layout. ``background(ctx, name)`` replaces
        the backdrop (the underlay diagnostic); ``arm`` takes the arm's
        pixels from the recordings (:class:`VideoArm`); ``on_render(ctx,
        images)`` sees every render (to save a few)."""
        import torch

        names = list(cameras) if cameras is not None else list(self.cameras)
        off = np.asarray(object_offset, dtype=np.float64)

        def inputs(ctx: Any) -> dict[str, Any]:
            if ctx.q_rad is None:
                raise ValueError("the step context carries no sim joint angles (q_rad) to render")
            obj = None if ctx.object_xyz is None else np.asarray(ctx.object_xyz, dtype=np.float64) + off
            bgs = None
            if background is not None:
                bgs = {n: b for n in names if (b := background(ctx, n)) is not None}
            imgs = self.render(ctx.q_rad, obj, bgs, names, arm=arm)
            if on_render is not None:
                on_render(ctx, imgs)
            return {prefix + n: torch.from_numpy(imgs[n]).permute(2, 0, 1).float().div(255.0) for n in names}

        return inputs


class VideoArm:
    """The arm's pixels from the robot's own recordings: the composite route.

    A MuJoCo render of the meshes is the right shape in the right place but
    does not look like the real arm (no cables, labels, white jaw tips,
    real shading), and a policy trained on the video notices. Every
    recorded frame of the ``pool`` episodes is an image of the real arm at
    a known pose, so for a sim pose this takes the recorded frame whose
    joint state is nearest (degrees RMS over every joint, the gripper
    included), cuts the arm out of it with the camera's background model
    (``masks[name](rgb) -> (H,W)`` mask, :meth:`antares.silhouette.BackgroundModel.mask`)
    and pastes it over the backdrop. That frame's own object is cut away
    (its position from ``objects[episode]``, a ``(T,3)`` track, rendered
    as a mask and dilated by ``object_margin_px``) so it does not appear
    twice. Pose error is the price: :attr:`last` records the nearest
    frame and its distance.

    The pool must not hold the episode being evaluated, or the policy is
    shown that episode's own video.
    """

    def __init__(
        self,
        pool: Sequence[Any],
        unit_map: Any,
        frame: Callable[[int, str, int], np.ndarray],
        masks: Mapping[str, Callable[[np.ndarray], np.ndarray]],
        objects: Mapping[int, np.ndarray] | None = None,
        stride: int = 1,
        object_margin_px: int = 12,
    ):
        from scipy.spatial import cKDTree

        keys, states = [], []
        for ep in pool:
            idx = np.arange(0, len(ep), max(1, int(stride)))
            keys += [(int(ep.index), int(t)) for t in idx]
            states.append(np.degrees(unit_map.to_rad(np.asarray(ep.state, dtype=np.float64)[idx])))
        if not keys:
            raise ValueError("empty pool")
        self.keys = keys
        self.episodes = {int(ep.index) for ep in pool}
        self.tree = cKDTree(np.concatenate(states))
        self.n_joints = int(self.tree.m)
        self.frame = frame
        self.masks = dict(masks)
        self.objects = dict(objects or {})
        self.margin = int(object_margin_px)
        self.last: tuple[int, int, float] | None = None
        self.distances: list[float] = []

    def nearest(self, q_rad: np.ndarray) -> tuple[int, int, float]:
        """``(episode, frame, distance)`` of the nearest recorded pose; the
        distance is the RMS over joints in degrees."""
        d, i = self.tree.query(np.degrees(np.asarray(q_rad, dtype=np.float64).reshape(-1)))
        ep, t = self.keys[int(i)]
        return ep, t, float(d) / math.sqrt(self.n_joints)

    def pixels(self, cams: SimCameras, q_rad: np.ndarray, names: Sequence[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Camera name -> ``(rgb, mask)`` of the real arm nearest ``q_rad``."""
        import cv2

        ep, t, dist = self.nearest(q_rad)
        self.last = (ep, t, dist)
        self.distances.append(dist)
        track = self.objects.get(ep)
        cut = None
        if track is not None and cams.has_object:
            lay = cams.layers(np.zeros(cams.n_arm_q), track[min(t, len(track) - 1)], names, object_only=True)
            cut = {n: v[2] for n, v in lay.items()}
        out = {}
        k = np.ones((2 * self.margin + 1, 2 * self.margin + 1), np.uint8)
        for n in names:
            rgb = np.asarray(self.frame(ep, n, t), dtype=np.uint8)
            m = np.asarray(self.masks[n](rgb)).astype(bool)
            if cut is not None:
                m &= ~cv2.dilate(cut[n].astype(np.uint8), k).astype(bool)
            out[n] = (rgb, m)
        return out


__all__ = ["ARM_RGBA", "IMAGE_PREFIX", "OBJECT_BODY", "OBJECT_GROUP", "SimCameras", "VideoArm"]
