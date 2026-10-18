"""Export a lock's calibrated simulator as a standalone MJCF.

A :class:`~antares.lock.RobotLock` holds fitted :class:`~antares.sim.SimParams`.
:func:`export_mjcf` bakes those values into a copy of the SO-101 MJCF so the
model can be dropped into any MuJoCo or RL pipeline without Antares:

* ``kp``/``kv`` go on the ``sts3215`` default class's ``<position>`` and
  ``damping``/``armature``/``frictionloss`` on its ``<joint>``. Per-joint
  values (a sequence of six) are written on the named ``<joint>`` /
  ``<position>`` elements instead.
* ``gravity_scale``, when the lock fitted one, scales ``<option gravity>``.
* ``meshdir`` is rewritten: either the STL meshes are copied into an
  ``assets/`` directory next to the output, or it points (absolute) at the
  package's own assets.
* A header comment records the lock (name, dataset, created_at, held-out
  score before/after, latency).

MJCF cannot express command latency or a servo deadband, so a sidecar ``<name>.antares.json``
is written next to the model with the full parameters (including
``latency_frames``) and the dataset-unit -> radian map, for a consumer to
apply in its control loop. :func:`load_exported_arm` builds an
:class:`~antares.sim.Arm` from an export, reading the latency back from the
sidecar.

Edits are done on the XML text with targeted substitutions (not a DOM
round-trip), so comments, formatting and everything else in the Menagerie
file are preserved verbatim.

Robots without a shared servo class (``export_class=None`` in their
profile, e.g. ALOHA) are exported through :class:`mujoco.MjSpec` instead:
the lock's parameters are applied to the compiled model exactly as the fit
saw them, then each mapped joint and actuator's values are written onto its
own element and the spec is saved flattened (includes inlined).
"""

from __future__ import annotations

import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from antares import __version__, factors
from antares.lock import RobotLock
from antares.sim import JOINTS, MESH_DIR, MODEL_PATH, STANDARD_GRAVITY, Arm, CommandMap, SimParams, UnitMap
from antares.task import OBJECT_BODY, OBJECT_JOINT, OBJECT_MASS, OBJECT_SIZE, PLACE_SITE, TaskDistribution, scene_xml

SCENE_PATH: Path = MODEL_PATH.parent / "scene.xml"
SIDECAR_SUFFIX = ".antares.json"
SCENE_FILENAME = "scene.xml"
TASK_FILENAME = "task.json"
OBJECTS_FILENAME = "objects.json"
ASSETS_DIRNAME = "assets"
EXPORT_FORMAT = "antares.export/1"

#: Default class in the MJCF that all six servos share.
SERVO_CLASS = "sts3215"
#: Floor and light for robots whose own scene file isn't fetched.
_GENERIC_SCENE = """<mujoco model="scene">
  <include file="robot.xml"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""
JOINT_PARAMS: tuple[str, ...] = factors.mjcf_attrs("joint")
ACTUATOR_PARAMS: tuple[str, ...] = factors.mjcf_attrs("position")
#: Written on ``<option>`` (``gravity_scale`` scales the gravity vector).
OPTION_PARAMS: tuple[str, ...] = ("gravity_scale",)
#: Everything :func:`bake_params` writes into the MJCF.
BAKED_PARAMS: tuple[str, ...] = JOINT_PARAMS + ACTUATOR_PARAMS + OPTION_PARAMS
#: Parameters MJCF cannot express; they travel in the sidecar for the
#: consumer's control loop.
SIDECAR_PARAMS: tuple[str, ...] = ("latency_frames", "deadband")

ParamValue = float | Sequence[float]


class ExportError(ValueError):
    """The MJCF did not have the structure the exporter expects."""


# --------------------------------------------------------------------------
# string-safe XML editing
# --------------------------------------------------------------------------


def _fmt(v: float) -> str:
    """Shortest decimal that parses back to the same double (``repr``)."""
    v = float(v)
    if not math.isfinite(v):
        raise ExportError(f"cannot write non-finite value {v!r} into MJCF")
    return repr(v)


def _set_attrs(tag: str, attrs: Mapping[str, str]) -> str:
    """Set/replace attributes on one start tag such as ``<joint a="1"/>``."""
    for name, value in attrs.items():
        pat = re.compile(r'(\s%s=")[^"]*(")' % re.escape(name))
        tag, n = pat.subn(lambda m: f"{m.group(1)}{value}{m.group(2)}", tag, count=1)
        if n == 0:
            close = tag.rfind("/>") if tag.rstrip().endswith("/>") else tag.rfind(">")
            tag = f'{tag[:close].rstrip()} {name}="{value}"{tag[close:]}'
    return tag


def _sub_one(xml: str, pattern: re.Pattern[str], attrs: Mapping[str, str], what: str) -> str:
    m = pattern.search(xml)
    if m is None:
        raise ExportError(f"could not find {what} in the MJCF")
    return xml[: m.start()] + _set_attrs(m.group(0), attrs) + xml[m.end() :]


def _class_block(xml: str, cls: str) -> tuple[int, int]:
    m = re.search(r'<default class="%s">.*?</default>' % re.escape(cls), xml, re.DOTALL)
    if m is None:
        raise ExportError(f"no <default class={cls!r}> block in the MJCF")
    return m.start(), m.end()


def _named(tag: str, name: str) -> re.Pattern[str]:
    # <joint ... name="X" .../> with attributes in any order
    return re.compile(r'<%s\b(?=[^>]*\bname="%s")[^>]*/>' % (tag, re.escape(name)))


def bake_params(xml: str, params: Mapping[str, ParamValue], joints: Sequence[str] = JOINTS) -> str:
    """Write ``params`` into the MJCF text.

    Scalars are set on the ``sts3215`` default class; a sequence of
    ``len(joints)`` values is written on each named joint/actuator element
    instead (explicit attributes override class defaults in MuJoCo).
    ``gravity_scale`` scales ``<option gravity>``. Keys outside
    :data:`BAKED_PARAMS` (``latency_frames``, ``deadband``: see
    :data:`SIDECAR_PARAMS`) are ignored. The rest of the text is untouched.
    """
    scalar: dict[str, str] = {}
    per_joint: dict[str, list[str]] = {}
    for key, value in params.items():
        if key not in BAKED_PARAMS or key in OPTION_PARAMS:
            continue
        if np.ndim(value) == 0:
            scalar[key] = _fmt(value)  # type: ignore[arg-type]
        else:
            vals = [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]
            if len(vals) != len(joints):
                raise ExportError(f"{key}: expected {len(joints)} per-joint values, got {len(vals)}")
            per_joint[key] = [_fmt(v) for v in vals]

    start, end = _class_block(xml, SERVO_CLASS)
    block = xml[start:end]
    j_attrs = {k: v for k, v in scalar.items() if k in JOINT_PARAMS}
    a_attrs = {k: v for k, v in scalar.items() if k in ACTUATOR_PARAMS}
    if j_attrs:
        block = _sub_one(block, re.compile(r"<joint\b[^>]*/>"), j_attrs, f"<joint> in class {SERVO_CLASS}")
    if a_attrs:
        block = _sub_one(block, re.compile(r"<position\b[^>]*/>"), a_attrs, f"<position> in class {SERVO_CLASS}")
    xml = xml[:start] + block + xml[end:]

    if "gravity_scale" in params:
        g = np.asarray(params["gravity_scale"], dtype=np.float64).reshape(-1)
        if np.any(g != g[0]):
            raise ExportError("gravity_scale must be one value for all joints")
        gz = _fmt(-STANDARD_GRAVITY * float(g[0]))
        xml = _sub_one(xml, re.compile(r"<option\b[^>]*/>"), {"gravity": f"0 0 {gz}"}, "<option>")

    for i, joint in enumerate(joints):
        j_attrs = {k: v[i] for k, v in per_joint.items() if k in JOINT_PARAMS}
        a_attrs = {k: v[i] for k, v in per_joint.items() if k in ACTUATOR_PARAMS}
        if j_attrs:
            xml = _sub_one(xml, _named("joint", joint), j_attrs, f"joint {joint!r}")
        if a_attrs:
            xml = _sub_one(xml, _named("position", joint), a_attrs, f"actuator {joint!r}")
    return xml


def set_meshdir(xml: str, meshdir: str | Path) -> str:
    """Point ``<compiler meshdir>`` at ``meshdir`` (any string MuJoCo accepts)."""
    return _sub_one(xml, re.compile(r"<compiler\b[^>]*/>"), {"meshdir": str(meshdir)}, "<compiler>")


def referenced_meshes(xml: str) -> list[str]:
    """``file`` attributes of every ``<mesh>`` asset, in document order, unique."""
    seen: dict[str, None] = {}
    for m in re.finditer(r'<mesh\b[^>]*\bfile="([^"]+)"', xml):
        seen.setdefault(m.group(1))
    return list(seen)


def _comment(lines: Sequence[str], indent: str = "  ") -> str:
    body = "\n".join(f"{indent}  {line}" if line else "" for line in lines)
    body = body.replace("--", "- -")  # '--' is illegal inside an XML comment
    return f"<!--\n{body}\n{indent}-->"


def insert_header(xml: str, comment: str) -> str:
    """Insert ``comment`` right after the root ``<mujoco ...>`` start tag."""
    m = re.search(r"<mujoco\b[^>]*>", xml)
    if m is None:
        raise ExportError("no <mujoco> root element")
    return f"{xml[: m.end()]}\n  {comment}{xml[m.end():]}"


def _vec(values: Sequence[float]) -> str:
    return " ".join(_fmt(v) for v in values)


def camera_elements(cameras: Mapping[str, Mapping[str, Any]]) -> str:
    """``<camera>`` elements (one per fitted camera) as MJCF text, with a comment each.

    ``cameras`` is a lock's ``cameras`` field: short name -> CameraFit dict
    (``pos``, ``quat`` wxyz, ``fovy``, ``width``, ``height``, IoUs). The
    cameras are in the world frame, which is the arm's base frame.
    """
    lines = []
    for name in sorted(cameras):
        c = cameras[name]
        fit = c.get("fit_iou")
        held = c.get("heldout_iou")
        note = (
            f"Antares camera fit for {c.get('key', name)} ({c.get('width', '?')}x{c.get('height', '?')}): "
            f"silhouette IoU {fit:.3f} fit / {held:.3f} held-out"
            if isinstance(fit, (int, float)) and isinstance(held, (int, float)) and math.isfinite(float(fit))
            else f"Antares camera fit for {c.get('key', name)}"
        )
        lines.append(f"<!-- {note.replace('--', '- -')} -->")
        lines.append(f'<camera name="{name}" pos="{_vec(c["pos"])}" quat="{_vec(c["quat"])}" fovy="{_fmt(c["fovy"])}"/>')
    return "\n".join(lines)


def insert_cameras(xml: str, cameras: Mapping[str, Mapping[str, Any]]) -> str:
    """Add a fixed ``<camera>`` per fitted camera at the top of ``<worldbody>``."""
    if not cameras:
        return xml
    for name in cameras:
        if not re.match(r"^[A-Za-z0-9_.-]+$", name):
            raise ExportError(f"camera name {name!r} is not a valid MJCF name")
        if re.search(r'<camera\b[^>]*\bname="%s"' % re.escape(name), xml):
            raise ExportError(f"the MJCF already has a camera named {name!r}")
    m = re.search(r"<worldbody\b[^>]*>", xml)
    if m is None:
        raise ExportError("no <worldbody> element")
    body = camera_elements(cameras).replace("\n", "\n    ")
    return f"{xml[: m.end()]}\n    {body}{xml[m.end():]}"


# --------------------------------------------------------------------------
# the static scene: table and backdrops (antares.scene)
# --------------------------------------------------------------------------

BACKDROPS_DIRNAME = "backdrops"
#: The table is drawn translucent so a backdrop behind it (a top-down
#: camera looks through the table at its own photo) stays visible.
TABLE_ALPHA = 0.55


def _normal_quat(normal: Sequence[float]) -> np.ndarray:
    """MuJoCo quaternion rotating +z onto ``normal``."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(1e-12, np.linalg.norm(n))
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, n)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if n[2] > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    angle = math.atan2(s, float(np.dot(z, n)))
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis / s, angle)
    return q


def scene_elements(
    scene: Any, cameras: Mapping[str, Mapping[str, Any]], backdrop_files: Mapping[str, str]
) -> tuple[str, str]:
    """``(asset xml, worldbody xml)`` for a :class:`antares.scene.SceneFit`.

    The table is a finite ``plane`` at the fitted height covering the
    workspace plus its margin, coloured like the visual table and
    translucent (:data:`TABLE_ALPHA`). Per camera with a backdrop image
    (``backdrop_files``: camera -> texture path relative to the model) a
    ``2d`` texture on an unlit plane sits ``distance`` metres along the
    optical axis, facing the camera and sized to fill its view, so a render
    from that camera shows the real background behind the sim arm.
    """
    from antares.scene import WORKSPACE_MARGIN, table_quad

    assets: list[str] = []
    body: list[str] = []
    t = scene.table
    ws = scene.workspace
    corners = table_quad(t, ws, WORKSPACE_MARGIN)
    cx, cy = float(corners[:, 0].mean()), float(corners[:, 1].mean())
    sx = (ws.max[0] - ws.min[0]) / 2 + WORKSPACE_MARGIN
    sy = (ws.max[1] - ws.min[1]) / 2 + WORKSPACE_MARGIN
    rgb = (0.82, 0.82, 0.8)
    for bd in scene.backdrops.values():
        rgb = tuple(v / 255.0 for v in bd.table_rgb)
        break
    body.append(
        f"<!-- Antares table from the logs: fingertips at {t.n_on_table} of {t.n_grasps} grasps sat at "
        f"z={_fmt(round(t.z_grasp, 4))}, minus an assumed object half-height of {_fmt(t.object_half_height)}; "
        f"fitted tilt {t.tilt_deg:.1f} deg{' (assumed horizontal)' if t.horizontal else ''} -->"
    )
    body.append(
        f'<geom name="table" type="plane" size="{_fmt(round(sx, 4))} {_fmt(round(sy, 4))} 0.05" '
        f'pos="{_vec((round(cx, 4), round(cy, 4), round(t.height_at(cx, cy), 5)))}" '
        f'quat="{_vec(np.round(_normal_quat(t.normal), 6))}" rgba="{_vec(tuple(round(v, 3) for v in rgb))} {TABLE_ALPHA}"/>'
    )
    for name in sorted(scene.backdrops):
        bd = scene.backdrops[name]
        cam = cameras.get(name)
        file = backdrop_files.get(name)
        if cam is None or file is None:
            continue
        from antares.project import Camera

        c = Camera.from_dict(cam)
        forward = -c.R[:, 2]
        pos = c.pos + bd.distance * forward
        assets.append(f'<texture name="backdrop_{name}" type="2d" file="{file}"/>')
        assets.append(
            f'<material name="backdrop_{name}" texture="backdrop_{name}" texrepeat="1 1" texuniform="false" '
            f'emission="1" specular="0" shininess="0" reflectance="0"/>'
        )
        body.append(
            f"<!-- Antares backdrop for camera {name}: its background image {bd.distance:g} m along the optical axis, "
            f"filling the view; table plane vs image overlap IoU {bd.iou:.2f} -->"
        )
        body.append(
            f'<geom name="backdrop_{name}" type="plane" size="{_fmt(round(bd.half_width, 4))} '
            f'{_fmt(round(bd.half_height, 4))} 1" pos="{_vec(np.round(pos, 4))}" quat="{_vec(np.round(c.quat, 6))}" '
            f'material="backdrop_{name}" contype="0" conaffinity="0"/>'
        )
    return "\n".join(assets), "\n".join(body)


def insert_scene(xml: str, scene: Any, cameras: Mapping[str, Mapping[str, Any]], backdrop_files: Mapping[str, str]) -> str:
    """Put the table and backdrops into a scene MJCF: assets into ``<asset>``,
    geoms at the top of ``<worldbody>``; the checker floor is dropped since the
    table is now the ground."""
    assets, body = scene_elements(scene, cameras, backdrop_files)
    if assets:
        m = re.search(r"<asset\b[^>]*>", xml)
        if m is None:
            raise ExportError("scene has no <asset> element")
        xml = f"{xml[: m.end()]}\n    {assets.replace(chr(10), chr(10) + '    ')}{xml[m.end():]}"
    m = re.search(r"<worldbody\b[^>]*>", xml)
    if m is None:
        raise ExportError("scene has no <worldbody> element")
    xml = f"{xml[: m.end()]}\n    {body.replace(chr(10), chr(10) + '    ')}{xml[m.end():]}"
    xml = re.sub(r'\n[ \t]*<geom name="floor"[^>]*/>', "", xml, count=1)
    return xml


def write_backdrops(scene: Any, lock_dir: Path | str, out_dir: Path) -> dict[str, str]:
    """Convert the saved backdrop JPEGs to PNGs (what MuJoCo textures read)
    under ``out_dir/backdrops/``; returns camera -> path relative to ``out_dir``.
    Cameras whose image is missing are skipped."""
    import cv2

    from antares.scene import load_backdrop

    out: dict[str, str] = {}
    for name, bd in scene.backdrops.items():
        try:
            rgb = load_backdrop(bd, lock_dir)
        except FileNotFoundError:
            continue
        dest = out_dir / BACKDROPS_DIRNAME / f"{name}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dest), np.ascontiguousarray(rgb[:, :, ::-1])):
            raise ExportError(f"could not write {dest}")
        out[name] = f"{BACKDROPS_DIRNAME}/{name}.png"
    return out


# --------------------------------------------------------------------------
# the manipulated object (antares.objects)
# --------------------------------------------------------------------------


def object_elements(fit: Any, mass: float | None = None) -> str:
    """``<worldbody>`` children for an :class:`antares.objects.ObjectFit`: a
    free box with the vision footprint at the mean vision start pose, the
    estimated mass (or :data:`antares.task.OBJECT_MASS` when the payload
    estimate was below the noise), coloured like the object, and a place
    marker at the mean vision end pose. Same body/joint/site names as the
    task cube, so :func:`antares.task.set_task` moves it too."""
    fp = fit.footprint
    if fp is None or fit.start is None:
        raise ValueError("objects fit has no footprint or start poses: nothing to put in the scene")
    hh = float(fit.object_half_height)
    half = (fp[0] / 2.0, fp[1] / 2.0, hh)
    pos = np.asarray(fit.start["mean"], dtype=np.float64)
    pos[2] = fit.table_z + hh
    m = mass if mass is not None else (fit.mass_kg if fit.mass_kg is not None else OBJECT_MASS)
    rgb = tuple(v / 255.0 for v in fit.prior.rgb)
    mass_note = (
        f"mass {m:.4g} kg from the residual" if fit.mass_kg is not None else
        f"mass {m:.4g} kg assumed (payload estimate below the noise: {(fit.payload or {}).get('verdict', 'not run')})"
    )
    place = np.asarray(fit.end["mean"], dtype=np.float64) if fit.end else pos
    return (
        f"    <!-- Antares object from the video: {fit.description}; footprint {fp[0] * 100:.1f} x {fp[1] * 100:.1f} cm "
        f"(height {2 * hh * 100:.1f} cm assumed); {mass_note}; start pose = mean of {fit.start['n']} vision start poses -->\n"
        f'    <body name="{OBJECT_BODY}" pos="{_vec(np.round(pos, 4))}">\n'
        f'      <freejoint name="{OBJECT_JOINT}"/>\n'
        f'      <geom type="box" size="{_vec(tuple(round(v, 4) for v in half))}" mass="{_fmt(round(m, 5))}" '
        f'friction="1 5e-3 1e-4" rgba="{_vec(tuple(round(v, 3) for v in rgb))} 1"/>\n'
        f"    </body>\n"
        f'    <site name="{PLACE_SITE}" pos="{_vec(np.round(place, 4))}" size="{_fmt(round(hh, 4))}" type="sphere" rgba="0.2 0.8 0.3 0.4"/>\n'
    )


def objects_sidecar(fit: Any) -> dict[str, Any]:
    """``objects.json`` next to the scene: the start/end pose distributions
    (for ``set_task``-style sampling), footprint, colour and the mass estimate."""
    return {
        "format": EXPORT_FORMAT,
        "lock": fit.lock,
        "dataset": fit.dataset,
        "camera": fit.camera,
        "description": fit.description,
        "rgb": list(fit.prior.rgb),
        "footprint_m": None if fit.footprint is None else list(fit.footprint),
        "half_height_m": fit.object_half_height,
        "table_z": fit.table_z,
        "mass_kg": fit.mass_kg,
        "measured_mass_kg": fit.measured_mass_kg,
        "payload_verdict": (fit.payload or {}).get("verdict"),
        "start": fit.start,
        "end": fit.end,
        "vision_vs_fk_m": fit.vision_vs_fk_m,
        "note": "start/end are vision poses on the table plane (base frame, m); draw a start pose and "
        "add ~1 cm of jitter, as antares.objects.ObjectFit.sample_start does.",
    }


# --------------------------------------------------------------------------
# lock -> files
# --------------------------------------------------------------------------


def _score(v: float) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def _score_text(v: float, unit: str = " deg") -> str:
    s = _score(v)
    return "n/a" if s is None else f"{s:.3f}{unit}"


def _lock_robot(lock: RobotLock):
    from antares.robots import get_robot

    return get_robot(getattr(lock, "robot", None))


def _generic(lock: RobotLock) -> bool:
    """Exported through MjSpec (no shared servo class to bake into)?"""
    return _lock_robot(lock).export_class is None


def _fmt_param(v: ParamValue) -> str:
    """A scalar as :func:`_fmt`; a per-joint tuple as ``[v0 v1 ... v5]``."""
    if isinstance(v, (tuple, list)):
        return "[" + " ".join(_fmt(x) for x in v) + "]"
    return _fmt(v)


def header_lines(lock: RobotLock) -> list[str]:
    r = lock.result
    p = lock.params
    robot = _lock_robot(lock)
    generic = _generic(lock)
    lines = [
        f"Antares calibrated {robot.label if generic else 'SO-101'} (antares {__version__}, lock format {EXPORT_FORMAT})",
        f"lock:        {lock.name}",
        f"dataset:     {lock.dataset} ({lock.robot_type}), data sha256 {lock.data_sha256}",
        f"created_at:  {lock.created_at}",
        f"held-out tracking RMSE: {_score_text(r.baseline_eval_score)} before -> "
        f"{_score_text(r.eval_score)} after ({len(r.eval_episodes)} episodes)",
    ]
    if generic:
        lines.append("fitted values are written on each mapped joint and actuator")
    else:
        lines.append(
            f"baked: kp={_fmt_param(p.kp)} kv={_fmt_param(p.kv)} damping={_fmt_param(p.damping)} "
            f"armature={_fmt_param(p.armature)} frictionloss={_fmt_param(p.frictionloss)} "
            f"gravity_scale={_fmt_param(float(p.per_joint('gravity_scale')[0]))} (scales option gravity)"
        )
    lines.append(
        f"latency_frames: {p.latency_frames} (command delay in dataset frames; MJCF cannot "
        f"express this, apply it in your control loop; see the .antares.json sidecar)"
    )
    if np.any(p.per_joint("deadband")):
        band = " ".join(_fmt(v) for v in p.per_joint("deadband")) if p.is_per_joint("deadband") else _fmt(p.deadband)
        lines.append(
            f"deadband: {band} rad (servo deadband; MJCF cannot express this, "
            f"apply it in your control loop; see the .antares.json sidecar)"
        )
    if lock.scene:
        lines.append(
            f"scene:       table at z={lock.scene.get('table_z', float('nan')):.4f} m from "
            f"{lock.scene.get('n_grasps', '?')} grasps (see scene.xml)"
        )
    if lock.cameras:
        lines.append(
            "cameras:     "
            + ", ".join(f"{n} ({c.get('key', n)})" for n, c in sorted(lock.cameras.items()))
            + " fitted from the dataset videos; see <camera> in <worldbody>"
        )
    if generic:
        lines.append(f"Base model: {_base_model_text(robot)} (see its LICENSE).")
    else:
        lines.append("Base model: MuJoCo Menagerie robotstudio_so101 (see assets LICENSE).")
    return lines


def _base_model_text(robot) -> str:
    m = robot.menagerie
    if m is None:
        return f"{robot.model_path.name} shipped with antares"
    return f"MuJoCo Menagerie {m.directory}/{m.xml} at {m.commit[:12]}"


def sidecar_dict(
    lock: RobotLock, model: mujoco.MjModel, model_file: str, meshdir: str, unit_map: UnitMap | None = None
) -> dict[str, Any]:
    """The ``.antares.json`` next to an export. ``unit_map`` is the fitted
    arm's map for robots exported through :func:`fitted_spec` (the SO arms
    read theirs off ``model``)."""
    r = lock.result
    robot = _lock_robot(lock)
    generic = robot.export_class is None
    return {
        "format": EXPORT_FORMAT,
        "antares_version": __version__,
        "model": model_file,
        "meshdir": meshdir,
        "lock": {
            "name": lock.name,
            "dataset": lock.dataset,
            "robot_type": lock.robot_type,
            "data_sha256": lock.data_sha256,
            "created_at": lock.created_at,
            "antares_version": lock.antares_version,
        },
        "params": lock.params.to_dict(),
        "baked_params": list(BAKED_PARAMS),
        "sidecar_params": list(SIDECAR_PARAMS),
        "latency_frames": lock.params.to_dict()["latency_frames"],  # int, or six ints for a per-joint lock
        "latency_note": (
            "Command delay in dataset frames (at the dataset's recording rate). "
            "Not representable in MJCF: delay each ctrl by latency_frames frames."
        ),
        "joints": list(robot.joints) if generic else list(JOINTS),
        "robot": getattr(lock, "robot", None),
        "deadband_note": (
            "Servo position-error deadband in radians (params.deadband, shared or per joint; "
            "0 or absent means none). Not representable in MJCF: command q + shrink(target - q) "
            "with |error| reduced by the band, as antares.sim.Arm.replay does."
        ),
        "command_map": {
            **CommandMap.from_dict(lock.command_map, robot.n_joints).to_dict(),
            "note": "Per-arm sensor calibration: correct each logged command as "
            "gain * u + bias (LeRobot units) before the unit map. Identity if the lock has none.",
        },
        "unit_map": _unit_map_entry(robot, model, unit_map),
        "scores": {
            "baseline_eval_score": _score(r.baseline_eval_score),
            "eval_score": _score(r.eval_score),
            "baseline_train_score": _score(r.baseline_train_score),
            "train_score": _score(r.train_score),
            "unit": "deg RMSE",
            "eval_episodes": list(r.eval_episodes),
            "train_episodes": list(r.train_episodes),
        },
    }


def _unit_map_entry(robot: Any, model: mujoco.MjModel, unit_map: UnitMap | None) -> dict[str, Any]:
    if robot.export_class is not None:
        return {
            **UnitMap.from_model(model).to_dict(),
            "note": "rad = scale * u + offset, clipped to [lo, hi]; u in LeRobot units "
            "(body joints -100..100, gripper 0..100).",
        }
    if unit_map is None:  # the arm the spec was fitted on is normally passed in
        unit_map = Arm(robot=robot.name).unit_map
    return {
        **unit_map.to_dict(),
        "units": list(robot.unit_kinds()),
        "note": "rad = scale * u + offset, clipped to [lo, hi]; u in the dataset's own units, "
        "one entry per column of 'joints'.",
    }


def sidecar_path(xml_path: Path) -> Path:
    xml_path = Path(xml_path)
    return xml_path.with_name(xml_path.stem + SIDECAR_SUFFIX)


def _resolve_out(out_path: Path, name: str) -> Path:
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".xml":  # a directory was given
        out_path = out_path / f"{name}.xml"
    return out_path


def _copy_assets(xml: str, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    files = referenced_meshes(xml)
    for f in files:
        src = MESH_DIR / f
        if not src.is_file():
            raise ExportError(f"mesh {f} referenced by the MJCF is missing from {MESH_DIR}")
        shutil.copyfile(src, dest / f)
    return files


def fitted_spec(lock: RobotLock, arm: Arm | None = None) -> mujoco.MjSpec:
    """The lock's robot as an :class:`mujoco.MjSpec` carrying the fitted values.

    The parameters (added factors included) are applied to a compiled
    :class:`~antares.sim.Arm` (``arm``, or one built for the lock's robot),
    so the export holds exactly what the fit simulated; each mapped joint
    gets its damping, armature and friction loss, each actuator its gains,
    bias and force range, and the global options (timestep, gravity) are
    copied.
    """
    if arm is None:
        arm = Arm(robot=_lock_robot(lock).name)
    arm.apply(lock.params)
    m = arm.model
    spec = mujoco.MjSpec.from_file(str(arm.model_path))
    for j, d in zip(arm.joint_ids, arm.dof_ids):
        el = spec.joint(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)))
        damping = np.array(el.damping, dtype=np.float64).reshape(-1)  # [linear, ...] in MuJoCo >= 3.5
        if damping.size > 1:
            damping[0] = float(m.dof_damping[d])
            el.damping = damping
        else:
            el.damping = float(m.dof_damping[d])
        el.armature = float(m.dof_armature[d])
        el.frictionloss = float(m.dof_frictionloss[d])
    for a in arm.actuator_ids:
        el = spec.actuator(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a)))
        el.gainprm[:3] = m.actuator_gainprm[a, :3]
        el.biasprm[:3] = m.actuator_biasprm[a, :3]
        el.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        el.forcerange = m.actuator_forcerange[a]
    spec.option.timestep = float(m.opt.timestep)
    spec.option.gravity = m.opt.gravity.copy()
    return spec


def _export_generic(lock: RobotLock, out_path: Path, bundle_assets: bool) -> Path:
    robot = _lock_robot(lock)
    arm = Arm(robot=robot.name)  # one compile: the spec's values and the sidecar's unit map
    spec = fitted_spec(lock, arm)
    src_dir = Path(spec.modelfiledir) if spec.modelfiledir else robot.model_path.parent
    mesh_src = src_dir / (spec.meshdir or "")
    tex_src = src_dir / (spec.texturedir or "")
    if bundle_assets:
        dest = out_path.parent / ASSETS_DIRNAME
        for mesh in spec.meshes:
            if mesh.file:
                (dest / mesh.file).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(mesh_src / mesh.file, dest / mesh.file)
        for tex in spec.textures:
            if tex.file:
                (dest / tex.file).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tex_src / tex.file, dest / tex.file)
        lic = src_dir / "LICENSE"
        if lic.is_file():
            shutil.copyfile(lic, dest / "LICENSE")
        meshdir = texturedir = ASSETS_DIRNAME
    else:
        meshdir, texturedir = str(mesh_src.resolve()), str(tex_src.resolve())
    xml = spec.to_xml()  # with the source dirs, so MuJoCo can still find the files
    xml = _sub_one(xml, re.compile(r"<compiler\b[^>]*/>"), {"meshdir": meshdir, "texturedir": texturedir}, "<compiler>")
    xml = insert_cameras(xml, lock.cameras)
    xml = insert_header(xml, _comment(header_lines(lock)))
    out_path.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(out_path))  # validates the export
    sidecar = sidecar_dict(lock, model, out_path.name, meshdir, unit_map=arm.unit_map)
    sidecar_path(out_path).write_text(json.dumps(sidecar, indent=2) + "\n")
    return out_path


def export_mjcf(lock: RobotLock, out_path: Path, bundle_assets: bool = True) -> Path:
    """Write ``lock``'s calibrated SO-101 as MJCF at ``out_path`` (``.xml``).

    If ``out_path`` has no ``.xml`` suffix it is taken as a directory and the
    file is named ``<lock.name>.xml``. With ``bundle_assets`` the referenced
    STL meshes are copied into ``assets/`` next to it and ``meshdir`` is
    relative; otherwise ``meshdir`` is the absolute package assets dir. A
    sidecar ``<stem>.antares.json`` is written alongside. The result is
    loaded with MuJoCo before returning, so a broken export raises.
    """
    out_path = _resolve_out(out_path, lock.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _generic(lock):
        return _export_generic(lock, out_path, bundle_assets)

    xml = MODEL_PATH.read_text()
    xml = bake_params(xml, lock.params.to_dict())
    if bundle_assets:
        meshdir = ASSETS_DIRNAME
        _copy_assets(xml, out_path.parent / ASSETS_DIRNAME)
    else:
        meshdir = str(MESH_DIR.resolve())
    xml = set_meshdir(xml, meshdir)
    xml = insert_cameras(xml, lock.cameras)
    xml = insert_header(xml, _comment(header_lines(lock)))
    out_path.write_text(xml)

    model = mujoco.MjModel.from_xml_path(str(out_path))  # validates the export
    sidecar = sidecar_dict(lock, model, out_path.name, meshdir)
    sidecar_path(out_path).write_text(json.dumps(sidecar, indent=2) + "\n")
    return out_path


def export_scene(
    lock: RobotLock,
    out_dir: Path,
    bundle_assets: bool = True,
    task: TaskDistribution | None = None,
    object_size: float = OBJECT_SIZE,
    scene_fit: Any | None = None,
    lock_dir: Path | str | None = None,
    objects_fit: Any | None = None,
) -> Path:
    """Export the arm into ``out_dir`` plus a ``scene.xml`` (light and skybox
    from the Menagerie scene) that includes it, so
    ``python -m mujoco.viewer --mjcf <out_dir>/scene.xml`` works. Returns the
    scene path; the arm is at ``<out_dir>/<lock.name>.xml``.

    With a ``task`` (see :mod:`antares.task`) the scene also gets a free cube
    at the logged mean pick position and a marker at the mean place, and the
    task is copied next to it as ``task.json`` for
    :meth:`~antares.task.TaskDistribution.sample` and
    :func:`~antares.task.set_task`.

    With a :class:`antares.scene.SceneFit` (``scene_fit``, or the one saved
    as ``<lock_dir>/<lock>.scene.json`` when ``lock_dir`` is given) the
    scene also carries the table at the fitted height and, per fitted
    camera, its backdrop image on a quad filling the camera's view (PNGs
    under ``backdrops/``); otherwise the Menagerie checker floor stays.

    With an :class:`antares.objects.ObjectFit` (``objects_fit``, or the one
    saved as ``<lock_dir>/<lock>.objects.json``) the free body is the object
    seen in the video instead of the task cube: its vision footprint, its
    estimated mass and the mean vision start pose, with the start/end
    distributions in ``objects.json``. The objects fit wins when both it
    and a ``task`` exist; the task is still written as ``task.json``, with
    its picks taken from vision (``source = "vision"``).
    """
    out_dir = Path(out_dir)
    if objects_fit is None and lock_dir is not None:
        from antares.objects import load_object_fit

        objects_fit = load_object_fit(lock.name, lock_dir)
    if objects_fit is not None and (objects_fit.footprint is None or objects_fit.start is None):
        objects_fit = None  # a fit that saw nothing: fall back to the task cube
    if objects_fit is not None:
        objects = object_elements(objects_fit)
        if task is not None:
            starts = {int(t["episode"]): t["start_pose"] for t in objects_fit.tracks if t.get("start_pose") is not None}
            task = task.with_vision_picks(starts)
    else:
        objects = scene_xml(task, object_size) if task is not None else None  # may raise: before anything is written
    arm_path = export_mjcf(lock, out_dir / f"{lock.name}.xml", bundle_assets=bundle_assets)

    if scene_fit is None and lock_dir is not None:
        from antares.scene import load_scene_fit

        scene_fit = load_scene_fit(lock.name, lock_dir)

    scene = _GENERIC_SCENE if _generic(lock) else SCENE_PATH.read_text()
    scene, n = re.subn(r'<include file="[^"]+"/>', f'<include file="{arm_path.name}"/>', scene, count=1)
    if n == 0:
        raise ExportError("scene.xml has no <include>")
    if objects is not None:
        if scene.count("</worldbody>") != 1:
            raise ExportError("scene.xml needs exactly one </worldbody>")
        scene = scene.replace("</worldbody>", objects + "  </worldbody>")
        if task is not None:
            task.save(out_dir / TASK_FILENAME)
        if objects_fit is not None:
            (out_dir / OBJECTS_FILENAME).write_text(json.dumps(objects_sidecar(objects_fit), indent=2) + "\n")
    lines = [
        f"Antares scene for lock {lock.name}: includes the calibrated arm {arm_path.name}.",
        f"View it: python -m mujoco.viewer, passing {SCENE_FILENAME} as the mjcf argument.",
    ]
    if scene_fit is not None:
        files = write_backdrops(scene_fit, lock_dir, out_dir) if lock_dir is not None else {}
        scene = insert_scene(scene, scene_fit, lock.cameras, files)
        lines.append(
            f"Static scene from the logs: table at z={scene_fit.table.z:.4f} m"
            + (f", backdrops for {', '.join(sorted(files))}" if files else "")
            + ". Look through a fitted camera to see the real cell behind the sim arm."
        )
    if objects_fit is not None:
        fp = objects_fit.footprint
        lines.append(
            f"Object from the video: {objects_fit.description} ({fp[0] * 100:.1f} x {fp[1] * 100:.1f} cm), "
            f"free body {OBJECT_BODY} at the mean vision start pose; start/end distributions in {OBJECTS_FILENAME}."
        )
    scene = insert_header(scene, _comment(lines))
    scene_path = out_dir / SCENE_FILENAME
    scene_path.write_text(scene)
    mujoco.MjModel.from_xml_path(str(scene_path))  # validates
    return scene_path


def export_zip(
    lock: RobotLock,
    targets: Sequence[str] = (),
    fps: float | None = None,
    lock_dir: Path | str | None = None,
) -> bytes:
    """Zip of ``export_scene(lock, bundle_assets=True)``: ``<name>/`` holding
    the arm XML, sidecar, ``scene.xml`` and ``assets/*.stl``, plus the files
    of each extra export target (:data:`antares.export_targets.TARGETS`) and,
    with a scene fit under ``lock_dir``, ``backdrops/*.png``."""
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / lock.name
        export_scene(lock, root, bundle_assets=True, lock_dir=lock_dir)
        if targets:
            from antares.export_targets import export_targets

            export_targets(lock, root, targets, fps=fps)
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(root.parent).as_posix())
    return buf.getvalue()


# --------------------------------------------------------------------------
# export -> Arm
# --------------------------------------------------------------------------


def read_sidecar(xml_path: Path) -> dict[str, Any]:
    return json.loads(sidecar_path(xml_path).read_text())


def load_exported_arm(xml_path: Path, contacts: bool = False) -> Arm:
    """An :class:`Arm` driven by an exported model. The parameters come from
    the sidecar when it is there (its ``params`` carry per-joint values,
    ``latency_frames`` and the servo ``deadband``, which the XML cannot), else the gains/joint values
    compiled into the XML with latency 0; the command map likewise comes
    from the sidecar (identity if it is missing)."""
    xml_path = Path(xml_path)
    params: SimParams | None = None
    cmap, robot = None, None
    sc = sidecar_path(xml_path)
    if sc.is_file():
        side = json.loads(sc.read_text())
        if isinstance(side.get("params"), dict):
            params = SimParams.from_dict(side["params"])
        cmap = CommandMap.from_dict(side.get("command_map"))
        robot = side.get("robot")
    arm = Arm(model_path=xml_path, contacts=contacts, command_map=cmap, robot=robot)
    if params is None:
        params = (
            arm.default_params(from_model=True)
            if arm.robot.export_class is None
            else SimParams.from_model(arm.model, latency_frames=0)
        )
    arm.apply(params)
    return arm


__all__ = [
    "ASSETS_DIRNAME",
    "BACKDROPS_DIRNAME",
    "BAKED_PARAMS",
    "EXPORT_FORMAT",
    "ExportError",
    "OBJECTS_FILENAME",
    "SCENE_FILENAME",
    "SCENE_PATH",
    "SIDECAR_SUFFIX",
    "bake_params",
    "export_mjcf",
    "export_scene",
    "export_zip",
    "fitted_spec",
    "header_lines",
    "insert_header",
    "insert_scene",
    "load_exported_arm",
    "object_elements",
    "objects_sidecar",
    "read_sidecar",
    "referenced_meshes",
    "scene_elements",
    "set_meshdir",
    "sidecar_dict",
    "sidecar_path",
    "write_backdrops",
]
