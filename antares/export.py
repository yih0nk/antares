"""Export a lock's calibrated simulator as a standalone MJCF.

A :class:`~antares.lock.RobotLock` holds fitted :class:`~antares.sim.SimParams`.
:func:`export_mjcf` bakes those values into a copy of the SO-101 MJCF so the
model can be dropped into any MuJoCo or RL pipeline without Antares:

* ``kp``/``kv`` go on the ``sts3215`` default class's ``<position>`` and
  ``damping``/``armature``/``frictionloss`` on its ``<joint>``. Per-joint
  values (a sequence of six) are written on the named ``<joint>`` /
  ``<position>`` elements instead.
* ``meshdir`` is rewritten: either the STL meshes are copied into an
  ``assets/`` directory next to the output, or it points (absolute) at the
  package's own assets.
* A header comment records the lock (name, dataset, created_at, held-out
  score before/after, latency).

MJCF cannot express command latency, so a sidecar ``<name>.antares.json``
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
from antares.sim import JOINTS, MESH_DIR, MODEL_PATH, Arm, CommandMap, SimParams, UnitMap

SCENE_PATH: Path = MODEL_PATH.parent / "scene.xml"
SIDECAR_SUFFIX = ".antares.json"
SCENE_FILENAME = "scene.xml"
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
BAKED_PARAMS: tuple[str, ...] = JOINT_PARAMS + ACTUATOR_PARAMS

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
    instead (explicit attributes override class defaults in MuJoCo). Keys
    other than the five baked parameters (e.g. ``latency_frames``) are
    ignored. The rest of the text is untouched.
    """
    scalar: dict[str, str] = {}
    per_joint: dict[str, list[str]] = {}
    for key, value in params.items():
        if key not in BAKED_PARAMS:
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


def header_lines(lock: RobotLock) -> list[str]:
    r = lock.result
    p = lock.params
    robot = _lock_robot(lock)
    if _generic(lock):
        return [
            f"Antares calibrated {robot.label} (antares {__version__}, lock format {EXPORT_FORMAT})",
            f"lock:        {lock.name}",
            f"dataset:     {lock.dataset} ({lock.robot_type}), data sha256 {lock.data_sha256}",
            f"created_at:  {lock.created_at}",
            f"held-out tracking RMSE: {_score_text(r.baseline_eval_score)} before -> "
            f"{_score_text(r.eval_score)} after ({len(r.eval_episodes)} episodes)",
            "fitted values are written on each mapped joint and actuator",
            f"latency_frames: {p.latency_frames} (command delay in dataset frames; MJCF cannot "
            f"express this, apply it in your control loop; see the .antares.json sidecar)",
            f"Base model: {_base_model_text(robot)} (see its LICENSE).",
        ]
    return [
        f"Antares calibrated SO-101 (antares {__version__}, lock format {EXPORT_FORMAT})",
        f"lock:        {lock.name}",
        f"dataset:     {lock.dataset} ({lock.robot_type}), data sha256 {lock.data_sha256}",
        f"created_at:  {lock.created_at}",
        f"held-out tracking RMSE: {_score_text(r.baseline_eval_score)} before -> "
        f"{_score_text(r.eval_score)} after ({len(r.eval_episodes)} episodes)",
        f"baked: kp={_fmt(p.kp)} kv={_fmt(p.kv)} damping={_fmt(p.damping)} "
        f"armature={_fmt(p.armature)} frictionloss={_fmt(p.frictionloss)}",
        f"latency_frames: {p.latency_frames} (command delay in dataset frames; MJCF cannot "
        f"express this, apply it in your control loop; see the .antares.json sidecar)",
    ]
    if lock.cameras:
        lines.append(
            "cameras:     "
            + ", ".join(f"{n} ({c.get('key', n)})" for n, c in sorted(lock.cameras.items()))
            + " fitted from the dataset videos; see <camera> in <worldbody>"
        )
    lines.append("Base model: MuJoCo Menagerie robotstudio_so101 (see assets LICENSE).")
    return lines


def _base_model_text(robot) -> str:
    m = robot.menagerie
    if m is None:
        return f"{robot.model_path.name} shipped with antares"
    return f"MuJoCo Menagerie {m.directory}/{m.xml} at {m.commit[:12]}"


def sidecar_dict(lock: RobotLock, model: mujoco.MjModel, model_file: str, meshdir: str) -> dict[str, Any]:
    r = lock.result
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
        "latency_frames": int(lock.params.latency_frames),
        "latency_note": (
            "Command delay in dataset frames (at the dataset's recording rate). "
            "Not representable in MJCF: delay each ctrl by latency_frames frames."
        ),
        "joints": list(_lock_robot(lock).joints) if _generic(lock) else list(JOINTS),
        "robot": getattr(lock, "robot", None),
        "command_map": {
            **CommandMap.from_dict(lock.command_map, _lock_robot(lock).n_joints).to_dict(),
            "note": "Per-arm sensor calibration: correct each logged command as "
            "gain * u + bias (LeRobot units) before the unit map. Identity if the lock has none.",
        },
        "unit_map": _unit_map_entry(lock, model),
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


def _unit_map_entry(lock: RobotLock, model: mujoco.MjModel) -> dict[str, Any]:
    if not _generic(lock):
        return {
            **UnitMap.from_model(model).to_dict(),
            "note": "rad = scale * u + offset, clipped to [lo, hi]; u in LeRobot units "
            "(body joints -100..100, gripper 0..100).",
        }
    robot = _lock_robot(lock)
    arm = Arm(robot=robot.name)
    return {
        **arm.unit_map.to_dict(),
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


def fitted_spec(lock: RobotLock) -> mujoco.MjSpec:
    """The lock's robot as an :class:`mujoco.MjSpec` carrying the fitted values.

    The parameters (added factors included) are applied to a compiled
    :class:`~antares.sim.Arm`, so the export holds exactly what the fit
    simulated; each mapped joint gets its damping, armature and friction
    loss, each actuator its gains, bias and force range, and the global
    options (timestep, gravity) are copied.
    """
    robot = _lock_robot(lock)
    arm = Arm(robot=robot.name)
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
    spec = fitted_spec(lock)
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
    sidecar = sidecar_dict(lock, model, out_path.name, meshdir)
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


def export_scene(lock: RobotLock, out_dir: Path, bundle_assets: bool = True) -> Path:
    """Export the arm into ``out_dir`` plus a ``scene.xml`` (floor, light,
    skybox from the Menagerie scene) that includes it, so
    ``python -m mujoco.viewer --mjcf <out_dir>/scene.xml`` works. Returns the
    scene path; the arm is at ``<out_dir>/<lock.name>.xml``."""
    out_dir = Path(out_dir)
    arm_path = export_mjcf(lock, out_dir / f"{lock.name}.xml", bundle_assets=bundle_assets)

    scene = _GENERIC_SCENE if _generic(lock) else SCENE_PATH.read_text()
    scene, n = re.subn(r'<include file="[^"]+"/>', f'<include file="{arm_path.name}"/>', scene, count=1)
    if n == 0:
        raise ExportError("scene.xml has no <include>")
    scene = insert_header(
        scene,
        _comment(
            [
                f"Antares scene for lock {lock.name}: includes the calibrated arm {arm_path.name}.",
                f"View it: python -m mujoco.viewer, passing {SCENE_FILENAME} as the mjcf argument.",
            ]
        ),
    )
    scene_path = out_dir / SCENE_FILENAME
    scene_path.write_text(scene)
    mujoco.MjModel.from_xml_path(str(scene_path))  # validates
    return scene_path


def export_zip(lock: RobotLock) -> bytes:
    """Zip of ``export_scene(lock, bundle_assets=True)``: ``<name>/`` holding
    the arm XML, sidecar, ``scene.xml`` and ``assets/*.stl``."""
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / lock.name
        export_scene(lock, root, bundle_assets=True)
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
    """An :class:`Arm` driven by an exported model: gains/joint values come
    from the XML, ``latency_frames`` and the command map from the sidecar
    (0 and identity if it is missing)."""
    xml_path = Path(xml_path)
    latency, cmap, robot = 0, None, None
    sc = sidecar_path(xml_path)
    if sc.is_file():
        side = json.loads(sc.read_text())
        latency = int(side.get("latency_frames", 0))
        cmap = CommandMap.from_dict(side.get("command_map"))
        robot = side.get("robot")
    arm = Arm(model_path=xml_path, contacts=contacts, command_map=cmap, robot=robot)
    if arm.robot.export_class is None:
        arm.apply(arm.default_params(from_model=True).replace(latency_frames=latency))
    else:
        arm.apply(SimParams.from_model(arm.model, latency_frames=latency))
    return arm


__all__ = [
    "ASSETS_DIRNAME",
    "BAKED_PARAMS",
    "EXPORT_FORMAT",
    "ExportError",
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
    "load_exported_arm",
    "read_sidecar",
    "referenced_meshes",
    "set_meshdir",
    "sidecar_dict",
    "sidecar_path",
]
