"""Export a lock's fitted actuator parameters for other simulators.

:mod:`antares.export` writes the calibrated MuJoCo MJCF. The SO-100/101
training ecosystem mostly runs on NVIDIA Isaac Lab, then ManiSkill (SAPIEN)
and mjlab, and none of them reads MJCF actuator defaults the way MuJoCo
does. This module renders the same fitted :class:`~antares.sim.SimParams`
as each target's own actuator configuration, with the command latency
(which no simulator expresses natively) documented as a control-loop delay
and the data-derived DR ranges and pinned parameters alongside when the
lock has them.

Targets (``TARGETS``):

``isaaclab``
    ``<name>_isaaclab.py`` with ``SO101_ACTUATOR_CFG`` (an
    ``ImplicitActuatorCfg``), ``SO101_ARTICULATION_CFG`` around a USD path
    placeholder, ``CONTROL_HZ`` / ``COMMAND_DELAY_STEPS`` and, when
    ``locks/<name>.dr.json`` exists, ``EventTerm`` snippets for
    ``randomize_actuator_gains`` / ``randomize_joint_parameters``; plus
    ``<name>_isaaclab.json`` with the same numbers.
``maniskill``
    ``<name>_maniskill.py`` with ``PDJointPosControllerConfig`` for the six
    joints (absolute and delta variants) and ``<name>_maniskill.json``.
``mjlab``
    ``<name>_mjlab.py`` building an ``EntityCfg`` from the exported MJCF
    with ``XmlActuatorCfg`` (gains are already baked into the XML) and the
    delay in physics steps.
``json``
    ``<name>.params.json``: the simulator-neutral sidecar
    (:func:`antares.export.sidecar_dict`) plus the per-target numbers,
    latency in seconds, pinned parameters and DR ranges.

Unit mapping (MuJoCo hinge joints, ``<compiler angle="radian">``):

* ``kp`` (``<position kp>``, N*m/rad) -> Isaac Lab / ManiSkill ``stiffness``
  and mjlab (baked). One-to-one: Isaac Lab stores drive gains per radian
  and converts to the USD per-degree convention itself
  (``isaaclab/sim/schemas/schemas.py``: ``# N-m/rad --> N-m/deg``); SAPIEN
  wraps PhysX articulations, whose joint coordinates are radians (the
  ManiSkill SO-100 agent's rest keyframe is ``-1.5708``).
* ``kv`` (``<position kv>``, N*m*s/rad) **plus** ``damping`` (``<joint
  damping>``, N*m*s/rad) -> ``damping``. MuJoCo applies both as
  ``-b * qdot`` on a position-only actuator (the target velocity is 0), and
  PhysX drives have one damping slot, so the sum is the exact equivalent
  at zero target velocity; the raw pair is kept in the JSON.
* ``armature`` (kg*m^2) -> Isaac Lab ``armature`` (on the actuator cfg,
  not the articulation), mjlab (baked); ManiSkill: not verified to be
  settable from the controller config, emitted as a constant.
* ``frictionloss`` (N*m, Coulomb torque) -> Isaac Lab ``friction`` is an
  effort (N*m) on Isaac Sim 5.0+ but a unitless coefficient on 4.5
  (actuator cfg docstring), so the value is emitted with that caveat;
  ManiSkill ``friction`` is the PhysX joint friction *coefficient* and
  cannot be derived from a torque without the load, so it is left at 0.0
  and the N*m value is emitted beside it; mjlab keeps the XML value.
* ``forcerange`` (N*m) -> Isaac Lab ``effort_limit_sim``, ManiSkill
  ``force_limit``, mjlab (baked).
* ``latency_frames`` (dataset frames at the recording fps) -> seconds =
  frames / fps; "delay each command by N control steps at F Hz". mjlab's
  ``XmlActuatorCfg`` has ``delay_min_lag`` / ``delay_max_lag`` in physics
  timesteps, so the delay is also given in steps of the MJCF timestep.

Verified against (2026-10-17):

* Isaac Lab ``main`` = 2.3.2 (``VERSION``), ``isaaclab`` extension 0.54.4:
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html
  (``ActuatorBaseCfg`` / ``ImplicitActuatorCfg`` fields, ``dict[str, float]``
  per joint, ``None`` = USD value),
  https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html,
  ``source/isaaclab/isaaclab/actuators/actuator_base_cfg.py`` (friction
  units note), ``source/isaaclab/isaaclab/sim/schemas/schemas.py`` (rad ->
  deg conversion), ``source/isaaclab/isaaclab/envs/mdp/events.py``
  (``randomize_actuator_gains``, ``randomize_joint_parameters``:
  ``operation`` in add/scale/abs, ``distribution`` in
  uniform/log_uniform/gaussian), and LeIsaac's SO-101 config
  (``source/leisaac/leisaac/assets/robots/lerobot.py``: ``ImplicitActuatorCfg``
  with the same six joint names, stiffness 17.8, damping 0.60,
  effort_limit_sim 10).
* ManiSkill 3.0.1 (``mani_skill/__init__.py``):
  https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html
  (PD law ``F = Kp(q_target - q) + Kd(qdot_target - qdot)``),
  ``mani_skill/agents/controllers/pd_joint_pos.py`` (fields ``lower``,
  ``upper``, ``stiffness``, ``damping``, ``force_limit=1e10``,
  ``friction=0.0``, ``use_delta``, ``use_target``, ``interpolate``,
  ``normalize_action=True``, ``drive_mode="force"``; applied with
  ``joint.set_drive_properties(...)`` and ``joint.set_friction(...)``) and
  ``mani_skill/agents/robots/so100/so_100.py`` (stiffness 1e3, damping 1e2,
  force_limit 100 on all six joints, delta limits +-0.05 / +-0.2 rad).
* mjlab 1.6.0 (``pyproject.toml``; mujoco / mujoco-warp ~=3.11):
  ``src/mjlab/entity/entity.py`` (``EntityCfg(spec_fn, articulation)``,
  ``EntityArticulationInfoCfg(actuators, soft_joint_pos_limit_factor)``),
  ``src/mjlab/actuator/actuator.py`` (``ActuatorCfg``: ``target_names_expr``,
  ``armature`` / ``frictionloss`` / ``viscous_damping`` with ``None`` =
  keep XML, ``delay_min_lag`` / ``delay_max_lag`` in physics timesteps),
  ``src/mjlab/actuator/xml_actuator.py`` (``XmlActuatorCfg`` wraps the
  ``<position>`` actuators already in the MJCF) and
  ``src/mjlab/actuator/builtin_actuator.py`` (``BuiltinPositionActuatorCfg``
  with ``stiffness`` / ``damping`` / ``effort_limit``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from antares import __version__
from antares.export import EXPORT_FORMAT, header_lines, sidecar_dict
from antares.lock import LOCK_DIR, RobotLock, param_spread, pinned_params
from antares.sim import JOINTS, MODEL_PATH, CommandMap, SimParams

#: Non-MJCF export targets, in the order ``all`` writes them.
TARGETS: tuple[str, ...] = ("isaaclab", "maniskill", "mjlab", "json")
#: What ``antares export --format`` accepts.
FORMATS: tuple[str, ...] = ("mjcf",) + TARGETS + ("all",)

#: Recording rate assumed when the dataset's ``meta/info.json`` is not in the
#: local Hugging Face cache. Every LeRobot SO-100/101 dataset Antares ships
#: locks for records at 30 fps.
DEFAULT_FPS = 30.0
#: Physics step of the shipped MJCF (``<option timestep>``), for mjlab's
#: delay in physics timesteps.
MJCF_TIMESTEP = 0.005

#: The MJCF's actuators drive one joint each under this default class.
ACTUATOR_GROUP = "sts3215"
GRIPPER = "gripper"

#: Delta-position action limits (rad) used by ManiSkill's own SO-100 agent.
MANISKILL_DELTA_ARM = 0.05
MANISKILL_DELTA_GRIPPER = 0.2

#: DR dimensions that map onto a target's actuator parameters.
DR_PARAMS: tuple[str, ...] = ("kp", "kv", "damping", "armature", "frictionloss", "latency_frames")

#: One-line description per target for CLI output.
TARGET_LABELS = {
    "isaaclab": "Isaac Lab ImplicitActuatorCfg / ArticulationCfg (+ DR EventTerms)",
    "maniskill": "ManiSkill 3 PDJointPosControllerConfig (abs + delta)",
    "mjlab": "mjlab EntityCfg over the MJCF, delay in physics steps",
    "json": "simulator-neutral params: sidecar + every target's numbers",
}

MEDIA_TYPES = {
    ".py": "text/x-python",
    ".json": "application/json",
    ".xml": "application/xml",
}


class TargetExportError(ValueError):
    """Unknown target or a lock the renderer cannot express."""


@dataclass(frozen=True)
class TargetExport:
    """One target's rendered files: relative path -> text (main file first)."""

    target: str
    files: dict[str, str]
    notes: list[str] = field(default_factory=list)

    @property
    def main(self) -> str:
        """Relative path of the file ``GET .../export?format=`` returns."""
        return next(iter(self.files))

    def write(self, out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for rel, text in self.files.items():
            p = out_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            paths.append(p)
        return paths


# --------------------------------------------------------------------------
# shared facts: fps, latency, provenance
# --------------------------------------------------------------------------


def dataset_fps(lock: RobotLock, fps: float | None = None, default: float = DEFAULT_FPS) -> tuple[float, str]:
    """The lock's recording rate and where it came from.

    An explicit ``fps`` wins. Otherwise the dataset's ``meta/info.json`` is
    read from the local Hugging Face cache (never downloaded); failing that
    ``default`` is used and the source says so, so the exported files can
    carry the assumption.
    """
    if fps is not None:
        if not (float(fps) > 0):
            raise TargetExportError("fps must be positive")
        return float(fps), "given"
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(lock.dataset, "meta/info.json", repo_type="dataset")
        if isinstance(path, str) and Path(path).is_file():
            info = json.loads(Path(path).read_text())
            value = float(info["fps"])
            if value > 0:
                return value, f"{lock.dataset} meta/info.json (local cache)"
    except Exception:  # noqa: BLE001 - any cache trouble means "assume"
        pass
    return float(default), "assumed (dataset not in the local cache)"


def _latency_frames(params: SimParams) -> int | tuple[int, ...]:
    v = params.latency_frames
    return tuple(int(x) for x in v) if isinstance(v, tuple) else int(v)


def latency_info(params: SimParams, fps: float) -> dict[str, Any]:
    """Command delay in frames and seconds; per-joint lists when the lock is."""
    frames = _latency_frames(params)
    per_joint = isinstance(frames, tuple)
    arr = np.asarray(frames if per_joint else [frames], dtype=np.int64)
    seconds = (arr / fps).tolist()
    return {
        "control_hz": float(fps),
        "frames": list(frames) if per_joint else frames,
        "seconds": seconds if per_joint else seconds[0],
        "frames_min": int(arr.min()),
        "frames_max": int(arr.max()),
        "per_joint": per_joint,
        "note": (
            f"Delay every position command by {'these' if per_joint else 'this many'} control steps at "
            f"{fps:g} Hz ({'per joint' if per_joint else 'all joints'}) before it reaches the drive. "
            "No simulator expresses the latency natively; apply it in the control loop."
        ),
    }


def _spread_text(spread: Mapping[str, Any] | None) -> str:
    if not spread:
        return "n/a (single-seed fit)"
    parts = []
    for k in sorted(spread):
        s = spread[k]
        if "ratio" in s:
            parts.append(f"{k} x{float(s['ratio']):.2f}")
        elif "range" in s:
            parts.append(f"{k} +-{s['range']} frames")
    return ", ".join(parts) or "n/a"


def provenance_lines(lock: RobotLock, fps: float, fps_source: str) -> list[str]:
    """Header lines every exported file starts with."""
    lat = latency_info(lock.params, fps)
    lines = [ln for ln in header_lines(lock) if not ln.startswith("latency_frames:")]
    secs = lat["seconds"]
    secs_text = (
        ", ".join(f"{s:.4f}" for s in secs) + " s per joint" if lat["per_joint"] else f"{secs:.4f} s"
    )
    lines += [
        f"pinned params: {', '.join(pinned_params(lock)) or 'none'} (antares.lock.pinned_params; "
        f"the others trade off against each other at equal tracking error)",
        f"seed spread:   {_spread_text(param_spread(lock))}",
        f"parameters:    {'per-joint (one value per joint)' if lock.params.has_per_joint else 'scalar (shared by all six joints)'}",
        f"latency:       {lat['frames']} frame(s) at {fps:g} Hz = {secs_text}; fps {fps_source}",
        "units: MuJoCo hinge joints in radians: kp N*m/rad, kv/damping N*m*s/rad, armature kg*m^2, "
        "frictionloss N*m, forcerange N*m. Isaac Lab and SAPIEN/PhysX drive gains are per radian too.",
    ]
    cmap = CommandMap.from_dict(lock.command_map)
    if not cmap.is_identity():
        lines.append(
            "command_map:   this lock has a fitted sensor map; correct each logged command as "
            "gain*u + bias (LeRobot units) before the unit map, see the JSON."
        )
    return lines


def _header(lines: Sequence[str], prefix: str = "# ") -> str:
    return "\n".join(f"{prefix}{ln}".rstrip() for ln in lines)


def _provenance_dict(lock: RobotLock, fps: float, fps_source: str) -> dict[str, Any]:
    r = lock.result
    return {
        "antares_version": __version__,
        "format": EXPORT_FORMAT,
        "lock": lock.name,
        "dataset": lock.dataset,
        "robot_type": lock.robot_type,
        "created_at": lock.created_at,
        "heldout_rmse_deg": {"before": _num(r.baseline_eval_score), "after": _num(r.eval_score)},
        "pinned_params": pinned_params(lock),
        "param_spread": param_spread(lock),
        "per_joint": lock.params.has_per_joint,
        "fps_source": fps_source,
    }


def _num(v: float) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------
# parameter mapping
# --------------------------------------------------------------------------


def _scalar_or_joint(params: SimParams, name: str) -> float | dict[str, float]:
    v = getattr(params, name)
    if isinstance(v, tuple):
        return {j: float(x) for j, x in zip(JOINTS, v)}
    return float(v)


def drive_damping(params: SimParams) -> float | dict[str, float]:
    """``kv + damping``: the single PhysX drive-damping slot (see module doc)."""
    if not params.is_per_joint("kv") and not params.is_per_joint("damping"):
        return float(params.kv) + float(params.damping)
    total = params.per_joint("kv") + params.per_joint("damping")
    return {j: float(x) for j, x in zip(JOINTS, total)}


def mapped_params(params: SimParams) -> dict[str, Any]:
    """The fitted values in target vocabulary (scalar or per-joint dict).

    A lock fitted with the STS3215 servo model is given as its PD
    equivalent (:meth:`~antares.sim.SimParams.linearized`): stiffness from
    the firmware gain and motor constants, drive damping from the back-EMF,
    effort limit at stall. The firmware's rate limit has no counterpart.
    """
    params = params.linearized()
    return {
        "stiffness": _scalar_or_joint(params, "kp"),
        "damping": drive_damping(params),
        "drive_damping_kv": _scalar_or_joint(params, "kv"),
        "joint_damping": _scalar_or_joint(params, "damping"),
        "armature": _scalar_or_joint(params, "armature"),
        "frictionloss": _scalar_or_joint(params, "frictionloss"),
        "effort_limit": _scalar_or_joint(params, "forcerange"),
    }


def joint_limits() -> tuple[list[float], list[float]]:
    """Joint ranges (rad) of the shipped MJCF, canonical joint order."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    return model.jnt_range[:, 0].tolist(), model.jnt_range[:, 1].tolist()


def load_dr_ranges(lock: RobotLock, lock_dir: Path | str | None = None) -> Any | None:
    """``locks/<name>.dr.json`` as :class:`~antares.dr.DRRanges`, or None."""
    from antares.dr import DR_SUFFIX, DRRanges

    p = Path(lock_dir if lock_dir is not None else LOCK_DIR) / f"{lock.name}{DR_SUFFIX}"
    if not p.is_file():
        return None
    return DRRanges.load(p)


def dr_dict(dr: Any | None, lock: RobotLock | None = None) -> dict[str, Any] | None:
    """DR ranges in target vocabulary: log-uniform ``[lo, hi]`` per parameter.

    With ``lock``, also records whether the ranges were derived around this
    lock's own values (``nominal_matches_lock``); a lock refit after
    ``antares dr`` ran keeps the older ranges, and consumers should know.
    """
    if dr is None:
        return None
    rng = {k: dr.ranges[k] for k in DR_PARAMS if k in dr.ranges}
    matches = None
    if lock is not None:
        matches = all(
            np.allclose(lock.params.per_joint(k), dr.nominal.per_joint(k), rtol=1e-9, atol=0) for k in rng
        )
    out: dict[str, Any] = {
        "source": f"{dr.n_fits} refits x {dr.n_train} episodes, percentiles {list(dr.percentiles)}, "
        f"seed {dr.seed}, created {dr.created_at}",
        "dataset": dr.dataset,
        "nominal": dr.nominal.to_dict(),
        "nominal_matches_lock": matches,
        "scale": "log-uniform between lo and hi (latency: uniform integer frames)",
        "mujoco": {k: {"lo": v.lo, "hi": v.hi} for k, v in rng.items()},
    }
    mapped: dict[str, Any] = {}
    if "kp" in rng:
        mapped["stiffness"] = [rng["kp"].lo, rng["kp"].hi]
    if "kv" in rng and "damping" in rng:
        mapped["damping"] = [rng["kv"].lo + rng["damping"].lo, rng["kv"].hi + rng["damping"].hi]
    if "armature" in rng:
        mapped["armature"] = [rng["armature"].lo, rng["armature"].hi]
    if "frictionloss" in rng:
        mapped["friction"] = [rng["frictionloss"].lo, rng["frictionloss"].hi]
    if "latency_frames" in rng:
        mapped["latency_frames"] = [int(rng["latency_frames"].lo), int(rng["latency_frames"].hi)]
    out["mapped"] = mapped
    return out


# --------------------------------------------------------------------------
# Python rendering helpers
# --------------------------------------------------------------------------


def _py(v: Any, indent: int = 0) -> str:
    """Python literal for a float, int, list, tuple or per-joint dict (multi-line).

    ``indent`` is the column the enclosing statement starts at, so a dict's
    closing brace lines up with it."""
    if isinstance(v, dict):
        pad = " " * (indent + 4)
        body = "".join(f"{pad}{k!r}: {_py(x, indent + 4)},\n" for k, x in v.items())
        return "{\n" + body + " " * indent + "}"
    if isinstance(v, tuple):
        inner = ", ".join(_py(x, indent) for x in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    if isinstance(v, list):
        return "[" + ", ".join(_py(x, indent) for x in v) + "]"
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, (int, np.integer)):
        return repr(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            raise TargetExportError(f"cannot write non-finite value {f!r}")
        return repr(f)
    if v is None:
        return "None"
    return repr(v)


def _as_list(v: float | dict[str, float]) -> list[float]:
    """Per-joint list in canonical order (scalars broadcast)."""
    if isinstance(v, dict):
        return [float(v[j]) for j in JOINTS]
    return [float(v)] * len(JOINTS)


def _delay_constants(lat: dict[str, Any]) -> str:
    frames = lat["frames"]
    secs = lat["seconds"]
    if lat["per_joint"]:
        frames_lit = "(" + ", ".join(str(int(f)) for f in frames) + ")"
        secs_lit = "(" + ", ".join(repr(float(s)) for s in secs) + ")"
        shape = "one entry per joint, in SO101_JOINTS order"
    else:
        frames_lit, secs_lit, shape = str(int(frames)), repr(float(secs)), "shared by all joints"
    return f"""#: Recording rate of the dataset the lock was fitted on: the control loop
#: the latency was measured against.
CONTROL_HZ = {lat['control_hz']!r}
#: Command latency the data shows: delay each position target by this many
#: control steps at CONTROL_HZ ({shape}). No simulator expresses this
#: natively; keep a FIFO of targets in your policy/env wrapper.
COMMAND_DELAY_STEPS = {frames_lit}
COMMAND_DELAY_S = {secs_lit}
"""


def _dr_python(dr_info: dict[str, Any] | None, pinned: Sequence[str]) -> str:
    if dr_info is None:
        return (
            "#: No data-derived DR ranges for this lock (run `antares dr <dataset>` to\n"
            "#: measure them; they are saved as locks/<name>.dr.json and exported here).\n"
            "DR_RANGES = None\n"
        )
    m = dr_info["mapped"]
    nom = dr_info["nominal"]
    nominal_text = ", ".join(f"{k}={nom[k]:.4g}" for k in DR_PARAMS if k in nom and not isinstance(nom[k], list))
    lines = [
        "#: Domain-randomization ranges measured from the robot's own data",
        f"#: ({dr_info['source']}). Sample log-uniformly between lo and hi; latency is an",
        "#: integer frame count. Keys are in this target's vocabulary (damping = kv + joint damping).",
        f"#: Pinned by the fit and not worth randomizing: {', '.join(pinned) or 'none'}.",
    ]
    if dr_info.get("nominal_matches_lock") is False:
        lines.append(
            f"#: NOTE: the refits were started from an earlier fit of this lock ({nominal_text}),"
            " not from the values above; rerun `antares dr` to re-derive them around this lock."
        )
    lines += ["DR_RANGES = " + _py({k: list(v) for k, v in m.items()}), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------


def _common(lock: RobotLock, fps: float | None, dr: Any | None, lock_dir: Path | str | None):
    fps_value, fps_source = dataset_fps(lock, fps)
    if dr is None:
        dr = load_dr_ranges(lock, lock_dir)
    lat = latency_info(lock.params, fps_value)
    prov = _provenance_dict(lock, fps_value, fps_source)
    prov["latency"] = lat
    header = provenance_lines(lock, fps_value, fps_source)
    return fps_value, fps_source, dr, lat, prov, header


def _command_map_dict(lock: RobotLock) -> dict[str, Any]:
    cmap = CommandMap.from_dict(lock.command_map)
    return {
        **cmap.to_dict(),
        "identity": cmap.is_identity(),
        "note": "Per-arm sensor calibration: correct each logged command as gain*u + bias "
        "(LeRobot units) before converting to radians. Identity if the lock has none.",
    }


def render_isaaclab(
    lock: RobotLock, fps: float | None = None, dr: Any | None = None, lock_dir: Path | str | None = None
) -> TargetExport:
    """Isaac Lab ``ImplicitActuatorCfg`` / ``ArticulationCfg`` + JSON twin."""
    fps_value, fps_source, dr, lat, prov, header = _common(lock, fps, dr, lock_dir)
    mp = mapped_params(lock.params)
    pinned = pinned_params(lock)
    dr_info = dr_dict(dr, lock)
    joints = _py(list(JOINTS))
    cmap = CommandMap.from_dict(lock.command_map)

    friction_note = (
        "        # MuJoCo frictionloss (Coulomb torque, N*m). Isaac Lab's `friction` is an effort\n"
        "        # on Isaac Sim 5.0+ (one-to-one) but a unitless coefficient on Isaac Sim 4.5:\n"
        "        # on 4.5 drop this line or retune it, the number below is a torque.\n"
    )
    damping_note = (
        "        # kv (drive damping) + MuJoCo joint damping: PhysX has one damping slot and\n"
        "        # both act as -b*qdot at zero target velocity, so the sum is exact there.\n"
    )
    cmap_note = (
        "# NOTE: this lock carries a fitted command map (see the JSON `command_map`):\n"
        "# apply gain*u + bias to logged commands (LeRobot units) before the unit map.\n\n"
        if not cmap.is_identity()
        else ""
    )

    dr_snippet = ""
    if dr_info is not None:
        m = dr_info["mapped"]
        lat_rng = m.get("latency_frames")
        dr_snippet = f'''
# -- domain randomization from data ------------------------------------------
# Event terms with the ranges Antares measured ({dr_info['source']}).
# `operation="abs"` + `distribution="log_uniform"` samples the absolute value
# log-uniformly in [lo, hi] (isaaclab.envs.mdp.events). Pinned parameters
# ({', '.join(pinned) or 'none'}) agreed across seeds and need no randomization.
# Attach these to your env cfg's EventsCfg, e.g. `class EventsCfg: gains = SO101_RANDOMIZE_GAINS`.
# Latency: the data puts it in [{lat_rng[0] if lat_rng else '?'}, {lat_rng[1] if lat_rng else '?'}] frames;
# randomize COMMAND_DELAY_STEPS in your wrapper, not here.
try:
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    import isaaclab.envs.mdp as mdp
except ImportError:  # keep the numbers importable without Isaac Lab
    EventTerm = SceneEntityCfg = mdp = None

if EventTerm is not None:
    SO101_RANDOMIZE_GAINS = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={{
            "asset_cfg": SceneEntityCfg("robot", joint_names=SO101_JOINTS),
            "stiffness_distribution_params": {_py(tuple(m['stiffness'])) if 'stiffness' in m else 'None'},
            "damping_distribution_params": {_py(tuple(m['damping'])) if 'damping' in m else 'None'},
            "operation": "abs",
            "distribution": "log_uniform",
        }},
    )
    SO101_RANDOMIZE_JOINTS = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={{
            "asset_cfg": SceneEntityCfg("robot", joint_names=SO101_JOINTS),
            "friction_distribution_params": {_py(tuple(m['friction'])) if 'friction' in m else 'None'},
            "armature_distribution_params": {_py(tuple(m['armature'])) if 'armature' in m else 'None'},
            "operation": "abs",
            "distribution": "log_uniform",
        }},
    )
'''

    py = f'''"""Isaac Lab actuator configuration for lock {lock.name} (generated by Antares).

{_header(header, prefix="")}

Verified against Isaac Lab 2.3.2 (isaaclab extension 0.54.4):
isaaclab.actuators.ImplicitActuatorCfg takes a float or a {{joint_name: value}}
dict per field; None means "keep the USD value". Armature and friction live on
the actuator cfg. Replace SO101_USD_PATH with your SO-101 USD (LeIsaac ships
one as robots/so101_follower.usd with these same six joint names).
"""

from __future__ import annotations

try:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
except ImportError:  # the numbers below stay importable without Isaac Lab
    sim_utils = ImplicitActuatorCfg = ArticulationCfg = None

{cmap_note}SO101_JOINTS = {joints}

#: Path to the SO-101 USD articulation (placeholder: point it at your asset).
SO101_USD_PATH = "/path/to/so101_follower.usd"

{_delay_constants(lat)}
# -- fitted actuator parameters (SI, per radian) ----------------------------
STIFFNESS = {_py(mp['stiffness'])}  # N*m/rad, MuJoCo kp
DAMPING = {_py(mp['damping'])}  # N*m*s/rad, MuJoCo kv + joint damping
ARMATURE = {_py(mp['armature'])}  # kg*m^2
FRICTION = {_py(mp['frictionloss'])}  # N*m (MuJoCo frictionloss; see note below)
EFFORT_LIMIT = {_py(mp['effort_limit'])}  # N*m, MuJoCo forcerange

{_dr_python(dr_info, pinned)}
if ImplicitActuatorCfg is not None:
    SO101_ACTUATOR_CFG = ImplicitActuatorCfg(
        joint_names_expr=list(SO101_JOINTS),
        stiffness=STIFFNESS,
{damping_note}        damping=DAMPING,
        armature=ARMATURE,
{friction_note}        friction=FRICTION,
        effort_limit_sim=EFFORT_LIMIT,
        # velocity_limit_sim is not fitted; None keeps the USD value.
    )

    SO101_ARTICULATION_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(usd_path=SO101_USD_PATH),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={{j: 0.0 for j in SO101_JOINTS}}),
        actuators={{"{ACTUATOR_GROUP}": SO101_ACTUATOR_CFG}},
        soft_joint_pos_limit_factor=1.0,
    )
{dr_snippet}'''

    data = {
        "target": "isaaclab",
        "verified": "Isaac Lab 2.3.2 (isaaclab 0.54.4), 2026-10-17",
        "header": list(header),
        "provenance": prov,
        "joints": list(JOINTS),
        "actuator": {
            "class": "isaaclab.actuators.ImplicitActuatorCfg",
            "joint_names_expr": list(JOINTS),
            "stiffness": mp["stiffness"],
            "damping": mp["damping"],
            "armature": mp["armature"],
            "friction": mp["frictionloss"],
            "effort_limit_sim": mp["effort_limit"],
            "velocity_limit_sim": None,
        },
        "mujoco_params": lock.params.to_dict(),
        "units": {
            "stiffness": "N*m/rad (MuJoCo kp)",
            "damping": "N*m*s/rad (MuJoCo kv + joint damping; PhysX has one damping slot)",
            "armature": "kg*m^2",
            "friction": "N*m on Isaac Sim 5.0+ (effort); unitless coefficient on 4.5, do not use the number there",
            "effort_limit_sim": "N*m (MuJoCo forcerange)",
        },
        "articulation": {
            "class": "isaaclab.assets.ArticulationCfg",
            "usd_path": "/path/to/so101_follower.usd",
            "actuators": {ACTUATOR_GROUP: "actuator"},
            "init_state_joint_pos": {j: 0.0 for j in JOINTS},
        },
        "latency": lat,
        "command_map": _command_map_dict(lock),
        "dr": dr_info,
    }
    notes = [
        "Isaac Lab: stiffness/damping/armature/friction/effort_limit_sim on ImplicitActuatorCfg; "
        "damping = kv + joint damping; friction is N*m only on Isaac Sim 5.0+.",
        f"latency: {lat['frames']} frame(s) at {fps_value:g} Hz, apply in the control loop.",
    ]
    if dr_info is None:
        notes.append("no DR ranges (locks/<name>.dr.json missing); EventTerm snippet omitted.")
    return TargetExport(
        "isaaclab",
        {f"{lock.name}_isaaclab.py": py, f"{lock.name}_isaaclab.json": _dumps(data)},
        notes,
    )


def render_maniskill(
    lock: RobotLock, fps: float | None = None, dr: Any | None = None, lock_dir: Path | str | None = None
) -> TargetExport:
    """ManiSkill 3 ``PDJointPosControllerConfig`` (absolute + delta) + JSON twin."""
    fps_value, fps_source, dr, lat, prov, header = _common(lock, fps, dr, lock_dir)
    mp = mapped_params(lock.params)
    pinned = pinned_params(lock)
    dr_info = dr_dict(dr, lock)
    lo, hi = joint_limits()
    cmap = CommandMap.from_dict(lock.command_map)
    per_joint = lock.params.has_per_joint

    def lit(v: float | dict[str, float]) -> str:
        return _py(_as_list(v)) if isinstance(v, dict) else _py(v)

    delta_lo = [-MANISKILL_DELTA_ARM] * (len(JOINTS) - 1) + [-MANISKILL_DELTA_GRIPPER]
    delta_hi = [MANISKILL_DELTA_ARM] * (len(JOINTS) - 1) + [MANISKILL_DELTA_GRIPPER]
    cmap_note = (
        "# NOTE: this lock carries a fitted command map (see the JSON `command_map`):\n"
        "# apply gain*u + bias to logged commands (LeRobot units) before converting to rad.\n\n"
        if not cmap.is_identity()
        else ""
    )
    scalar_note = (
        "one value per joint, in SO101_JOINTS order" if per_joint else "one value shared by all six joints"
    )

    py = f'''"""ManiSkill 3 controller configuration for lock {lock.name} (generated by Antares).

{_header(header, prefix="")}

Verified against ManiSkill 3.0.1: mani_skill.agents.controllers.PDJointPosControllerConfig
(joint_names, lower, upper, stiffness, damping, force_limit, friction, use_delta,
use_target, normalize_action, drive_mode) applies stiffness/damping/force_limit with
joint.set_drive_properties and friction with joint.set_friction; values are a float
or one per joint. SAPIEN/PhysX joints are in radians. ManiSkill's own SO-100 agent
uses stiffness 1e3, damping 1e2, force_limit 100 on all six joints; these are the
values fitted to this arm instead ({scalar_note}).
Drop these into your agent's `_controller_configs`; joint names must match your URDF.
"""

from __future__ import annotations

try:
    from mani_skill.agents.controllers import PDJointPosControllerConfig
except ImportError:  # the numbers below stay importable without ManiSkill
    PDJointPosControllerConfig = None

{cmap_note}SO101_JOINTS = {_py(list(JOINTS))}

#: Joint limits (rad) of the SO-101 MJCF, used as the absolute action range.
JOINT_LOWER = {_py(lo)}
JOINT_UPPER = {_py(hi)}

{_delay_constants(lat)}
# -- fitted actuator parameters (SI, per radian) ----------------------------
STIFFNESS = {lit(mp['stiffness'])}  # N*m/rad, MuJoCo kp
# kv (drive damping) + MuJoCo joint damping: SAPIEN's drive has one damping slot
# and both act as -b*qdot at zero target velocity, so the sum is exact there.
DAMPING = {lit(mp['damping'])}  # N*m*s/rad
FORCE_LIMIT = {lit(mp['effort_limit'])}  # N*m, MuJoCo forcerange
#: MuJoCo frictionloss is a Coulomb torque (N*m). ManiSkill's `friction` is the
#: PhysX joint friction *coefficient* (torque = coefficient * transmitted force),
#: which cannot be derived from a torque without the load, so it stays 0.0 and
#: the fitted torque is kept here for your own friction model.
JOINT_FRICTIONLOSS_NM = {lit(mp['frictionloss'])}
FRICTION = 0.0
#: Reflected rotor inertia (kg*m^2). Not part of the controller config; set it on
#: the SAPIEN joints if your build exposes it (not verified against SAPIEN 3).
JOINT_ARMATURE = {lit(mp['armature'])}

{_dr_python(dr_info, pinned)}
if PDJointPosControllerConfig is not None:
    pd_joint_pos = PDJointPosControllerConfig(
        SO101_JOINTS,
        lower=JOINT_LOWER,
        upper=JOINT_UPPER,
        stiffness=STIFFNESS,
        damping=DAMPING,
        force_limit=FORCE_LIMIT,
        friction=FRICTION,
        normalize_action=False,
    )
    # delta variant with the step limits ManiSkill's SO-100 agent uses
    # (+-{MANISKILL_DELTA_ARM} rad arm, +-{MANISKILL_DELTA_GRIPPER} rad gripper); actions normalized to [-1, 1]
    pd_joint_delta_pos = PDJointPosControllerConfig(
        SO101_JOINTS,
        lower={_py(delta_lo)},
        upper={_py(delta_hi)},
        stiffness=STIFFNESS,
        damping=DAMPING,
        force_limit=FORCE_LIMIT,
        friction=FRICTION,
        use_delta=True,
        use_target=False,
    )
    controller_configs = dict(pd_joint_pos=pd_joint_pos, pd_joint_delta_pos=pd_joint_delta_pos)
'''

    data = {
        "target": "maniskill",
        "verified": "ManiSkill 3.0.1, 2026-10-17",
        "header": list(header),
        "provenance": prov,
        "joints": list(JOINTS),
        "controller": {
            "class": "mani_skill.agents.controllers.PDJointPosControllerConfig",
            "joint_names": list(JOINTS),
            "lower": lo,
            "upper": hi,
            "stiffness": mp["stiffness"],
            "damping": mp["damping"],
            "force_limit": mp["effort_limit"],
            "friction": 0.0,
            "normalize_action": False,
        },
        "delta_controller": {"lower": delta_lo, "upper": delta_hi, "use_delta": True, "normalize_action": True},
        "joint_frictionloss_nm": mp["frictionloss"],
        "joint_armature": mp["armature"],
        "mujoco_params": lock.params.to_dict(),
        "units": {
            "stiffness": "N*m/rad (MuJoCo kp)",
            "damping": "N*m*s/rad (MuJoCo kv + joint damping; one drive damping slot)",
            "force_limit": "N*m (MuJoCo forcerange)",
            "friction": "PhysX coefficient, not derivable from frictionloss (N*m); left 0.0",
            "joint_armature": "kg*m^2, not settable from the controller config",
        },
        "latency": lat,
        "command_map": _command_map_dict(lock),
        "dr": dr_info,
    }
    notes = [
        "ManiSkill: stiffness/damping/force_limit on PDJointPosControllerConfig; damping = kv + joint damping; "
        "frictionloss has no coefficient equivalent, kept as JOINT_FRICTIONLOSS_NM.",
        f"latency: {lat['frames']} frame(s) at {fps_value:g} Hz, apply in the control loop.",
    ]
    return TargetExport(
        "maniskill",
        {f"{lock.name}_maniskill.py": py, f"{lock.name}_maniskill.json": _dumps(data)},
        notes,
    )


def mjlab_delay_steps(lat: dict[str, Any], timestep: float = MJCF_TIMESTEP) -> tuple[int, int]:
    """Latency as mjlab physics steps: ``round(frames / fps / timestep)`` (min, max)."""
    hz = lat["control_hz"]
    return (
        int(round(lat["frames_min"] / hz / timestep)),
        int(round(lat["frames_max"] / hz / timestep)),
    )


def render_mjlab(
    lock: RobotLock, fps: float | None = None, dr: Any | None = None, lock_dir: Path | str | None = None
) -> TargetExport:
    """mjlab ``EntityCfg`` over the exported MJCF with ``XmlActuatorCfg`` + delay."""
    fps_value, fps_source, dr, lat, prov, header = _common(lock, fps, dr, lock_dir)
    mp = mapped_params(lock.params)
    pinned = pinned_params(lock)
    dr_info = dr_dict(dr, lock)
    lo_steps, hi_steps = mjlab_delay_steps(lat)
    cmap = CommandMap.from_dict(lock.command_map)
    cmap_note = (
        "# NOTE: this lock carries a fitted command map (see <name>.antares.json / .params.json):\n"
        "# apply gain*u + bias to logged commands (LeRobot units) before the unit map.\n\n"
        if not cmap.is_identity()
        else ""
    )

    py = f'''"""mjlab entity configuration for lock {lock.name} (generated by Antares).

{_header(header, prefix="")}

Verified against mjlab 1.6.0 (mujoco / mujoco-warp ~=3.11). mjlab consumes MJCF
through mujoco.MjSpec, so the calibrated arm is the exported {lock.name}.xml next
to this file: kp/kv sit on its <position> actuators and damping/armature/
frictionloss/forcerange on its joints. XmlActuatorCfg wraps those actuators
without re-stating gains (ActuatorCfg's armature/frictionloss/viscous_damping
default to None = keep the XML value). Its delay_min_lag/delay_max_lag are in
physics timesteps, so the measured latency is converted with the MJCF timestep.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

try:
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
except ImportError:  # the numbers below stay importable without mjlab
    XmlActuatorCfg = EntityArticulationInfoCfg = EntityCfg = None

{cmap_note}SO101_JOINTS = {_py(list(JOINTS))}
#: The calibrated MJCF from `antares export` (same directory as this file).
SO101_MJCF = Path(__file__).with_name({lock.name + '.xml'!r})
#: <option timestep> of the exported MJCF.
PHYSICS_DT = {MJCF_TIMESTEP!r}

{_delay_constants(lat)}
#: The same delay in physics steps for XmlActuatorCfg (round(frames / CONTROL_HZ / PHYSICS_DT)).
#: For a per-joint lock the min..max over joints becomes mjlab's per-env lag range.
COMMAND_DELAY_PHYSICS_STEPS = ({lo_steps}, {hi_steps})

# For reference (already baked into the XML; SI, per radian):
STIFFNESS = {_py(mp['stiffness'])}  # kp, N*m/rad
DRIVE_DAMPING = {_py(mp['drive_damping_kv'])}  # kv, N*m*s/rad
JOINT_DAMPING = {_py(mp['joint_damping'])}  # <joint damping>, N*m*s/rad
ARMATURE = {_py(mp['armature'])}  # kg*m^2
FRICTIONLOSS = {_py(mp['frictionloss'])}  # N*m
EFFORT_LIMIT = {_py(mp['effort_limit'])}  # forcerange, N*m

{_dr_python(dr_info, pinned)}

def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(SO101_MJCF))


if EntityCfg is not None:
    SO101_ACTUATOR_CFG = XmlActuatorCfg(
        target_names_expr=tuple(SO101_JOINTS),
        delay_min_lag=COMMAND_DELAY_PHYSICS_STEPS[0],
        delay_max_lag=COMMAND_DELAY_PHYSICS_STEPS[1],
    )
    SO101_ARTICULATION = EntityArticulationInfoCfg(
        actuators=(SO101_ACTUATOR_CFG,),
        soft_joint_pos_limit_factor=1.0,
    )
    SO101_ENTITY_CFG = EntityCfg(spec_fn=get_spec, articulation=SO101_ARTICULATION)
    # Alternative: drop the XML actuators and add BuiltinPositionActuatorCfg(
    #     target_names_expr=..., stiffness=STIFFNESS, damping=DRIVE_DAMPING,
    #     effort_limit=EFFORT_LIMIT, armature=ARMATURE, frictionloss=FRICTIONLOSS)
    # per joint when you want mjlab to own the gains (per-joint locks need one cfg per joint).
'''
    notes = [
        f"mjlab: load {lock.name}.xml via mujoco.MjSpec; XmlActuatorCfg keeps the baked gains; "
        f"delay {lo_steps}..{hi_steps} physics steps at dt={MJCF_TIMESTEP}.",
    ]
    return TargetExport("mjlab", {f"{lock.name}_mjlab.py": py}, notes)


def render_json(
    lock: RobotLock, fps: float | None = None, dr: Any | None = None, lock_dir: Path | str | None = None
) -> TargetExport:
    """Simulator-neutral ``<name>.params.json``: the sidecar plus target numbers."""
    fps_value, fps_source, dr, lat, prov, header = _common(lock, fps, dr, lock_dir)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    side = sidecar_dict(lock, model, f"{lock.name}.xml", "assets")
    mp = mapped_params(lock.params)
    lo_steps, hi_steps = mjlab_delay_steps(lat)
    data = {
        **side,
        "command_map": _command_map_dict(lock),
        "header": list(header),
        "provenance": prov,
        "latency": lat,
        "pinned_params": pinned_params(lock),
        "param_spread": param_spread(lock),
        "mapped": {
            **mp,
            "units": {
                "stiffness": "N*m/rad",
                "damping": "N*m*s/rad (kv + joint damping)",
                "armature": "kg*m^2",
                "frictionloss": "N*m",
                "effort_limit": "N*m",
            },
        },
        "targets": {
            "isaaclab": {
                "stiffness": mp["stiffness"],
                "damping": mp["damping"],
                "armature": mp["armature"],
                "friction": mp["frictionloss"],
                "effort_limit_sim": mp["effort_limit"],
            },
            "maniskill": {
                "stiffness": mp["stiffness"],
                "damping": mp["damping"],
                "force_limit": mp["effort_limit"],
                "friction": 0.0,
                "joint_frictionloss_nm": mp["frictionloss"],
            },
            "mjlab": {
                "mjcf": f"{lock.name}.xml",
                "delay_physics_steps": [lo_steps, hi_steps],
                "physics_dt": MJCF_TIMESTEP,
            },
        },
        "dr": dr_dict(dr, lock),
    }
    return TargetExport("json", {f"{lock.name}.params.json": _dumps(data)}, ["neutral JSON with every target's numbers"])


def _dumps(d: Mapping[str, Any]) -> str:
    return json.dumps(d, indent=2, allow_nan=False) + "\n"


RENDERERS = {
    "isaaclab": render_isaaclab,
    "maniskill": render_maniskill,
    "mjlab": render_mjlab,
    "json": render_json,
}


def unsupported_reason(lock: RobotLock) -> str | None:
    """Why the framework configs cannot be written for ``lock``, or None.

    They name the SO-101's six joints, limits and USD asset, so they are
    only written for locks on the SO-101 model; other robots export as MJCF.
    """
    from antares.robots import get_robot

    robot = get_robot(getattr(lock, "robot", None))
    if Path(robot.model_path).resolve() == Path(MODEL_PATH).resolve() and tuple(robot.joints) == tuple(JOINTS):
        return None
    return (
        f"Isaac Lab, ManiSkill, mjlab and JSON configs are written for the SO-101 model only; "
        f"lock {lock.name} is on {robot.name}, export it as MJCF (--format mjcf)"
    )


def render(
    lock: RobotLock,
    target: str,
    fps: float | None = None,
    dr: Any | None = None,
    lock_dir: Path | str | None = None,
) -> TargetExport:
    """Render one target (``TARGETS``) for ``lock``.

    ``fps`` overrides the dataset rate (else the local HF cache, else
    :data:`DEFAULT_FPS`); ``dr`` overrides the ranges (else
    ``<lock_dir>/<name>.dr.json`` when it exists).
    """
    try:
        fn = RENDERERS[target]
    except KeyError:
        raise TargetExportError(f"unknown export target {target!r}; known: {', '.join(TARGETS)}") from None
    reason = unsupported_reason(lock)
    if reason is not None:
        raise TargetExportError(reason)
    return fn(lock, fps=fps, dr=dr, lock_dir=lock_dir)


def resolve_targets(fmt: str) -> list[str]:
    """``--format`` value -> non-MJCF targets to render (``mjcf`` -> none)."""
    if fmt == "all":
        return list(TARGETS)
    if fmt == "mjcf":
        return []
    if fmt in TARGETS:
        return [fmt]
    raise TargetExportError(f"unknown export format {fmt!r}; known: {', '.join(FORMATS)}")


def export_targets(
    lock: RobotLock,
    out_dir: Path,
    targets: Sequence[str] = TARGETS,
    fps: float | None = None,
    dr: Any | None = None,
    lock_dir: Path | str | None = None,
) -> list[Path]:
    """Write the given targets into ``out_dir``; returns the files written."""
    paths: list[Path] = []
    for t in targets:
        paths += render(lock, t, fps=fps, dr=dr, lock_dir=lock_dir).write(out_dir)
    return paths


def media_type(filename: str) -> str:
    return MEDIA_TYPES.get(Path(filename).suffix.lower(), "text/plain")


__all__ = [
    "DEFAULT_FPS",
    "DR_PARAMS",
    "FORMATS",
    "MJCF_TIMESTEP",
    "TARGETS",
    "TARGET_LABELS",
    "TargetExport",
    "TargetExportError",
    "dataset_fps",
    "dr_dict",
    "drive_damping",
    "export_targets",
    "joint_limits",
    "latency_info",
    "load_dr_ranges",
    "mapped_params",
    "media_type",
    "mjlab_delay_steps",
    "provenance_lines",
    "render",
    "render_isaaclab",
    "render_json",
    "render_maniskill",
    "render_mjlab",
    "resolve_targets",
    "unsupported_reason",
]
