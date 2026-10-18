"""Read the follower arm's LeRobot motor calibration and turn it into a unit map.

LeRobot calibrates every arm once and keeps the result per motor: the
homing offset written into the servo and the encoder range the joint was
moved through. The dataset's normalized values are defined by that file,
so reading it gives the exact tick-to-angle map of *this* arm instead of
the one :meth:`antares.sim.UnitMap.from_model` infers from the MJCF joint
ranges.

Where the file lives (LeRobot 0.4.0 through 0.6.1, the current PyPI release
of 2026-08-03, and ``main`` as of 2026-10-17)::

    $HF_LEROBOT_CALIBRATION/robots/<robot.name>/<robot.id>.json
    $HF_LEROBOT_CALIBRATION/teleoperators/<teleop.name>/<teleop.id>.json

``HF_LEROBOT_CALIBRATION`` defaults to ``$HF_LEROBOT_HOME/calibration``,
which defaults to ``$HF_HOME/lerobot/calibration`` (``HF_HOME`` is the
huggingface_hub cache, ``~/.cache/huggingface`` unless set); see
``src/lerobot/utils/constants.py``. ``<robot.name>`` is the robot *class*
name, not the ``--robot.type`` on the command line: LeRobot 0.4.x had one
class per arm (``so101_follower``, ``so100_follower``); 0.5.0 merged them
into ``SOFollower`` with ``name = "so_follower"``, while ``--robot.type``
still accepts ``so101_follower`` and ``so100_follower``
(``src/lerobot/robots/so_follower/so_follower.py``,
``config_so_follower.py``). :func:`find_calibration` therefore looks in
both spellings. The path is built in ``src/lerobot/robots/robot.py``::

    self.calibration_dir = config.calibration_dir or HF_LEROBOT_CALIBRATION / ROBOTS / self.name
    self.calibration_fpath = self.calibration_dir / f"{self.id}.json"

File schema (``draccus`` dump of ``dict[str, MotorCalibration]``,
``src/lerobot/motors/motors_bus.py``)::

    {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1023,
                      "range_min": 741, "range_max": 3352}, ...}

How LeRobot turns ticks into the dataset's numbers (``MotorsBus._normalize``
and ``_unnormalize`` in ``motors_bus.py``; ``present`` is the position the
servo reports after its ``Homing_Offset`` register is applied, clipped to
``[range_min, range_max]`` on the way in and to the normalized bounds on
the way out)::

    RANGE_M100_100 (body joints, LeRobot <= 0.4.x default):
        n = ((present - range_min) / (range_max - range_min)) * 200 - 100
        present = ((n + 100) / 200) * (range_max - range_min) + range_min
    RANGE_0_100 (gripper, always):
        n = ((present - range_min) / (range_max - range_min)) * 100
    DEGREES (body joints, LeRobot >= 0.5.0 default, ``use_degrees=True``):
        n = (present - (range_min + range_max) / 2) * 360 / (resolution - 1)

With ``drive_mode == 1`` the sign is flipped (``-n``, or ``100 - n`` for the
gripper); the SO follower writes ``drive_mode=0`` for every motor. The
STS3215 counts ``resolution = 4096`` ticks per turn, so one tick is
``2*pi/4096`` rad of the joint (the SO-101 joints sit directly on the servo
horns). The homing offset is set at calibration time so that the pose the
user holds (LeRobot's "middle of the range of motion") reads a half turn,
``present = (resolution - 1) // 2 = 2047``::

    offset = sign * ((resolution - 1) // 2 - raw)   # motors_bus._get_half_turn_homings
    present = raw + sign * offset                    # docstring of the same method

``range_min``/``range_max`` are recorded afterwards, on ``present`` values,
so the homing offset is already folded into them and does not enter the
normalized <-> tick formula; it is kept in the lock for the record.

The formula this module implements, for a body joint with ``k = 2*pi/4096``
and ``present(n)`` the inverse normalization above::

    rad(n) = anchor + sign * k * (present(n) - 2047)      sign = -1 if drive_mode else +1

which is affine in ``n``: ``scale = k * (range_max - range_min) / 200`` and
``offset = anchor + sign * k * ((range_min + range_max) / 2 - 2047)``. The
one thing the file cannot say is where the servo's half turn sits in the
MJCF's joint coordinates; ``anchor`` is that assumption. The default,
``"homing"``, takes the calibration pose (half turn) to be the MJCF
mid-range ``(lo + hi) / 2``: LeRobot asks for the arm at the middle of its
range and the Menagerie SO-101 ranges are symmetric about the straight-up
pose. ``"range"`` instead centres the recorded range on the MJCF mid-range
(LeRobot's own DEGREES convention), which ignores where the user held the
arm and differs from ``"homing"`` by the asymmetry of the recorded range
about 2047. The gripper is anchored at its closed stop in both cases:
``n = 0`` (``range_min``) is the MJCF ``lo``, with ``scale = k * (range_max -
range_min) / 100``.

Compared with :meth:`UnitMap.from_model`, which spreads -100..100 over the
MJCF range ``[lo, hi]`` (``scale = (hi - lo) / 200``, ``offset = (lo + hi) /
2``), the calibration map agrees exactly when the recorded range spans
``(hi - lo) * 4096 / (2*pi)`` ticks centred on 2047. A narrower or shifted
range is a real difference in the physical angle the dataset's numbers
mean, which is what :func:`compare_unit_maps` and :func:`calibration_check`
report. The replay residual in dataset units is almost blind to it (the
follower closes its loop in its own ticks, so command and state share the
map); what changes is the physical pose the sim is driven to and reads
back, which gravity, forward kinematics, the viewer, camera fits and
exports all depend on.

Only the FOLLOWER calibration matters for replay. A teleoperation dataset's
``action`` is the leader's normalized reading, but the follower
un-normalizes it with its *own* calibration before writing
``Goal_Position``, and ``observation.state`` is the follower's reading in
that same calibration; the leader's file never enters what the arm did.

``meta/info.json`` does not embed the calibration (``DatasetInfo`` in
``src/lerobot/datasets/utils.py`` has ``robot_type`` and nothing else about
the robot), so there is nothing to load from the dataset itself.

Sources (read 2026-10-17):

* https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py
* https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/robot.py
* https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py
* https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py
* https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/robots/so101_follower/so101_follower.py
* https://github.com/huggingface/lerobot/blob/main/src/lerobot/utils/constants.py
* https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/utils.py
* https://huggingface.co/docs/lerobot/so101 (calibration procedure)

Pure numpy; MuJoCo enters only through the compiled model passed in.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from antares.sim import GRIPPER_INDEX, JOINTS, CommandMap, UnitMap

#: Encoder steps per turn of the STS3215 (12-bit).
RESOLUTION = 4096
#: The ``present`` tick LeRobot's half-turn homing assigns to the calibration pose.
HALF_TURN = (RESOLUTION - 1) // 2
#: Radians per encoder tick.
RAD_PER_TICK = 2.0 * math.pi / RESOLUTION

NORM_RANGE = "range_m100_100"
NORM_DEGREES = "degrees"
NORM_MODES = (NORM_RANGE, NORM_DEGREES)

ANCHOR_HOMING = "homing"
ANCHOR_RANGE = "range"
ANCHORS = (ANCHOR_HOMING, ANCHOR_RANGE)

DEFAULT_ROBOT_TYPE = "so101_follower"
#: Disagreement between the model's and the arm's map worth a warning.
FLAG_THRESHOLD_DEG = 2.0

#: Calibration directories LeRobot has used for the SO arms, by ``--robot.type``.
_TYPE_DIRS: dict[str, tuple[str, ...]] = {
    "so101_follower": ("so101_follower", "so_follower"),
    "so100_follower": ("so100_follower", "so_follower"),
    "so_follower": ("so_follower", "so101_follower", "so100_follower"),
}

_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


@dataclass(frozen=True)
class MotorCalibration:
    """One motor's entry of a LeRobot calibration file (same fields as LeRobot's)."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    def __post_init__(self) -> None:
        if self.drive_mode not in (0, 1):
            raise ValueError(f"drive_mode must be 0 or 1, got {self.drive_mode}")
        if self.range_max <= self.range_min:
            raise ValueError(f"range_max ({self.range_max}) must exceed range_min ({self.range_min})")

    @property
    def span(self) -> int:
        return self.range_max - self.range_min

    @property
    def mid(self) -> float:
        return (self.range_min + self.range_max) / 2.0

    @property
    def sign(self) -> int:
        return -1 if self.drive_mode else 1

    @property
    def homing_pose_units(self) -> float:
        """Where the calibration pose (half turn) lands in RANGE_M100_100 units.

        0 for a range centred on 2047; the asymmetry of the recorded range
        about the pose the user held, in dataset units.
        """
        n = ((HALF_TURN - self.range_min) / self.span) * 200.0 - 100.0
        return -n if self.drive_mode else n

    def to_dict(self) -> dict[str, int]:
        return {f: int(getattr(self, f)) for f in _FIELDS}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], motor: str = "") -> MotorCalibration:
        missing = [f for f in _FIELDS if f not in d]
        if missing:
            raise ValueError(f"motor {motor!r}: missing {', '.join(missing)}")
        vals = {}
        for f in _FIELDS:
            v = d[f]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
                raise ValueError(f"motor {motor!r}: {f} must be an integer, got {v!r}")
            vals[f] = int(v)
        return cls(**vals)


Calibration = dict[str, MotorCalibration]


# --------------------------------------------------------------------------
# File discovery and parsing
# --------------------------------------------------------------------------


def calibration_root(root: str | Path | None = None) -> Path:
    """LeRobot's calibration directory: ``root`` if given, else the env vars
    LeRobot reads (``HF_LEROBOT_CALIBRATION``, ``HF_LEROBOT_HOME``,
    ``HF_HOME``), else ``~/.cache/huggingface/lerobot/calibration``."""
    if root is not None:
        return Path(root).expanduser()
    env = os.environ.get("HF_LEROBOT_CALIBRATION")
    if env:
        return Path(env).expanduser()
    home = os.environ.get("HF_LEROBOT_HOME")
    if home:
        return Path(home).expanduser() / "calibration"
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(hf_home).expanduser() / "lerobot" / "calibration"


def calibration_dirs(robot_type: str = DEFAULT_ROBOT_TYPE, root: str | Path | None = None) -> list[Path]:
    """Directories a follower of ``robot_type`` may have saved its file in,
    most likely first (the type's own name, then LeRobot's other spelling)."""
    rt = (robot_type or DEFAULT_ROBOT_TYPE).strip().lower()
    names = _TYPE_DIRS.get(rt, (rt,))
    base = calibration_root(root) / "robots"
    return [base / n for n in names]


def find_calibration(
    robot_id: str, robot_type: str = DEFAULT_ROBOT_TYPE, root: str | Path | None = None
) -> Path | None:
    """Path of ``<robot_id>.json`` for a follower of ``robot_type``, or None.

    ``root`` overrides the calibration directory (default: see
    :func:`calibration_root`). Both the 0.4.x (``so101_follower``) and the
    0.5+ (``so_follower``) directory names are tried.
    """
    rid = str(robot_id).strip()
    if not rid or "/" in rid or os.sep in rid:
        raise ValueError(f"robot_id must be a bare name, got {robot_id!r}")
    for d in calibration_dirs(robot_type, root):
        p = d / f"{rid}.json"
        if p.is_file():
            return p
    return None


def parse_calibration(data: Any) -> Calibration:
    """``{"<motor>": {"id", "drive_mode", "homing_offset", "range_min", "range_max"}}``
    -> :class:`MotorCalibration` per motor. Unknown motors are kept; the
    joints a map needs are checked in :func:`unit_map_from_calibration`."""
    if not isinstance(data, Mapping) or not data:
        raise ValueError("calibration must be a non-empty JSON object keyed by motor name")
    out: Calibration = {}
    for motor, entry in data.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"motor {motor!r}: expected an object, got {type(entry).__name__}")
        out[str(motor)] = MotorCalibration.from_dict(entry, motor=str(motor))
    return out


def load_calibration(path: str | Path) -> Calibration:
    """Read and parse a LeRobot calibration file."""
    try:
        data = json.loads(Path(path).read_text())
    except ValueError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from None
    return parse_calibration(data)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_to_dict(calib: Calibration) -> dict[str, dict[str, int]]:
    return {m: c.to_dict() for m, c in calib.items()}


def _check_mode(norm_mode: str, anchor: str) -> None:
    if norm_mode not in NORM_MODES:
        raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {norm_mode!r}")
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")


def _motors_for(calib: Calibration, joint_names: Sequence[str]) -> list[MotorCalibration]:
    missing = [j for j in joint_names if j not in calib]
    if missing:
        raise ValueError(f"calibration has no motor named {', '.join(missing)} (has {', '.join(calib)})")
    return [calib[j] for j in joint_names]


# --------------------------------------------------------------------------
# Unit map
# --------------------------------------------------------------------------


def unit_map_from_calibration(
    calib: Calibration,
    model: Any,
    joint_names: Sequence[str] = JOINTS,
    norm_mode: str = NORM_RANGE,
    anchor: str = ANCHOR_HOMING,
) -> UnitMap:
    """Dataset units -> physical radians of *this* arm, in the MJCF's joint coordinates.

    ``model`` is the compiled MuJoCo model whose joints are in canonical
    order; only its ``jnt_range`` is read (the anchor, and the gripper's
    closed stop). ``joint_names`` are the calibration's motor names in that
    order. ``norm_mode`` says how the dataset's body joints were normalized
    (``"range_m100_100"``, the default and LeRobot's until 0.4.x, or
    ``"degrees"``, the default since 0.5.0); the gripper is always 0..100.
    See the module docstring for the formula and the ``anchor`` choice.
    The map's ``lo``/``hi`` are the recorded range of motion in radians.
    """
    _check_mode(norm_mode, anchor)
    motors = _motors_for(calib, joint_names)
    n = len(joint_names)
    jr = np.asarray(model.jnt_range, dtype=np.float64)
    if jr.shape[0] < n:
        raise ValueError(f"model has {jr.shape[0]} joints, need {n}")
    lo_m, hi_m = jr[:n, 0], jr[:n, 1]
    scale = np.empty(n)
    offset = np.empty(n)
    lo = np.empty(n)
    hi = np.empty(n)
    g = joint_names.index("gripper") if "gripper" in joint_names else GRIPPER_INDEX
    for i, m in enumerate(motors):
        if i == g:
            scale[i] = RAD_PER_TICK * m.span / 100.0
            offset[i] = lo_m[i]
            lo[i], hi[i] = offset[i], offset[i] + scale[i] * 100.0
            continue
        mid_model = (lo_m[i] + hi_m[i]) / 2.0
        if norm_mode == NORM_RANGE:
            scale[i] = RAD_PER_TICK * m.span / 200.0
            ends = (-100.0, 100.0)
        else:
            scale[i] = RAD_PER_TICK * (RESOLUTION - 1) / 360.0
            half = (m.span / 2.0) * 360.0 / (RESOLUTION - 1)
            ends = (-half, half)
        if anchor == ANCHOR_HOMING:
            offset[i] = mid_model + m.sign * RAD_PER_TICK * (m.mid - HALF_TURN)
        else:
            offset[i] = mid_model
        a, b = scale[i] * ends[0] + offset[i], scale[i] * ends[1] + offset[i]
        lo[i], hi[i] = min(a, b), max(a, b)
    return UnitMap(scale=scale, offset=offset, lo=lo, hi=hi)


def _samples_for(joint: int, n_joints: int, samples: int | Sequence[float] | None) -> np.ndarray:
    if samples is not None and not isinstance(samples, int):
        return np.asarray(samples, dtype=np.float64)
    count = int(samples) if isinstance(samples, int) else 201
    if joint == GRIPPER_INDEX and n_joints > GRIPPER_INDEX:
        return np.linspace(0.0, 100.0, count)
    return np.linspace(-100.0, 100.0, count)


def compare_unit_maps(
    a: UnitMap,
    b: UnitMap,
    samples: int | Sequence[float] | None = None,
    joint_names: Sequence[str] = JOINTS,
) -> dict[str, dict[str, Any]]:
    """Per-joint disagreement between two unit maps, in degrees.

    The affine maps are evaluated unclipped at ``samples`` dataset units
    (default: 201 points over -100..100, 0..100 for the gripper; an int
    changes the count, a sequence gives the points). Returns, per joint,
    ``max_deg``, ``mean_deg`` and ``at``: the signed difference ``a - b`` at
    -100/0/+100 (0/50/100 for the gripper).
    """
    if a.scale.shape != b.scale.shape:
        raise ValueError("unit maps have different shapes")
    n = a.scale.shape[0]
    out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(joint_names[:n]):
        u = _samples_for(i, n, samples)
        d = np.degrees((a.scale[i] - b.scale[i]) * u + (a.offset[i] - b.offset[i]))
        marks = (0.0, 50.0, 100.0) if i == GRIPPER_INDEX and n > GRIPPER_INDEX else (-100.0, 0.0, 100.0)
        at = {
            f"{m:g}": float(np.degrees((a.scale[i] - b.scale[i]) * m + (a.offset[i] - b.offset[i])))
            for m in marks
        }
        out[name] = {"max_deg": float(np.max(np.abs(d))), "mean_deg": float(np.mean(np.abs(d))), "at": at}
    return out


def command_map_prior(calib: Calibration, model: Any, joint_names: Sequence[str] = JOINTS) -> CommandMap:
    """The sensor fit's starting :class:`~antares.sim.CommandMap` under the
    calibration-derived unit map: identity.

    A :class:`CommandMap` corrects the *commands* in dataset units before the
    unit map, which is the right place for a disagreement between what the
    commands and the measured state mean. The follower's calibration leaves
    none to express: it un-normalizes every command and normalizes every
    reading with the same ``range_min``/``range_max``, so a command of ``n``
    units drives the arm to the very tick a reading of ``n`` reports. What
    the calibration changes is how both sides turn into radians, and that
    lives in the unit map. Whatever steady offset the sensor fit still finds
    afterwards (servo load error, dead band) is not in the file, so the fit
    starts from identity and measures only that.
    """
    _motors_for(calib, joint_names)
    return CommandMap.identity(len(joint_names))


# --------------------------------------------------------------------------
# Diagnostic and lock record
# --------------------------------------------------------------------------

FLAG_MESSAGE = "the model's assumed range does not match this arm's calibration; use --calibration when calibrating"


def calibration_check(
    calib: Calibration,
    model: Any,
    joint_names: Sequence[str] = JOINTS,
    norm_mode: str = NORM_RANGE,
    anchor: str = ANCHOR_HOMING,
    threshold_deg: float = FLAG_THRESHOLD_DEG,
) -> dict[str, Any]:
    """Model-range map vs calibration map, joint by joint, for ``antares inspect``
    and ``GET /api/datasets/calibration-check``.

    Per joint: the calibration entry, the physical span of the recorded
    range and of the MJCF range in degrees, where the calibration pose sits
    in units, the signed disagreement at -100/0/+100 and its max, and
    ``flagged`` when the max exceeds ``threshold_deg``.
    """
    calib_map = unit_map_from_calibration(calib, model, joint_names, norm_mode=norm_mode, anchor=anchor)
    model_map = UnitMap.from_model(model)
    cmp = compare_unit_maps(calib_map, model_map, joint_names=joint_names)
    jr = np.asarray(model.jnt_range, dtype=np.float64)
    rows = []
    flagged = []
    for i, name in enumerate(joint_names):
        m = calib[name]
        c = cmp[name]
        over = c["max_deg"] > threshold_deg
        if over:
            flagged.append(name)
        rows.append(
            {
                "joint": name,
                **m.to_dict(),
                "calibration_span_deg": float(m.span * 360.0 / RESOLUTION),
                "model_span_deg": float(np.degrees(jr[i, 1] - jr[i, 0])),
                "homing_pose_units": None if i == GRIPPER_INDEX else float(m.homing_pose_units),
                "at": c["at"],
                "max_deg": c["max_deg"],
                "mean_deg": c["mean_deg"],
                "flagged": bool(over),
            }
        )
    return {
        "norm_mode": norm_mode,
        "anchor": anchor,
        "threshold_deg": float(threshold_deg),
        "joints": rows,
        "flagged": flagged,
        "message": FLAG_MESSAGE if flagged else None,
        "unit_map": calib_map.to_dict(),
    }


def calibration_record(
    calib: Calibration,
    model: Any,
    *,
    path: str | Path | None = None,
    robot_id: str | None = None,
    robot_type: str = DEFAULT_ROBOT_TYPE,
    joint_names: Sequence[str] = JOINTS,
    norm_mode: str = NORM_RANGE,
    anchor: str = ANCHOR_HOMING,
) -> dict[str, Any]:
    """The ``calibration`` block of a lock: enough to rebuild the unit map
    without the file (per-joint ranges and offsets, the map itself, the
    mode and anchor) plus where it came from and its digest."""
    um = unit_map_from_calibration(calib, model, joint_names, norm_mode=norm_mode, anchor=anchor)
    cmp = compare_unit_maps(um, UnitMap.from_model(model), joint_names=joint_names)
    return {
        "source": None if path is None else str(path),
        "robot_id": robot_id,
        "robot_type": robot_type,
        "sha256": None if path is None else file_sha256(path),
        "norm_mode": norm_mode,
        "anchor": anchor,
        "motors": {j: calib[j].to_dict() for j in joint_names},
        "unit_map": um.to_dict(),
        "unit_map_disagreement_deg": {j: round(cmp[j]["max_deg"], 4) for j in joint_names},
    }


def unit_map_from_record(record: Mapping[str, Any] | None, model: Any = None) -> UnitMap | None:
    """Rebuild the unit map a lock's ``calibration`` block was fitted with.

    The stored ``unit_map`` is what the fit used and is preferred; without
    it the map is rebuilt from the per-joint ``motors`` and ``model``. None
    for a lock without a calibration block.
    """
    if not record:
        return None
    um = record.get("unit_map")
    if um is not None:
        return UnitMap.from_dict(um)
    motors = record.get("motors")
    if motors is None or model is None:
        return None
    calib = parse_calibration(motors)
    return unit_map_from_calibration(
        calib, model, tuple(motors), norm_mode=record.get("norm_mode", NORM_RANGE),
        anchor=record.get("anchor", ANCHOR_HOMING),
    )


__all__ = [
    "ANCHORS",
    "ANCHOR_HOMING",
    "ANCHOR_RANGE",
    "DEFAULT_ROBOT_TYPE",
    "FLAG_MESSAGE",
    "FLAG_THRESHOLD_DEG",
    "HALF_TURN",
    "NORM_DEGREES",
    "NORM_MODES",
    "NORM_RANGE",
    "RAD_PER_TICK",
    "RESOLUTION",
    "Calibration",
    "MotorCalibration",
    "calibration_check",
    "calibration_dirs",
    "calibration_record",
    "calibration_root",
    "calibration_to_dict",
    "command_map_prior",
    "compare_unit_maps",
    "file_sha256",
    "find_calibration",
    "load_calibration",
    "parse_calibration",
    "unit_map_from_calibration",
    "unit_map_from_record",
]
