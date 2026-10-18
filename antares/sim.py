"""MuJoCo side: load the SO-101 arm, map dataset units to joint space, replay.

The MJCF is MuJoCo Menagerie's ``robotstudio_so101``. Its six hinge joints
are named exactly like the dataset's canonical joints and appear in the
same order, so a dataset row maps onto ``qpos`` index for index.

Dataset values are LeRobot's normalized units (body joints in [-100, 100],
gripper in [0, 100]); :class:`UnitMap` turns them into radians. The real
initial state seeds the sim, the recorded commands drive the position
actuators, and :meth:`Arm.replay` returns the simulated joint trajectory
at the dataset's frame rate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from antares.data import Episode

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
N_JOINTS = len(JOINTS)
GRIPPER_INDEX = JOINTS.index("gripper")

MODEL_PATH: Path = Path(__file__).resolve().parent / "assets" / "so101" / "so101.xml"

#: |gravity| in the stock MJCF (m/s^2); ``gravity_scale`` multiplies it.
STANDARD_GRAVITY = 9.81

# Physics step the MJCF was tuned for; replay picks a substep count so the
# effective timestep stays close to this.
BASE_TIMESTEP = 0.005

#: A parameter is either one value shared by all joints or one value per joint.
Scalar = float | int
PerJoint = tuple[float, ...] | tuple[int, ...]
ParamValue = Scalar | PerJoint

#: Fields that are integers (dataset frames) rather than physical quantities.
INT_FIELDS = frozenset({"latency_frames"})


def _coerce(name: str, value: Any) -> ParamValue:
    """Normalize a field value to ``float``/``int`` or a tuple of them (one per joint).

    The tuple's length is checked against the robot when the parameters are
    applied (:meth:`Arm.apply`), not here: a ``SimParams`` does not know
    which arm it is for.
    """
    as_int = name in INT_FIELDS
    if isinstance(value, (list, tuple, np.ndarray)) and np.ndim(value) > 0:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 1:
            raise ValueError(f"{name}: per-joint value needs at least one entry")
        if as_int:
            return tuple(int(round(float(v))) for v in arr)
        return tuple(float(v) for v in arr)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name}: expected a number, got bool")
    return int(round(float(value))) if as_int else float(value)


def _extra_items(extra: Any) -> tuple[tuple[str, ParamValue], ...]:
    items = extra.items() if isinstance(extra, Mapping) else extra
    out = []
    for k, v in items:
        out.append((str(k), _coerce(str(k), v)))
    return tuple(sorted(out))


@dataclass(frozen=True)
class SimParams:
    """Calibratable simulator parameters. Defaults are the SO-101 MJCF values.

    Every field is either a scalar shared by all joints or a tuple of
    per-joint values (one per joint of the robot, in its canonical order).
    Scalars are how the original locks were written and stay the default;
    :meth:`per_joint` broadcasts either form to an ``(n,)`` array when
    writing into the model.

    ``extra`` holds values of factors added to :mod:`antares.factors` (a
    config or plugin factor such as a gravity scale); they read like fields
    (``params.gravity_scale``), serialize next to them and are applied by
    the factor's own hook. Built-in parameters never go there.
    """

    kp: ParamValue = 998.22
    kv: ParamValue = 2.731
    damping: ParamValue = 0.6
    armature: ParamValue = 0.028
    frictionloss: ParamValue = 0.052
    latency_frames: ParamValue = 0  # command delay in dataset frames (>= 0)
    forcerange: ParamValue = 2.94  # actuator force limit, symmetric (N*m)
    #: Gravity load the servos feel, as a fraction of the MJCF's link weights.
    #: The position servo's static error under load is about ``gravity_scale *
    #: g(q) / kp``, so this sets how much the arm sags in a given pose
    #: independently of how fast ``kp`` makes it track. 1.0 is the stock model.
    #: Shared by all joints (it scales the gravity vector).
    gravity_scale: ParamValue = 1.0
    #: Position-error deadband of the servo (rad, >= 0): within it the servo
    #: pushes no torque, past it the gain acts on the excess. Lumps the
    #: STS3215's firmware deadzone and its gearbox backlash, which sit inside
    #: the loop because the encoder reads the output shaft. A held arm then
    #: rests at the edge of the band on the side gravity pulls it to, and a
    #: moving one trails by it, so the error flips sign with the motion. 0 is
    #: the stock smooth actuator.
    deadband: ParamValue = 0.0
    extra: tuple[tuple[str, ParamValue], ...] = ()

    def __post_init__(self) -> None:
        for n in self.field_names():
            object.__setattr__(self, n, _coerce(n, getattr(self, n)))
        object.__setattr__(self, "extra", _extra_items(self.extra))
        for k, _ in self.extra:
            if k in self.field_names() or k == "extra":
                raise ValueError(f"{k!r} is a built-in parameter, not an extra factor")
        if np.any(self.per_joint("latency_frames") < 0):
            raise ValueError("latency_frames must be >= 0")
        g = self.per_joint("gravity_scale")
        if np.any(g != g[0]):
            raise ValueError("gravity_scale is shared by all joints; per-joint values must be equal")
        if g[0] < 0:
            raise ValueError("gravity_scale must be >= 0")
        if np.any(self.per_joint("deadband") < 0):
            raise ValueError("deadband must be >= 0")

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails: extra factor values read like fields.
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            extra = object.__getattribute__(self, "extra")
        except AttributeError:
            raise AttributeError(name) from None
        for k, v in extra:
            if k == name:
                return v
        raise AttributeError(f"SimParams has no parameter {name!r}")

    @classmethod
    def default(cls) -> SimParams:
        return cls()

    @classmethod
    def from_model(cls, model: mujoco.MjModel, latency_frames: int = 0) -> SimParams:
        """Read the actuator/joint values back out of a compiled model.

        Uses the first actuator and dof (the SO-101 shares one ``sts3215``
        class across all six). The model cannot carry command latency, so
        it is passed in (default 0). ``gravity_scale`` is the model's gravity
        relative to :data:`STANDARD_GRAVITY`.
        """
        return cls(
            gravity_scale=float(np.linalg.norm(model.opt.gravity)) / STANDARD_GRAVITY,
            kp=float(model.actuator_gainprm[0, 0]),
            kv=float(-model.actuator_biasprm[0, 2]),
            damping=float(model.dof_damping[0]),
            armature=float(model.dof_armature[0]),
            frictionloss=float(model.dof_frictionloss[0]),
            latency_frames=latency_frames,
        )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """The built-in parameters (not ``extra``)."""
        return tuple(f.name for f in fields(cls) if f.name != "extra")

    def names(self) -> tuple[str, ...]:
        """Built-in parameters, then this instance's extra factors."""
        return self.field_names() + tuple(k for k, _ in self.extra)

    @property
    def extras(self) -> dict[str, ParamValue]:
        return dict(self.extra)

    # -- shape helpers ------------------------------------------------------

    def is_per_joint(self, name: str) -> bool:
        return isinstance(getattr(self, name), tuple)

    @property
    def has_per_joint(self) -> bool:
        """True if any field carries per-joint values."""
        return any(self.is_per_joint(n) for n in self.names())

    def per_joint(self, name: str, n: int | None = None) -> np.ndarray:
        """Value of ``name`` broadcast to an ``(n,)`` array (int for frame counts).

        ``n`` defaults to the tuple's own length, or :data:`N_JOINTS` for a
        scalar; a tuple of another length than ``n`` is an error.
        """
        v = getattr(self, name)
        dtype = np.int64 if name in INT_FIELDS else np.float64
        if isinstance(v, tuple):
            if n is not None and len(v) != n:
                raise ValueError(f"{name}: per-joint value has {len(v)} entries, the robot has {n} joints")
            return np.asarray(v, dtype=dtype)
        return np.full(N_JOINTS if n is None else n, v, dtype=dtype)

    def broadcast(self, n: int | None = None) -> SimParams:
        """Copy with every field in per-joint form; replays identically."""
        core = {k: tuple(self.per_joint(k, n).tolist()) for k in self.field_names()}
        extra = {k: tuple(self.per_joint(k, n).tolist()) for k, _ in self.extra}
        return SimParams(**core, extra=extra)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Scalars stay scalars; per-joint values become lists. Extras follow
        the built-ins under their own names.

        ``gravity_scale`` and ``deadband`` are left out at their scalar
        defaults (1.0 and 0), so locks fitted without them serialize exactly
        as before; in per-joint form they are always written, so
        :meth:`broadcast` round-trips through :meth:`from_dict`.
        """
        out: dict[str, Any] = {}
        for n in self.names():
            v = getattr(self, n)
            if isinstance(v, tuple):
                out[n] = list(v)
            elif (n == "gravity_scale" and v == 1.0) or (n == "deadband" and v == 0.0):
                continue
            else:
                out[n] = v
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimParams:
        """Build from a dict; missing keys default.

        Scalars (possibly as strings) and per-joint sequences are accepted for
        every field, so old locks (all scalars, no ``forcerange``) still load.
        A key that names a factor registered in :mod:`antares.factors` becomes
        an extra; any other unknown key is ignored.
        """
        from antares import factors

        known = set(cls.field_names())
        core = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known and k in factors.REGISTRY}
        return cls(**core, extra=extra)

    def replace(self, **changes: Any) -> SimParams:
        known = set(self.field_names())
        core = {k: v for k, v in changes.items() if k in known}
        rest = {k: v for k, v in changes.items() if k not in known}
        if "extra" in rest:
            raise ValueError("pass extra factors by name, not as extra=")
        if rest:
            merged = {**dict(self.extra), **rest}
            core["extra"] = merged
        return replace(self, **core)


@dataclass(frozen=True, eq=False)
class UnitMap:
    """Per-joint affine map dataset units -> radians: ``rad = scale*u + offset``.

    ``lo``/``hi`` (radians) are the joint limits used to clip ``to_rad``;
    they are optional so a map can be built from a bare ``scale``/``offset``.
    """

    scale: np.ndarray  # (6,)
    offset: np.ndarray  # (6,)
    lo: np.ndarray | None = None  # (6,) joint range low, rad
    hi: np.ndarray | None = None  # (6,) joint range high, rad

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", np.asarray(self.scale, dtype=np.float64))
        object.__setattr__(self, "offset", np.asarray(self.offset, dtype=np.float64))
        if self.scale.shape != self.offset.shape:
            raise ValueError("scale and offset must have the same shape")
        for name in ("lo", "hi"):
            v = getattr(self, name)
            if v is not None:
                object.__setattr__(self, name, np.asarray(v, dtype=np.float64))

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> UnitMap:
        """Body joints: -100..100 -> joint range; gripper: 0..100 -> joint range."""
        lo = model.jnt_range[:, 0].copy()
        hi = model.jnt_range[:, 1].copy()
        scale = (hi - lo) / 200.0
        offset = (hi + lo) / 2.0
        g = GRIPPER_INDEX
        scale[g] = (hi[g] - lo[g]) / 100.0
        offset[g] = lo[g]
        return cls(scale=scale, offset=offset, lo=lo, hi=hi)

    @classmethod
    def from_kinds(
        cls,
        lo: np.ndarray,
        hi: np.ndarray,
        kinds: Sequence[str],
        extra_scale: Sequence[float] | None = None,
    ) -> UnitMap:
        """One column per joint range ``[lo, hi]`` and unit kind (see
        :data:`antares.robots.UNIT_KINDS`); ``extra_scale`` multiplies each
        column after its unit (a sign flip, a jaw opening that drives one
        finger)."""
        lo = np.asarray(lo, dtype=np.float64).copy()
        hi = np.asarray(hi, dtype=np.float64).copy()
        scale = np.empty_like(lo)
        offset = np.empty_like(lo)
        for i, k in enumerate(kinds):
            if k == "range_m100_100":
                scale[i], offset[i] = (hi[i] - lo[i]) / 200.0, (hi[i] + lo[i]) / 2.0
            elif k == "range_0_100":
                scale[i], offset[i] = (hi[i] - lo[i]) / 100.0, lo[i]
            elif k == "range_0_1":
                scale[i], offset[i] = hi[i] - lo[i], lo[i]
            elif k == "degrees":
                scale[i], offset[i] = np.pi / 180.0, (hi[i] + lo[i]) / 2.0
            elif k == "degrees_raw":
                scale[i], offset[i] = np.pi / 180.0, 0.0
            elif k in ("radians", "native"):
                scale[i], offset[i] = 1.0, 0.0
            else:
                raise ValueError(f"unknown unit kind {k!r}")
        if extra_scale is not None:
            scale = scale * np.asarray(extra_scale, dtype=np.float64)
        return cls(scale=scale, offset=offset, lo=lo, hi=hi)

    def to_rad(self, u: np.ndarray) -> np.ndarray:
        """Dataset units -> radians, clipped to the joint range. (6,) or (T,6)."""
        q = np.asarray(u, dtype=np.float64) * self.scale + self.offset
        if self.lo is not None and self.hi is not None:
            q = np.clip(q, self.lo, self.hi)
        return q

    def to_units(self, q: np.ndarray) -> np.ndarray:
        """Radians -> dataset units (inverse of the affine map, no clipping)."""
        return (np.asarray(q, dtype=np.float64) - self.offset) / self.scale

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scale": self.scale.tolist(),
            "offset": self.offset.tolist(),
        }
        if self.lo is not None and self.hi is not None:
            d["lo"] = self.lo.tolist()
            d["hi"] = self.hi.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UnitMap:
        return cls(
            scale=np.asarray(d["scale"], dtype=np.float64),
            offset=np.asarray(d["offset"], dtype=np.float64),
            lo=None if d.get("lo") is None else np.asarray(d["lo"], dtype=np.float64),
            hi=None if d.get("hi") is None else np.asarray(d["hi"], dtype=np.float64),
        )


@dataclass(frozen=True, eq=False)
class CommandMap:
    """Per-joint correction applied to commands before :class:`UnitMap`:
    ``u' = gain*u + bias``, in dataset units.

    In teleoperated LeRobot data the commands are the leader arm's readings
    in the leader's own calibration, while the state is the follower's in
    its own, so the two are normalized separately and need not agree. This
    map captures that per-arm disagreement; identity is the old behaviour.
    """

    gain: np.ndarray  # (6,)
    bias: np.ndarray  # (6,) dataset units

    def __post_init__(self) -> None:
        object.__setattr__(self, "gain", np.asarray(self.gain, dtype=np.float64))
        object.__setattr__(self, "bias", np.asarray(self.bias, dtype=np.float64))
        if self.gain.shape != self.bias.shape:
            raise ValueError("gain and bias must have the same shape")

    @classmethod
    def identity(cls, n: int = len(JOINTS)) -> CommandMap:
        return cls(gain=np.ones(n), bias=np.zeros(n))

    def is_identity(self) -> bool:
        return bool(np.all(self.gain == 1.0) and np.all(self.bias == 0.0))

    def apply(self, u: np.ndarray) -> np.ndarray:
        """Correct raw command units. (6,) or (T,6)."""
        return np.asarray(u, dtype=np.float64) * self.gain + self.bias

    def invert(self, u: np.ndarray) -> np.ndarray:
        """Corrected units -> raw command units (inverse of :meth:`apply`)."""
        return (np.asarray(u, dtype=np.float64) - self.bias) / self.gain

    def to_dict(self) -> dict[str, Any]:
        return {"gain": self.gain.tolist(), "bias": self.bias.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None, n: int = len(JOINTS)) -> CommandMap:
        """``None`` (e.g. a lock saved before command maps existed) is the
        identity over ``n`` columns."""
        if d is None:
            return cls.identity(n)
        return cls(gain=np.asarray(d["gain"], dtype=np.float64), bias=np.asarray(d["bias"], dtype=np.float64))


def load_model(path: str | Path = MODEL_PATH) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path))


class Arm:
    """One robot in MuJoCo with a reusable ``MjData``.

    Contacts are disabled by default (``contacts=False``). The scene holds
    only the arm, so the sole possible contacts are self-collisions between
    the MJCF's coarse collision boxes, and those fire spuriously: at the
    real robot's folded rest pose the SO-101's shoulder boxes already
    penetrate the lower-arm and wrist boxes by 1-2 mm, which wedges the sim
    elbow for dozens of frames while the real arm moves freely. Object
    contacts are not logged either, so nothing physical is lost.

    Not thread-safe: ``replay`` and ``fk`` share state. Use one ``Arm`` per
    thread, or serialize calls.

    ``robot`` names an entry of :data:`antares.robots.REGISTRY` (default
    ``so101``), which supplies the MJCF, the mesh directory, the dataset
    columns and the MJCF joint behind each, and the unit map. Each column
    drives the position actuator on its joint and is read back from that
    joint's ``qpos``; joints the MJCF couples to a column's joint with a
    joint equality (the second finger of a parallel gripper) follow it, and
    MJCF joints no column names are left at their reference pose. For the
    SO arms every column is one joint and one actuator, in order.

    ``model_path`` loads another MJCF with the same joints (for example one
    written by :mod:`antares.export`). With ``params=None`` the values
    already compiled into that model are kept (see
    :meth:`SimParams.from_model`); for the stock SO-101 that is the default
    ``SimParams()``.
    """

    def __init__(
        self,
        params: SimParams | None = None,
        unit_map: UnitMap | None = None,
        contacts: bool = False,
        model_path: str | Path | None = None,
        command_map: CommandMap | None = None,
        robot: str | None = None,
    ):
        from antares.robots import get_robot

        self.robot = get_robot(robot).resolve()
        self.model_path = Path(model_path) if model_path is not None else self.robot.model_path
        self.model = load_model(self.model_path)
        if not contacts:
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.data = mujoco.MjData(self.model)
        self._fk_data = mujoco.MjData(self.model)
        self.body_names: list[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)
        ]
        self._map_joints()
        self.unit_map = unit_map if unit_map is not None else self._default_unit_map()
        self.action_unit_map = (
            self.unit_map
            if unit_map is not None or self.robot.action_units is None
            else UnitMap.from_kinds(
                self.unit_map.lo, self.unit_map.hi, self.robot.unit_kinds(action=True), self.robot.unit_scale
            )
        )
        self.command_map = command_map if command_map is not None else CommandMap.identity(self.n_joints)
        self._ctrl_lo = np.where(self.model.actuator_ctrllimited[self.actuator_ids] > 0,
                                 self.model.actuator_ctrlrange[self.actuator_ids, 0], -np.inf)
        self._ctrl_hi = np.where(self.model.actuator_ctrllimited[self.actuator_ids] > 0,
                                 self.model.actuator_ctrlrange[self.actuator_ids, 1], np.inf)
        self._base_timestep = float(self.model.opt.timestep)
        #: Original model values that added factors scale (see antares.factors).
        self._factor_base: dict[Any, np.ndarray] = {}
        self._extras_applied: set[str] = set()
        # Unit-length-of-9.81 gravity, so gravity_scale=1 is the stock model
        # whatever gravity an exported model was baked with.
        g = self.model.opt.gravity.copy()
        norm = float(np.linalg.norm(g))
        self._base_gravity = g / norm * STANDARD_GRAVITY if norm > 0 else np.array([0.0, 0.0, -STANDARD_GRAVITY])
        self._limit_forces()
        if params is None:
            params = self.default_params(from_model=True)
        self.params = params
        self.apply(params)

    # -- joint map ----------------------------------------------------------

    def _map_joints(self) -> None:
        """Column -> MJCF joint, qpos/dof address and the actuator driving it."""
        m = self.model
        joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        want = tuple(self.robot.joint_names)
        missing = [j for j in want if j not in joint_names]
        if missing:
            raise RuntimeError(f"MJCF joints {joint_names} != expected {list(want)} for {self.robot.name} "
                               f"(missing {missing})")
        self.n_joints = len(want)
        self.joint_ids = np.array([joint_names.index(j) for j in want], dtype=np.int64)
        self.qpos_ids = np.asarray(m.jnt_qposadr[self.joint_ids], dtype=np.int64)
        self.dof_ids = np.asarray(m.jnt_dofadr[self.joint_ids], dtype=np.int64)
        acts = []
        for j, name in zip(self.joint_ids, want):
            hits = [a for a in range(m.nu)
                    if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == j]
            if len(hits) != 1:
                raise RuntimeError(f"{self.robot.name}: joint {name!r} needs exactly one actuator, has {len(hits)}")
            acts.append(hits[0])
        self.actuator_ids = np.array(acts, dtype=np.int64)
        # Joints the MJCF slaves to a mapped joint: qpos[follower] = c0 + c1 * qpos[driver].
        self._followers: list[tuple[int, int, float, float]] = []
        for e in range(m.neq):
            if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            if j2 in set(self.joint_ids.tolist()) and j1 not in set(self.joint_ids.tolist()):
                c0, c1 = float(m.eq_data[e, 0]), float(m.eq_data[e, 1])
                self._followers.append((int(m.jnt_qposadr[j1]), int(m.jnt_qposadr[j2]), c0, c1))
        self._identity_map = (
            self.n_joints == m.nq == m.nu
            and np.array_equal(self.qpos_ids, np.arange(m.nq))
            and np.array_equal(self.actuator_ids, np.arange(m.nu))
        )
        self._qpos0 = m.qpos0.copy()

    def _default_unit_map(self) -> UnitMap:
        if self.robot.unit_map_factory is not None:
            return self.robot.unit_map_factory(self.model)
        rng = self.model.jnt_range[self.joint_ids]
        return UnitMap.from_kinds(rng[:, 0], rng[:, 1], self.robot.unit_kinds(), self.robot.unit_scale)

    def _limit_forces(self) -> None:
        """Make every mapped actuator force-limited so ``forcerange`` acts.

        An actuator without a force range gets the joint's actuator force
        range when it has one, else an effectively unlimited one; the SO
        models already limit theirs and are left alone.
        """
        m = self.model
        for a, j in zip(self.actuator_ids, self.joint_ids):
            if m.actuator_forcelimited[a]:
                continue
            hi = float(m.jnt_actfrcrange[j, 1]) if m.jnt_actfrclimited[j] else 1e6
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = (-hi, hi)

    def default_params(self, from_model: bool = False) -> SimParams:
        """The robot's uncalibrated parameters.

        SO arms (``default_params="sts3215"``) start from :class:`SimParams`
        defaults, or with ``from_model`` the first actuator's compiled values
        (the old ``Arm(params=None)`` behaviour). Other robots read every
        mapped joint and actuator of their own MJCF, so the starting point is
        the model as its authors tuned it.
        """
        if self.robot.default_params == "sts3215":
            return SimParams.from_model(self.model) if from_model else SimParams()
        m = self.model
        a, d = self.actuator_ids, self.dof_ids

        def per(x: np.ndarray) -> tuple[float, ...]:
            return tuple(float(v) for v in x)

        return SimParams(
            kp=per(m.actuator_gainprm[a, 0]),
            kv=per(np.maximum(-m.actuator_biasprm[a, 2], 1e-6)),
            damping=per(np.maximum(m.dof_damping[d], 1e-6)),
            armature=per(np.maximum(m.dof_armature[d], 1e-6)),
            frictionloss=per(np.maximum(m.dof_frictionloss[d], 1e-6)),
            latency_frames=0,
            forcerange=per(m.actuator_forcerange[a, 1]),
        )

    # -- parameters ---------------------------------------------------------

    def apply(self, params: SimParams) -> None:
        """Write ``params`` into the compiled model in place (cheap, idempotent).

        Scalar fields are broadcast to all joints; per-joint tuples are
        written joint by joint (on the actuator and dof behind each column).
        Values of added factors go through their own hooks (see
        :mod:`antares.factors`). Armature feeds the qpos0-derived constants
        the compiler precomputes (``dof_M0``, ``dof_invweight0``, which scale
        the frictionloss and limit constraints), so they are recomputed with
        ``mj_setConst``; the result is then identical to a model compiled
        with these values.
        """
        m = self.model
        n = self.n_joints
        kp = params.per_joint("kp", n)
        fr = params.per_joint("forcerange", n)
        kv = params.per_joint("kv", n)
        if self._identity_map:
            m.actuator_gainprm[:, 0] = kp
            m.actuator_biasprm[:, 1] = -kp
            m.actuator_biasprm[:, 2] = -kv
            m.actuator_forcerange[:, 0] = -fr
            m.actuator_forcerange[:, 1] = fr
            m.dof_damping[:] = params.per_joint("damping", n)
            m.dof_armature[:] = params.per_joint("armature", n)
            m.dof_frictionloss[:] = params.per_joint("frictionloss", n)
        else:
            a, d = self.actuator_ids, self.dof_ids
            m.actuator_gainprm[a, 0] = kp
            m.actuator_biasprm[a, 1] = -kp
            m.actuator_biasprm[a, 2] = -kv
            m.actuator_forcerange[a, 0] = -fr
            m.actuator_forcerange[a, 1] = fr
            m.dof_damping[d] = params.per_joint("damping", n)
            m.dof_armature[d] = params.per_joint("armature", n)
            m.dof_frictionloss[d] = params.per_joint("frictionloss", n)
        m.opt.gravity[:] = self._base_gravity * params.per_joint("gravity_scale", n)[0]
        self._apply_extras(params)
        mujoco.mj_setConst(m, self.data)  # ~20 us; uses self.data as scratch
        self.params = params

    def _apply_extras(self, params: SimParams) -> None:
        if not params.extra and not self._extras_applied:
            return
        from antares import factors

        now = set()
        for name, value in params.extra:
            if not factors.is_extra(name):
                raise ValueError(f"parameter {name!r} is not a registered factor (load its factor file first)")
            v = params.per_joint(name, self.n_joints) if isinstance(value, tuple) else np.asarray(float(value))
            factors.apply_hook(name)(self, v)
            now.add(name)
        for name in self._extras_applied - now:  # dropped since the last apply: back to neutral
            f = factors.REGISTRY.get(name)
            if f is not None and f.neutral is not None:
                factors.apply_hook(name)(self, np.asarray(float(f.neutral)))
        self._extras_applied = now

    # -- kinematics ---------------------------------------------------------

    def _set_q(self, d: mujoco.MjData, q: np.ndarray) -> None:
        """Column angles into ``d.qpos`` (followers too, the rest at qpos0)."""
        if self._identity_map:
            d.qpos[:] = q
            return
        d.qpos[:] = self._qpos0
        d.qpos[self.qpos_ids] = q
        for f, drv, c0, c1 in self._followers:
            d.qpos[f] = c0 + c1 * d.qpos[drv]

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Joint angles (rad) -> body poses ``[x,y,z, qw,qx,qy,qz]`` in the world frame.

        ``q`` is (n,) or (T,n) with one value per column; the result is
        (nbody,7) or (T,nbody,7). Pure forward kinematics, no dynamics.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.ndim == 1:
            return self._fk_one(q)
        if q.ndim != 2 or q.shape[1] != self.n_joints:
            raise ValueError(f"q must be ({self.n_joints},) or (T,{self.n_joints}), got {q.shape}")
        out = np.empty((q.shape[0], self.model.nbody, 7), dtype=np.float64)
        d = self._fk_data
        for t in range(q.shape[0]):
            self._set_q(d, q[t])
            mujoco.mj_kinematics(self.model, d)
            out[t, :, :3] = d.xpos
            out[t, :, 3:] = d.xquat
        return out

    def _fk_one(self, q: np.ndarray) -> np.ndarray:
        d = self._fk_data
        self._set_q(d, q)
        mujoco.mj_kinematics(self.model, d)
        out = np.empty((self.model.nbody, 7), dtype=np.float64)
        out[:, :3] = d.xpos
        out[:, 3:] = d.xquat
        return out

    # -- replay -------------------------------------------------------------

    @property
    def robot_name(self) -> str:
        """Registry name of the model this arm simulates (``so101`` by default)."""
        return self.robot.name

    @property
    def joint_columns(self) -> tuple[str, ...]:
        """Dataset column names, in the order of every per-joint array."""
        return tuple(self.robot.joints)

    @property
    def ctrl_range(self) -> tuple[np.ndarray, np.ndarray]:
        """Actuator command limits ``(lo, hi)`` in radians, one per column (copies)."""
        return self._ctrl_lo.copy(), self._ctrl_hi.copy()

    def real_q(self, episode: Episode) -> np.ndarray:
        """Measured joint positions in radians, (T,n)."""
        return self.unit_map.to_rad(episode.state)

    def replay(self, episode: Episode, fps: float | None = None) -> np.ndarray:
        """Drive the sim with the episode's commands; return sim joint angles (T,n) rad.

        Frame 0 is the real initial state (the sim starts there, at rest).
        Commands pass through ``command_map`` first. For each later frame the
        command from ``latency_frames`` frames ago is applied (per joint, when
        latency is per joint) and the physics is advanced one dataset frame.
        This is :meth:`rollout` with the episode's own commands and initial
        state.
        """
        fps = float(fps if fps is not None else episode.fps)
        n = self.n_joints
        if episode.state.shape[1] != n or episode.action.shape[1] != n:
            raise ValueError(f"episode must have {n} joints")
        ctrl = self.action_unit_map.to_rad(self.command_map.apply(episode.action))
        q0 = self.unit_map.to_rad(episode.state[0])
        return self.rollout(ctrl, q0, fps)

    def rollout(
        self,
        ctrl_rad: np.ndarray,
        q0_rad: np.ndarray,
        fps: float,
        latency_frames: int | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Drive the sim with an arbitrary command sequence; return joint angles (T,n) rad.

        ``ctrl_rad[t]`` is the command *sent* at frame ``t`` (radians; clipped
        to the actuator ctrlrange). The servo applies the command sent
        ``latency_frames`` frames earlier (an int, or one per joint; default: the
        current ``params.latency_frames``); frames before the first command arrives
        hold ``ctrl_rad[0]``. The sim starts at rest at ``q0_rad``, which is
        returned unchanged as row 0.
        """
        fps = float(fps)
        if fps <= 0:
            raise ValueError("fps must be positive")
        m, d = self.model, self.data
        nq = self.n_joints
        ctrl = np.asarray(ctrl_rad, dtype=np.float64)
        q0 = np.asarray(q0_rad, dtype=np.float64).reshape(-1)
        if ctrl.ndim != 2 or ctrl.shape[1] != nq:
            raise ValueError(f"ctrl_rad must be (T,{nq}), got {ctrl.shape}")
        if q0.shape[0] != nq:
            raise ValueError(f"q0_rad must have {nq} joints, got {q0.shape[0]}")
        T = ctrl.shape[0]
        if T == 0:
            raise ValueError("ctrl_rad must have at least one frame")
        if latency_frames is None:
            lat = self.params.per_joint("latency_frames", nq)
        else:
            lat = np.asarray(latency_frames, dtype=np.int64).reshape(-1)
            if lat.size == 1:
                lat = np.full(nq, lat[0], dtype=np.int64)
            elif lat.size != nq:
                raise ValueError(f"latency_frames must be an int or {nq} ints, got {lat.size}")
        if np.any(lat < 0):
            raise ValueError("latency_frames must be >= 0")

        frame_dt = 1.0 / fps
        n_sub = max(1, int(round(frame_dt / self._base_timestep)))

        ctrl = np.clip(ctrl, self._ctrl_lo, self._ctrl_hi)
        # Command index per frame and joint with latency; frames before the
        # first command arrives hold the first command.
        src = np.maximum(np.arange(T)[:, None] - lat[None, :], 0)
        ctrl = np.ascontiguousarray(np.take_along_axis(ctrl, src, axis=0))

        sim_q = np.empty((T, nq), dtype=np.float64)

        old_dt = m.opt.timestep
        m.opt.timestep = frame_dt / n_sub
        try:
            mujoco.mj_resetData(m, d)
            step = mujoco.mj_step
            a, qi = self.actuator_ids, self.qpos_ids
            if self._identity_map:
                d.qpos[:] = q0
                d.qvel[:] = 0.0
                d.ctrl[:] = ctrl[0]
            else:
                self._set_q(d, q0)
                d.qvel[:] = 0.0
                # Actuators no column drives hold their joint where it starts.
                d.ctrl[:] = 0.0
                d.ctrl[a] = ctrl[0]
            mujoco.mj_forward(m, d)
            qpos = d.qpos
            dctrl = d.ctrl
            sim_q[0] = qpos if self._identity_map else qpos[qi]
            band = self.params.per_joint("deadband", nq)
            if not np.any(band):
                if self._identity_map:
                    for t in range(1, T):
                        dctrl[:] = ctrl[t]
                        step(m, d, n_sub)
                        sim_q[t] = qpos
                else:
                    for t in range(1, T):
                        dctrl[a] = ctrl[t]
                        step(m, d, n_sub)
                        sim_q[t] = qpos[qi]
            else:
                # The position actuator pushes kp * (ctrl - q); feeding it
                # q + shrink(target - q) makes that kp * shrink(error). Updated
                # every physics substep, so the band is exact, not per frame.
                # Only the joints with a band are touched: the others keep
                # their exact command, as on the plain path above. This is
                # the slow path (a Python loop per substep); a zero band
                # everywhere, the default, takes the C-side multi-step above.
                idx = np.flatnonzero(band > 0)
                lo, hi, band = self._ctrl_lo[idx], self._ctrl_hi[idx], band[idx]
                ai, qii = a[idx], qi[idx]
                for t in range(1, T):
                    target = ctrl[t]
                    dctrl[a] = target
                    want = target[idx]
                    for _ in range(n_sub):
                        q = qpos[qii]
                        e = want - q
                        dctrl[ai] = np.clip(q + np.sign(e) * np.maximum(np.abs(e) - band, 0.0), lo, hi)
                        step(m, d)
                    sim_q[t] = qpos[qi]
        finally:
            m.opt.timestep = old_dt
        return sim_q


def mesh_files(model_path: str | Path) -> dict[str, tuple[str, list[float]]]:
    """Mesh name -> (file relative to the MJCF's directory, scale), read with
    :class:`mujoco.MjSpec` so includes and ``meshdir`` are resolved."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    base = Path(spec.meshdir or "")
    return {m.name: ((base / m.file).as_posix(), [float(v) for v in m.scale]) for m in spec.meshes if m.file}


def model_geometry(
    model: mujoco.MjModel | None = None, meshes: dict[str, tuple[str, list[float]]] | None = None
) -> dict[str, Any]:
    """Compile-time geometry of the visual mesh geoms, for the web viewer.

    Returns bodies (all, in id order, with parent ids), visual mesh geoms
    (group 2) with the body-frame pose of the raw STL file and colour, and
    the joints. With ``meshes`` (see :func:`mesh_files`; robots other than
    the SO arms) each geom names its file relative to the MJCF and its scale,
    visual geoms are the ones that collide with nothing, and a geom's
    material colour is used when it has one.
    """
    if model is None:
        model = load_model()
    bodies = [
        {
            "id": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i),
            "parent": int(model.body_parentid[i]),
        }
        for i in range(model.nbody)
    ]
    geoms = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if meshes is None and model.geom_group[g] != 2:
            continue
        if meshes is not None and (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        mesh_id = int(model.geom_dataid[g])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        # The compiler recentres each mesh on its inertial frame and folds that
        # into geom_pos/geom_quat; undo it so the pose places the raw STL file.
        inv_pos, inv_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_negPose(inv_pos, inv_quat, model.mesh_pos[mesh_id], model.mesh_quat[mesh_id])
        pos, quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(pos, quat, model.geom_pos[g], model.geom_quat[g], inv_pos, inv_quat)
        if meshes is None:
            geoms.append(
                {
                    "body": int(model.geom_bodyid[g]),
                    "mesh": f"{mesh_name}.stl",
                    "pos": pos.tolist(),
                    "quat": quat.tolist(),
                    "rgba": model.geom_rgba[g].tolist(),
                }
            )
            continue
        file, scale = meshes[mesh_name]
        mat = int(model.geom_matid[g])
        geoms.append(
            {
                "body": int(model.geom_bodyid[g]),
                "mesh": file,
                "scale": scale,
                "pos": pos.tolist(),
                "quat": quat.tolist(),
                "rgba": (model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]).tolist(),
            }
        )
    joints = [
        {
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
            "body": int(model.jnt_bodyid[j]),
            "axis": model.jnt_axis[j].tolist(),
            "range": model.jnt_range[j].tolist(),
        }
        for j in range(model.njnt)
    ]
    return {"bodies": bodies, "geoms": geoms, "joints": joints}


MESH_DIR: Path = MODEL_PATH.parent / "assets"
