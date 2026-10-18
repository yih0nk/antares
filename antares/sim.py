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

from collections.abc import Sequence
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
#: Fields that name a model rather than hold a number (never per joint).
STR_FIELDS = frozenset({"servo"})

#: Actuator models :attr:`SimParams.servo` can select. ``position`` is
#: MuJoCo's ideal position actuator (``kp``, ``kv``, ``forcerange``);
#: ``sts3215`` is the Feetech STS3215's firmware and DC motor as identified on
#: a test bench by BAM (Rhoban, https://bam.readthedocs.io), see
#: :meth:`SimParams.sts3215`.
SERVOS = ("position", "sts3215")

#: Duty cycle per (firmware P coefficient x radian of error) of the STS3215,
#: measured by BAM with an oscilloscope (``error_gain`` in bam/feetech).
STS3215_ERROR_GAIN = 0.166
#: Largest duty cycle the STS3215's H-bridge reaches (BAM, oscilloscope).
STS3215_MAX_PWM = 0.97
#: Servo-model fields; written to a lock only when :attr:`SimParams.servo` is
#: not ``position``.
SERVO_FIELDS = ("servo_gain", "p_coefficient", "vin", "kt", "resistance", "max_velocity")


def _coerce(name: str, value: Any) -> ParamValue:
    """Normalize a field value to ``float``/``int`` or a 6-tuple of them."""
    if name in STR_FIELDS:
        if not isinstance(value, str):
            raise TypeError(f"{name}: expected a string, got {type(value).__name__}")
        return value
    as_int = name in INT_FIELDS
    if isinstance(value, (list, tuple, np.ndarray)) and np.ndim(value) > 0:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] != N_JOINTS:
            raise ValueError(f"{name}: per-joint value needs {N_JOINTS} entries, got {arr.shape[0]}")
        if as_int:
            return tuple(int(round(float(v))) for v in arr)
        return tuple(float(v) for v in arr)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name}: expected a number, got bool")
    return int(round(float(value))) if as_int else float(value)


@dataclass(frozen=True)
class SimParams:
    """Calibratable simulator parameters. Defaults are the MJCF values.

    Every field is either a scalar shared by all six joints or a tuple of six
    per-joint values (one per entry of :data:`JOINTS`). Scalars are how the
    original locks were written and stay the default; :meth:`per_joint`
    broadcasts either form to a ``(6,)`` array when writing into the model.
    """

    #: Position-actuator gain, velocity gain and symmetric force limit (N*m).
    #: With a servo model (:attr:`servo` not ``position``) these three are
    #: *derived*: they always hold the model's PD equivalent (see
    #: :meth:`linearized`), and any value passed for them is replaced, so a
    #: consumer that does not know the servo model (an older antares, the
    #: fleet table, :mod:`antares.sag`) still reads the right stiffness.
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
    #: Actuator model, one of :data:`SERVOS`. With ``sts3215`` the servo is
    #: simulated as its firmware and motor instead of a position actuator:
    #: the firmware moves an internal target toward the command at most at
    #: ``max_velocity``, its P loop turns the error into a PWM duty cycle
    #: ``clip(error * p_coefficient * STS3215_ERROR_GAIN * servo_gain, +-STS3215_MAX_PWM)``,
    #: and the motor pushes ``kt / resistance * (vin * duty - kt * qvel)``
    #: (back-EMF makes it weaker the faster it turns). ``kp``, ``kv`` and
    #: ``forcerange`` are then derived from the motor constants (the PD
    #: equivalent, :meth:`linearized`) rather than fitted; ``damping``,
    #: ``armature`` and ``frictionloss`` stay the joint's viscous friction,
    #: rotor inertia and Coulomb friction, which is BAM's M1 friction model.
    servo: str = "position"
    #: Multiplies the firmware's position-to-duty gain (BAM's
    #: ``error_gain_ratio``; 1.005 on BAM's bench). What the data has to pin
    #: for an STS3215 instead of ``kp``: it lumps supply sag and unit spread.
    servo_gain: ParamValue = 1.0
    #: Firmware P coefficient written to the servo. LeRobot writes 16 to SO-100
    #: and SO-101 followers (``position_p_coefficient``); the factory value is 32.
    p_coefficient: ParamValue = 16.0
    #: Supply voltage (V); the SO-101 follower's STS3215s are the 7.4 V kind.
    vin: ParamValue = 7.4
    #: Motor torque constant at the output shaft (N*m/A, also V*s/rad), BAM.
    kt: ParamValue = 1.1776311631627974
    #: Motor winding resistance (ohm), BAM.
    resistance: ParamValue = 2.478681934101461
    #: Rate limit of the firmware's internal target (rad/s), BAM.
    max_velocity: ParamValue = 5.2883481319359

    def __post_init__(self) -> None:
        for f in fields(self):
            object.__setattr__(self, f.name, _coerce(f.name, getattr(self, f.name)))
        if np.any(self.per_joint("latency_frames") < 0):
            raise ValueError("latency_frames must be >= 0")
        g = self.per_joint("gravity_scale")
        if np.any(g != g[0]):
            raise ValueError("gravity_scale is shared by all joints; per-joint values must be equal")
        if g[0] < 0:
            raise ValueError("gravity_scale must be >= 0")
        if np.any(self.per_joint("deadband") < 0):
            raise ValueError("deadband must be >= 0")
        if self.servo not in SERVOS:
            raise ValueError(f"servo must be one of {', '.join(SERVOS)}, got {self.servo!r}")
        for n in ("vin", "kt", "resistance", "max_velocity"):
            if np.any(self.per_joint(n) <= 0):
                raise ValueError(f"{n} must be > 0")
        if np.any(self.per_joint("servo_gain") < 0) or np.any(self.per_joint("p_coefficient") < 0):
            raise ValueError("servo_gain and p_coefficient must be >= 0")
        if self.uses_servo_model:
            for name, value in self._pd_equivalent().items():
                object.__setattr__(self, name, value)

    @classmethod
    def default(cls) -> SimParams:
        return cls()

    @classmethod
    def sts3215(cls, **changes: Any) -> SimParams:
        """The STS3215 servo model with BAM's bench-identified values.

        BAM fitted a 7.4 V STS3215 on a pendulum bench (``feetech_sts3215_7_4V``,
        friction model ``m1``, https://github.com/Rhoban/bam, Apache-2.0):
        torque constant, resistance, gain ratio, rate limit, rotor inertia
        (``armature``), Coulomb (``frictionloss``) and viscous (``damping``)
        friction. ``p_coefficient`` is LeRobot's 16, not BAM's bench value of
        32. ``changes`` override any field. At these values the servo is as
        stiff as a position actuator with ``kp`` about 9.4 (see
        :meth:`linearized`).
        """
        base = dict(
            servo="sts3215",
            servo_gain=1.0049367060648848,
            armature=0.02608291141307205,
            frictionloss=0.05152047758105413,
            damping=0.05989161871380604,
        )
        return cls(**{**base, **changes})

    @property
    def uses_servo_model(self) -> bool:
        return self.servo != "position"

    def _pd_equivalent(self) -> dict[str, ParamValue]:
        """``kp``, ``kv`` and ``forcerange`` of the servo model's linear part.

        Away from saturation and the rate limit the STS3215 model is a PD
        servo: ``kp = kt / resistance * vin * p_coefficient * STS3215_ERROR_GAIN
        * servo_gain`` and ``kv = kt**2 / resistance`` (back-EMF), with a stall
        torque of ``kt / resistance * vin * STS3215_MAX_PWM`` as ``forcerange``.
        Each is per joint exactly when a servo field it depends on is.
        """
        kt, r, vin = self.per_joint("kt"), self.per_joint("resistance"), self.per_joint("vin")
        kp = kt / r * vin * self.per_joint("p_coefficient") * STS3215_ERROR_GAIN * self.per_joint("servo_gain")
        kv = kt**2 / r
        fr = kt / r * vin * STS3215_MAX_PWM

        def pack(a: np.ndarray, *deps: str) -> ParamValue:
            if any(self.is_per_joint(d) for d in deps):
                return tuple(float(v) for v in a)
            return float(a[0])

        return {
            "kp": pack(kp, "kt", "resistance", "vin", "p_coefficient", "servo_gain"),
            "kv": pack(kv, "kt", "resistance"),
            "forcerange": pack(fr, "kt", "resistance", "vin"),
        }

    def linearized(self) -> SimParams:
        """Position-actuator equivalent of the servo model (``self`` if there is none).

        A servo-model ``SimParams`` already carries its PD equivalent in
        ``kp``, ``kv`` and ``forcerange`` (:meth:`_pd_equivalent`); this
        drops the servo model and its fields so the result replays as that
        position actuator. This is what an MJCF can carry; the rate limit
        and the duty-cycle (rather than torque) saturation are lost.
        """
        if not self.uses_servo_model:
            return self
        return self.replace(
            servo="position", **{n: SimParams.__dataclass_fields__[n].default for n in SERVO_FIELDS}
        )

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
        """The numeric parameters (everything :meth:`per_joint` accepts).

        ``servo``, which names the actuator model, is not one of them.
        """
        return tuple(f.name for f in fields(cls) if f.name not in STR_FIELDS)

    # -- shape helpers ------------------------------------------------------

    def is_per_joint(self, name: str) -> bool:
        return isinstance(getattr(self, name), tuple)

    @property
    def has_per_joint(self) -> bool:
        """True if any field carries per-joint values."""
        return any(self.is_per_joint(n) for n in self.field_names())

    def per_joint(self, name: str) -> np.ndarray:
        """Value of ``name`` broadcast to a ``(6,)`` array (int for frame counts)."""
        if name in STR_FIELDS:
            raise ValueError(f"{name} is not a numeric parameter")
        v = getattr(self, name)
        dtype = np.int64 if name in INT_FIELDS else np.float64
        if isinstance(v, tuple):
            return np.asarray(v, dtype=dtype)
        return np.full(N_JOINTS, v, dtype=dtype)

    def broadcast(self) -> SimParams:
        """Copy with every numeric field in per-joint (6-tuple) form; replays identically."""
        return SimParams(servo=self.servo, **{n: tuple(self.per_joint(n).tolist()) for n in self.field_names()})

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Scalars stay scalars; per-joint values become 6-element lists.

        ``gravity_scale`` and ``deadband`` are left out at their scalar
        defaults (1.0 and 0), so locks fitted without them serialize exactly
        as before; in per-joint form they are always written, so
        :meth:`broadcast` round-trips through :meth:`from_dict`. ``servo`` and
        the servo-model fields (:data:`SERVO_FIELDS`) are written only when a
        servo model is selected (or, like the others, in per-joint form), so
        position-actuator locks are unchanged. A servo-model dict also carries
        the derived ``kp``/``kv``/``forcerange``, so a reader that drops the
        unknown servo keys gets the PD equivalent, not the stock actuator.
        """
        out: dict[str, Any] = {}
        for n in self.field_names():
            v = getattr(self, n)
            if n in SERVO_FIELDS and not self.uses_servo_model and not isinstance(v, tuple):
                continue
            if isinstance(v, tuple):
                out[n] = list(v)
            elif (n == "gravity_scale" and v == 1.0) or (n == "deadband" and v == 0.0):
                continue
            else:
                out[n] = v
        if self.uses_servo_model:
            out["servo"] = self.servo
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimParams:
        """Build from a dict; unknown keys are ignored, missing keys default.

        Scalars (possibly as strings) and 6-element sequences are accepted for
        every field, so old locks (all scalars, no ``forcerange``) still load.
        """
        known = set(cls.field_names()) | STR_FIELDS
        return cls(**{k: v for k, v in d.items() if k in known})

    def replace(self, **changes: Any) -> SimParams:
        return replace(self, **changes)


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
    def from_dict(cls, d: dict[str, Any] | None) -> CommandMap:
        """``None`` (e.g. a lock saved before command maps existed) is identity."""
        if d is None:
            return cls.identity()
        return cls(gain=np.asarray(d["gain"], dtype=np.float64), bias=np.asarray(d["bias"], dtype=np.float64))


def load_model(path: str | Path = MODEL_PATH) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path))


class Arm:
    """One SO-101 in MuJoCo with a reusable ``MjData``.

    Contacts are disabled by default (``contacts=False``). The scene holds
    only the arm, so the sole possible contacts are self-collisions between
    the MJCF's coarse collision boxes, and those fire spuriously: at the
    real robot's folded rest pose the shoulder boxes already penetrate the
    lower-arm and wrist boxes by 1-2 mm, which wedges the sim elbow for
    dozens of frames while the real arm moves freely. Object contacts are
    not logged either, so nothing physical is lost.

    Not thread-safe: ``replay`` and ``fk`` share state. Use one ``Arm`` per
    thread, or serialize calls.

    ``robot`` names an entry of :data:`antares.robots.REGISTRY` (default
    ``so101``), which supplies the MJCF, the mesh directory, the MJCF's own
    joint names (checked against the compiled model, in canonical order) and
    the unit map. ``model_path`` loads another MJCF with the same six joints
    (for example one written by :mod:`antares.export`). With ``params=None``
    the values already compiled into that model are kept (see
    :meth:`SimParams.from_model`); for the stock model that is the default
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

        self.robot = get_robot(robot)
        self.model_path = Path(model_path) if model_path is not None else self.robot.model_path
        self.model = load_model(self.model_path)
        if not contacts:
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.data = mujoco.MjData(self.model)
        self._fk_data = mujoco.MjData(self.model)
        self.unit_map = unit_map if unit_map is not None else self.robot.unit_map_factory(self.model)
        self.command_map = command_map if command_map is not None else CommandMap.identity()
        self.body_names: list[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)
        ]
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        if tuple(joint_names) != tuple(self.robot.joint_names):
            raise RuntimeError(f"MJCF joints {joint_names} != expected {list(self.robot.joint_names)} for {self.robot.name}")
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self._base_timestep = float(self.model.opt.timestep)
        # Unit-length-of-9.81 gravity, so gravity_scale=1 is the stock model
        # whatever gravity an exported model was baked with.
        g = self.model.opt.gravity.copy()
        norm = float(np.linalg.norm(g))
        self._base_gravity = g / norm * STANDARD_GRAVITY if norm > 0 else np.array([0.0, 0.0, -STANDARD_GRAVITY])
        if params is None:
            params = SimParams.from_model(self.model)
        self.params = params
        self.apply(params)

    @property
    def robot_name(self) -> str:
        """Registry name of the model this arm simulates (``so101`` by default)."""
        return self.robot.name

    # -- parameters ---------------------------------------------------------

    def apply(self, params: SimParams) -> None:
        """Write ``params`` into the compiled model in place (cheap, idempotent).

        Scalar fields are broadcast to all six joints; per-joint tuples are
        written joint by joint (actuator ``i`` drives joint ``i``). Armature
        feeds the qpos0-derived constants the compiler precomputes (``dof_M0``,
        ``dof_invweight0``, which scale the frictionloss and limit constraints),
        so they are recomputed with ``mj_setConst``; the result is then identical
        to a model compiled with these values.

        With a servo model (``params.servo != "position"``) the position
        actuators carry the servo's linear part (:meth:`SimParams.linearized`:
        P loop and back-EMF, so MuJoCo integrates the back-EMF implicitly)
        with no force or control limit; :meth:`rollout` feeds them a control
        that reproduces the firmware's rate limit and duty-cycle saturation.
        """
        m = self.model
        lin = params.linearized()
        kp = lin.per_joint("kp")
        kv = lin.per_joint("kv")
        fr = lin.per_joint("forcerange")
        if params.uses_servo_model:
            fr = np.full_like(fr, 1e9)
            m.actuator_ctrlrange[:, 0] = -1e3
            m.actuator_ctrlrange[:, 1] = 1e3
        else:
            m.actuator_ctrlrange[:, 0] = self._ctrl_lo
            m.actuator_ctrlrange[:, 1] = self._ctrl_hi
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        m.actuator_biasprm[:, 2] = -kv
        m.actuator_forcerange[:, 0] = -fr
        m.actuator_forcerange[:, 1] = fr
        m.dof_damping[:] = params.per_joint("damping")
        m.dof_armature[:] = params.per_joint("armature")
        m.dof_frictionloss[:] = params.per_joint("frictionloss")
        m.opt.gravity[:] = self._base_gravity * params.per_joint("gravity_scale")[0]
        mujoco.mj_setConst(m, self.data)  # ~20 us; uses self.data as scratch
        self.params = params

    # -- kinematics ---------------------------------------------------------

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Joint angles (rad) -> body poses ``[x,y,z, qw,qx,qy,qz]`` in the world frame.

        ``q`` is (6,) or (T,6); the result is (nbody,7) or (T,nbody,7).
        Pure forward kinematics, no dynamics.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.ndim == 1:
            return self._fk_one(q)
        if q.ndim != 2 or q.shape[1] != self.model.nq:
            raise ValueError(f"q must be (6,) or (T,6), got {q.shape}")
        out = np.empty((q.shape[0], self.model.nbody, 7), dtype=np.float64)
        d = self._fk_data
        for t in range(q.shape[0]):
            d.qpos[:] = q[t]
            mujoco.mj_kinematics(self.model, d)
            out[t, :, :3] = d.xpos
            out[t, :, 3:] = d.xquat
        return out

    def _fk_one(self, q: np.ndarray) -> np.ndarray:
        d = self._fk_data
        d.qpos[:] = q
        mujoco.mj_kinematics(self.model, d)
        out = np.empty((self.model.nbody, 7), dtype=np.float64)
        out[:, :3] = d.xpos
        out[:, 3:] = d.xquat
        return out

    # -- replay -------------------------------------------------------------

    @property
    def ctrl_range(self) -> tuple[np.ndarray, np.ndarray]:
        """Actuator command limits ``(lo, hi)`` in radians, (6,) each (copies)."""
        return self._ctrl_lo.copy(), self._ctrl_hi.copy()

    def real_q(self, episode: Episode) -> np.ndarray:
        """Measured joint positions in radians, (T,6)."""
        return self.unit_map.to_rad(episode.state)

    def replay(self, episode: Episode, fps: float | None = None) -> np.ndarray:
        """Drive the sim with the episode's commands; return sim joint angles (T,6) rad.

        Frame 0 is the real initial state (the sim starts there, at rest).
        Commands pass through ``command_map`` first. For each later frame the
        command from ``latency_frames`` frames ago is applied (per joint, when
        latency is per joint) and the physics is advanced one dataset frame.
        This is :meth:`rollout` with the episode's own commands and initial
        state.
        """
        fps = float(fps if fps is not None else episode.fps)
        nq = self.model.nq
        if episode.state.shape[1] != nq or episode.action.shape[1] != nq:
            raise ValueError("episode must have 6 joints")
        ctrl = self.unit_map.to_rad(self.command_map.apply(episode.action))
        q0 = self.unit_map.to_rad(episode.state[0])
        return self.rollout(ctrl, q0, fps)

    def rollout(
        self,
        ctrl_rad: np.ndarray,
        q0_rad: np.ndarray,
        fps: float,
        latency_frames: int | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Drive the sim with an arbitrary command sequence; return joint angles (T,6) rad.

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
        nq = m.nq
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
            lat = self.params.per_joint("latency_frames")
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
            d.qpos[:] = q0
            d.qvel[:] = 0.0
            d.ctrl[:] = ctrl[0]
            mujoco.mj_forward(m, d)
            sim_q[0] = d.qpos
            qpos = d.qpos
            dctrl = d.ctrl
            step = mujoco.mj_step
            band = self.params.per_joint("deadband")
            if self.params.uses_servo_model:
                self._servo_steps(ctrl, sim_q, n_sub, band)
            elif not np.any(band):
                for t in range(1, T):
                    dctrl[:] = ctrl[t]
                    step(m, d, n_sub)
                    sim_q[t] = qpos
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
                for t in range(1, T):
                    target = ctrl[t]
                    dctrl[:] = target
                    want = target[idx]
                    for _ in range(n_sub):
                        q = qpos[idx]
                        e = want - q
                        dctrl[idx] = np.clip(q + np.sign(e) * np.maximum(np.abs(e) - band, 0.0), lo, hi)
                        step(m, d)
                    sim_q[t] = qpos
        finally:
            m.opt.timestep = old_dt
        return sim_q

    def _servo_steps(self, ctrl: np.ndarray, sim_q: np.ndarray, n_sub: int, band: np.ndarray) -> None:
        """Advance frames ``1..T-1`` with the STS3215 servo model (see :attr:`SimParams.servo`).

        Runs inside :meth:`rollout` after the reset, so ``sim_q[0]`` is set
        and ``m.opt.timestep`` is the substep. Every substep the firmware's
        internal target moves toward the latency-shifted command by at most
        ``max_velocity * dt``; the (deadband-shrunk) error to it is clipped
        where the duty cycle saturates, and the actuator, which :meth:`apply`
        set to the servo's linear gains, is given ``q + error``. Its force is
        then ``kt / resistance * (vin * duty - kt * qvel)`` exactly, with the
        back-EMF integrated implicitly. The internal target starts at the
        first command, as if the servo had been holding it, which is what the
        position actuator assumes too.
        """
        m, d = self.model, self.data
        p = self.params
        dt = float(m.opt.timestep)
        gain = p.per_joint("p_coefficient") * STS3215_ERROR_GAIN * p.per_joint("servo_gain")
        # Error at which the duty cycle saturates (no limit for a zero gain).
        with np.errstate(divide="ignore"):
            sat = np.where(gain > 0, STS3215_MAX_PWM / gain, np.inf)
        step_max = p.per_joint("max_velocity") * dt
        banded = bool(np.any(band))
        target = ctrl[0].copy()
        qpos, dctrl = d.qpos, d.ctrl
        step = mujoco.mj_step
        for t in range(1, ctrl.shape[0]):
            want = ctrl[t]
            for _ in range(n_sub):
                target = np.clip(want, target - step_max, target + step_max)
                q = qpos.copy()
                e = target - q
                if banded:
                    e = np.sign(e) * np.maximum(np.abs(e) - band, 0.0)
                dctrl[:] = q + np.clip(e, -sat, sat)
                step(m, d)
            sim_q[t] = qpos


def model_geometry(model: mujoco.MjModel | None = None) -> dict[str, Any]:
    """Compile-time geometry of the visual mesh geoms, for the web viewer.

    Returns bodies (all, in id order, with parent ids), visual mesh geoms
    (group 2) with the body-frame pose of the raw STL file and colour, and
    the joints.
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
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 2:
            continue
        mesh_id = int(model.geom_dataid[g])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        # The compiler recentres each mesh on its inertial frame and folds that
        # into geom_pos/geom_quat; undo it so the pose places the raw STL file.
        inv_pos, inv_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_negPose(inv_pos, inv_quat, model.mesh_pos[mesh_id], model.mesh_quat[mesh_id])
        pos, quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(pos, quat, model.geom_pos[g], model.geom_quat[g], inv_pos, inv_quat)
        geoms.append(
            {
                "body": int(model.geom_bodyid[g]),
                "mesh": f"{mesh_name}.stl",
                "pos": pos.tolist(),
                "quat": quat.tolist(),
                "rgba": model.geom_rgba[g].tolist(),
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
