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
    """Normalize a field value to ``float``/``int`` or a 6-tuple of them."""
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

    kp: ParamValue = 998.22
    kv: ParamValue = 2.731
    damping: ParamValue = 0.6
    armature: ParamValue = 0.028
    frictionloss: ParamValue = 0.052
    latency_frames: ParamValue = 0  # command delay in dataset frames (>= 0)
    forcerange: ParamValue = 2.94  # actuator force limit, symmetric (N*m)

    def __post_init__(self) -> None:
        for f in fields(self):
            object.__setattr__(self, f.name, _coerce(f.name, getattr(self, f.name)))
        if np.any(self.per_joint("latency_frames") < 0):
            raise ValueError("latency_frames must be >= 0")

    @classmethod
    def default(cls) -> SimParams:
        return cls()

    @classmethod
    def from_model(cls, model: mujoco.MjModel, latency_frames: int = 0) -> SimParams:
        """Read the actuator/joint values back out of a compiled model.

        Uses the first actuator and dof (the SO-101 shares one ``sts3215``
        class across all six). The model cannot carry command latency, so
        it is passed in (default 0).
        """
        return cls(
            kp=float(model.actuator_gainprm[0, 0]),
            kv=float(-model.actuator_biasprm[0, 2]),
            damping=float(model.dof_damping[0]),
            armature=float(model.dof_armature[0]),
            frictionloss=float(model.dof_frictionloss[0]),
            latency_frames=latency_frames,
        )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    # -- shape helpers ------------------------------------------------------

    def is_per_joint(self, name: str) -> bool:
        return isinstance(getattr(self, name), tuple)

    @property
    def has_per_joint(self) -> bool:
        """True if any field carries per-joint values."""
        return any(self.is_per_joint(n) for n in self.field_names())

    def per_joint(self, name: str) -> np.ndarray:
        """Value of ``name`` broadcast to a ``(6,)`` array (int for frame counts)."""
        v = getattr(self, name)
        dtype = np.int64 if name in INT_FIELDS else np.float64
        if isinstance(v, tuple):
            return np.asarray(v, dtype=dtype)
        return np.full(N_JOINTS, v, dtype=dtype)

    def broadcast(self) -> SimParams:
        """Copy with every field in per-joint (6-tuple) form; replays identically."""
        return SimParams(**{n: tuple(self.per_joint(n).tolist()) for n in self.field_names()})

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Scalars stay scalars; per-joint values become 6-element lists."""
        out: dict[str, Any] = {}
        for n in self.field_names():
            v = getattr(self, n)
            out[n] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimParams:
        """Build from a dict; unknown keys are ignored, missing keys default.

        Scalars (possibly as strings) and 6-element sequences are accepted for
        every field, so old locks (all scalars, no ``forcerange``) still load.
        """
        known = set(cls.field_names())
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

    ``model_path`` loads another MJCF with the same six joints (for example
    one written by :mod:`antares.export`). With ``params=None`` the values
    already compiled into that model are kept (see
    :meth:`SimParams.from_model`); for the stock model that is the default
    ``SimParams()``.
    """

    def __init__(
        self,
        params: SimParams | None = None,
        unit_map: UnitMap | None = None,
        contacts: bool = False,
        model_path: str | Path | None = None,
    ):
        self.model_path = Path(model_path) if model_path is not None else MODEL_PATH
        self.model = load_model(self.model_path)
        if not contacts:
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.data = mujoco.MjData(self.model)
        self._fk_data = mujoco.MjData(self.model)
        self.unit_map = unit_map if unit_map is not None else UnitMap.from_model(self.model)
        self.body_names: list[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)
        ]
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        if tuple(joint_names) != JOINTS:
            raise RuntimeError(f"MJCF joints {joint_names} != expected {JOINTS}")
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self._base_timestep = float(self.model.opt.timestep)
        if params is None:
            params = SimParams.from_model(self.model)
        self.params = params
        self.apply(params)

    # -- parameters ---------------------------------------------------------

    def apply(self, params: SimParams) -> None:
        """Write ``params`` into the compiled model in place (cheap, idempotent).

        Scalar fields are broadcast to all six joints; per-joint tuples are
        written joint by joint (actuator ``i`` drives joint ``i``). Armature
        feeds the qpos0-derived constants the compiler precomputes (``dof_M0``,
        ``dof_invweight0``, which scale the frictionloss and limit constraints),
        so they are recomputed with ``mj_setConst``; the result is then identical
        to a model compiled with these values.
        """
        m = self.model
        kp = params.per_joint("kp")
        fr = params.per_joint("forcerange")
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        m.actuator_biasprm[:, 2] = -params.per_joint("kv")
        m.actuator_forcerange[:, 0] = -fr
        m.actuator_forcerange[:, 1] = fr
        m.dof_damping[:] = params.per_joint("damping")
        m.dof_armature[:] = params.per_joint("armature")
        m.dof_frictionloss[:] = params.per_joint("frictionloss")
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
        For each later frame the command from ``latency_frames`` frames ago
        is applied (per joint, when latency is per joint) and the physics is
        advanced one dataset frame. This is :meth:`rollout` with the episode's
        own commands and initial state.
        """
        fps = float(fps if fps is not None else episode.fps)
        nq = self.model.nq
        if episode.state.shape[1] != nq or episode.action.shape[1] != nq:
            raise ValueError("episode must have 6 joints")
        ctrl = self.unit_map.to_rad(episode.action)
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
            for t in range(1, T):
                dctrl[:] = ctrl[t]
                step(m, d, n_sub)
                sim_q[t] = qpos
        finally:
            m.opt.timestep = old_dt
        return sim_q


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
