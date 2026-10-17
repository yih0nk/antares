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

from dataclasses import asdict, dataclass, fields, replace
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
GRIPPER_INDEX = JOINTS.index("gripper")

MODEL_PATH: Path = Path(__file__).resolve().parent / "assets" / "so101" / "so101.xml"

# Physics step the MJCF was tuned for; replay picks a substep count so the
# effective timestep stays close to this.
BASE_TIMESTEP = 0.005


@dataclass(frozen=True)
class SimParams:
    """Calibratable simulator parameters. Defaults are the MJCF values."""

    kp: float = 998.22
    kv: float = 2.731
    damping: float = 0.6
    armature: float = 0.028
    frictionloss: float = 0.052
    latency_frames: int = 0  # command delay in dataset frames (>= 0)

    def __post_init__(self) -> None:
        if self.latency_frames < 0:
            raise ValueError("latency_frames must be >= 0")
        object.__setattr__(self, "latency_frames", int(self.latency_frames))

    @classmethod
    def default(cls) -> SimParams:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimParams:
        """Build from a dict; unknown keys are ignored, missing keys default."""
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k not in known:
                continue
            kwargs[k] = int(round(float(v))) if k == "latency_frames" else float(v)
        return cls(**kwargs)

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
    """

    def __init__(
        self,
        params: SimParams = SimParams(),
        unit_map: UnitMap | None = None,
        contacts: bool = False,
    ):
        self.model = load_model()
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
        self.params = params
        self.apply(params)

    # -- parameters ---------------------------------------------------------

    def apply(self, params: SimParams) -> None:
        """Write ``params`` into the compiled model in place (cheap, idempotent)."""
        m = self.model
        m.actuator_gainprm[:, 0] = params.kp
        m.actuator_biasprm[:, 1] = -params.kp
        m.actuator_biasprm[:, 2] = -params.kv
        m.dof_damping[:] = params.damping
        m.dof_armature[:] = params.armature
        m.dof_frictionloss[:] = params.frictionloss
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
        is applied and the physics is advanced one dataset frame. This is
        :meth:`rollout` with the episode's own commands and initial state.
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
        latency_frames: int | None = None,
    ) -> np.ndarray:
        """Drive the sim with an arbitrary command sequence; return joint angles (T,6) rad.

        ``ctrl_rad[t]`` is the command *sent* at frame ``t`` (radians; clipped
        to the actuator ctrlrange). The servo applies the command sent
        ``latency_frames`` frames earlier (default: the current
        ``params.latency_frames``); frames before the first command arrives
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
        latency = self.params.latency_frames if latency_frames is None else int(latency_frames)
        if latency < 0:
            raise ValueError("latency_frames must be >= 0")

        frame_dt = 1.0 / fps
        n_sub = max(1, int(round(frame_dt / self._base_timestep)))

        ctrl = np.clip(ctrl, self._ctrl_lo, self._ctrl_hi)
        # Command index per frame with latency; frames before the first
        # command arrives hold the first command.
        src = np.maximum(np.arange(T) - latency, 0)
        ctrl = np.ascontiguousarray(ctrl[src])

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
    (group 2) with their body-frame pose and colour, and the joints.
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
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[g]))
        geoms.append(
            {
                "body": int(model.geom_bodyid[g]),
                "mesh": f"{mesh_name}.stl",
                "pos": model.geom_pos[g].tolist(),
                "quat": model.geom_quat[g].tolist(),
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
