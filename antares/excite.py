"""A calibration dance: a short excitation trajectory for the SO-101.

Pick-and-place logs move the arm through a few smooth reaches, which pin
latency and the sensor map but let ``kp``, ``kv``, damping, armature and
friction trade off at the same tracking error. :func:`dance` writes about
a minute of commands built to separate them, one body joint at a time,
around a raised home pose:

* **steps** of two sizes. A large step saturates the actuator's force
  limit, where ``kv`` (inside the limit) and joint ``damping`` (outside
  it) stop acting alike; a small one stays in the linear range, where the
  rise time measures ``kp`` and the stall measures ``frictionloss``.
* a **chirp**, a log sweep from ``f0`` to ``f1`` Hz with tapered ends,
  whose phase lag at the top of the sweep measures ``armature`` and the
  latency.
* **holds** in poses with different gravity torque, where the static sag
  is ``g(q) / kp`` because the fit keeps ``gravity_scale`` at 1.

Commands are in dataset units (LeRobot's normalized ``-100..100`` for the
body joints, ``0..100`` for the gripper), so the same file drives the sim
and, through LeRobot's ``send_action``, an arm. :func:`check_safety`
verifies the whole trajectory with forward kinematics: inside the joint
limits with a margin and the fingertip and links well above the table.

The trajectory and every claim about what it identifies were evaluated in
the sim only (``scripts/excite_study.py``); it has not been run on an arm.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from antares.data import Episode
from antares.sim import JOINTS

#: Raised home pose, dataset units: upper arm tilted back, forearm forward
#: and level-ish, fingertip about 25 cm above the base plane.
HOME: tuple[float, ...] = (0.0, -30.0, 30.0, 0.0, 0.0, 30.0)

#: Poses held to load the shoulder and elbow differently against gravity
#: (arm leaning forward, forearm raised, arm folded back).
GRAVITY_POSES: tuple[tuple[float, ...], ...] = (
    (0.0, 15.0, -15.0, 0.0, 0.0, 30.0),
    (0.0, -10.0, -40.0, 20.0, 0.0, 30.0),
    (0.0, -50.0, 40.0, 30.0, 0.0, 30.0),
)

#: Body joints excited one at a time, in this order.
BODY = tuple(j for j in JOINTS if j != "gripper")

#: Table height (m, world z) of the SO-101 demo scene fit
#: (``locks/svla_so101_pickplace.scene.json``), and how far above it the
#: fingertip and every link below the shoulder must stay.
TABLE_Z = -0.022
CLEARANCE = 0.08
#: Joint-limit margin, dataset units (100 is half a joint's range).
LIMIT = 80.0
#: The largest commanded speed outside a step (rad/s). BAM measured the
#: STS3215's rate limit at about 5.3 rad/s; this keeps well inside it.
MAX_SPEED = 3.0


@dataclass(frozen=True)
class Dance:
    """A command sequence: ``commands`` is (T, 6) in dataset units at ``fps``.

    ``segments`` names each part as ``(label, first frame, end frame)``;
    ``steps`` lists the frames where a step changes the command at once.
    """

    fps: float
    commands: np.ndarray
    segments: tuple[tuple[str, int, int], ...]
    steps: tuple[int, ...] = ()
    joint_names: tuple[str, ...] = JOINTS
    config: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.commands) / self.fps

    def episode(self, index: int = 0) -> Episode:
        """The dance as a commanded episode (state = commands), for a replay."""
        return Episode(index=index, fps=self.fps, joint_names=self.joint_names,
                       state=self.commands.copy(), action=self.commands.copy())


def _frames(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))


def _minjerk(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """``n`` frames from ``a`` to ``b`` (excluding ``a``) on a minimum-jerk profile."""
    s = np.arange(1, n + 1) / n
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    return a + s[:, None] * (b - a)


def chirp(n: int, fps: float, f0: float, f1: float, taper: float = 0.5) -> np.ndarray:
    """Unit-amplitude log sweep from ``f0`` to ``f1`` Hz over ``n`` frames.

    Starts and ends at zero; the first and last ``taper`` seconds are faded
    in and out with a raised cosine.
    """
    t = np.arange(n) / fps
    T = n / fps
    k = f1 / f0
    phase = 2 * np.pi * f0 * T / np.log(k) * (k ** (t / T) - 1)
    w = np.ones(n)
    m = min(_frames(taper, fps), n // 2)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
    w[:m] = ramp
    w[n - m:] = ramp[::-1]
    return w * np.sin(phase)


#: Per-joint step and chirp sizes, dataset units (about a degree each).
BIG_STEP = {"shoulder_pan": 25.0, "shoulder_lift": 20.0, "elbow_flex": 20.0, "wrist_flex": 25.0, "wrist_roll": 25.0}
SMALL_STEP = 4.0
CHIRP_AMP = {"shoulder_pan": 10.0, "shoulder_lift": 8.0, "elbow_flex": 8.0, "wrist_flex": 10.0, "wrist_roll": 8.0}


def dance(
    fps: float = 30.0,
    home: Sequence[float] = HOME,
    gravity_poses: Sequence[Sequence[float]] = GRAVITY_POSES,
    big_step: dict[str, float] | None = None,
    small_step: float = SMALL_STEP,
    chirp_amp: dict[str, float] | None = None,
    f0: float = 0.3,
    f1: float = 2.0,
    step_hold: float = 0.8,
    small_hold: float = 0.5,
    chirp_seconds: float = 5.0,
    move_seconds: float = 1.2,
    pose_hold: float = 1.5,
) -> Dance:
    """The calibration dance, about 60 s at the defaults.

    Starts and ends at ``home`` (move the arm there before replaying). For
    each body joint in turn: a large step up, back, down and back, the same
    with ``small_step``, then a chirp of ``chirp_amp`` units from ``f0`` to
    ``f1`` Hz. Then each of ``gravity_poses`` is approached on a
    minimum-jerk path and held for ``pose_hold`` seconds.
    """
    big_step = {**BIG_STEP, **(big_step or {})}
    chirp_amp = {**CHIRP_AMP, **(chirp_amp or {})}
    home_v = np.asarray(home, dtype=np.float64)
    if home_v.shape != (len(JOINTS),):
        raise ValueError(f"home needs {len(JOINTS)} values, got {home_v.shape}")
    parts: list[np.ndarray] = []
    segments: list[tuple[str, int, int]] = []
    steps: list[int] = []
    n = 0

    def add(label: str, block: np.ndarray, step: bool = False) -> None:
        nonlocal n
        if step:
            steps.append(n)
        parts.append(block)
        segments.append((label, n, n + len(block)))
        n += len(block)

    def hold(q: np.ndarray, seconds: float) -> np.ndarray:
        return np.repeat(q[None], _frames(seconds, fps), axis=0)

    add("home", hold(home_v, 1.0))
    for j, name in enumerate(BODY):
        for size, secs in ((big_step[name], step_hold), (small_step, small_hold)):
            for sign in (+1, 0, -1, 0):
                q = home_v.copy()
                q[j] += sign * size
                add(f"{name} step {sign * size:+g}", hold(q, secs), step=True)
        m = _frames(chirp_seconds, fps)
        block = np.repeat(home_v[None], m, axis=0)
        block[:, j] += chirp_amp[name] * chirp(m, fps, f0, f1)
        add(f"{name} chirp {f0:g}-{f1:g} Hz", block)
    last = home_v
    for i, pose in enumerate(list(gravity_poses) + [home]):
        p = np.asarray(pose, dtype=np.float64)
        label = "back home" if i == len(gravity_poses) else f"gravity pose {i + 1}"
        add(f"move to {label}", _minjerk(last, p, _frames(move_seconds, fps)))
        add(label, hold(p, pose_hold if i < len(gravity_poses) else 0.5))
        last = p
    config = {
        "fps": fps, "home": list(home_v), "gravity_poses": [list(map(float, p)) for p in gravity_poses],
        "big_step": big_step, "small_step": small_step, "chirp_amp": chirp_amp, "f0": f0, "f1": f1,
        "step_hold": step_hold, "small_hold": small_hold, "chirp_seconds": chirp_seconds,
        "move_seconds": move_seconds, "pose_hold": pose_hold,
    }
    return Dance(fps=fps, commands=np.concatenate(parts), segments=tuple(segments), steps=tuple(steps),
                 config=config)


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------


def check_safety(
    commands: np.ndarray,
    fps: float,
    steps: Sequence[int] = (),
    arm=None,
    table_z: float = TABLE_Z,
    clearance: float = CLEARANCE,
    limit: float = LIMIT,
    max_speed: float = MAX_SPEED,
) -> dict:
    """Forward-kinematics check of every frame of ``commands`` (dataset units).

    ``ok`` when every body joint stays inside ``±limit`` units (the gripper
    inside 0..100), the fingertip and every link from the forearm out stay
    ``clearance`` metres above ``table_z``, and the commanded speed outside
    the ``steps`` frames stays below ``max_speed`` rad/s. Returns the
    margins alongside the verdict.
    """
    import mujoco

    from antares.sim import Arm

    commands = np.asarray(commands, dtype=np.float64)
    arm = arm or Arm()
    m = arm.model
    q = arm.unit_map.to_rad(commands)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    low_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                  for b in ("lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")]
    d = mujoco.MjData(m)
    tip_z = np.empty(len(q))
    link_z = np.empty(len(q))
    for t in range(len(q)):
        d.qpos[arm.qpos_ids] = q[t]
        mujoco.mj_kinematics(m, d)
        tip_z[t] = d.site_xpos[site, 2]
        link_z[t] = d.xpos[low_bodies, 2].min()
    speed = np.abs(np.diff(q, axis=0)) * fps
    jumps = {s - 1 for s in steps if s > 0}
    smooth = np.array([t for t in range(len(speed)) if t not in jumps], dtype=int)
    peak = speed[smooth].max(axis=0) if smooth.size else np.zeros(q.shape[1])
    body = commands[:, : len(BODY)]
    grip = commands[:, len(BODY):]
    floor = table_z + clearance
    out = {
        "n_frames": int(len(commands)),
        "duration_s": float(len(commands) / fps),
        "max_abs_units": float(np.abs(body).max()),
        "limit_units": float(limit),
        "gripper_range": [float(grip.min()), float(grip.max())],
        "min_tip_z": float(tip_z.min()),
        "min_link_z": float(link_z.min()),
        "floor_z": float(floor),
        "peak_speed_rad_s": {n: float(v) for n, v in zip(JOINTS, peak)},
        "max_speed_rad_s": float(max_speed),
        "max_step_units": float(np.abs(np.diff(commands, axis=0)).max()) if len(commands) > 1 else 0.0,
    }
    out["ok"] = bool(
        out["max_abs_units"] <= limit
        and grip.min() >= 0 and grip.max() <= 100
        and min(out["min_tip_z"], out["min_link_z"]) >= floor
        and peak[: len(BODY)].max() <= max_speed
    )
    return out


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def write(path: str | Path, d: Dance) -> Path:
    """Save ``d`` by extension: ``.csv`` (``timestamp`` plus one ``<joint>.pos``
    column per joint, LeRobot's action names), ``.json`` (the same plus the
    segments and the generator settings) or ``.npy`` (the bare (T, 6) array)."""
    path = Path(path)
    names = [f"{j}.pos" for j in d.joint_names]
    if path.suffix == ".npy":
        np.save(path, d.commands)
    elif path.suffix == ".csv":
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", *names])
            for t, row in enumerate(d.commands):
                w.writerow([f"{t / d.fps:.4f}", *(f"{v:.3f}" for v in row)])
    elif path.suffix == ".json":
        path.write_text(json.dumps({
            "fps": d.fps,
            "names": names,
            "units": "LeRobot normalized: body joints -100..100, gripper 0..100",
            "action": np.round(d.commands, 3).tolist(),
            "segments": [{"label": s, "start": a, "end": b} for s, a, b in d.segments],
            "config": d.config,
        }) + "\n")
    else:
        raise ValueError(f"{path}: expected a .csv, .json or .npy path")
    return path


def read(path: str | Path) -> tuple[np.ndarray, float]:
    """``(commands (T, 6), fps)`` from a file :func:`write` saved (``.npy``
    carries no frame rate, so it reads as 30 fps)."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path), 30.0
    if path.suffix == ".json":
        doc = json.loads(path.read_text())
        return np.asarray(doc["action"], dtype=np.float64), float(doc["fps"])
    if path.suffix == ".csv":
        rows = list(csv.reader(path.open()))
        a = np.array([[float(v) for v in r] for r in rows[1:]])
        # timestamps are rounded to 0.1 ms, so take the rate over the whole file
        fps = (len(a) - 1) / (a[-1, 0] - a[0, 0]) if len(a) > 1 else 30.0
        return a[:, 1:], float(round(fps, 3))
    raise ValueError(f"{path}: expected a .csv, .json or .npy path")


__all__ = ["BODY", "Dance", "GRAVITY_POSES", "HOME", "check_safety", "chirp", "dance", "read", "write"]
