"""Payload mass from how the arm sags while it holds the object.

The calibrated sim replays the arm with nothing in its gripper. Whatever
the real arm does differently while it carries something is in the residual
``real - sim``. A position servo settles short of its command by the load
it holds divided by its gain, so an extra mass ``m`` at the fingertip adds,
on joint ``j``, a steady deflection

    dq_j = -m * g * Jz_j / kp_j

where ``Jz_j`` is the fingertip's height per radian of joint ``j`` (the
``z`` row of the site Jacobian, the lever arm with its sign) and ``kp_j`` the
lock's position gain in N*m/rad (MuJoCo's ``<position kp>``; the actuator
force is ``kp * (ctrl - q)``, clipped to ``forcerange``). ``shoulder_lift``
and ``elbow_flex`` carry the load; the pan and roll axes are vertical or
nearly so.

Lifting the object also reverses the motion (the arm comes down empty and
goes up loaded), and a servo deadband or gearbox backlash flips the
tracking error with the direction of motion. The two are separated with a
differential design: around the *pick* the residual steps by ``P + D``
(payload plus motion flip), around the *release* by ``-P + D`` (the same
down-then-up motion, the load leaving instead of arriving). Half the
difference is the payload, half the sum the motion term, reported as a
nuisance. Each step is the median residual over a short window after the
event minus the median over a window before it; the lever arm is the
median over the same windows.

Assumptions, stated rather than hidden: quasi-static (medians over ~0.6 s
windows average the tracking lag, which is what the motion term absorbs);
the sim's servo is the real one (the mass scales 1:1 with ``kp``, which a
pick-and-place log pins down poorly, see the README); the gripper joint is
excluded; the object is a point mass at the ``gripperframe`` site; the
sim's own link masses are the real ones (the lock's ``gravity_scale`` says
how far that holds). There is no ground truth; consistency across
episodes and a plausible size are the only checks, and when the measured
deflection is below the encoder step the result says so.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from antares import __version__
from antares.data import Dataset, Episode
from antares.events import residuals
from antares.lock import now_iso
from antares.scene import TIP_SITE
from antares.sim import JOINTS, STANDARD_GRAVITY

#: Joints that carry the load (their axes are horizontal).
LOAD_JOINTS: tuple[str, ...] = ("shoulder_lift", "elbow_flex")
#: Window length either side of an event (s) and the gap left around it:
#: the jaw is still closing for ``SETTLE_S`` after the command flips.
WINDOW_S = 0.6
SETTLE_S = 0.3
MARGIN_S = 0.1
MIN_WINDOW_FRAMES = 6
#: Joints whose lever arm is shorter than this (m) say nothing about the mass.
MIN_LEVER_M = 0.03
#: One encoder step of the STS3215 (deg); deflections below it are noise.
ENCODER_STEP_DEG = 360.0 / 4096


def fingertip_jz(arm: Any, q: np.ndarray, site: str = TIP_SITE) -> np.ndarray:
    """``d(tip z)/dq`` per joint, (6,) m/rad at ``q`` (rad) or (T,6) for (T,6)."""
    m = arm.model
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        raise ValueError(f"MJCF has no site {site!r}")
    d = mujoco.MjData(m)
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    q = np.asarray(q, dtype=np.float64)
    one = q.ndim == 1
    qs = q[None] if one else q
    out = np.empty((qs.shape[0], m.nv))
    for t in range(qs.shape[0]):
        d.qpos[:] = qs[t]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        out[t] = jp[2]
    return out[0] if one else out


@dataclass(frozen=True)
class Step:
    """The residual step across one gripper event on one joint."""

    before_deg: float
    after_deg: float
    lever_m: float  # median Jz over both windows
    n_before: int
    n_after: int

    @property
    def delta_deg(self) -> float:
        return self.after_deg - self.before_deg

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_deg": self.before_deg,
            "after_deg": self.after_deg,
            "delta_deg": self.delta_deg,
            "lever_m": self.lever_m,
            "n_before": self.n_before,
            "n_after": self.n_after,
        }


@dataclass(frozen=True)
class JointPayload:
    joint: str
    pick: Step
    release: Step
    kp: float
    payload_deg: float  # (pick - release) / 2
    motion_deg: float  # (pick + release) / 2
    torque_nm: float  # payload deflection * kp
    mass_kg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "pick": self.pick.to_dict(),
            "release": self.release.to_dict(),
            "kp": self.kp,
            "payload_deg": self.payload_deg,
            "motion_deg": self.motion_deg,
            "torque_nm": self.torque_nm,
            "mass_kg": self.mass_kg,
        }


@dataclass(frozen=True)
class EpisodePayload:
    episode: int
    grasp: int
    release: int
    joints: dict[str, JointPayload]
    mass_kg: float  # lever-weighted combination of the joints (NaN if none usable)
    ok: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "grasp": self.grasp,
            "release": self.release,
            "joints": {k: v.to_dict() for k, v in self.joints.items()},
            "mass_kg": self.mass_kg,
            "ok": self.ok,
            "note": self.note,
        }


def _quartiles(v: Sequence[float]) -> dict[str, float] | None:
    a = np.asarray([x for x in v if x is not None and math.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return None
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "std": float(a.std())}


@dataclass
class PayloadEstimate:
    lock: str
    dataset: str
    kp: dict[str, float]
    episodes: list[EpisodePayload]
    mass_kg: float | None  # median over usable episodes
    mass_iqr_kg: float | None
    mass: dict[str, float] | None  # quartiles
    mass_by_joint: dict[str, dict[str, float] | None]
    payload_deg: dict[str, dict[str, float] | None]  # per joint, across episodes
    motion_deg: dict[str, dict[str, float] | None]
    n_ok: int
    n_episodes: int
    #: Mass one encoder step of deflection stands for on each joint with the
    #: lock's kp and the median lever arm: the resolution of the method here.
    mass_per_step_kg: dict[str, float | None] = field(default_factory=dict)
    #: Mass half the across-episode IQR of the deflection stands for: the noise floor.
    mass_noise_floor_kg: dict[str, float | None] = field(default_factory=dict)
    lever_m: dict[str, float | None] = field(default_factory=dict)
    window_s: float = WINDOW_S
    assumptions: list[str] = field(default_factory=list)
    verdict: str = ""
    created_at: str = ""
    antares_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return {
            "lock": self.lock,
            "dataset": self.dataset,
            "kp": dict(self.kp),
            "episodes": [e.to_dict() for e in self.episodes],
            "mass_kg": self.mass_kg,
            "mass_iqr_kg": self.mass_iqr_kg,
            "mass": self.mass,
            "mass_by_joint": dict(self.mass_by_joint),
            "payload_deg": dict(self.payload_deg),
            "motion_deg": dict(self.motion_deg),
            "n_ok": self.n_ok,
            "n_episodes": self.n_episodes,
            "mass_per_step_kg": dict(self.mass_per_step_kg),
            "mass_noise_floor_kg": dict(self.mass_noise_floor_kg),
            "lever_m": dict(self.lever_m),
            "window_s": self.window_s,
            "encoder_step_deg": ENCODER_STEP_DEG,
            "assumptions": list(self.assumptions),
            "verdict": self.verdict,
            "created_at": self.created_at,
            "antares_version": self.antares_version,
        }


ASSUMPTIONS = [
    "quasi-static: medians over short windows before/after each gripper event stand for the steady-state deflection",
    "servo torque = kp * (cmd - q) with the lock's kp (N*m/rad), so the mass scales 1:1 with kp, which pick-and-place logs pin down poorly",
    "the object is a point mass at the gripperframe site; the gripper joint is excluded",
    "the sim's link masses are the real ones (gravity_scale of the lock)",
    "the motion-direction term (deadband/backlash) is the same at the pick and at the release and cancels in the difference",
]


def _window(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo, hi = max(0, lo), min(len(arr), hi)
    return arr[lo:hi] if hi > lo else arr[:0]


def episode_payload(
    episode: Episode,
    arm: Any,
    grasp: int,
    release: int,
    *,
    window_s: float = WINDOW_S,
    settle_s: float = SETTLE_S,
    margin_s: float = MARGIN_S,
    sim_q: np.ndarray | None = None,
) -> EpisodePayload:
    """The residual steps around one carried pick (``grasp`` .. ``release``) and the mass they imply."""
    fps = float(episode.fps)
    real = arm.real_q(episode)
    sim = arm.replay(episode) if sim_q is None else np.asarray(sim_q, dtype=np.float64)
    res = residuals(real, sim)
    jz = fingertip_jz(arm, real)
    kp = arm.params.per_joint("kp")
    W, settle, margin = int(round(window_s * fps)), int(round(settle_s * fps)), int(round(margin_s * fps))
    carry = release - grasp
    # Shrink the windows so the pick's after-window and the release's before-window do not overlap.
    W = max(MIN_WINDOW_FRAMES, min(W, (carry - 2 * settle) // 2))
    note = ""
    if carry < 2 * settle + 2 * MIN_WINDOW_FRAMES:
        return EpisodePayload(episode.index, grasp, release, {}, float("nan"), False, f"carry too short ({carry} frames)")
    joints: dict[str, JointPayload] = {}
    masses, weights = [], []
    for name in LOAD_JOINTS:
        j = JOINTS.index(name)
        pb, pa = _window(res[:, j], grasp - margin - W, grasp - margin), _window(res[:, j], grasp + settle, grasp + settle + W)
        rb, ra = _window(res[:, j], release - margin - W, release - margin), _window(res[:, j], release + settle, release + settle + W)
        lp = np.concatenate([_window(jz[:, j], grasp - margin - W, grasp - margin), _window(jz[:, j], grasp + settle, grasp + settle + W)])
        lr = np.concatenate([_window(jz[:, j], release - margin - W, release - margin), _window(jz[:, j], release + settle, release + settle + W)])
        if min(len(pb), len(pa), len(rb), len(ra)) < MIN_WINDOW_FRAMES:
            note = "windows run off the episode"
            continue
        pick = Step(float(np.median(pb)), float(np.median(pa)), float(np.median(lp)), len(pb), len(pa))
        rel = Step(float(np.median(rb)), float(np.median(ra)), float(np.median(lr)), len(rb), len(ra))
        payload = 0.5 * (pick.delta_deg - rel.delta_deg)
        motion = 0.5 * (pick.delta_deg + rel.delta_deg)
        # dq_pick = -m g Jp / kp, dq_release = +m g Jr / kp  ->  (pick - release) / 2 = -m g (Jp + Jr) / (2 kp)
        lever = 0.5 * (pick.lever_m + rel.lever_m)
        torque = float(kp[j] * math.radians(payload))
        mass = float(-torque / (STANDARD_GRAVITY * lever)) if abs(lever) >= MIN_LEVER_M else float("nan")
        joints[name] = JointPayload(name, pick, rel, float(kp[j]), payload, motion, torque, mass)
        if math.isfinite(mass):
            masses.append(mass)
            weights.append(abs(lever))
    combined = float(np.average(masses, weights=weights)) if masses else float("nan")
    return EpisodePayload(episode.index, grasp, release, joints, combined, bool(masses), note)


def estimate_payload(
    dataset: Dataset,
    arm: Any,
    lock_name: str = "",
    *,
    picks: dict[int, list[tuple[int, int]]] | None = None,
    episodes: Sequence[int] | None = None,
    window_s: float = WINDOW_S,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> PayloadEstimate:
    """Replay every episode with a carried pick on ``arm`` (the lock's
    parameters) and estimate the payload mass per episode.

    ``picks`` (episode -> ``[(grasp, release)]`` held windows, as
    :func:`antares.objects.held_windows` gives them) default to the task
    module's carried picks.
    """
    from antares.task import Tip, dataset_thresholds, empty_level, find_grasps

    eps = [ep for ep in dataset.episodes if episodes is None or ep.index in set(episodes)]
    if picks is None:
        thr = dataset_thresholds(list(dataset.episodes))
        empty = empty_level(list(dataset.episodes), thr)
        tip = Tip(arm.model)
        picks = {}
        for ep in eps:
            picks[ep.index] = [
                (g.frame, g.release_frame) for g in find_grasps(ep, thr, tip, arm.unit_map, empty) if g.carried and g.release_frame is not None
            ]
    out: list[EpisodePayload] = []
    stop = should_stop or (lambda: False)
    for i, ep in enumerate(eps):
        if stop():
            break
        wins = [(g, r) for g, r in picks.get(ep.index, []) if r is not None and r < len(ep)]
        if not wins:
            continue
        sim = arm.replay(ep)
        for g, r in wins:
            out.append(episode_payload(ep, arm, int(g), int(r), window_s=window_s, sim_q=sim))
        if on_progress is not None:
            on_progress({"stage": "payload", "episode": ep.index, "i": i + 1, "n": len(eps), "mass_kg": out[-1].mass_kg})
    ok = [e for e in out if e.ok]
    mass_q = _quartiles([e.mass_kg for e in ok])
    by_joint = {j: _quartiles([e.joints[j].mass_kg for e in ok if j in e.joints]) for j in LOAD_JOINTS}
    payload_deg = {j: _quartiles([e.joints[j].payload_deg for e in ok if j in e.joints]) for j in LOAD_JOINTS}
    motion_deg = {j: _quartiles([e.joints[j].motion_deg for e in ok if j in e.joints]) for j in LOAD_JOINTS}
    kp = arm.params.per_joint("kp")
    lever = {
        j: (float(np.median([abs(0.5 * (e.joints[j].pick.lever_m + e.joints[j].release.lever_m)) for e in ok if j in e.joints])) if any(j in e.joints for e in ok) else None)
        for j in LOAD_JOINTS
    }

    def mass_for(deg: float, j: str) -> float | None:
        lv = lever[j]
        if lv is None or lv < MIN_LEVER_M:
            return None
        return float(kp[JOINTS.index(j)] * math.radians(deg) / (STANDARD_GRAVITY * lv))

    per_step = {j: mass_for(ENCODER_STEP_DEG, j) for j in LOAD_JOINTS}
    floor = {j: (mass_for(0.5 * payload_deg[j]["iqr"], j) if payload_deg[j] else None) for j in LOAD_JOINTS}
    verdict = _verdict(mass_q, payload_deg, by_joint)
    return PayloadEstimate(
        lock=lock_name,
        dataset=dataset.repo_id,
        kp={j: float(kp[JOINTS.index(j)]) for j in LOAD_JOINTS},
        episodes=out,
        mass_kg=None if mass_q is None else mass_q["median"],
        mass_iqr_kg=None if mass_q is None else mass_q["iqr"],
        mass=mass_q,
        mass_by_joint=by_joint,
        payload_deg=payload_deg,
        motion_deg=motion_deg,
        n_ok=len(ok),
        n_episodes=len(out),
        mass_per_step_kg=per_step,
        mass_noise_floor_kg=floor,
        lever_m=lever,
        window_s=window_s,
        assumptions=list(ASSUMPTIONS),
        verdict=verdict,
        created_at=now_iso(),
    )


def _verdict(
    mass_q: dict[str, float] | None,
    payload_deg: dict[str, dict[str, float] | None],
    mass_by_joint: dict[str, dict[str, float] | None],
) -> str:
    """One line on whether the deflection is a measurement or noise.

    A payload shows as a deflection above the encoder step and above half
    its own spread across episodes, on every load joint, with the same sign
    (positive mass) on each; anything else is reported as what it is.
    """
    if mass_q is None:
        return "no carried picks to measure"
    sig = {j: d for j, d in payload_deg.items() if d}
    if not sig:
        return "no joint with a lever arm"
    weakest = min(abs(d["median"]) for d in sig.values())
    if weakest < ENCODER_STEP_DEG:
        return f"below the noise: the extra deflection ({weakest:.3f} deg median on the weakest joint) is under one encoder step ({ENCODER_STEP_DEG:.3f} deg)"
    if any(abs(d["median"]) < 0.5 * d["iqr"] for d in sig.values()):
        worst = max(d["iqr"] / max(1e-9, abs(d["median"])) for d in sig.values())
        return f"below the noise: the extra deflection is smaller than half its spread across episodes (IQR up to {worst:.1f}x the median)"
    masses = {j: m["median"] for j, m in mass_by_joint.items() if m}
    if len(masses) > 1 and (min(masses.values()) <= 0 < max(masses.values())):
        return "the load joints disagree in sign (" + ", ".join(f"{j} {m * 1000:+.0f} g" for j, m in masses.items()) + "): not a payload"
    if mass_q["median"] <= 0:
        return f"the deflection has the wrong sign for a payload (median {mass_q['median'] * 1000:.0f} g): the servo model does not hold here"
    return f"consistent: median {mass_q['median'] * 1000:.0f} g, IQR {mass_q['iqr'] * 1000:.0f} g over {mass_q['n']} picks"


__all__ = [
    "ASSUMPTIONS",
    "ENCODER_STEP_DEG",
    "EpisodePayload",
    "JointPayload",
    "LOAD_JOINTS",
    "PayloadEstimate",
    "Step",
    "WINDOW_S",
    "episode_payload",
    "estimate_payload",
    "fingertip_jz",
]
