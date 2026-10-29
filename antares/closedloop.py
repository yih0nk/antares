"""Closed-loop evaluation: the policy drives the sim, in the default and the
calibrated one, with no objects.

Starting from the real initial state of a held-out episode the policy
observes the *sim's* joint state (radians -> dataset units through the
variant's :class:`~antares.sim.UnitMap`), emits a chunk, the sim executes
the chunk's first action with the lock's latency, and so on for the length
of the episode. The policy is given the same goal (the real episode's grasp
point) and clock in every sim. Metrics per sim:

* the closed-loop trajectory against the real episode's (deg RMS over the
  body joints): did the arm go where the operator took it?
* the fingertip distance to the episode's logged grasp point at the frame
  the policy closes the gripper: did it reach where the real grasp happened?
* the share of held-out episodes where that distance is within
  :data:`SUCCESS_CM` (2 cm): a proxy success rate with no objects, plus the
  minimum fingertip distance to the grasp point over the whole run, for
  the episodes where the policy never closes.
* with an object (``objects=``, a :class:`~antares.task.KinematicObject`
  per episode, starting where the operator picked it), the task itself:
  did the policy pick it up, lift it, and let go of it within
  :data:`~antares.task.PLACE_CM` of where the operator put it down
  (:meth:`KinematicObject.result`). The object is carried by rule, not
  simulated (the sim has no contacts); an image policy sees it through
  :mod:`antares.simcam`, which renders the frame from ``ctx.q_rad`` and
  ``ctx.object_xyz``.

The point is to measure, not to force a difference: if the default and the
calibrated sim rank the policy the same, the numbers say so.

**Timing.** :meth:`antares.sim.Arm.rollout` applies at frame ``t`` the
command sent at ``t - L`` (``L`` the lock's latency) and the policy at
frame ``t`` sees the state the sim has reached at ``t``. The command it
emits cannot act on the step that produced the state it just read, so a
zero-latency servo is stepped with a latency of one frame in closed loop
(the observe -> command tick); any ``L >= 1`` is used as is. With the logged
commands fed back regardless of the observation (:class:`LoggedPolicy`)
the closed loop reproduces :meth:`Arm.replay` exactly, which the tests
check.

**Chunk shift.** A chunk covers ``H`` future commands, so a cheap lag
compensation is to send ``chunk[L]`` instead of ``chunk[0]``: the command
the policy meant for the frame at which this one will arrive
(:func:`antares.compensate.shift_compensate` applied inside the chunk).
``shift=True`` adds a ``<variant>+shift`` run for every variant with
``L >= 1``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from antares.data import Episode
from antares.events import gripper_state
from antares.gap import GapReport, aggregate, tracking_error
from antares.policy import GraspContext, Policy, StepContext, real_grasp
from antares.shadow import SIM_ONLY_NOTE, SimVariant, episode_goal
from antares.sim import GRIPPER_INDEX, JOINTS, Arm
from antares.task import KinematicObject, PlaceResult

#: Fingertip within this distance of the real grasp point when the policy
#: closes the gripper counts as reaching it.
SUCCESS_CM = 2.0


# --------------------------------------------------------------------------
# Stepping the sim one frame at a time
# --------------------------------------------------------------------------


class FrameStepper:
    """Advance an :class:`Arm` one dataset frame per call, like
    :meth:`Arm.rollout` does internally (same substeps, clipping, deadband).

    Use as a context manager so the model's timestep is restored. ``step``
    takes the command *applied* this frame (radians); the caller handles
    latency.
    """

    def __init__(self, arm: Arm, q0_rad: np.ndarray, fps: float):
        self.arm = arm
        m, d = arm.model, arm.data
        fps = float(fps)
        if fps <= 0:
            raise ValueError("fps must be positive")
        frame_dt = 1.0 / fps
        self.n_sub = max(1, int(round(frame_dt / arm._base_timestep)))
        self._old_dt = m.opt.timestep
        m.opt.timestep = frame_dt / self.n_sub
        q0 = np.asarray(q0_rad, dtype=np.float64).reshape(-1)
        if q0.shape[0] != m.nq:
            raise ValueError(f"q0_rad must have {m.nq} joints, got {q0.shape[0]}")
        self.lo, self.hi = arm.ctrl_range
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        d.ctrl[:] = np.clip(q0, self.lo, self.hi)
        mujoco.mj_forward(m, d)
        self.band = arm.params.per_joint("deadband")
        self.open = True

    def __enter__(self) -> FrameStepper:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.open:
            self.arm.model.opt.timestep = self._old_dt
            self.open = False

    @property
    def q(self) -> np.ndarray:
        return self.arm.data.qpos.copy()

    def step(self, ctrl_rad: np.ndarray) -> np.ndarray:
        """Apply ``ctrl_rad`` for one frame; return the joint angles after it (rad)."""
        m, d = self.arm.model, self.arm.data
        target = np.clip(np.asarray(ctrl_rad, dtype=np.float64).reshape(-1), self.lo, self.hi)
        if not np.any(self.band):
            d.ctrl[:] = target
            mujoco.mj_step(m, d, self.n_sub)
        else:
            idx = np.flatnonzero(self.band > 0)
            lo, hi, band = self.lo[idx], self.hi[idx], self.band[idx]
            want = target[idx]
            d.ctrl[:] = target
            for _ in range(self.n_sub):
                q = d.qpos[idx]
                e = want - q
                d.ctrl[idx] = np.clip(q + np.sign(e) * np.maximum(np.abs(e) - band, 0.0), lo, hi)
                mujoco.mj_step(m, d)
        return d.qpos.copy()


# --------------------------------------------------------------------------
# One episode in one sim
# --------------------------------------------------------------------------


def run_closed_loop(
    policy: Policy,
    arm: Arm,
    q0_rad: np.ndarray,
    n_frames: int,
    fps: float,
    exec_index: int = 0,
    goal: np.ndarray | None = None,
    obj: KinematicObject | None = None,
    episode: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive ``arm`` with ``policy`` for ``n_frames`` from ``q0_rad``.

    Returns ``(sim_q, actions)``: joint angles ``(T,6)`` rad with row 0 the
    start state, and the command the policy sent at each frame ``(T,6)`` in
    dataset units (``chunk[exec_index]``; the index is clipped to the
    chunk). The sim applies at frame ``t`` the command sent at ``t - L``
    with ``L = max(latency, 1)`` per joint; frames before the first command
    arrives hold the command sent at frame 0. ``goal`` is handed to the
    policy in every frame's :class:`StepContext`, with the sim's joint
    angles and, with ``obj``, the object's position (``obj`` is reset and
    stepped every frame with the gripper command the sim applies, and the
    policy reads the jaw through :meth:`KinematicObject.observe`).
    """
    T = int(n_frames)
    if T < 1:
        raise ValueError("n_frames must be >= 1")
    nq = arm.model.nq
    k = int(policy.k)
    lat = np.maximum(arm.params.per_joint("latency_frames"), 1)
    ex = int(np.clip(exec_index, 0, max(int(policy.h) - 1, 0)))
    um, cm = arm.unit_map, arm.command_map
    sim_q = np.empty((T, nq), dtype=np.float64)
    sent_units = np.empty((T, nq), dtype=np.float64)
    sent_rad = np.empty((T, nq), dtype=np.float64)
    policy.reset()
    if obj is not None:
        obj.reset()
    with FrameStepper(arm, q0_rad, fps) as st:
        q = st.q
        sim_q[0] = q
        oxyz = None if obj is None else obj.xyz.copy()

        def seen(q: np.ndarray) -> np.ndarray:
            u = um.to_units(q)
            return u if obj is None else obj.observe(u)

        hist = np.repeat(seen(q)[None, :], k, axis=0)  # (k,6) dataset units

        def decide(t: int) -> None:
            ctx = StepContext(t, fps, goal, q_rad=q.copy(), object_xyz=oxyz, episode=episode)
            chunk = np.asarray(policy.act(hist, ctx), dtype=np.float64)
            a = chunk[min(ex, chunk.shape[0] - 1)]
            sent_units[t] = a
            sent_rad[t] = um.to_rad(cm.apply(a))

        decide(0)
        rows = np.arange(nq)
        for t in range(1, T):
            src = np.maximum(t - lat, 0)
            applied = sent_rad[src, rows]
            q = st.step(applied)
            sim_q[t] = q
            if obj is not None:
                oxyz = obj.step(t, q, float(sent_units[src[GRIPPER_INDEX], GRIPPER_INDEX]))
            u = seen(q)
            hist = np.vstack([hist[1:], u[None, :]]) if k > 1 else u[None, :]
            decide(t)
    return sim_q, sent_units


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass
class ClosedLoopRun:
    """One episode driven by the policy in one sim variant."""

    variant: str
    episode: int
    n_frames: int
    sim_q: np.ndarray  # (T,6) rad
    actions: np.ndarray  # (T,6) dataset units
    gap: GapReport  # closed-loop trajectory vs the real one
    real_grasp_frame: int | None
    real_grasp_xyz: tuple[float, float, float] | None
    close_frame: int | None  # when the policy's gripper command first closes
    grasp_error_cm: float | None  # fingertip vs the real grasp point at close_frame
    min_tip_distance_cm: float | None  # over every frame
    min_tip_frame: int | None
    success: bool | None  # grasp_error_cm <= SUCCESS_CM; None without a real grasp
    final_error_deg: float  # body-joint RMS at the last frame
    approach_rms_deg: float | None = None  # trajectory RMS up to the real grasp frame
    exec_index: int = 0
    place: PlaceResult | None = None  # the object's fate, with ``objects=``

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "variant": self.variant,
            "episode": self.episode,
            "n_frames": self.n_frames,
            "trajectory_rms_deg": float(self.gap.overall_deg),
            "approach_rms_deg": self.approach_rms_deg,
            "per_joint_deg": dict(self.gap.per_joint_deg),
            "real_grasp_frame": self.real_grasp_frame,
            "real_grasp_xyz": None if self.real_grasp_xyz is None else list(self.real_grasp_xyz),
            "close_frame": self.close_frame,
            "grasp_error_cm": self.grasp_error_cm,
            "min_tip_distance_cm": self.min_tip_distance_cm,
            "min_tip_frame": self.min_tip_frame,
            "success": self.success,
            "final_error_deg": self.final_error_deg,
            "exec_index": self.exec_index,
            "place": None if self.place is None else self.place.to_dict(),
        }
        if include_arrays:
            d["sim_q_deg"] = np.degrees(self.sim_q).tolist()
            d["actions"] = self.actions.tolist()
        return d


def score_run(
    variant: SimVariant,
    arm: Arm,
    ep: Episode,
    sim_q: np.ndarray,
    actions: np.ndarray,
    ctx: GraspContext,
    exec_index: int = 0,
) -> ClosedLoopRun:
    """Trajectory and grasp-point metrics of one closed-loop run against the real episode."""
    real_q = arm.real_q(ep)
    gap = tracking_error(real_q, sim_q)
    grasp = real_grasp(ep, ctx, arm.unit_map)
    body_idx = [JOINTS.index(j) for j in gap.scored_joints]
    final = float(np.sqrt(np.mean(np.degrees(sim_q[-1, body_idx] - real_q[-1, body_idx]) ** 2)))
    close_frame = grasp_err = min_d = approach = None
    min_f = None
    success = None
    gxyz = None
    if grasp is not None:
        gxyz = tuple(float(v) for v in grasp.xyz)
        # the approach: the policy only knows where the pick is, so judge the
        # trajectory up to the operator's grasp separately from the whole episode
        upto = int(grasp.frame) + 1
        approach = float(tracking_error(real_q[:upto], sim_q[:upto]).overall_deg)
        tips = ctx.tip(sim_q)  # (T,3)
        dists = np.linalg.norm(tips - np.asarray(gxyz), axis=1) * 100.0
        min_f = int(np.argmin(dists))
        min_d = float(dists[min_f])
        s = gripper_state(actions[:, GRIPPER_INDEX], ctx.thresholds).astype(np.int64)
        closes = np.flatnonzero(np.diff(s) == -1) + 1
        if closes.size:
            close_frame = int(closes[0])
            grasp_err = float(dists[close_frame])
            success = bool(grasp_err <= SUCCESS_CM)
        else:
            success = False
    return ClosedLoopRun(
        variant=variant.name,
        episode=int(ep.index),
        n_frames=len(ep),
        sim_q=sim_q,
        actions=actions,
        gap=gap,
        real_grasp_frame=None if grasp is None else int(grasp.frame),
        real_grasp_xyz=gxyz,
        close_frame=close_frame,
        grasp_error_cm=grasp_err,
        min_tip_distance_cm=min_d,
        min_tip_frame=min_f,
        success=success,
        final_error_deg=final,
        approach_rms_deg=approach,
        exec_index=exec_index,
    )


# --------------------------------------------------------------------------
# Over the held-out split
# --------------------------------------------------------------------------


def _mean(xs: Sequence[float | None]) -> float | None:
    v = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


@dataclass
class ClosedLoopSummary:
    variants: list[SimVariant]
    runs: list[ClosedLoopRun]  # every (variant, episode)
    shift_variants: list[str] = field(default_factory=list)  # names of the "+shift" runs
    skipped: list[int] = field(default_factory=list)  # episodes without a grasp point
    note: str = SIM_ONLY_NOTE
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def run_names(self) -> list[str]:
        names = [v.name for v in self.variants]
        return names + [s for s in self.shift_variants if s not in names]

    def by_variant(self, name: str) -> list[ClosedLoopRun]:
        return [r for r in self.runs if r.variant == name]

    def aggregate(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_episodes": len({r.episode for r in self.runs}),
            "skipped_episodes": list(self.skipped),
            "variants": {},
        }
        for n in self.run_names:
            rs = self.by_variant(n)
            if not rs:
                continue
            judged = [r for r in rs if r.success is not None]
            closed = [r for r in judged if r.close_frame is not None]
            out["variants"][n] = {
                "trajectory_rms_deg": float(aggregate([r.gap for r in rs]).overall_deg),
                "approach_rms_deg": _mean([r.approach_rms_deg for r in rs]),
                "per_joint_deg": dict(aggregate([r.gap for r in rs]).per_joint_deg),
                "final_error_deg": _mean([r.final_error_deg for r in rs]),
                "grasp_error_cm": _mean([r.grasp_error_cm for r in closed]),
                "min_tip_distance_cm": _mean([r.min_tip_distance_cm for r in judged]),
                "close_frame_offset": _mean(
                    [r.close_frame - r.real_grasp_frame for r in closed if r.real_grasp_frame is not None]
                ),
                "n_judged": len(judged),
                "n_closed": len(closed),
                "n_success": int(sum(1 for r in judged if r.success)),
                "success_rate": (float(sum(1 for r in judged if r.success)) / len(judged)) if judged else None,
                "success_cm": SUCCESS_CM,
            }
            placed = [r.place for r in rs if r.place is not None]
            if placed:
                out["variants"][n].update(
                    n_objects=len(placed),
                    n_picked=sum(1 for p in placed if p.picked),
                    n_lifted=sum(1 for p in placed if p.lifted),
                    n_placed=sum(1 for p in placed if p.success),
                    place_rate=sum(1 for p in placed if p.success) / len(placed),
                    place_error_cm=_mean([p.place_error_cm for p in placed if p.released]),
                    lift_cm=_mean([p.lift_cm for p in placed if p.picked]),
                    place_cm=placed[0].place_cm,
                )
        return out

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "mode": "closedloop",
            "validation": "sim-only",
            "note": self.note,
            "success_cm": SUCCESS_CM,
            "variants": [v.to_dict() for v in self.variants],
            "run_names": self.run_names,
            "summary": self.aggregate(),
            "runs": [r.to_dict(include_arrays) for r in self.runs],
            **self.extra,
        }


def closed_loop_episodes(
    policy: Policy,
    episodes: Sequence[Episode],
    variants: Sequence[SimVariant],
    ctx: GraspContext,
    shift: bool = True,
    on_progress: Callable[[int, int, list[ClosedLoopRun]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    objects: Callable[[Episode], KinematicObject | None] | None = None,
) -> ClosedLoopSummary:
    """Run the policy closed-loop in every variant on every episode.

    With ``shift`` every variant with a latency of at least one frame also
    runs as ``<name>+shift``, executing ``chunk[L]`` (see the module
    docstring). Starts from each episode's real initial state; episodes
    without a grasp point are skipped for a policy that needs one.
    ``objects(ep)`` gives the episode's object (None for none), used fresh
    in every variant.
    """
    episodes = list(episodes)
    variants = list(variants)
    if not episodes:
        raise ValueError("no episodes to evaluate")
    if not variants:
        raise ValueError("no sim variants")
    arms = {v.name: v.arm() for v in variants}
    shift_names: list[str] = []
    plan: list[tuple[SimVariant, str, int]] = []
    for v in variants:
        plan.append((v, v.name, 0))
        if shift:
            L = int(np.min(v.latency_frames))
            if L >= 1 and policy.h > 1:
                name = f"{v.name}+shift"
                shift_names.append(name)
                plan.append((v, name, min(L, int(policy.h) - 1)))
    runs: list[ClosedLoopRun] = []
    skipped: list[int] = []
    for i, ep in enumerate(episodes):
        if should_stop is not None and should_stop():
            break
        goal = episode_goal(ep, ctx)
        if goal is None and getattr(policy, "needs_goal", False):
            skipped.append(int(ep.index))
            continue
        obj = objects(ep) if objects is not None else None
        batch: list[ClosedLoopRun] = []
        for v, name, ex in plan:
            arm = arms[v.name]
            q0 = arm.real_q(ep)[0]
            sim_q, actions = run_closed_loop(
                policy, arm, q0, len(ep), ep.fps, exec_index=ex, goal=goal, obj=obj, episode=int(ep.index)
            )
            run = score_run(v, arm, ep, sim_q, actions, ctx, exec_index=ex)
            run.variant = name
            if obj is not None:
                run.place = obj.result()
            batch.append(run)
        runs.extend(batch)
        if on_progress is not None:
            on_progress(i + 1, len(episodes), batch)
    return ClosedLoopSummary(variants, runs, shift_variants=shift_names, skipped=skipped)


__all__ = [
    "SUCCESS_CM",
    "ClosedLoopRun",
    "ClosedLoopSummary",
    "FrameStepper",
    "closed_loop_episodes",
    "run_closed_loop",
    "score_run",
]
