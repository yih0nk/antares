"""Open-loop shadow evaluation: the policy's commands, rolled through each sim.

For a held-out episode the policy is fed the *real* observations frame by
frame and its first predicted action per frame (receding horizon) is
collected, which is what it would have sent had it been driving. Those
commands are then rolled through :meth:`antares.sim.Arm.rollout` from the
real initial state in the default sim and in the calibrated lock(s). What
comes out:

* **policy vs logged**: how far the policy's commands are from what the
  operator sent (deg RMS over the body joints), the plain imitation error;
* **sim vs real**, per sim: how far each sim's trajectory under the policy's
  commands is from the real trajectory;
* **default vs lock**: how far the two sims' trajectories are from *each
  other* under the same commands. That number is how much the stock sim
  misjudges this policy: it is zero only if calibration did not matter for
  what the policy does;
* where they disagree most (per joint, and the worst frames), and the
  fingertip position error at the real grasp instants (FK at the frames the
  operator closed the gripper): would the stock sim have predicted the
  grasp at the right place?

Pure kinematics and dynamics in the sim; nothing is driven on hardware.
Angles are radians internally and degrees in every report, distances in
centimetres.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from antares.data import Episode
from antares.events import GripperThresholds, gripper_transitions
from antares.gap import GapReport, aggregate, tracking_error
from antares.policy import GraspContext, Policy, StepContext, real_grasp
from antares.robots import DEFAULT_ROBOT, get_robot
from antares.sim import GRIPPER_INDEX, JOINTS, Arm, CommandMap, SimParams, UnitMap, load_model

#: How many frames of largest default-vs-lock disagreement to list per episode.
N_WORST_FRAMES = 5

SIM_ONLY_NOTE = "Sim-only: the policy's commands are rolled through MuJoCo; no hardware was driven."


# --------------------------------------------------------------------------
# Which sims
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SimVariant:
    """One simulator to judge the policy in: parameters, unit map, command map."""

    name: str  # "default", "lock", "perjoint" or a lock name
    params: SimParams
    robot: str = DEFAULT_ROBOT
    command_map: CommandMap | None = None
    unit_map: UnitMap | None = None  # None: the model's own
    lock: str | None = None  # the lock this came from, if any
    label: str = ""

    @property
    def latency_frames(self) -> np.ndarray:
        return self.params.per_joint("latency_frames")

    def arm(self) -> Arm:
        """A private :class:`Arm` configured for this variant."""
        return Arm(self.params, unit_map=self.unit_map, command_map=self.command_map, robot=self.robot)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label or self.name,
            "lock": self.lock,
            "robot": self.robot,
            "params": self.params.to_dict(),
            "latency_frames": self.latency_frames.tolist(),
            "command_map": None if self.command_map is None else self.command_map.to_dict(),
        }


def default_variant(robot: str | None = None) -> SimVariant:
    return SimVariant("default", SimParams.default(), robot=get_robot(robot).name, label="default sim")


def variant_from_lock(lock: dict[str, Any], name: str, label: str | None = None) -> SimVariant:
    """A variant with a lock dict's parameters, command map and calibration-derived unit map."""
    from antares.lerobot_calib import unit_map_from_record

    robot = get_robot(lock.get("robot")).name
    params = SimParams.from_dict(lock["params"])
    cmap = CommandMap.from_dict(lock["command_map"]) if lock.get("command_map") else None
    unit_map = unit_map_from_record(lock.get("calibration"), load_model(get_robot(robot).model_path))
    return SimVariant(
        name,
        params,
        robot=robot,
        command_map=cmap,
        unit_map=unit_map,
        lock=str(lock.get("name") or name),
        label=label or f"lock {lock.get('name') or name}",
    )


def close_frames(cmd: np.ndarray, thr: GripperThresholds) -> list[int]:
    """Frames where a gripper command flips open -> closed."""
    g, _ = gripper_transitions(np.asarray(cmd, dtype=np.float64), thr)
    return [int(f) for f in g]


def episode_goal(ep: Episode, ctx: GraspContext) -> np.ndarray | None:
    """The episode's grasp point (metres, base frame) through the context's unit map."""
    g = real_grasp(ep, ctx)
    return None if g is None else np.asarray(g.xyz, dtype=np.float64)


# --------------------------------------------------------------------------
# The policy's commands on the real observations
# --------------------------------------------------------------------------


def shadow_actions(policy: Policy, ep: Episode, goal: np.ndarray | None = None) -> np.ndarray:
    """First action of the policy's chunk at every frame, fed the real states.

    Observation windows pad the start by repeating the first frame, as
    training did; ``goal`` is the episode's grasp point for a policy that
    takes one. ``(T,6)`` in dataset units.
    """
    state = np.asarray(ep.state, dtype=np.float64)
    T = state.shape[0]
    k = int(policy.k)
    oi = np.clip(np.arange(T)[:, None] - np.arange(k)[::-1][None, :], 0, T - 1)
    obs = state[oi]  # (T,k,6)
    ctxs = [StepContext(t, ep.fps, goal) for t in range(T)]
    policy.reset()
    batch = getattr(policy, "act_batch", None)
    features = getattr(policy, "features", None)
    if callable(batch) and features is not None:
        extras = np.stack([features.extras(c) for c in ctxs]) if features.n_extra else None
        return np.asarray(batch(obs, extras), dtype=np.float64)[:, 0, :]
    return np.stack([np.asarray(policy.act(obs[t], ctxs[t]), dtype=np.float64)[0] for t in range(T)])


# --------------------------------------------------------------------------
# Per episode
# --------------------------------------------------------------------------


def _deg_rms_frames(a: np.ndarray, b: np.ndarray, scored_idx: Sequence[int]) -> np.ndarray:
    """Per-frame RMS difference over the scored joints, degrees. (T,)"""
    d = np.degrees(a[:, scored_idx] - b[:, scored_idx])
    return np.sqrt(np.mean(d**2, axis=1))


def _worst_frames(per_frame: np.ndarray, n: int = N_WORST_FRAMES) -> list[dict[str, float | int]]:
    order = np.argsort(-per_frame)[:n]
    return [{"frame": int(i), "deg": float(per_frame[i])} for i in order]


def _per_joint_peak(a: np.ndarray, b: np.ndarray) -> dict[str, dict[str, float | int]]:
    d = np.abs(np.degrees(a - b))
    return {j: {"frame": int(np.argmax(d[:, i])), "deg": float(d[:, i].max())} for i, j in enumerate(JOINTS)}


def _cm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) * 100.0)


@dataclass
class ShadowEpisode:
    episode: int
    fps: float
    n_frames: int
    actions: np.ndarray  # policy's first actions, (T,6) dataset units
    action_gap: GapReport  # policy commands vs logged commands (as angles)
    sim_q: dict[str, np.ndarray]  # variant -> (T,6) rad under the policy's commands
    real_q: dict[str, np.ndarray]  # variant -> (T,6) rad (its unit map)
    gaps: dict[str, GapReport]  # variant -> sim vs real
    replay_gaps: dict[str, GapReport]  # variant -> sim with the *logged* commands vs real
    divergence: dict[str, GapReport]  # variant -> sim vs the baseline variant (same commands)
    worst_frames: dict[str, list[dict[str, float | int]]]
    per_joint_peak: dict[str, dict[str, dict[str, float | int]]]
    grasp_frames: list[int]  # real close frames
    grasp_errors: dict[str, list[float]]  # variant -> fingertip error vs real at each close, cm
    grasp_divergence: dict[str, list[float]]  # variant -> fingertip distance to the baseline sim at each close, cm
    goal: np.ndarray | None = None
    baseline: str = "default"

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "episode": self.episode,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "goal": None if self.goal is None else [float(v) for v in self.goal],
            "action_error_deg": float(self.action_gap.overall_deg),
            "action_error_per_joint_deg": dict(self.action_gap.per_joint_deg),
            "sim_vs_real_deg": {k: float(g.overall_deg) for k, g in self.gaps.items()},
            "replay_vs_real_deg": {k: float(g.overall_deg) for k, g in self.replay_gaps.items()},
            "divergence_deg": {k: float(g.overall_deg) for k, g in self.divergence.items()},
            "divergence_per_joint_deg": {k: dict(g.per_joint_deg) for k, g in self.divergence.items()},
            "worst_frames": self.worst_frames,
            "per_joint_peak": self.per_joint_peak,
            "grasp_frames": list(self.grasp_frames),
            "grasp_errors_cm": {k: [float(x) for x in v] for k, v in self.grasp_errors.items()},
            "grasp_divergence_cm": {k: [float(x) for x in v] for k, v in self.grasp_divergence.items()},
            "baseline": self.baseline,
        }
        if include_arrays:
            d["actions"] = self.actions.tolist()
            d["sim_q_deg"] = {k: np.degrees(v).tolist() for k, v in self.sim_q.items()}
            d["real_q_deg"] = {k: np.degrees(v).tolist() for k, v in self.real_q.items()}
        return d


def shadow_episode(
    policy: Policy,
    ep: Episode,
    variants: Sequence[SimVariant],
    ctx: GraspContext,
    arms: dict[str, Arm] | None = None,
    baseline: str = "default",
) -> ShadowEpisode:
    """Run the policy on the real observations and roll its commands through every variant.

    Raises ``ValueError`` when the policy needs a goal and the episode has
    no grasp.
    """
    variants = list(variants)
    if not variants:
        raise ValueError("no sim variants")
    names = [v.name for v in variants]
    if baseline not in names:
        baseline = names[0]
    arms = arms if arms is not None else {}
    goal = episode_goal(ep, ctx)
    if goal is None and getattr(policy, "needs_goal", False):
        raise ValueError(f"episode {ep.index} has no grasp to give the policy as its goal")
    actions = shadow_actions(policy, ep, goal)
    fps = float(ep.fps)
    sim_q: dict[str, np.ndarray] = {}
    real_q: dict[str, np.ndarray] = {}
    gaps: dict[str, GapReport] = {}
    replay_gaps: dict[str, GapReport] = {}
    for v in variants:
        arm = arms.get(v.name)
        if arm is None:
            arm = arms[v.name] = v.arm()
        rq = arm.real_q(ep)
        ctrl = arm.unit_map.to_rad(arm.command_map.apply(actions))
        sq = arm.rollout(ctrl, rq[0], fps)
        real_q[v.name], sim_q[v.name] = rq, sq
        gaps[v.name] = tracking_error(rq, sq)
        replay_gaps[v.name] = tracking_error(rq, arm.replay(ep))
    # Imitation error, as angles through the baseline's unit map.
    um = arms[baseline].unit_map
    action_gap = tracking_error(um.to_rad(ep.action), um.to_rad(actions))
    scored_idx = [JOINTS.index(j) for j in gaps[baseline].scored_joints]
    divergence, worst, peaks = {}, {}, {}
    for v in variants:
        if v.name == baseline:
            continue
        divergence[v.name] = tracking_error(sim_q[baseline], sim_q[v.name])
        per_frame = _deg_rms_frames(sim_q[v.name], sim_q[baseline], scored_idx)
        worst[v.name] = _worst_frames(per_frame)
        peaks[v.name] = _per_joint_peak(sim_q[v.name], sim_q[baseline])
    # Fingertip at the real grasp instants: where did each sim think the jaws were?
    frames = close_frames(ep.action[:, GRIPPER_INDEX], ctx.thresholds)
    grasp_errors: dict[str, list[float]] = {v.name: [] for v in variants}
    grasp_div: dict[str, list[float]] = {v.name: [] for v in variants if v.name != baseline}
    for g in frames:
        tips = {v.name: ctx.tip(sim_q[v.name][g]) for v in variants}
        for v in variants:
            grasp_errors[v.name].append(_cm(tips[v.name], ctx.tip(real_q[v.name][g])))
            if v.name != baseline:
                grasp_div[v.name].append(_cm(tips[v.name], tips[baseline]))
    return ShadowEpisode(
        episode=int(ep.index),
        fps=fps,
        n_frames=len(ep),
        actions=actions,
        action_gap=action_gap,
        sim_q=sim_q,
        real_q=real_q,
        gaps=gaps,
        replay_gaps=replay_gaps,
        divergence=divergence,
        worst_frames=worst,
        per_joint_peak=peaks,
        grasp_frames=frames,
        grasp_errors=grasp_errors,
        grasp_divergence=grasp_div,
        goal=goal,
        baseline=baseline,
    )


# --------------------------------------------------------------------------
# Over the held-out split
# --------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float | None:
    xs = [float(x) for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


@dataclass
class ShadowSummary:
    mode: str
    variants: list[SimVariant]
    episodes: list[ShadowEpisode]
    baseline: str = "default"
    skipped: list[int] = field(default_factory=list)  # episodes without a grasp point
    note: str = SIM_ONLY_NOTE
    extra: dict[str, Any] = field(default_factory=dict)

    def aggregate(self) -> dict[str, Any]:
        eps = self.episodes
        names = [v.name for v in self.variants]
        out: dict[str, Any] = {
            "n_episodes": len(eps),
            "n_frames": int(sum(e.n_frames for e in eps)),
            "skipped_episodes": list(self.skipped),
            "action_error_deg": float(aggregate([e.action_gap for e in eps]).overall_deg) if eps else None,
            "action_error_per_joint_deg": dict(aggregate([e.action_gap for e in eps]).per_joint_deg) if eps else {},
            "variants": {},
        }
        for n in names:
            gaps = [e.gaps[n] for e in eps]
            rep = [e.replay_gaps[n] for e in eps]
            row: dict[str, Any] = {
                "sim_vs_real_deg": float(aggregate(gaps).overall_deg) if gaps else None,
                "sim_vs_real_per_joint_deg": dict(aggregate(gaps).per_joint_deg) if gaps else {},
                "replay_vs_real_deg": float(aggregate(rep).overall_deg) if rep else None,
                "grasp_error_cm": _mean([x for e in eps for x in e.grasp_errors.get(n, [])]),
                "n_grasps": int(sum(len(e.grasp_errors.get(n, [])) for e in eps)),
            }
            if n != self.baseline:
                div = [e.divergence[n] for e in eps if n in e.divergence]
                row["divergence_deg"] = float(aggregate(div).overall_deg) if div else None
                row["divergence_per_joint_deg"] = dict(aggregate(div).per_joint_deg) if div else {}
                row["grasp_divergence_cm"] = _mean([x for e in eps for x in e.grasp_divergence.get(n, [])])
                worst = sorted(
                    ((e.episode, w["frame"], w["deg"]) for e in eps for w in e.worst_frames.get(n, [])),
                    key=lambda t: -t[2],
                )[:N_WORST_FRAMES]
                row["worst_frames"] = [{"episode": a, "frame": b, "deg": c} for a, b, c in worst]
                pj = {}
                for j in JOINTS:
                    best = max(
                        ((e.episode, e.per_joint_peak[n][j]["frame"], e.per_joint_peak[n][j]["deg"]) for e in eps if n in e.per_joint_peak),
                        key=lambda t: t[2],
                        default=None,
                    )
                    if best is not None:
                        pj[j] = {"episode": best[0], "frame": best[1], "deg": best[2]}
                row["per_joint_peak"] = pj
            out["variants"][n] = row
        return out

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "baseline": self.baseline,
            "validation": "sim-only",
            "note": self.note,
            "variants": [v.to_dict() for v in self.variants],
            "summary": self.aggregate(),
            "episodes": [e.to_dict(include_arrays) for e in self.episodes],
            **self.extra,
        }


def shadow_episodes(
    policy: Policy,
    episodes: Sequence[Episode],
    variants: Sequence[SimVariant],
    ctx: GraspContext,
    baseline: str = "default",
    on_progress: Callable[[int, int, ShadowEpisode], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ShadowSummary:
    """:func:`shadow_episode` over a split, with one Arm per variant reused.
    Episodes without a grasp point are skipped for a policy that needs one."""
    episodes = list(episodes)
    if not episodes:
        raise ValueError("no episodes to evaluate")
    arms: dict[str, Arm] = {}
    out: list[ShadowEpisode] = []
    skipped: list[int] = []
    for i, ep in enumerate(episodes):
        if should_stop is not None and should_stop():
            break
        if getattr(policy, "needs_goal", False) and episode_goal(ep, ctx) is None:
            skipped.append(int(ep.index))
            continue
        se = shadow_episode(policy, ep, variants, ctx, arms, baseline)
        out.append(se)
        if on_progress is not None:
            on_progress(i + 1, len(episodes), se)
    return ShadowSummary("shadow", list(variants), out, baseline=baseline, skipped=skipped)


__all__ = [
    "N_WORST_FRAMES",
    "SIM_ONLY_NOTE",
    "GraspContext",
    "ShadowEpisode",
    "ShadowSummary",
    "SimVariant",
    "close_frames",
    "default_variant",
    "episode_goal",
    "real_grasp",
    "shadow_actions",
    "shadow_episode",
    "shadow_episodes",
    "variant_from_lock",
]
