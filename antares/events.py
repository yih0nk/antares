"""The calibrated sim as a sensor: detect contact events from the residual.

Once the simulator reproduces the robot's free-space motion, whatever is
left in ``real - sim`` is something the sim does not model. In a
pick-and-place log that is mostly the arm touching, grasping or pushing an
object, so a sustained deviation is a contact event detected without any
force sensor.

Pipeline (pure numpy, no MuJoCo):

1. :func:`residuals` turns real and simulated joint trajectories into a
   signed per-joint residual in degrees.
2. :func:`residual_baseline` estimates, per joint, how large the residual
   is when nothing is happening: a robust centre and scale (median and MAD)
   plus the lag-1 autocorrelation, from the free-motion frames of held-out
   episodes. Free motion is read off the gripper command: frames away from
   any open/close transition while the gripper is not holding anything.
3. :func:`detect_events` z-scores a residual against the baseline and runs
   a hysteresis detector (enter at ``z``, exit at ``z/2``), merges nearby
   excursions and drops short ones.
4. The gripper command is also the proxy label: an open -> closed
   transition marks a grasp, so :func:`validate_episode` scores detected
   events against windows around the grasps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from antares.sim import JOINTS

GRIPPER_INDEX = JOINTS.index("gripper")
SCORED_JOINTS: tuple[str, ...] = tuple(j for j in JOINTS if j != "gripper")

# Frames around a gripper transition that count as "the grasp".
GRASP_PAD = 15
# A free-motion frame must be at least this far from any gripper transition.
FREE_MARGIN = 15
# Floor on the per-joint scale so a very clean joint cannot turn 0.1 deg
# into a z of 40.
MIN_SCALE_DEG = 0.1


# --------------------------------------------------------------------------
# residual
# --------------------------------------------------------------------------


def residuals(real_q: np.ndarray, sim_q: np.ndarray) -> np.ndarray:
    """Signed residual ``real - sim`` in degrees, (T, J). Inputs are radians."""
    real = np.asarray(real_q, dtype=np.float64)
    sim = np.asarray(sim_q, dtype=np.float64)
    if real.shape != sim.shape or real.ndim != 2:
        raise ValueError(f"expected matching (T, J) arrays, got {real.shape} and {sim.shape}")
    return np.degrees(real - sim)


# --------------------------------------------------------------------------
# gripper command as a proxy label
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GripperThresholds:
    """Hysteresis thresholds on the gripper command (dataset units, 0..100).

    The command is *closed* below ``closed_below`` and *open* above
    ``open_above``; in between it keeps its previous state.
    """

    closed_below: float
    open_above: float

    def to_dict(self) -> dict[str, float]:
        return {"closed_below": float(self.closed_below), "open_above": float(self.open_above)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GripperThresholds:
        return cls(float(d["closed_below"]), float(d["open_above"]))


def gripper_thresholds(cmd: np.ndarray, iters: int = 50) -> GripperThresholds:
    """Fit two clusters to the gripper command and put thresholds at 1/3 and 2/3.

    Pick-and-place commands are bimodal (jaw shut vs. jaw open); 2-means
    finds the two modes without assuming where on the 0..100 scale they sit.
    """
    x = np.asarray(cmd, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no gripper command values")
    lo, hi = np.percentile(x, 5.0), np.percentile(x, 95.0)
    if hi - lo < 1e-9:
        return GripperThresholds(float(lo), float(hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a, b = x[x <= mid], x[x > mid]
        if a.size == 0 or b.size == 0:
            break
        new_lo, new_hi = float(a.mean()), float(b.mean())
        if abs(new_lo - lo) < 1e-9 and abs(new_hi - hi) < 1e-9:
            break
        lo, hi = new_lo, new_hi
    gap = hi - lo
    return GripperThresholds(closed_below=lo + gap / 3.0, open_above=lo + 2.0 * gap / 3.0)


def gripper_state(cmd: np.ndarray, thr: GripperThresholds) -> np.ndarray:
    """Binarize the gripper command with hysteresis: 1 = open, 0 = closed, (T,)."""
    x = np.asarray(cmd, dtype=np.float64).ravel()
    T = x.size
    if T == 0:
        return np.zeros(0, dtype=np.int8)
    s = np.full(T, -1, dtype=np.int64)
    s[x > thr.open_above] = 1
    s[x < thr.closed_below] = 0
    if s[0] < 0:
        s[0] = 1 if x[0] > 0.5 * (thr.closed_below + thr.open_above) else 0
    # forward-fill undecided frames with the last decided state
    idx = np.where(s >= 0, np.arange(T), 0)
    idx = np.maximum.accumulate(idx)
    return s[idx].astype(np.int8)


def gripper_transitions(cmd: np.ndarray, thr: GripperThresholds) -> tuple[np.ndarray, np.ndarray]:
    """Frames where the command flips open -> closed (grasps) and closed -> open (releases)."""
    s = gripper_state(cmd, thr).astype(np.int64)
    d = np.diff(s)
    grasps = np.flatnonzero(d == -1) + 1
    releases = np.flatnonzero(d == 1) + 1
    return grasps, releases


def merge_windows(windows: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching half-open ``[start, end)`` windows."""
    out: list[tuple[int, int]] = []
    for s, e in sorted((int(s), int(e)) for s, e in windows):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def windows_around(frames: np.ndarray, n_frames: int, pad: int = GRASP_PAD) -> list[tuple[int, int]]:
    """``[f - pad, f + pad]`` as half-open windows clipped to ``[0, n_frames)``, merged."""
    return merge_windows(
        [(max(0, int(f) - pad), min(n_frames, int(f) + pad + 1)) for f in np.asarray(frames).ravel()]
    )


def grasp_windows(cmd: np.ndarray, thr: GripperThresholds, pad: int = GRASP_PAD) -> list[tuple[int, int]]:
    """Windows of ``+-pad`` frames around every open -> closed transition."""
    grasps, _ = gripper_transitions(cmd, thr)
    return windows_around(grasps, len(np.asarray(cmd).ravel()), pad)


def windows_mask(windows: Sequence[tuple[int, int]], n_frames: int) -> np.ndarray:
    m = np.zeros(n_frames, dtype=bool)
    for s, e in windows:
        m[max(0, s) : max(0, min(n_frames, e))] = True
    return m


def free_motion_mask(cmd: np.ndarray, thr: GripperThresholds, margin: int = FREE_MARGIN) -> np.ndarray:
    """Frames where the arm is moving freely: nothing in the gripper and the
    gripper command is not changing.

    "Holding" runs from an open -> closed transition to the next closed -> open
    one (or the end of the episode); frames within ``margin`` of any
    transition are excluded too, since the command settles before the jaw
    does.
    """
    x = np.asarray(cmd, dtype=np.float64).ravel()
    T = x.size
    grasps, releases = gripper_transitions(x, thr)
    holding = np.zeros(T, dtype=bool)
    for g in grasps:
        nxt = releases[releases > g]
        holding[g : (int(nxt[0]) if nxt.size else T)] = True
    near = windows_mask(windows_around(np.concatenate([grasps, releases]), T, margin), T)
    return ~holding & ~near


# --------------------------------------------------------------------------
# baseline
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualBaseline:
    """What the residual looks like when nothing is happening, per joint.

    ``center``/``scale`` are the median and ``1.4826 * MAD`` of the residual
    (degrees) over free-motion frames; ``acf1`` is its lag-1 autocorrelation
    there, i.e. how strongly consecutive frames of the residual move
    together (close to 1 means an excursion persists on its own for
    ``correlation_frames`` frames, so short events are not independent
    evidence).
    """

    joints: tuple[str, ...]
    center: np.ndarray  # (J,) deg
    scale: np.ndarray  # (J,) deg
    acf1: np.ndarray  # (J,)
    n_frames: int  # free-motion frames the estimate used
    n_episodes: int
    episodes: tuple[int, ...] = ()
    free_frames_only: bool = True

    def __post_init__(self) -> None:
        for name in ("center", "scale", "acf1"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        if not (self.center.shape == self.scale.shape == self.acf1.shape == (len(self.joints),)):
            raise ValueError("center, scale and acf1 must be (J,) with J = len(joints)")

    @property
    def correlation_frames(self) -> np.ndarray:
        """Decay time of the residual autocorrelation in frames, ``-1 / ln(acf1)``."""
        a = np.clip(self.acf1, 1e-6, 1 - 1e-6)
        return -1.0 / np.log(a)

    def z(self, residual: np.ndarray) -> np.ndarray:
        """Robust z-score of a (T, J) residual in degrees."""
        r = np.asarray(residual, dtype=np.float64)
        if r.ndim != 2 or r.shape[1] != len(self.joints):
            raise ValueError(f"residual must be (T, {len(self.joints)}), got {r.shape}")
        return (r - self.center) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "joints": list(self.joints),
            "center_deg": self.center.tolist(),
            "scale_deg": self.scale.tolist(),
            "acf1": self.acf1.tolist(),
            "correlation_frames": self.correlation_frames.tolist(),
            "n_frames": int(self.n_frames),
            "n_episodes": int(self.n_episodes),
            "episodes": [int(i) for i in self.episodes],
            "free_frames_only": bool(self.free_frames_only),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResidualBaseline:
        return cls(
            joints=tuple(d["joints"]),
            center=np.asarray(d["center_deg"], dtype=np.float64),
            scale=np.asarray(d["scale_deg"], dtype=np.float64),
            acf1=np.asarray(d["acf1"], dtype=np.float64),
            n_frames=int(d["n_frames"]),
            n_episodes=int(d["n_episodes"]),
            episodes=tuple(int(i) for i in d.get("episodes", ())),
            free_frames_only=bool(d.get("free_frames_only", True)),
        )


def _acf1(x: np.ndarray) -> float:
    x = x - x.mean()
    if x.size < 3:
        return 0.0
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / denom)


def baseline_from_residuals(
    residual_list: Sequence[np.ndarray],
    masks: Sequence[np.ndarray] | None = None,
    joints: Sequence[str] = JOINTS,
    episodes: Sequence[int] = (),
    min_scale_deg: float = MIN_SCALE_DEG,
    min_free_frames: int = 50,
) -> ResidualBaseline:
    """Robust per-joint statistics of residuals restricted to ``masks``.

    The autocorrelation is measured within each masked run (never across
    the gap between two runs or two episodes). If the masks leave fewer
    than ``min_free_frames`` frames, every frame is used instead and
    ``free_frames_only`` is False.
    """
    residual_list = [np.asarray(r, dtype=np.float64) for r in residual_list]
    if not residual_list:
        raise ValueError("no residuals")
    J = len(joints)
    for r in residual_list:
        if r.ndim != 2 or r.shape[1] != J:
            raise ValueError(f"residual must be (T, {J}), got {r.shape}")
    if masks is None:
        masks = [np.ones(len(r), dtype=bool) for r in residual_list]
    masks = [np.asarray(m, dtype=bool) for m in masks]
    free_only = True
    if sum(int(m.sum()) for m in masks) < min_free_frames:
        masks = [np.ones(len(r), dtype=bool) for r in residual_list]
        free_only = False

    kept = np.concatenate([r[m] for r, m in zip(residual_list, masks)])
    if kept.shape[0] == 0:
        raise ValueError("no frames to estimate a baseline from")
    center = np.median(kept, axis=0)
    mad = np.median(np.abs(kept - center), axis=0)
    scale = np.maximum(1.4826 * mad, min_scale_deg)

    # lag-1 autocorrelation within contiguous masked runs, frame-weighted
    num = np.zeros(J)
    wsum = 0.0
    for r, m in zip(residual_list, masks):
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(idx) != 1) + 1
        for run in np.split(idx, breaks):
            if run.size < 3:
                continue
            seg = r[run] - center
            num += np.array([_acf1(seg[:, j]) for j in range(J)]) * run.size
            wsum += run.size
    acf1 = num / wsum if wsum > 0 else np.zeros(J)

    return ResidualBaseline(
        joints=tuple(joints),
        center=center,
        scale=scale,
        acf1=np.clip(acf1, -1.0, 1.0),
        n_frames=int(kept.shape[0]),
        n_episodes=len(residual_list),
        episodes=tuple(int(i) for i in episodes),
        free_frames_only=free_only,
    )


def residual_baseline(
    episodes: Sequence[Any],
    arm: Any,
    thresholds: GripperThresholds | None = None,
    margin: int = FREE_MARGIN,
    min_scale_deg: float = MIN_SCALE_DEG,
) -> ResidualBaseline:
    """Replay ``episodes`` on ``arm`` and estimate the free-motion residual baseline.

    ``arm`` is anything with ``real_q(episode)`` and ``replay(episode)``
    returning (T, 6) radians, normally an :class:`antares.sim.Arm` carrying
    the lock's parameters. Use held-out episodes so the baseline is not
    flattered by the fit. ``thresholds`` default to ones fitted on these
    episodes' gripper commands.
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("no episodes")
    if thresholds is None:
        thresholds = gripper_thresholds(np.concatenate([ep.action[:, GRIPPER_INDEX] for ep in episodes]))
    res, masks = [], []
    for ep in episodes:
        res.append(residuals(arm.real_q(ep), arm.replay(ep)))
        masks.append(free_motion_mask(ep.action[:, GRIPPER_INDEX], thresholds, margin))
    return baseline_from_residuals(
        res, masks, joints=JOINTS, episodes=[ep.index for ep in episodes], min_scale_deg=min_scale_deg
    )


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """A sustained deviation of the real arm from the calibrated sim.

    ``[start, end)`` are frame indices (half-open). ``joints`` are the ones
    whose |z| crossed the entry threshold during the event; ``peak_z`` is
    the largest |z| over the scored joints and ``peak_deg`` the signed
    residual (real - sim, degrees) at that peak. ``kind`` is "contact" for a
    long, one-sided deviation (the arm is being held back or loaded) and
    "anomaly" for a brief or sign-flipping one (a glitch or oscillation).
    """

    start: int
    end: int
    joints: tuple[str, ...]
    peak_z: float
    peak_deg: float
    kind: str  # "contact" | "anomaly"

    @property
    def frames(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": int(self.start),
            "end": int(self.end),
            "joints": list(self.joints),
            "peak_z": float(self.peak_z),
            "peak_deg": float(self.peak_deg),
            "kind": self.kind,
        }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` runs of True in a boolean vector."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def hysteresis(x: np.ndarray, enter: float, exit: float) -> np.ndarray:
    """Boolean (T,) mask that turns on when ``x >= enter`` and off when ``x < exit``."""
    x = np.asarray(x, dtype=np.float64)
    T = x.size
    if T == 0:
        return np.zeros(0, dtype=bool)
    if exit > enter:
        raise ValueError("exit threshold must not exceed enter threshold")
    s = np.full(T, -1, dtype=np.int64)
    s[x >= enter] = 1
    s[x < exit] = 0
    if s[0] < 0:
        s[0] = 0
    idx = np.maximum.accumulate(np.where(s >= 0, np.arange(T), 0))
    return s[idx] == 1


def detect_events(
    residual: np.ndarray,
    baseline: ResidualBaseline,
    z: float = 4.0,
    min_frames: int = 5,
    joints: Sequence[str] = SCORED_JOINTS,
    merge_frames: int = 3,
    contact_min_frames: int = 10,
    one_sided: float = 0.8,
    warmup: int = 0,
) -> list[Event]:
    """Find sustained deviations of a (T, J) residual (degrees) from ``baseline``.

    Per scored joint, |z| enters an excursion at ``z`` and leaves at ``z/2``.
    Excursions of any scored joint are unioned over time, gaps of at most
    ``merge_frames`` are bridged, and runs shorter than ``min_frames`` are
    dropped. An event is a "contact" when it lasts at least
    ``contact_min_frames`` and the peak joint's residual keeps the sign of
    its peak for at least ``one_sided`` of the event; otherwise "anomaly".
    The first ``warmup`` frames are ignored: the sim starts at rest at the
    real pose, so its first few frames are a start-up transient.
    """
    if z <= 0:
        raise ValueError("z must be positive")
    if min_frames < 1:
        raise ValueError("min_frames must be >= 1")
    r = np.asarray(residual, dtype=np.float64)
    zs = baseline.z(r)
    names = list(baseline.joints)
    scored = [names.index(j) for j in joints]
    if not scored:
        raise ValueError("no joints to score")
    T = r.shape[0]
    if T == 0:
        return []

    az = np.abs(zs[:, scored])  # (T, S)
    active = np.zeros(T, dtype=bool)
    for k in range(len(scored)):
        active |= hysteresis(az[:, k], z, z / 2.0)
    if warmup > 0:
        active[:warmup] = False
        az = az.copy()
        az[:warmup] = 0.0

    runs = _runs(active)
    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= merge_frames:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    events: list[Event] = []
    for s, e in merged:
        if e - s < min_frames:
            continue
        seg = az[s:e]  # (n, S)
        crossed = tuple(joints[k] for k in range(len(scored)) if np.any(seg[:, k] >= z))
        if not crossed:
            continue  # an exit-only tail left over after merging; cannot happen in practice
        t_rel, k = np.unravel_index(int(np.argmax(seg)), seg.shape)
        jpeak = scored[int(k)]
        peak_z = float(seg[t_rel, k])
        peak_deg = float(r[s + int(t_rel), jpeak])
        sign = np.sign(r[s:e, jpeak] - baseline.center[jpeak])
        frac_same = float(np.mean(sign == np.sign(peak_deg - baseline.center[jpeak]))) if peak_deg != baseline.center[jpeak] else 0.0
        kind = "contact" if (e - s >= contact_min_frames and frac_same >= one_sided) else "anomaly"
        events.append(Event(start=s, end=e, joints=crossed, peak_z=peak_z, peak_deg=peak_deg, kind=kind))
    return events


# --------------------------------------------------------------------------
# validation against the gripper proxy label
# --------------------------------------------------------------------------


def _overlaps(a: tuple[int, int], windows: Sequence[tuple[int, int]]) -> bool:
    return any(a[0] < e and s < a[1] for s, e in windows)


@dataclass(frozen=True)
class EpisodeValidation:
    episode: int
    n_frames: int
    n_grasps: int
    n_grasps_hit: int  # grasps with a detected event inside their window
    n_events: int
    n_events_in_grasp: int  # events overlapping a grasp window
    n_events_near_transition: int  # events overlapping any transition window
    n_events_far: int  # events far from every gripper transition
    n_frames_in_grasp: int
    energy_total: float  # sum of residual^2 over scored joints, deg^2
    energy_in_grasp: float
    grasp_windows: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        d = {k: (list(map(list, v)) if k == "grasp_windows" else v) for k, v in self.__dict__.items()}
        return d


def validate_episode(
    residual: np.ndarray,
    events: Sequence[Event],
    gripper_cmd: np.ndarray,
    thresholds: GripperThresholds,
    episode: int = -1,
    pad: int = GRASP_PAD,
    joints: Sequence[str] = SCORED_JOINTS,
    joint_names: Sequence[str] = JOINTS,
) -> EpisodeValidation:
    """Compare detected events with the gripper command's open -> closed transitions."""
    r = np.asarray(residual, dtype=np.float64)
    T = r.shape[0]
    cmd = np.asarray(gripper_cmd, dtype=np.float64).ravel()
    if cmd.size != T:
        raise ValueError("gripper_cmd must have one value per residual frame")
    grasps, releases = gripper_transitions(cmd, thresholds)
    gw = windows_around(grasps, T, pad)
    tw = windows_around(np.concatenate([grasps, releases]), T, pad)
    spans = [(ev.start, ev.end) for ev in events]

    names = list(joint_names)
    cols = [names.index(j) for j in joints]
    e2 = np.sum(r[:, cols] ** 2, axis=1)
    gmask = windows_mask(gw, T)

    hits = 0
    for g in grasps:
        w = (max(0, int(g) - pad), min(T, int(g) + pad + 1))
        hits += int(_overlaps(w, spans))
    return EpisodeValidation(
        episode=int(episode),
        n_frames=T,
        n_grasps=int(grasps.size),
        n_grasps_hit=hits,
        n_events=len(spans),
        n_events_in_grasp=sum(_overlaps(s, gw) for s in spans),
        n_events_near_transition=sum(_overlaps(s, tw) for s in spans),
        n_events_far=sum(not _overlaps(s, tw) for s in spans),
        n_frames_in_grasp=int(gmask.sum()),
        energy_total=float(e2.sum()),
        energy_in_grasp=float(e2[gmask].sum()),
        grasp_windows=tuple(gw),
    )


def summarize_validation(items: Sequence[EpisodeValidation]) -> dict[str, Any]:
    """Pool per-episode counts into recall/precision-like rates.

    ``recall``: grasps whose window contains a detected event.
    ``precision``: events overlapping a grasp window; ``near_transition``
    relaxes that to any gripper transition. ``far_fraction`` is the
    complement, the false-ish positives. ``energy_in_grasp_fraction`` is
    the share of residual energy inside grasp windows; ``chance`` is the
    share of frames there, so ``lift`` > 1 means the residual concentrates
    around grasps.
    """
    items = list(items)

    def tot(name: str) -> float:
        return float(sum(getattr(v, name) for v in items))

    def ratio(a: float, b: float) -> float | None:
        return None if b <= 0 else a / b

    n_grasps, n_events = tot("n_grasps"), tot("n_events")
    energy_frac = ratio(tot("energy_in_grasp"), tot("energy_total"))
    chance = ratio(tot("n_frames_in_grasp"), tot("n_frames"))
    return {
        "n_episodes": len(items),
        "n_frames": int(tot("n_frames")),
        "n_grasps": int(n_grasps),
        "n_grasps_hit": int(tot("n_grasps_hit")),
        "n_events": int(n_events),
        "n_events_in_grasp": int(tot("n_events_in_grasp")),
        "n_events_near_transition": int(tot("n_events_near_transition")),
        "n_events_far": int(tot("n_events_far")),
        "recall": ratio(tot("n_grasps_hit"), n_grasps),
        "precision": ratio(tot("n_events_in_grasp"), n_events),
        "near_transition": ratio(tot("n_events_near_transition"), n_events),
        "far_fraction": ratio(tot("n_events_far"), n_events),
        "energy_in_grasp_fraction": energy_frac,
        "chance": chance,
        "lift": None if energy_frac is None or not chance else energy_frac / chance,
    }


__all__ = [
    "Event",
    "EpisodeValidation",
    "GripperThresholds",
    "ResidualBaseline",
    "SCORED_JOINTS",
    "baseline_from_residuals",
    "detect_events",
    "free_motion_mask",
    "grasp_windows",
    "gripper_state",
    "gripper_thresholds",
    "gripper_transitions",
    "hysteresis",
    "residual_baseline",
    "residuals",
    "summarize_validation",
    "validate_episode",
    "windows_around",
]
