"""Fit simulator parameters to one robot's logs with CMA-ES.

The search space is six scalars of :class:`antares.sim.SimParams`
(actuator gain and velocity gain, joint damping, armature, friction loss,
action latency). CMA-ES works on a normalized ``[0, 1]^6`` vector; the
physical parameters live on log scales because they span orders of
magnitude, and latency is an integer number of frames encoded as a
continuous dimension and rounded on decode.

The objective is the overall tracking error (degrees) of the simulated
replay against the real trajectory, aggregated over the training episodes.
Held-out episodes are scored once before and once after the fit. Every
objective evaluation emits a :class:`ProgressEvent` so a UI can stream the
fit live.

Simulation is injected as a ``simulate(params, episode) -> (real_q, sim_q)``
callable so this module (and its tests) do not need MuJoCo.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from antares.data import Episode

# --------------------------------------------------------------------------
# Lazy imports of modules written in parallel. The fallbacks only exist so
# that this module can be imported and tested with a fake simulate function
# before antares.sim / antares.gap land; they mirror the agreed contract.
# --------------------------------------------------------------------------

try:  # pragma: no cover - which branch runs depends on sibling modules
    from antares.sim import SimParams
except (ImportError, AttributeError):  # pragma: no cover

    @dataclass(frozen=True)
    class SimParams:  # type: ignore[no-redef]
        kp: float = 998.22
        kv: float = 2.731
        damping: float = 0.6
        armature: float = 0.028
        frictionloss: float = 0.052
        latency_frames: int = 0

        def to_dict(self) -> dict:
            return {
                "kp": float(self.kp),
                "kv": float(self.kv),
                "damping": float(self.damping),
                "armature": float(self.armature),
                "frictionloss": float(self.frictionloss),
                "latency_frames": int(self.latency_frames),
            }

        @classmethod
        def from_dict(cls, d: dict) -> "SimParams":
            return cls(
                kp=float(d["kp"]),
                kv=float(d["kv"]),
                damping=float(d["damping"]),
                armature=float(d["armature"]),
                frictionloss=float(d["frictionloss"]),
                latency_frames=int(d["latency_frames"]),
            )

        @classmethod
        def default(cls) -> "SimParams":
            return cls()


def _gap_functions() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return ``(tracking_error, aggregate)`` from :mod:`antares.gap`, or
    minimal stand-ins with the same shape if that module is not ready."""
    try:
        from antares.gap import aggregate, tracking_error

        return tracking_error, aggregate
    except (ImportError, AttributeError):
        return _fallback_tracking_error, _fallback_aggregate


_FALLBACK_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class _FallbackGapReport:
    per_joint_deg: dict[str, float]
    overall_deg: float
    scored_joints: tuple[str, ...]
    n_frames: int

    def to_dict(self) -> dict:
        return {
            "per_joint_deg": dict(self.per_joint_deg),
            "overall_deg": float(self.overall_deg),
            "scored_joints": list(self.scored_joints),
            "n_frames": int(self.n_frames),
        }


def _fallback_tracking_error(
    real_q: np.ndarray,
    sim_q: np.ndarray,
    joint_names: Sequence[str] = _FALLBACK_JOINTS,
    exclude: Sequence[str] = ("gripper",),
) -> _FallbackGapReport:
    real_q = np.asarray(real_q, dtype=np.float64)
    sim_q = np.asarray(sim_q, dtype=np.float64)
    n = min(len(real_q), len(sim_q))
    err_deg = np.degrees(sim_q[:n] - real_q[:n])
    rms = np.sqrt(np.mean(err_deg**2, axis=0)) if n else np.full(err_deg.shape[1], np.nan)
    per_joint = {name: float(rms[i]) for i, name in enumerate(joint_names) if i < len(rms)}
    scored = tuple(j for j in joint_names if j not in exclude and j in per_joint)
    overall = float(np.mean([per_joint[j] for j in scored])) if scored else float("nan")
    return _FallbackGapReport(per_joint, overall, scored, n)


def _fallback_aggregate(reports: Sequence[Any]) -> _FallbackGapReport:
    reports = list(reports)
    if not reports:
        return _FallbackGapReport({}, float("nan"), (), 0)
    weights = np.array([max(r.n_frames, 1) for r in reports], dtype=np.float64)
    joints = list(reports[0].per_joint_deg)
    per_joint = {
        j: float(
            np.sqrt(np.sum(weights * np.array([r.per_joint_deg[j] ** 2 for r in reports])) / weights.sum())
        )
        for j in joints
    }
    scored = reports[0].scored_joints
    overall = float(np.mean([per_joint[j] for j in scored])) if scored else float("nan")
    return _FallbackGapReport(per_joint, overall, scored, int(weights.sum()))


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------

#: name -> (lo, hi, scale). ``log`` dims are searched in log10, ``int`` dims
#: are continuous during the search and rounded on decode.
SPACE: dict[str, tuple[float, float, str]] = {
    "kp": (50.0, 5000.0, "log"),
    "kv": (0.1, 50.0, "log"),
    "damping": (0.01, 10.0, "log"),
    "armature": (1e-4, 0.5, "log"),
    "frictionloss": (1e-3, 2.0, "log"),
    "latency_frames": (0.0, 5.0, "int"),
}

DIMS: tuple[str, ...] = tuple(SPACE)
N_DIMS = len(DIMS)

#: Objective value used when the simulator returns NaN/inf (unstable sim).
PENALTY = 1e3


def _to_unit(value: float, lo: float, hi: float, scale: str) -> float:
    value = min(max(float(value), lo), hi)
    if scale == "log":
        return (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return (value - lo) / (hi - lo)


def _from_unit(x: float, lo: float, hi: float, scale: str) -> float | int:
    x = min(max(float(x), 0.0), 1.0)
    if scale == "log":
        v = 10 ** (math.log10(lo) + x * (math.log10(hi) - math.log10(lo)))
        return float(min(max(v, lo), hi))
    v = lo + x * (hi - lo)
    if scale == "int":
        return int(min(max(round(v), lo), hi))
    return float(v)


def encode(params: SimParams) -> np.ndarray:
    """Map parameters to the normalized ``[0, 1]^6`` search vector (clipped)."""
    return np.array(
        [_to_unit(getattr(params, name), *SPACE[name]) for name in DIMS],
        dtype=np.float64,
    )


def decode(x: np.ndarray) -> SimParams:
    """Map a search vector back to parameters, clipping to the bounds."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] != N_DIMS:
        raise ValueError(f"expected {N_DIMS} dims, got {x.shape[0]}")
    kwargs = {name: _from_unit(x[i], *SPACE[name]) for i, name in enumerate(DIMS)}
    return SimParams(**kwargs)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

SimulateFn = Callable[[SimParams, Episode], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class ProgressEvent:
    evaluation: int
    generation: int
    candidate_score: float
    best_score: float
    best_params: SimParams
    best_per_joint_deg: dict[str, float]
    elapsed_s: float

    def to_dict(self) -> dict:
        return {
            "evaluation": int(self.evaluation),
            "generation": int(self.generation),
            "candidate_score": _json_float(self.candidate_score),
            "best_score": _json_float(self.best_score),
            "best_params": self.best_params.to_dict(),
            "best_per_joint_deg": {k: _json_float(v) for k, v in self.best_per_joint_deg.items()},
            "elapsed_s": float(self.elapsed_s),
        }


@dataclass(frozen=True)
class CalibrationResult:
    params: SimParams
    baseline: SimParams
    train_score: float
    eval_score: float
    baseline_train_score: float
    baseline_eval_score: float
    per_joint_before: dict[str, float]
    per_joint_after: dict[str, float]
    history: list[tuple[int, float]]
    n_evaluations: int
    elapsed_s: float
    train_episodes: list[int] = field(default_factory=list)
    eval_episodes: list[int] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Fraction of the held-out gap closed (1.0 = perfect, 0 = none)."""
        if not math.isfinite(self.baseline_eval_score) or self.baseline_eval_score <= 0:
            return 0.0
        return 1.0 - self.eval_score / self.baseline_eval_score

    def to_dict(self) -> dict:
        return {
            "params": self.params.to_dict(),
            "baseline": self.baseline.to_dict(),
            "train_score": _json_float(self.train_score),
            "eval_score": _json_float(self.eval_score),
            "baseline_train_score": _json_float(self.baseline_train_score),
            "baseline_eval_score": _json_float(self.baseline_eval_score),
            "per_joint_before": {k: _json_float(v) for k, v in self.per_joint_before.items()},
            "per_joint_after": {k: _json_float(v) for k, v in self.per_joint_after.items()},
            "history": [[int(i), _json_float(s)] for i, s in self.history],
            "n_evaluations": int(self.n_evaluations),
            "elapsed_s": float(self.elapsed_s),
            "train_episodes": [int(i) for i in self.train_episodes],
            "eval_episodes": [int(i) for i in self.eval_episodes],
            "improvement": _json_float(self.improvement),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationResult":
        return cls(
            params=SimParams.from_dict(d["params"]),
            baseline=SimParams.from_dict(d["baseline"]),
            train_score=_parse_float(d["train_score"]),
            eval_score=_parse_float(d["eval_score"]),
            baseline_train_score=_parse_float(d["baseline_train_score"]),
            baseline_eval_score=_parse_float(d["baseline_eval_score"]),
            per_joint_before={k: _parse_float(v) for k, v in d.get("per_joint_before", {}).items()},
            per_joint_after={k: _parse_float(v) for k, v in d.get("per_joint_after", {}).items()},
            history=[(int(i), _parse_float(s)) for i, s in d.get("history", [])],
            n_evaluations=int(d["n_evaluations"]),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            train_episodes=[int(i) for i in d.get("train_episodes", [])],
            eval_episodes=[int(i) for i in d.get("eval_episodes", [])],
        )


def _json_float(v: float) -> float | None:
    """NaN/inf are not valid JSON; serialize them as null."""
    v = float(v)
    return v if math.isfinite(v) else None


def _parse_float(v: Any) -> float:
    return float("nan") if v is None else float(v)


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------


def default_simulate() -> SimulateFn:
    """Build one MuJoCo :class:`antares.sim.Arm` and wrap it as a ``SimulateFn``.

    The arm is reused across calls (``apply`` is cheap); the real trajectory
    does not depend on the parameters, so it is cached per episode.
    """
    from antares.sim import Arm

    arm = Arm()
    real_cache: dict[int, tuple[Episode, np.ndarray]] = {}

    def simulate(params: SimParams, episode: Episode) -> tuple[np.ndarray, np.ndarray]:
        arm.apply(params)
        hit = real_cache.get(id(episode))
        if hit is None or hit[0] is not episode:
            hit = (episode, arm.real_q(episode))
            real_cache[id(episode)] = hit
        return hit[1], arm.replay(episode)

    return simulate


def score(params: SimParams, episodes: Sequence[Episode], simulate: SimulateFn):
    """Tracking-error report for ``params`` aggregated over ``episodes``."""
    tracking_error, aggregate = _gap_functions()
    reports = []
    for ep in episodes:
        real_q, sim_q = simulate(params, ep)
        reports.append(tracking_error(real_q, sim_q))
    return aggregate(reports)


def _objective_value(report: Any) -> float:
    """``overall_deg`` guarded against NaN/inf from an unstable sim."""
    try:
        v = float(report.overall_deg)
    except (TypeError, ValueError, AttributeError):
        return PENALTY
    return v if math.isfinite(v) else PENALTY


def _safe_score(params: SimParams, episodes: Sequence[Episode], simulate: SimulateFn):
    """Like :func:`score`, but a simulator exception also counts as a penalty."""
    try:
        report = score(params, episodes, simulate)
    except (FloatingPointError, ValueError, ArithmeticError):
        return None, PENALTY
    return report, _objective_value(report)


def _evaluate_candidates(
    candidates: Sequence[SimParams],
    episodes: Sequence[Episode],
    simulate: SimulateFn,
):
    """Yield ``(report, value)`` for each candidate, in order, as computed so
    progress can stream between evaluations. Single-process; this is the
    seam where a ``workers`` pool would plug in later (order must match)."""
    for p in candidates:
        yield _safe_score(p, episodes, simulate)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def calibrate(
    train: Sequence[Episode],
    eval: Sequence[Episode],
    simulate: SimulateFn | None = None,
    start: SimParams | None = None,
    max_evaluations: int = 150,
    popsize: int | None = None,
    sigma0: float = 0.3,
    seed: int = 0,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> CalibrationResult:
    """Fit :class:`SimParams` to ``train`` episodes; report on ``eval`` episodes.

    ``on_progress`` fires once per objective evaluation, starting with the
    starting point as evaluation 0, so it is called ``n_evaluations + 1``
    times. ``should_stop`` is polled between evaluations; when it returns
    True the fit ends early and the best parameters found so far are
    returned. A partially evaluated last generation is never fed back to
    CMA-ES, but its candidates still count toward the best.
    """
    import cma

    t0 = time.perf_counter()
    train = list(train)
    eval = list(eval)
    if not train:
        raise ValueError("calibrate needs at least one training episode")
    if simulate is None:
        simulate = default_simulate()
    if start is None:
        start = SimParams()

    stop = should_stop or (lambda: False)
    history: list[tuple[int, float]] = []
    evaluations = 0
    generation = 0

    # Evaluation 0: the starting point.
    start_report, start_score = _safe_score(start, train, simulate)
    best_params, best_score, best_report = start, start_score, start_report
    history.append((0, best_score))
    if on_progress is not None:
        on_progress(
            ProgressEvent(
                evaluation=0,
                generation=0,
                candidate_score=start_score,
                best_score=best_score,
                best_params=best_params,
                best_per_joint_deg=_per_joint(best_report),
                elapsed_s=time.perf_counter() - t0,
            )
        )

    opts: dict[str, Any] = {
        "bounds": [0.0, 1.0],
        # cma treats seed 0 as "use the clock"; derive a positive seed instead.
        "seed": int(np.random.default_rng(seed).integers(1, 2**31 - 1)),
        "verbose": -9,
        "verb_log": 0,
        "maxfevals": max_evaluations,
    }
    if popsize is not None:
        opts["popsize"] = int(popsize)

    x0 = np.clip(encode(start), 1e-3, 1 - 1e-3)
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    while evaluations < max_evaluations and not es.stop() and not stop():
        generation += 1
        solutions = es.ask()
        budget = max_evaluations - evaluations
        batch = solutions[:budget]
        candidates = [decode(x) for x in batch]

        values: list[float] = []
        cancelled = False
        for i, (params, (report, value)) in enumerate(
            zip(candidates, _evaluate_candidates(candidates, train, simulate))
        ):
            evaluations += 1
            values.append(value)
            if value < best_score:
                best_params, best_score, best_report = params, value, report
            history.append((evaluations, best_score))
            if on_progress is not None:
                on_progress(
                    ProgressEvent(
                        evaluation=evaluations,
                        generation=generation,
                        candidate_score=value,
                        best_score=best_score,
                        best_params=best_params,
                        best_per_joint_deg=_per_joint(best_report),
                        elapsed_s=time.perf_counter() - t0,
                    )
                )
            if i + 1 < len(candidates) and stop():
                cancelled = True
                break

        if cancelled or len(values) != len(solutions):
            break  # partial generation: do not update the distribution
        es.tell(solutions, values)

    # Held-out evaluation, before and after. With no eval episodes the
    # training reports stand in so the result stays fully populated.
    if eval:
        before_report, before_score = _safe_score(start, eval, simulate)
        after_report, after_score = _safe_score(best_params, eval, simulate)
    else:
        before_report, before_score = start_report, start_score
        after_report, after_score = best_report, best_score

    return CalibrationResult(
        params=best_params,
        baseline=start,
        train_score=best_score,
        eval_score=after_score,
        baseline_train_score=start_score,
        baseline_eval_score=before_score,
        per_joint_before=_per_joint(before_report),
        per_joint_after=_per_joint(after_report),
        history=history,
        n_evaluations=evaluations,
        elapsed_s=time.perf_counter() - t0,
        train_episodes=[ep.index for ep in train],
        eval_episodes=[ep.index for ep in eval],
    )


def _per_joint(report: Any) -> dict[str, float]:
    if report is None:
        return {}
    return {k: float(v) for k, v in report.per_joint_deg.items()}


def split_episodes(
    episodes: Sequence[Episode], n_train: int, n_eval: int, seed: int = 0
) -> tuple[list[Episode], list[Episode]]:
    """Deterministic disjoint train/eval split (each sorted by episode index).

    If the dataset is smaller than ``n_train + n_eval`` the counts are
    clamped: training takes up to ``n_train`` and eval gets what is left.
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("no episodes to split")
    if n_train < 1:
        raise ValueError("n_train must be >= 1")
    n_train = min(n_train, len(episodes))
    n_eval = max(0, min(n_eval, len(episodes) - n_train))
    perm = np.random.default_rng(seed).permutation(len(episodes))
    train = [episodes[i] for i in perm[:n_train]]
    eval = [episodes[i] for i in perm[n_train : n_train + n_eval]]
    key = lambda ep: ep.index  # noqa: E731
    return sorted(train, key=key), sorted(eval, key=key)


__all__ = [
    "SPACE",
    "DIMS",
    "PENALTY",
    "SimulateFn",
    "ProgressEvent",
    "CalibrationResult",
    "encode",
    "decode",
    "score",
    "calibrate",
    "default_simulate",
    "split_episodes",
]
