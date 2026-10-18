"""Fit simulator parameters to one robot's logs with CMA-ES.

The shared search space is six scalars of :class:`antares.sim.SimParams`
(actuator gain and velocity gain, joint damping, armature, friction loss,
action latency), one value for all joints. CMA-ES works on a normalized
``[0, 1]^n`` vector; the physical parameters live on log scales because
they span orders of magnitude, and latency is an integer number of frames
encoded as a continuous dimension and rounded on decode.

A per-joint fit refines that: every STS3215 carries a different load, so
after the shared fit a second CMA-ES stage searches one value per joint
for each parameter (plus the actuator force limit), warm-started from the
shared optimum with a smaller step size. See :func:`calibrate`.

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from antares import factors
from antares.data import Episode
from antares.sim import JOINTS, SimParams

# --------------------------------------------------------------------------
# Lazy import of antares.gap, written in parallel. The fallbacks only exist
# so that this module can be imported and tested with a fake simulate
# function before antares.gap lands; they mirror the agreed contract.
# --------------------------------------------------------------------------


def _gap_functions() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return ``(tracking_error, aggregate)`` from :mod:`antares.gap`, or
    minimal stand-ins with the same shape if that module is not ready."""
    try:
        from antares.gap import aggregate, tracking_error

        return tracking_error, aggregate
    except (ImportError, AttributeError):
        return _fallback_tracking_error, _fallback_aggregate


_FALLBACK_JOINTS = JOINTS


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
#: are continuous during the search and rounded on decode. This is the
#: shared space: one value of each parameter for all six joints. Bounds and
#: order come from :mod:`antares.factors`.
SPACE: dict[str, tuple[float, float, str]] = factors.space("shared")

#: Bounds for the per-joint refinement: the shared parameters plus the
#: actuator force limit. Each is searched once per joint, so the space has
#: ``6 * len(PER_JOINT_SPACE)`` dims.
PER_JOINT_SPACE: dict[str, tuple[float, float, str]] = {
    **SPACE,
    **factors.space("per_joint", per_joint=True),
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


@dataclass(frozen=True)
class SearchSpace:
    """Normalized ``[0, 1]^n`` coordinates over a set of :class:`SimParams` fields.

    ``shared`` searches one value per field; ``per_joint`` searches one value
    per field *and* joint (field-major: ``kp.shoulder_pan, kp.shoulder_lift,
    ...``). Fields outside ``bounds`` are not searched and keep whatever the
    ``base`` parameters passed to :meth:`decode` hold.
    """

    bounds: Mapping[str, tuple[float, float, str]]
    per_joint: bool = False
    #: Column names, one per joint (the robot's; default the SO arms' six).
    joints: tuple[str, ...] = JOINTS

    @property
    def name(self) -> str:
        return "per_joint" if self.per_joint else "shared"

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.bounds)

    @property
    def dims(self) -> tuple[str, ...]:
        if not self.per_joint:
            return self.fields
        return tuple(f"{f}.{j}" for f in self.fields for j in self.joints)

    @property
    def n_dims(self) -> int:
        return len(self.fields) * (len(self.joints) if self.per_joint else 1)

    def encode(self, params: SimParams) -> np.ndarray:
        """Parameters -> search vector, clipped to the bounds.

        Shared spaces take a per-joint field's mean (in search coordinates)
        when a per-joint parameter is passed; per-joint spaces broadcast
        scalars.
        """
        out = []
        for f in self.fields:
            b = self.bounds[f]
            if self.per_joint:
                out.extend(_to_unit(v, *b) for v in params.per_joint(f, len(self.joints)))
            elif params.is_per_joint(f):
                out.append(float(np.mean([_to_unit(v, *b) for v in params.per_joint(f)])))
            else:
                out.append(_to_unit(getattr(params, f), *b))
        return np.asarray(out, dtype=np.float64)

    def decode(self, x: np.ndarray, base: SimParams | None = None) -> SimParams:
        """Search vector -> parameters; fields not searched come from ``base``."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_dims:
            raise ValueError(f"expected {self.n_dims} dims, got {x.shape[0]}")
        kwargs: dict[str, Any] = {}
        if self.per_joint:
            nj = len(self.joints)
            for i, f in enumerate(self.fields):
                b = self.bounds[f]
                kwargs[f] = tuple(_from_unit(v, *b) for v in x[i * nj : (i + 1) * nj])
        else:
            for i, f in enumerate(self.fields):
                kwargs[f] = _from_unit(x[i], *self.bounds[f])
        return (base or SimParams()).replace(**kwargs)


SHARED = SearchSpace(SPACE, per_joint=False)
PER_JOINT = SearchSpace(PER_JOINT_SPACE, per_joint=True)


def spaces_for(
    robot: Any | None = None,
    fit_factors: Sequence[str] | None = None,
    per_joint_factors: Sequence[str] | None = None,
) -> tuple[SearchSpace, SearchSpace]:
    """``(shared, per_joint)`` search spaces for a robot and a factor selection.

    ``robot`` (a :class:`antares.robots.RobotModel`) supplies the joint
    names and any bound overrides; ``fit_factors`` picks the shared stage's
    factors (default: those :mod:`antares.factors` fits by default) and
    ``per_joint_factors`` the per-joint stage's (default: the shared ones
    plus the per-joint defaults). With nothing given these are
    :data:`SHARED` and :data:`PER_JOINT`.
    """
    if robot is None and fit_factors is None and per_joint_factors is None:
        return SHARED, PER_JOINT
    joints = tuple(robot.joints) if robot is not None else JOINTS
    overrides = dict(robot.bounds) if robot is not None else {}
    shared_names = tuple(fit_factors) if fit_factors is not None else factors.names("shared")
    if per_joint_factors is None:
        per_joint_factors = shared_names + tuple(n for n in factors.names("per_joint") if n not in shared_names)
    shared = SearchSpace(factors.space("shared", factors=shared_names, overrides=overrides), False, joints)
    per_joint = SearchSpace(
        factors.space("per_joint", per_joint=True, factors=tuple(per_joint_factors), overrides=overrides), True, joints
    )
    return shared, per_joint


def encode(params: SimParams, space: SearchSpace = SHARED) -> np.ndarray:
    """Map parameters to the normalized search vector (clipped); shared by default."""
    return space.encode(params)


def decode(x: np.ndarray, space: SearchSpace = SHARED, base: SimParams | None = None) -> SimParams:
    """Map a search vector back to parameters, clipping to the bounds."""
    return space.decode(x, base)


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
    stage: str = "shared"  # which search space produced this evaluation
    #: Index of the CMA-ES seed run this evaluation belongs to (0-based; a
    #: single-seed fit is all seed 0). ``best_score`` is the best of that run.
    seed: int = 0

    def to_dict(self) -> dict:
        return {
            "evaluation": int(self.evaluation),
            "generation": int(self.generation),
            "candidate_score": _json_float(self.candidate_score),
            "best_score": _json_float(self.best_score),
            "best_params": self.best_params.to_dict(),
            "best_per_joint_deg": {k: _json_float(v) for k, v in self.best_per_joint_deg.items()},
            "elapsed_s": float(self.elapsed_s),
            "stage": self.stage,
            "seed": int(self.seed),
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
    #: ``"shared"`` or ``"per_joint"``: the space the final parameters live in.
    space: str = "shared"
    #: One entry per CMA-ES stage run, in order: ``{"name", "n_evaluations",
    #: "train_score", "params"}`` with the stage's best on the training set,
    #: plus ``"seed"`` (the CMA-ES seed) for stages fitted since seeds were
    #: recorded. A best-of-N fit has one ``"shared"`` entry per seed.
    stages: list[dict] = field(default_factory=list)
    #: Best-of-N seed summary (see :func:`seed_summary`), or None for a
    #: single-seed fit and for locks written before seeds were recorded.
    seeds: dict | None = None

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
            "space": self.space,
            "stages": [_stage_to_dict(s, _json_float) for s in self.stages],
            **({"seeds": _seeds_to_dict(self.seeds)} if self.seeds is not None else {}),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationResult":
        seeds = d.get("seeds")
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
            space=str(d.get("space", "shared")),
            stages=[_stage_to_dict(s, _parse_float) for s in d.get("stages", [])],
            seeds=_seeds_from_dict(seeds) if isinstance(seeds, dict) else None,
        )


def _json_float(v: float) -> float | None:
    """NaN/inf are not valid JSON; serialize them as null."""
    v = float(v)
    return v if math.isfinite(v) else None


def _parse_float(v: Any) -> float:
    return float("nan") if v is None else float(v)


def _stage_to_dict(s: Mapping[str, Any], score: Callable[[Any], Any]) -> dict:
    """Normalize one ``stages`` entry; ``seed`` only when the stage recorded one."""
    out = {
        "name": str(s["name"]),
        "n_evaluations": int(s["n_evaluations"]),
        "train_score": score(s["train_score"]),
        "params": dict(s["params"]),
    }
    if s.get("seed") is not None:
        out["seed"] = int(s["seed"])
    return out


# --------------------------------------------------------------------------
# Best-of-N seeds: spread and pinned parameters
# --------------------------------------------------------------------------

#: A log-scaled parameter is pinned when its max/min across seeds is at most this.
PINNED_RATIO = 2.0
#: ``latency_frames`` is pinned when it moves by at most this many frames across seeds.
PINNED_LATENCY_FRAMES = 1


def param_spread_of(params: Sequence[SimParams]) -> tuple[dict[str, dict[str, float]], list[str]]:
    """``(spread, pinned)`` of the shared search parameters across several fits.

    ``spread[name]`` is ``{"min", "max", "ratio"}`` for a log-scaled
    parameter and ``{"min", "max", "range"}`` for ``latency_frames``, over
    every fit (and every joint, for per-joint values). A parameter is
    *pinned* when the fits agree on it: ratio at most :data:`PINNED_RATIO`,
    or latency within :data:`PINNED_LATENCY_FRAMES` frames. Other parameters
    moved between fits of equal quality and are not physical measurements.
    """
    params = list(params)
    if len(params) < 2:
        raise ValueError("spread needs at least two fits")
    spread: dict[str, dict[str, float]] = {}
    pinned: list[str] = []
    for name, (_, _, scale) in SPACE.items():
        values = np.concatenate([np.asarray(p.per_joint(name), dtype=np.float64) for p in params])
        lo, hi = float(values.min()), float(values.max())
        if scale == "log":
            ratio = hi / lo if lo > 0 else float("inf")
            spread[name] = {"min": lo, "max": hi, "ratio": ratio}
            if ratio <= PINNED_RATIO:
                pinned.append(name)
        else:
            spread[name] = {"min": lo, "max": hi, "range": hi - lo}
            if hi - lo <= PINNED_LATENCY_FRAMES:
                pinned.append(name)
    return spread, pinned


def seed_summary(
    seeds: Sequence[int],
    chosen: int,
    params: Sequence[SimParams],
    eval_scores: Sequence[float],
    train_scores: Sequence[float],
) -> dict:
    """The ``seeds`` block of a :class:`CalibrationResult`: every seed's
    shared optimum and scores, which one was kept (best held-out score), and
    the spread/pinned verdict from :func:`param_spread_of`."""
    params = list(params)
    spread, pinned = param_spread_of(params)
    return {
        "n": len(params),
        "seeds": [int(s) for s in seeds],
        "chosen": int(chosen),
        "eval_scores": [float(v) for v in eval_scores],
        "train_scores": [float(v) for v in train_scores],
        "params": [p.to_dict() for p in params],
        "spread": spread,
        "pinned": pinned,
    }


def _seeds_to_dict(s: Mapping[str, Any]) -> dict:
    return {
        "n": int(s["n"]),
        "seeds": [int(v) for v in s["seeds"]],
        "chosen": int(s["chosen"]),
        "eval_scores": [_json_float(v) for v in s["eval_scores"]],
        "train_scores": [_json_float(v) for v in s["train_scores"]],
        "params": [dict(p) for p in s["params"]],
        "spread": {
            k: {kk: _json_float(vv) for kk, vv in v.items()} for k, v in (s.get("spread") or {}).items()
        },
        "pinned": [str(p) for p in (s.get("pinned") or [])],
    }


def _seeds_from_dict(s: Mapping[str, Any]) -> dict:
    return {
        "n": int(s.get("n", len(s.get("params", [])))),
        "seeds": [int(v) for v in s.get("seeds", [])],
        "chosen": int(s.get("chosen", 0)),
        "eval_scores": [_parse_float(v) for v in s.get("eval_scores", [])],
        "train_scores": [_parse_float(v) for v in s.get("train_scores", [])],
        "params": [dict(p) for p in s.get("params", [])],
        "spread": {
            k: {kk: _parse_float(vv) for kk, vv in v.items()} for k, v in (s.get("spread") or {}).items()
        },
        "pinned": [str(p) for p in (s.get("pinned") or [])],
    }


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------


def default_simulate(command_map: Any | None = None, robot: str | None = None) -> SimulateFn:
    """Build one MuJoCo :class:`antares.sim.Arm` and wrap it as a ``SimulateFn``.

    The arm is reused across calls (``apply`` is cheap); the real trajectory
    does not depend on the parameters, so it is cached per episode.
    ``command_map`` (an :class:`antares.sim.CommandMap`) corrects every
    replay's commands, e.g. a lock's sensor calibration held fixed.
    ``robot`` picks the model from :data:`antares.robots.REGISTRY` (default
    SO-101).
    """
    from antares.sim import Arm

    arm = Arm(command_map=command_map, robot=robot)
    real_cache: dict[int, tuple[Episode, np.ndarray]] = {}

    def simulate(params: SimParams, episode: Episode) -> tuple[np.ndarray, np.ndarray]:
        arm.apply(params)
        hit = real_cache.get(id(episode))
        if hit is None or hit[0] is not episode:
            hit = (episode, arm.real_q(episode))
            real_cache[id(episode)] = hit
        return hit[1], arm.replay(episode)

    return describe_simulate(simulate, arm)


def describe_simulate(simulate: Callable[..., Any], arm: Any) -> Callable[..., Any]:
    """Tag a simulate callable with the robot it runs, so scoring knows the
    columns and which are grippers, and a fit knows the robot's defaults."""
    simulate.robot = arm.robot  # type: ignore[attr-defined]
    simulate.default_params = arm.default_params()  # type: ignore[attr-defined]
    return simulate


def robot_of(simulate: Any) -> Any | None:
    """The :class:`antares.robots.RobotModel` a simulate callable runs, if tagged."""
    return getattr(simulate, "robot", None)


def scoring_names(simulate: Any) -> dict[str, Any]:
    """``joint_names``/``exclude`` keyword arguments for scoring ``simulate``'s
    output: the robot's columns and grippers, or the SO arms' by default."""
    r = robot_of(simulate)
    if r is None:
        return {}
    return {"joint_names": tuple(r.joints), "exclude": tuple(r.grippers)}


def score(params: SimParams, episodes: Sequence[Episode], simulate: SimulateFn):
    """Tracking-error report for ``params`` aggregated over ``episodes``."""
    tracking_error, aggregate = _gap_functions()
    names = scoring_names(simulate)
    reports = []
    for ep in episodes:
        real_q, sim_q = simulate(params, ep)
        reports.append(tracking_error(real_q, sim_q, **names))
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


#: Default evaluation budget of the per-joint refinement stage. At ~0.12 s
#: per evaluation (20 episodes) this is about a minute.
PER_JOINT_MAX_EVALUATIONS = 600

#: Default CMA-ES step size (in unit coordinates) of the per-joint stage. It
#: starts at the shared optimum, so it only needs to move locally: on both
#: demo datasets 0.05 beat 0.1 clearly within 600 evaluations, and 0.2
#: never produced a candidate better than the warm start.
PER_JOINT_SIGMA0 = 0.05


@dataclass
class _Fit:
    """Mutable state shared by the CMA-ES stages of one :func:`calibrate` call."""

    train: list[Episode]
    simulate: SimulateFn
    on_progress: Callable[[ProgressEvent], None] | None
    stop: Callable[[], bool]
    t0: float
    best_params: SimParams
    best_score: float
    best_report: Any
    evaluations: int = 0
    generation: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)
    stages: list[dict] = field(default_factory=list)
    #: Which seed run is in progress (0-based); stamped on every progress event.
    seed_index: int = 0

    def reset_best(self, params: SimParams, report: Any, value: float) -> None:
        """Start tracking the best from ``params`` again (a new seed run)."""
        self.best_params, self.best_score, self.best_report = params, value, report

    def record(self, params: SimParams, report: Any, value: float, stage: str) -> None:
        """Account one evaluated candidate: best-so-far, history, progress."""
        if value < self.best_score:
            self.best_params, self.best_score, self.best_report = params, value, report
        self.history.append((self.evaluations, self.best_score))
        if self.on_progress is not None:
            self.on_progress(
                ProgressEvent(
                    evaluation=self.evaluations,
                    generation=self.generation,
                    candidate_score=value,
                    best_score=self.best_score,
                    best_params=self.best_params,
                    best_per_joint_deg=_per_joint(self.best_report),
                    elapsed_s=time.perf_counter() - self.t0,
                    stage=stage,
                    seed=self.seed_index,
                )
            )

    def run_stage(
        self,
        space: SearchSpace,
        start: SimParams,
        max_evaluations: int,
        sigma0: float,
        popsize: int | None,
        seed: int,
    ) -> bool:
        """One CMA-ES run in ``space`` from ``start``. Returns False if cancelled.

        A partially evaluated last generation is never fed back to CMA-ES,
        but its candidates still count toward the best.
        """
        import cma

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

        x0 = np.clip(space.encode(start), 1e-3, 1 - 1e-3)
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        first_eval = self.evaluations
        cancelled = False

        while self.evaluations - first_eval < max_evaluations and not es.stop() and not self.stop():
            self.generation += 1
            solutions = es.ask()
            budget = max_evaluations - (self.evaluations - first_eval)
            candidates = [space.decode(x, base=start) for x in solutions[:budget]]

            values: list[float] = []
            for i, (params, (report, value)) in enumerate(
                zip(candidates, _evaluate_candidates(candidates, self.train, self.simulate))
            ):
                self.evaluations += 1
                values.append(value)
                self.record(params, report, value, space.name)
                if i + 1 < len(candidates) and self.stop():
                    cancelled = True
                    break

            if cancelled or len(values) != len(solutions):
                break  # partial generation: do not update the distribution
            es.tell(solutions, values)

        self.stages.append(
            {
                "name": space.name,
                "n_evaluations": self.evaluations - first_eval,
                "train_score": self.best_score,
                "params": self.best_params.to_dict(),
                "seed": int(seed),
            }
        )
        return not cancelled and not self.stop()


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
    per_joint: bool = False,
    per_joint_max_evaluations: int = PER_JOINT_MAX_EVALUATIONS,
    per_joint_sigma0: float = PER_JOINT_SIGMA0,
    per_joint_popsize: int | None = None,
    n_seeds: int = 1,
    seeds: Sequence[int] | None = None,
    fit_factors: Sequence[str] | None = None,
    per_joint_factors: Sequence[str] | None = None,
) -> CalibrationResult:
    """Fit :class:`SimParams` to ``train`` episodes; report on ``eval`` episodes.

    The shared stage searches :data:`SPACE` (one value per parameter for all
    joints) for ``max_evaluations`` evaluations. With ``per_joint=True`` a
    second stage then searches :data:`PER_JOINT_SPACE` (one value per
    parameter *and* joint, plus the force limit) for another
    ``per_joint_max_evaluations`` evaluations, warm-started from the shared
    optimum with step size ``per_joint_sigma0``; ``per_joint_popsize`` of
    None takes CMA-ES's default for that dimensionality.

    With ``n_seeds > 1`` (or an explicit ``seeds`` list) the shared stage
    runs once per seed (``seed``, ``seed + 100``, ``seed + 200``, ...), each
    from the same ``start`` with the full ``max_evaluations`` budget, and the
    candidate with the best held-out score is kept (the training score
    decides when there are no eval episodes). The per-joint stage then runs
    once from the winner. ``result.stages`` has one ``"shared"`` entry per
    seed and ``result.seeds`` records every candidate, the spread of each
    parameter across them and which parameters are *pinned* (agree across
    seeds); a single-seed fit leaves ``seeds`` None. The same seed run twice
    gives the same fit, so one seed is repeatable but not representative:
    on the demo arms different seeds land at kp 10 and kp 260 with the same
    error.

    ``on_progress`` fires once per objective evaluation, starting with the
    starting point as evaluation 0, so it is called ``n_evaluations + 1``
    times; each event names its ``stage`` and ``seed`` index, and
    ``best_score`` is the best of the seed run in progress. ``should_stop``
    is polled between evaluations; when it returns True the fit ends early
    (skipping later seeds and any later stage) and the best parameters found
    so far are returned.

    ``fit_factors`` / ``per_joint_factors`` choose which factors of
    :mod:`antares.factors` each stage searches (the rest keep ``start``'s
    values); by default the stages search what they always have. When
    ``simulate`` runs a robot other than the SO arms (see
    :func:`describe_simulate`), its columns, grippers and bounds shape the
    spaces and the score, and ``start`` defaults to its own MJCF values.
    """
    t0 = time.perf_counter()
    train = list(train)
    eval = list(eval)
    if not train:
        raise ValueError("calibrate needs at least one training episode")
    if seeds is not None:
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            raise ValueError("seeds must name at least one seed")
    else:
        if n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        seed_list = [seed + 100 * i for i in range(n_seeds)]
    if simulate is None:
        simulate = default_simulate()
    robot = robot_of(simulate)
    so_arm = robot is None or robot.default_params == "sts3215"
    if start is None:
        start = SimParams() if so_arm else simulate.default_params  # type: ignore[attr-defined]
    shared_space, per_joint_space = spaces_for(
        None if so_arm and robot is not None and tuple(robot.joints) == JOINTS and not robot.bounds else robot,
        fit_factors,
        per_joint_factors,
    )

    # Evaluation 0: the starting point.
    start_report, start_score = _safe_score(start, train, simulate)
    fit = _Fit(
        train=train,
        simulate=simulate,
        on_progress=on_progress,
        stop=should_stop or (lambda: False),
        t0=t0,
        best_params=start,
        best_score=start_score,
        best_report=start_report,
    )
    fit.record(start, start_report, start_score, shared_space.name)

    def held_out(params: SimParams, report: Any, value: float) -> tuple[Any, float]:
        """Held-out score; the training score stands in without eval episodes."""
        return _safe_score(params, eval, simulate) if eval else (report, value)

    # Shared stage, once per seed from the same start. No shared budget:
    # skip it (a per-joint refit starts in its own space).
    finished = True
    chosen = 0
    candidates: list[tuple[SimParams, Any, float]] = []  # (params, train report, train score)
    candidate_eval: list[tuple[Any, float]] = []
    if max_evaluations > 0:
        for i, s in enumerate(seed_list):
            fit.seed_index = i
            if i > 0:
                fit.reset_best(start, start_report, start_score)
            finished = fit.run_stage(shared_space, start, max_evaluations, sigma0, popsize, s)
            candidates.append((fit.best_params, fit.best_report, fit.best_score))
            candidate_eval.append(held_out(fit.best_params, fit.best_report, fit.best_score))
            if not finished:
                break
        chosen = int(np.argmin([v for _, v in candidate_eval]))
        fit.reset_best(*candidates[chosen])
    seeds_block = (
        seed_summary(
            seed_list[: len(candidates)],
            chosen,
            [p for p, _, _ in candidates],
            [v for _, v in candidate_eval],
            [v for _, _, v in candidates],
        )
        if len(candidates) >= 2
        else None
    )

    if per_joint and finished:
        # Refine from the shared optimum; a fresh seed so the two stages do
        # not share a random stream.
        fit.run_stage(
            per_joint_space,
            fit.best_params,
            per_joint_max_evaluations,
            per_joint_sigma0,
            per_joint_popsize,
            seed + 1,
        )
    best_params, best_score, best_report = fit.best_params, fit.best_score, fit.best_report

    # Held-out evaluation, before and after. With no eval episodes the
    # training reports stand in so the result stays fully populated. The
    # chosen seed's held-out score is reused when nothing refined it.
    before_report, before_score = held_out(start, start_report, start_score)
    if candidates and best_params is candidates[chosen][0]:
        after_report, after_score = candidate_eval[chosen]
    else:
        after_report, after_score = held_out(best_params, best_report, best_score)

    return CalibrationResult(
        params=best_params,
        baseline=start,
        train_score=best_score,
        eval_score=after_score,
        baseline_train_score=start_score,
        baseline_eval_score=before_score,
        per_joint_before=_per_joint(before_report),
        per_joint_after=_per_joint(after_report),
        history=fit.history,
        n_evaluations=fit.evaluations,
        elapsed_s=time.perf_counter() - t0,
        train_episodes=[ep.index for ep in train],
        eval_episodes=[ep.index for ep in eval],
        space=per_joint_space.name if best_params.has_per_joint else shared_space.name,
        stages=fit.stages,
        seeds=seeds_block,
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
    "PER_JOINT_SPACE",
    "DIMS",
    "PENALTY",
    "PER_JOINT_MAX_EVALUATIONS",
    "PER_JOINT_SIGMA0",
    "PINNED_RATIO",
    "PINNED_LATENCY_FRAMES",
    "SearchSpace",
    "SHARED",
    "PER_JOINT",
    "SimulateFn",
    "ProgressEvent",
    "CalibrationResult",
    "encode",
    "decode",
    "score",
    "calibrate",
    "default_simulate",
    "describe_simulate",
    "robot_of",
    "scoring_names",
    "spaces_for",
    "param_spread_of",
    "seed_summary",
    "split_episodes",
]
