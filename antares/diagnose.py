"""Name the cause of drift: what changed in the arm since its lock was fitted.

:func:`antares.check.check` says *whether* a robot still matches its lock.
:func:`diagnose` says *what* changed. It refits the lock on fresh episodes,
but instead of searching the parameters from scratch it searches one
*change* per parameter on top of the lock: a factor on each of ``kp``,
``kv``, ``damping``, ``armature`` and ``frictionloss`` (x1/4 to x4,
log-uniform) and a shift of whole frames on ``latency_frames``. A per-joint
lock keeps its per-joint shape; every joint gets the same change.

The search prefers few, small changes. It minimizes the refit error plus a
cost per change (:func:`complexity`), so a factor has to earn its size,
and afterwards drops any kept change that lowers the error by less than
:data:`STEP_GAIN_DEG`. Without that, a free search over all six can trade
one frame of lag for a large velocity-gain swing that mimics it, or pick up
a small gain from parameters pulling against each other on a recording
where nothing changed. Latency is a handful of whole frames, so each shift
is scored first and the factors are refitted by CMA-ES for the most
promising few.

The refit runs on half of the fresh episodes and both the lock and the
refit are scored on the other half, so the gap it closes is real and not
overfitting.

The six parameters partly trade off against each other (a stiffer servo
with more damping can track like a softer one), so a parameter that moved
a lot in the refit is not necessarily the one that changed. Each change is
therefore also tried *alone* on the lock: a parameter is named as a cause
when that single change recovers a real share of the gap
(:data:`CAUSE_FRACTION`). When the refit itself recovers little of the
excess error, the change is not one the sim's dynamics parameters can
express (sensor calibration, contacts, a different task), and the
diagnosis says so instead of naming a parameter.

The simulator is injected as a ``simulate(params, episode) -> (real_q,
sim_q)`` callable like everywhere else, so this module is testable without
MuJoCo.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from antares.calibrate import PENALTY, _per_joint, _safe_score
from antares.data import Episode

#: Largest factor searched for each scaled parameter, either way.
FACTOR_RANGE: dict[str, float] = {
    "kp": 4.0,
    "kv": 4.0,
    "damping": 4.0,
    "armature": 4.0,
    "frictionloss": 4.0,
}
#: Frames ``latency_frames`` may shift by, (fewest, most).
LATENCY_SHIFT: tuple[int, int] = (-2, 3)
FIELDS: tuple[str, ...] = (*FACTOR_RANGE, "latency_frames")

#: Latency shifts (best first, by the scan with no other change) whose
#: factors are each refitted.
SHIFT_CANDIDATES = 3
#: What a change costs in the search, in degrees of refit error: per unit of
#: |ln factor| for the scaled parameters (x2 costs 0.14 deg, x3 costs 0.22),
#: and per frame of latency shift. Lag is kept cheap on purpose: a whole
#: frame is the most specific explanation, and on the real arm a large
#: velocity-gain swing can otherwise stand in for it.
COMPLEXITY_DEG = 0.2
LATENCY_COMPLEXITY_DEG = 0.01
#: A kept change that lowers the refit error by less than this (degrees) is
#: dropped after the search.
STEP_GAIN_DEG = 0.05
#: A parameter is a cause when changing it alone recovers this share of the gap.
CAUSE_FRACTION = 0.2
#: A refit that improves on the lock by less than this (degrees) is noise.
MIN_GAP_DEG = 0.05
#: Below this share of the excess error recovered, the drift is reported as
#: something the dynamics parameters cannot explain.
EXPLAINED_FRACTION = 0.5
#: Smallest refit budget: enough for the latency scan and a little more.
MIN_EVALUATIONS = 10
#: A factor this close to 1 is "no change" (it would print as +0%).
NEUTRAL_TOL = 0.005


# --------------------------------------------------------------------------
# Change space
# --------------------------------------------------------------------------


def neutral() -> dict[str, float]:
    """No change: factor 1 on every scaled parameter, latency shift 0."""
    return {**{f: 1.0 for f in FACTOR_RANGE}, "latency_frames": 0}


def decode_change(x: np.ndarray) -> dict[str, float]:
    """``[0, 1]^6`` search vector -> ``{param: factor or frame shift}``."""
    x = np.clip(np.asarray(x, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if x.shape[0] != len(FIELDS):
        raise ValueError(f"expected {len(FIELDS)} dims, got {x.shape[0]}")
    out: dict[str, float] = {}
    for i, f in enumerate(FACTOR_RANGE):
        out[f] = float(FACTOR_RANGE[f] ** (2.0 * x[i] - 1.0))
    lo, hi = LATENCY_SHIFT
    out["latency_frames"] = int(round(lo + x[-1] * (hi - lo)))
    return out


def encode_change(change: Mapping[str, float]) -> np.ndarray:
    """Inverse of :func:`decode_change` (clipped to the ranges)."""
    x = []
    for f, r in FACTOR_RANGE.items():
        x.append(0.5 * (math.log(float(change.get(f, 1.0))) / math.log(r) + 1.0))
    lo, hi = LATENCY_SHIFT
    x.append((float(change.get("latency_frames", 0)) - lo) / (hi - lo))
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def apply_change(params, change: Mapping[str, float]):
    """``params`` with each factor multiplied in and the latency shift added."""
    from antares.synth import worn

    return worn(params, change)


def describe_change(name: str, value: float) -> str:
    if name == "latency_frames":
        n = int(value)
        return f"{n:+d} frame" + ("" if abs(n) == 1 else "s")
    return f"{(float(value) - 1.0) * 100.0:+.0f}%"


def _is_neutral(name: str, value: float) -> bool:
    return int(value) == 0 if name == "latency_frames" else abs(float(value) - 1.0) < NEUTRAL_TOL


def effective_shift(shift: int, params) -> int:
    """The latency shift :func:`antares.synth.worn` really applies to ``params``:
    the same on every joint, limited so none goes below zero."""
    return max(int(shift), -int(np.min(np.asarray(params.latency_frames))))


def complexity(change: Mapping[str, float]) -> float:
    """What ``change`` costs in the search, in degrees (see :data:`COMPLEXITY_DEG`)."""
    cost = 0.0
    for f, v in change.items():
        if f == "latency_frames":
            cost += LATENCY_COMPLEXITY_DEG * abs(int(v))
        else:
            cost += COMPLEXITY_DEG * abs(math.log(float(v)))
    return cost


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamChange:
    """One parameter's change from the lock in the refit.

    ``alone_deg`` is the held-out error of the lock with only this change
    applied, so ``recovers_deg`` is how much of the stale error that single
    change removes on its own.
    """

    name: str
    value: float  # factor, or frame shift for latency_frames
    alone_deg: float
    stale_deg: float

    @property
    def recovers_deg(self) -> float:
        return self.stale_deg - self.alone_deg

    @property
    def change(self) -> str:
        return describe_change(self.name, self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "change": self.change,
            "alone_deg": _json_float(self.alone_deg),
            "recovers_deg": _json_float(self.recovers_deg),
        }


@dataclass(frozen=True)
class Diagnosis:
    lock: str
    dataset: str
    reference_deg: float | None  # the lock's held-out score when it was fitted
    stale_deg: float  # the lock on the fresh held-out episodes
    refit_deg: float  # the lock with the refit's changes, same episodes
    per_joint_stale: dict[str, float]
    per_joint_refit: dict[str, float]
    changes: list[ParamChange]
    refit_episodes: list[int]
    held_out_episodes: list[int]
    n_evaluations: int
    elapsed_s: float
    refit_params: Any = field(repr=False, default=None)

    @property
    def gap_deg(self) -> float:
        """How much the refit improves on the lock, in degrees."""
        return self.stale_deg - self.refit_deg

    @property
    def excess_deg(self) -> float | None:
        """How far the lock is above its own reference on the fresh episodes;
        None for a lock that carries no reference score."""
        if self.reference_deg is None:
            return None
        return self.stale_deg - self.reference_deg

    @property
    def recovered(self) -> float | None:
        """Share of the excess error the refit removes (1.0 = all of it);
        None when the lock has no reference to measure the excess against."""
        excess = self.excess_deg
        if excess is None:
            return None
        if not math.isfinite(excess) or excess <= 0:
            return 1.0
        return max(0.0, min(1.0, self.gap_deg / excess))

    @property
    def explained(self) -> bool:
        """True when the dynamics parameters can account for the drift. A lock
        without a reference score cannot be judged, so this is False for it."""
        return (
            self.gap_deg > MIN_GAP_DEG
            and self.recovered is not None
            and self.recovered >= EXPLAINED_FRACTION
        )

    @property
    def causes(self) -> list[ParamChange]:
        """Changes that alone recover a real share of the gap, largest first.
        Empty unless the drift is :attr:`explained`, so a consumer that reads
        only the causes agrees with the summary."""
        if not self.explained:
            return []
        cut = CAUSE_FRACTION * self.gap_deg
        hits = [c for c in self.changes if math.isfinite(c.recovers_deg) and c.recovers_deg >= cut]
        return sorted(hits, key=lambda c: -c.recovers_deg)

    def summary(self) -> str:
        recovered = (
            "reference unknown" if self.recovered is None
            else f"{100 * self.recovered:.0f}% of the excess recovered"
        )
        head = (
            f"lock {self.lock} on {self.dataset}: {self.stale_deg:.3f} deg, "
            f"{self.refit_deg:.3f} deg after refitting the changes ({recovered})"
        )
        if self.recovered is None:
            return f"{head}; the lock carries no held-out score, so the drift cannot be judged"
        if not self.explained:
            return (
                f"{head}; not explained by the sim's dynamics parameters "
                f"(look at sensor calibration, contacts, or the task itself)"
            )
        causes = self.causes
        if not causes:
            return f"{head}; several parameters changed together, none explains it alone"
        named = ", ".join(
            f"{c.name} {c.change} (alone recovers {c.recovers_deg:.2f} deg)" for c in causes
        )
        return f"{head}; likely cause: {named}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "lock": self.lock,
            "dataset": self.dataset,
            "reference_deg": _json_float(self.reference_deg),
            "stale_deg": _json_float(self.stale_deg),
            "refit_deg": _json_float(self.refit_deg),
            "gap_deg": _json_float(self.gap_deg),
            "excess_deg": _json_float(self.excess_deg),
            "recovered": _json_float(self.recovered),
            "explained": bool(self.explained),
            "causes": [c.name for c in self.causes],
            "changes": [c.to_dict() for c in self.changes],
            "per_joint_stale": {k: _json_float(v) for k, v in self.per_joint_stale.items()},
            "per_joint_refit": {k: _json_float(v) for k, v in self.per_joint_refit.items()},
            "refit_params": None if self.refit_params is None else self.refit_params.to_dict(),
            "refit_episodes": [int(i) for i in self.refit_episodes],
            "held_out_episodes": [int(i) for i in self.held_out_episodes],
            "n_evaluations": int(self.n_evaluations),
            "elapsed_s": float(self.elapsed_s),
            "summary": self.summary(),
        }


def _json_float(v: float | None) -> float | None:
    """Like :func:`antares.calibrate._json_float`, but None passes through."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------


def split_alternating(episodes: Sequence[Episode]) -> tuple[list[Episode], list[Episode]]:
    """Episodes in index order, alternately to the refit and to the held-out half."""
    eps = sorted(episodes, key=lambda e: e.index)
    return eps[0::2], eps[1::2]


def diagnose(
    episodes: Sequence[Episode],
    lock: Any,
    simulate: Callable[..., Any] | None = None,
    max_evaluations: int = 300,
    sigma0: float = 0.15,
    seed: int = 0,
    dataset: str | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[int, float], None] | None = None,
) -> Diagnosis:
    """Refit ``lock`` as per-parameter changes on fresh ``episodes`` and name
    the changes that explain the drift.

    ``lock`` is a :class:`~antares.lock.RobotLock` or its JSON dict. Needs at
    least two episodes, split alternately into refit and held-out halves.
    ``max_evaluations`` (at least :data:`MIN_EVALUATIONS`) is an exact cap on
    the objective evaluations spent. ``should_stop`` is polled between
    evaluations; when it returns True the search ends and the best change
    so far is scored. ``on_progress(evaluations, best_value)`` is called
    after every evaluation.
    """
    import cma

    from antares.check import _lock_view

    if max_evaluations < MIN_EVALUATIONS:
        raise ValueError(f"max_evaluations must be >= {MIN_EVALUATIONS}, got {max_evaluations}")
    t0 = time.perf_counter()
    name, lock_dataset, params, reference, _ = _lock_view(lock)
    refit_eps, held_out = split_alternating(episodes)
    if not refit_eps or not held_out:
        raise ValueError("diagnose needs at least two episodes")
    if simulate is None:
        from antares.calibrate import default_simulate

        simulate = default_simulate()
    stop = should_stop or (lambda: False)

    evaluations = 0
    best: dict[str, float] = neutral()
    best_value = math.inf

    def objective(change: Mapping[str, float]) -> float:
        nonlocal evaluations, best, best_value
        _, error = _safe_score(apply_change(params, change), refit_eps, simulate)
        value = error + complexity(change)
        evaluations += 1
        if value < best_value:
            best, best_value = dict(change), value
        if on_progress is not None:
            on_progress(evaluations, best_value)
        return value

    # Latency is a handful of whole frames, which CMA-ES steps over poorly,
    # so each shift is scored with no other change, and the factors are then
    # refitted by CMA-ES separately for each of the most promising shifts.
    # A slower servo and extra lag look alike until the factors move, so
    # the best shift before the refit is not always the best one after it.
    # Shifts that the lock's latency clamps to the same value are scanned once.
    lo, hi = LATENCY_SHIFT
    scanned = []
    seen: set[int] = set()
    for shift in range(lo, hi + 1):
        shift = effective_shift(shift, params)
        if shift in seen:
            continue
        seen.add(shift)
        change = {**neutral(), "latency_frames": shift}
        scanned.append((objective(change), shift != 0, change))
        if stop():
            break
    scanned.sort(key=lambda t: (t[0], t[1]))
    candidates = [c for _, _, c in scanned[:SHIFT_CANDIDATES]]
    factors = list(FACTOR_RANGE)
    for i, start in enumerate(candidates):
        if stop():
            break
        budget = (max_evaluations - evaluations) // (len(candidates) - i)
        if budget < 1:
            continue
        shift = start["latency_frames"]
        es = cma.CMAEvolutionStrategy(
            encode_change(start)[: len(factors)],
            sigma0,
            {
                "bounds": [0.0, 1.0],
                "seed": int(np.random.default_rng([seed, i]).integers(1, 2**31 - 1)),
                "verbose": -9,
                "verb_log": 0,
                "maxfevals": budget,
            },
        )
        spent = 0
        while spent < budget and not es.stop() and not stop():
            xs = es.ask()
            values = []
            # Never past the budget; a partial generation still counts toward
            # the best but is not fed back to CMA-ES (as in calibrate).
            for x in xs[: budget - spent]:
                change = decode_change(np.append(x, 0.0))
                change["latency_frames"] = shift
                values.append(objective(change))
                spent += 1
                if spent < budget and stop():
                    break
            if len(values) != len(xs):
                break
            es.tell(xs, values)

    # Prune: drop a kept change while the unpenalized refit error rises by
    # less than STEP_GAIN_DEG without it, smallest cost first. What is left
    # is the fewest changes that still explain the recording.
    def refit_error(change: Mapping[str, float]) -> float:
        return _safe_score(apply_change(params, change), refit_eps, simulate)[1]

    if not stop():
        kept_error = refit_error(best)
        while True:
            costs = []
            for f in FIELDS:
                if _is_neutral(f, best[f]):
                    continue
                trial = {**best, f: neutral()[f]}
                costs.append((refit_error(trial) - kept_error, f, trial))
            if not costs:
                break
            cost, f, trial = min(costs)
            if cost >= STEP_GAIN_DEG:
                break
            best, kept_error = trial, kept_error + cost

    stale_report, stale = _safe_score(params, held_out, simulate)
    refit_params = apply_change(params, best)
    refit_report, refit = _safe_score(refit_params, held_out, simulate)
    if refit > stale:  # the refit did not generalize; the lock stands
        best, refit_params, refit_report, refit = neutral(), params, stale_report, stale

    changes = []
    for f in FIELDS:
        v = best[f]
        if _is_neutral(f, v):
            alone = stale
        else:
            _, alone = _safe_score(apply_change(params, {f: v}), held_out, simulate)
        changes.append(ParamChange(name=f, value=v, alone_deg=alone, stale_deg=stale))

    return Diagnosis(
        lock=name,
        dataset=dataset if dataset is not None else lock_dataset,
        reference_deg=reference,
        stale_deg=stale,
        refit_deg=refit,
        per_joint_stale=_per_joint(stale_report),
        per_joint_refit=_per_joint(refit_report),
        changes=changes,
        refit_episodes=[e.index for e in refit_eps],
        held_out_episodes=[e.index for e in held_out],
        n_evaluations=evaluations,
        elapsed_s=time.perf_counter() - t0,
        refit_params=refit_params,
    )


__all__ = [
    "CAUSE_FRACTION",
    "COMPLEXITY_DEG",
    "LATENCY_COMPLEXITY_DEG",
    "SHIFT_CANDIDATES",
    "EXPLAINED_FRACTION",
    "FACTOR_RANGE",
    "FIELDS",
    "LATENCY_SHIFT",
    "MIN_EVALUATIONS",
    "MIN_GAP_DEG",
    "NEUTRAL_TOL",
    "PENALTY",
    "STEP_GAIN_DEG",
    "complexity",
    "Diagnosis",
    "ParamChange",
    "apply_change",
    "decode_change",
    "describe_change",
    "diagnose",
    "effective_shift",
    "encode_change",
    "neutral",
    "split_alternating",
]
