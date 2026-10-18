"""Name the cause of drift: what changed in the arm since its lock was fitted.

:func:`antares.check.check` says *whether* a robot still matches its lock.
:func:`diagnose` says *what* changed. It refits the lock on fresh episodes,
but instead of searching the parameters from scratch it searches one
*change* per parameter on top of the lock: a factor on each of ``kp``,
``kv``, ``damping``, ``armature`` and ``frictionloss`` (x1/4 to x4,
log-uniform) and a shift of whole frames on ``latency_frames``. A per-joint
lock keeps its per-joint shape; every joint gets the same change. The search
starts at "no change" for each of the most promising latency shifts, so a
parameter only moves when the new data asks for it.

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
#: A parameter is a cause when changing it alone recovers this share of the gap.
CAUSE_FRACTION = 0.2
#: A refit that improves on the lock by less than this (degrees) is noise.
MIN_GAP_DEG = 0.05
#: Below this share of the excess error recovered, the drift is reported as
#: something the dynamics parameters cannot explain.
EXPLAINED_FRACTION = 0.5


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
    return int(value) == 0 if name == "latency_frames" else abs(float(value) - 1.0) < 1e-9


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
    def excess_deg(self) -> float:
        """How far the lock is above its own reference on the fresh episodes."""
        ref = self.reference_deg if self.reference_deg is not None else self.refit_deg
        return self.stale_deg - ref

    @property
    def recovered(self) -> float:
        """Share of the excess error the refit removes (1.0 = all of it)."""
        if not math.isfinite(self.excess_deg) or self.excess_deg <= 0:
            return 1.0
        return max(0.0, min(1.0, self.gap_deg / self.excess_deg))

    @property
    def explained(self) -> bool:
        """True when the dynamics parameters can account for the drift."""
        return self.gap_deg > MIN_GAP_DEG and self.recovered >= EXPLAINED_FRACTION

    @property
    def causes(self) -> list[ParamChange]:
        """Changes that alone recover a real share of the gap, largest first."""
        if self.gap_deg <= MIN_GAP_DEG:
            return []
        cut = CAUSE_FRACTION * self.gap_deg
        hits = [c for c in self.changes if math.isfinite(c.recovers_deg) and c.recovers_deg >= cut]
        return sorted(hits, key=lambda c: -c.recovers_deg)

    def summary(self) -> str:
        head = (
            f"lock {self.lock} on {self.dataset}: {self.stale_deg:.3f} deg, "
            f"{self.refit_deg:.3f} deg after refitting the changes "
            f"({100 * self.recovered:.0f}% of the excess recovered)"
        )
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
    max_evaluations: int = 240,
    sigma0: float = 0.15,
    seed: int = 0,
    dataset: str | None = None,
) -> Diagnosis:
    """Refit ``lock`` as per-parameter changes on fresh ``episodes`` and name
    the changes that explain the drift.

    ``lock`` is a :class:`~antares.lock.RobotLock` or its JSON dict. Needs at
    least two episodes, split alternately into refit and held-out halves.
    """
    import cma

    from antares.check import _lock_view

    t0 = time.perf_counter()
    name, lock_dataset, params, reference, _ = _lock_view(lock)
    refit_eps, held_out = split_alternating(episodes)
    if not refit_eps or not held_out:
        raise ValueError("diagnose needs at least two episodes")
    if simulate is None:
        from antares.calibrate import default_simulate

        simulate = default_simulate()

    def objective(change: Mapping[str, float]) -> float:
        _, value = _safe_score(apply_change(params, change), refit_eps, simulate)
        return value

    # Latency is a handful of whole frames, which CMA-ES steps over poorly,
    # so each shift is scored with no other change, and the factors are then
    # refitted by CMA-ES separately for each of the most promising shifts.
    # A slower servo and extra lag look alike until the factors move, so
    # the best shift before the refit is not always the best one after it.
    lo, hi = LATENCY_SHIFT
    scanned = []
    for shift in range(lo, hi + 1):
        change = {**neutral(), "latency_frames": shift}
        scanned.append((objective(change), shift != 0, change))
    evaluations = len(scanned)
    scanned.sort(key=lambda t: (t[0], t[1]))
    best_value, _, best = scanned[0]
    candidates = [c for _, _, c in scanned[:SHIFT_CANDIDATES]]
    factors = list(FACTOR_RANGE)
    for i, start in enumerate(candidates):
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
        while spent < budget and not es.stop():
            xs = es.ask()
            values = []
            for x in xs:
                change = decode_change(np.append(x, 0.0))
                change["latency_frames"] = shift
                v = objective(change)
                values.append(v)
                spent += 1
                if v < best_value:
                    best, best_value = change, v
            es.tell(xs, values)
        evaluations += spent

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


#: Objective value for a change that makes the sim unstable (NaN/inf).
PENALTY = 1e3


def _safe_score(params, episodes: Sequence[Episode], simulate: Callable[..., Any]):
    """``(report, overall_deg)``; an unstable or failing sim scores PENALTY."""
    from antares.gap import aggregate, tracking_error

    try:
        report = aggregate([tracking_error(*simulate(params, ep)) for ep in episodes])
    except (FloatingPointError, ValueError, ArithmeticError):
        return None, PENALTY
    v = float(report.overall_deg)
    return report, (v if math.isfinite(v) else PENALTY)


def _per_joint(report: Any) -> dict[str, float]:
    if report is None:
        return {}
    return {k: float(v) for k, v in report.per_joint_deg.items()}


__all__ = [
    "CAUSE_FRACTION",
    "EXPLAINED_FRACTION",
    "FACTOR_RANGE",
    "FIELDS",
    "LATENCY_SHIFT",
    "MIN_GAP_DEG",
    "SHIFT_CANDIDATES",
    "Diagnosis",
    "ParamChange",
    "apply_change",
    "decode_change",
    "describe_change",
    "diagnose",
    "encode_change",
    "neutral",
    "split_alternating",
]
