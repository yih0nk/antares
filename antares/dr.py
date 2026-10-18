"""Domain-randomization ranges derived from one robot's data.

RL and imitation pipelines usually randomize simulator parameters inside
hand-picked ranges ("friction +-30%"). Once a lock exists for a robot we
can measure instead of guess: refit the simulator on several random
subsets of that robot's episodes, warm-started from the lock, and read the
spread of the fitted parameters. The spread is how much the data itself
leaves the parameters undetermined, which is exactly what a policy should
be robust to.

:func:`bootstrap_fits` runs the cheap warm-started refits (it reuses
:func:`antares.calibrate.calibrate`), :class:`DRRanges` turns their
fitted parameters into per-parameter ``[lo, hi]`` ranges with provenance
and saves them next to the lock as ``locks/<name>.dr.json``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from antares import __version__
from antares.calibrate import (
    SERVO_SPACE,
    SPACE,
    CalibrationResult,
    ProgressEvent,
    SimParams,
    SimulateFn,
    calibrate,
    default_simulate,
)
from antares.data import Episode
from antares.lock import JsonFile, now_iso, slug

DR_SUFFIX = ".dr.json"

#: Default percentiles of the fitted values that become ``lo`` / ``hi``.
PERCENTILES = (5.0, 95.0)


def _as_params(lock_or_start: Any) -> SimParams:
    """Accept a :class:`SimParams`, anything with a ``.params`` (a lock), or a dict."""
    if isinstance(lock_or_start, SimParams):
        return lock_or_start
    if isinstance(lock_or_start, dict):
        return SimParams.from_dict(lock_or_start)
    params = getattr(lock_or_start, "params", None)
    if isinstance(params, SimParams):
        return params
    raise TypeError(f"expected SimParams, a RobotLock or a dict, got {type(lock_or_start).__name__}")


# --------------------------------------------------------------------------
# Bootstrap fits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FitProgress:
    """One objective evaluation inside one bootstrap fit."""

    fit: int
    n_fits: int
    event: ProgressEvent

    def to_dict(self) -> dict:
        return {"fit": int(self.fit), "n_fits": int(self.n_fits), **self.event.to_dict()}


def bootstrap_fits(
    episodes: Sequence[Episode],
    lock_or_start: Any,
    n_fits: int = 8,
    n_train: int = 12,
    max_evaluations: int = 60,
    seed: int = 0,
    on_progress: Callable[[FitProgress], None] | None = None,
    *,
    sigma0: float = 0.1,
    simulate: SimulateFn | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_fit: Callable[[int, CalibrationResult], None] | None = None,
) -> list[CalibrationResult]:
    """Refit the simulator on ``n_fits`` random subsets of ``episodes``.

    Each fit trains on ``n_train`` episodes drawn without replacement (a
    subsampling bootstrap), warm-started from ``lock_or_start`` with a small
    CMA-ES step ``sigma0`` so it only explores the neighbourhood the data
    cannot pin down. No held-out split is scored (the training report stands
    in), which keeps each fit at ``max_evaluations * n_train`` replays.

    ``on_progress`` fires per objective evaluation, ``on_fit(i, result)``
    once per finished fit. ``should_stop`` is polled between fits and inside
    each fit; fits that finished before it fired are returned. The draw
    order is fixed by ``seed``, so the same inputs give the same subsets and
    the same fits.
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("bootstrap_fits needs at least one episode")
    if n_fits < 1:
        raise ValueError("n_fits must be >= 1")
    if n_train < 1:
        raise ValueError("n_train must be >= 1")
    start = _as_params(lock_or_start)
    n_train = min(n_train, len(episodes))
    rng = np.random.default_rng(seed)
    if simulate is None:
        simulate = default_simulate()
    stop = should_stop or (lambda: False)

    fits: list[CalibrationResult] = []
    for i in range(n_fits):
        if stop():
            break
        idx = np.sort(rng.choice(len(episodes), size=n_train, replace=False))
        train = [episodes[j] for j in idx]
        fit_seed = int(rng.integers(1, 2**31 - 1))
        cb = None
        if on_progress is not None:

            def cb(ev: ProgressEvent, _i: int = i) -> None:
                on_progress(FitProgress(_i, n_fits, ev))

        result = calibrate(
            train,
            [],
            simulate=simulate,
            start=start,
            max_evaluations=max_evaluations,
            sigma0=sigma0,
            seed=fit_seed,
            on_progress=cb,
            should_stop=should_stop,
        )
        fits.append(result)
        if on_fit is not None:
            on_fit(i, result)
    return fits


# --------------------------------------------------------------------------
# Ranges
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamRange:
    """Randomization range of one parameter.

    ``scale`` is ``"log"`` (sample log-uniformly) or ``"int"`` (uniform
    integer, inclusive). ``samples`` are the fitted values the range was
    read from, kept for provenance; they are empty for hand-picked ranges.
    """

    nominal: float
    lo: float
    hi: float
    scale: str
    samples: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.scale not in ("log", "int"):
            raise ValueError(f"unknown scale {self.scale!r}")
        if not (self.lo <= self.nominal <= self.hi):
            raise ValueError(f"need lo <= nominal <= hi, got {self.lo}, {self.nominal}, {self.hi}")
        if self.scale == "log" and self.lo <= 0:
            raise ValueError("log-scale range must be positive")
        object.__setattr__(self, "samples", tuple(float(s) for s in self.samples))
        if self.scale == "int":
            for name in ("nominal", "lo", "hi"):
                object.__setattr__(self, name, int(round(getattr(self, name))))

    def to_dict(self) -> dict:
        return {
            "nominal": self.nominal,
            "lo": self.lo,
            "hi": self.hi,
            "scale": self.scale,
            "samples": list(self.samples),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParamRange":
        return cls(
            nominal=float(d["nominal"]),
            lo=float(d["lo"]),
            hi=float(d["hi"]),
            scale=str(d["scale"]),
            samples=tuple(float(s) for s in d.get("samples", [])),
        )


def space_of(params: SimParams) -> dict[str, tuple[float, float, str]]:
    """The shared search space a refit from ``params`` explores, and so the
    dimensions the randomization covers: :data:`~antares.calibrate.SERVO_SPACE`
    for a servo-model lock (:func:`~antares.calibrate.calibrate` switches to
    it when the start selects a servo), else :data:`~antares.calibrate.SPACE`."""
    return SERVO_SPACE if params.uses_servo_model else SPACE


def dims_of(params: SimParams) -> tuple[str, ...]:
    """Names of the randomized parameters for ``params`` (see :func:`space_of`)."""
    return tuple(space_of(params))


def _range_from_values(name: str, nominal: float, values: Sequence[float], percentiles, scale: str) -> ParamRange:
    vals = np.asarray(values, dtype=np.float64)
    if scale == "int":
        lo, hi = int(vals.min()), int(vals.max())
    else:
        p_lo, p_hi = np.percentile(np.log10(vals), percentiles)
        lo, hi = float(10**p_lo), float(10**p_hi)
    # The calibrated point is always inside its own randomization range.
    lo, hi = min(lo, nominal), max(hi, nominal)
    return ParamRange(nominal=nominal, lo=lo, hi=hi, scale=scale, samples=tuple(vals.tolist()))


@dataclass(frozen=True)
class DRRanges(JsonFile):
    """Per-parameter randomization ranges for one robot, with provenance."""

    FILE_SUFFIX: ClassVar[str] = DR_SUFFIX

    name: str
    nominal: SimParams
    ranges: dict[str, ParamRange]
    dataset: str = ""
    lock: str | None = None
    n_fits: int = 0
    n_train: int = 0
    max_evaluations: int = 0
    seed: int = 0
    percentiles: tuple[float, float] = PERCENTILES
    fit_train_scores: tuple[float, ...] = ()
    fit_episodes: tuple[tuple[int, ...], ...] = ()
    created_at: str = field(default_factory=now_iso)
    antares_version: str = __version__

    @classmethod
    def from_fits(
        cls,
        fits: Sequence[CalibrationResult],
        nominal: Any,
        dataset: str,
        lock: str | None = None,
        name: str | None = None,
        seed: int = 0,
        percentiles: tuple[float, float] = PERCENTILES,
    ) -> "DRRanges":
        """Read ``lo``/``hi`` off the fitted parameters of ``fits``.

        Log-scale parameters take the ``percentiles`` of the fitted values
        in log space; latency takes the integer min/max. Every range is
        widened, if needed, to contain the nominal (lock) value. The ranges
        cover the dimensions the fits searched (:func:`dims_of`): the servo
        gain, rate limit, friction, inertia and latency for a servo-model
        lock, the position actuator's gains otherwise.
        """
        fits = list(fits)
        if not fits:
            raise ValueError("from_fits needs at least one fit")
        nominal = _as_params(nominal)
        space = space_of(nominal)
        ranges = {
            dim: _range_from_values(
                dim, getattr(nominal, dim), [getattr(f.params, dim) for f in fits], percentiles, scale
            )
            for dim, (_, _, scale) in space.items()
        }
        n_train = max(len(f.train_episodes) for f in fits)
        return cls(
            name=name or lock or slug(dataset),
            nominal=nominal,
            ranges=ranges,
            dataset=dataset,
            lock=lock,
            n_fits=len(fits),
            n_train=n_train,
            max_evaluations=max(f.n_evaluations for f in fits),
            seed=seed,
            percentiles=(float(percentiles[0]), float(percentiles[1])),
            fit_train_scores=tuple(float(f.train_score) for f in fits),
            fit_episodes=tuple(tuple(int(i) for i in f.train_episodes) for f in fits),
        )

    def __getitem__(self, name: str) -> ParamRange:
        return self.ranges[name]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "lock": self.lock,
            "nominal": self.nominal.to_dict(),
            "ranges": {k: v.to_dict() for k, v in self.ranges.items()},
            "n_fits": int(self.n_fits),
            "n_train": int(self.n_train),
            "max_evaluations": int(self.max_evaluations),
            "seed": int(self.seed),
            "percentiles": list(self.percentiles),
            "fit_train_scores": [_json_float(s) for s in self.fit_train_scores],
            "fit_episodes": [list(e) for e in self.fit_episodes],
            "created_at": self.created_at,
            "antares_version": self.antares_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DRRanges":
        pct = d.get("percentiles") or list(PERCENTILES)
        return cls(
            name=str(d["name"]),
            nominal=SimParams.from_dict(d["nominal"]),
            ranges={k: ParamRange.from_dict(v) for k, v in d["ranges"].items()},
            dataset=str(d.get("dataset", "")),
            lock=d.get("lock"),
            n_fits=int(d.get("n_fits", 0)),
            n_train=int(d.get("n_train", 0)),
            max_evaluations=int(d.get("max_evaluations", 0)),
            seed=int(d.get("seed", 0)),
            percentiles=(float(pct[0]), float(pct[1])),
            fit_train_scores=tuple(float("nan") if s is None else float(s) for s in d.get("fit_train_scores", [])),
            fit_episodes=tuple(tuple(int(i) for i in e) for e in d.get("fit_episodes", [])),
            created_at=str(d.get("created_at", "")),
            antares_version=str(d.get("antares_version", "")),
        )


def _json_float(v: float) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def uniform_ranges(nominal: Any, frac: float = 0.3, name: str | None = None) -> DRRanges:
    """The hand-picked alternative: ``nominal * [1 - frac, 1 + frac]`` per parameter.

    Latency is widened to the rounded integer bounds. Useful as the baseline
    data-derived ranges are compared against; it carries no provenance.
    """
    if not 0 <= frac < 1:
        raise ValueError("frac must be in [0, 1)")
    nominal = _as_params(nominal)
    ranges = {}
    for dim, (_, _, scale) in space_of(nominal).items():
        v = float(getattr(nominal, dim))
        lo, hi = v * (1 - frac), v * (1 + frac)
        if scale == "int":
            lo, hi = max(0, int(round(lo))), int(round(hi))
        ranges[dim] = ParamRange(nominal=v, lo=min(lo, v), hi=max(hi, v), scale=scale)
    return DRRanges(name=name or f"pm{int(round(frac * 100))}", nominal=nominal, ranges=ranges)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def sample(ranges: DRRanges, rng: np.random.Generator | int | None, n: int) -> list[SimParams]:
    """Draw ``n`` parameter sets: log-uniform inside ``[lo, hi]`` for log-scale
    parameters, uniform integers (inclusive) for latency.

    Sets are drawn one at a time, so ``sample(r, rng, n)[:k]`` equals
    ``sample(r, rng_copy, k)``: drawing in a loop or in a batch gives the
    same sequence for the same generator state.

    Only the parameters ``ranges`` carries are drawn (the dimensions the
    fits searched, :func:`dims_of`); every other field (``forcerange``,
    ``gravity_scale``, ``deadband``, the servo's motor constants) comes
    from ``ranges.nominal``, so a lock fitted with them keeps them in every
    draw, and a servo-model lock stays a servo-model lock whose ``kp``
    follows the drawn gain.
    """
    rng = np.random.default_rng(rng)
    if n < 0:
        raise ValueError("n must be >= 0")
    out: list[SimParams] = []
    for _ in range(n):
        kwargs: dict[str, float | int] = {}
        for dim, r in ranges.ranges.items():
            if r.scale == "int":
                kwargs[dim] = int(rng.integers(int(r.lo), int(r.hi) + 1))
            else:
                kwargs[dim] = float(10 ** rng.uniform(math.log10(r.lo), math.log10(r.hi)))
        out.append(ranges.nominal.replace(**kwargs))
    return out


def apply(arm: Any, params: SimParams) -> SimParams:
    """Write ``params`` into ``arm`` (an :class:`antares.sim.Arm`) and return them."""
    arm.apply(params)
    return params


def randomize(arm: Any, ranges: DRRanges, rng: np.random.Generator | int | None = None) -> SimParams:
    """Sample one parameter set from ``ranges`` and apply it to ``arm``."""
    return apply(arm, sample(ranges, rng, 1)[0])


__all__ = [
    "DR_SUFFIX",
    "PERCENTILES",
    "DRRanges",
    "FitProgress",
    "ParamRange",
    "apply",
    "bootstrap_fits",
    "dims_of",
    "randomize",
    "sample",
    "space_of",
    "uniform_ranges",
]
