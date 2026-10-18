"""Parameter drift over a recording session.

Episodes are recorded in order, so if a servo heats up or a joint wears
during the session, the simulator parameters that fit the first episodes
should stop fitting the last ones. :func:`drift_scan` slides a window over
the episodes (by ``episode_index``), refits the sim on each window starting
from a saved lock's parameters, and records how the fitted parameters and
the lock's own tracking error move across the session.

Each per-window fit is cheap: it is warm-started from the lock with a small
CMA-ES step size and a short evaluation budget, so a nine-window scan of a
50-episode dataset takes well under a minute.

Trend statistics per parameter (and for the static lock score) are an
ordinary least-squares slope against the window's mean episode index, in
log10 space for the log-scaled parameters, with r² and a 95% percentile
bootstrap interval over windows. A quantity is flagged ``drifting`` only
when the interval excludes zero *and* the fitted line moves by more than a
relative threshold (10% by default) from the first window to the last.
Overlapping windows are not independent, so the interval is optimistic;
treat the flag as "worth a look", not as a measurement of wear.

A parameter trend is only *announced* as drift when the lock pins that
parameter (:func:`antares.lock.pinned_params`): several parameter sets
reproduce an arm equally well, so a parameter that moves across seeds at
the same error can trend across windows without anything in the arm having
changed. Scores always count. Scans saved without a ``pinned`` list keep
the old, ungated verdict.

The simulator is injected as a ``simulate(params, episode) -> (real_q,
sim_q)`` callable like in :mod:`antares.calibrate`, so this module and its
tests run without MuJoCo.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from antares import __version__
from antares.data import Episode
from antares.lock import now_iso

DRIFT_DIR = Path("locks")
DRIFT_SUFFIX = ".drift.json"

#: Relative change across the session above which a significant slope counts.
DEFAULT_THRESHOLD = 0.10

#: Keys of the trend table that are scores, not parameters.
SCORE_KEYS = ("lock_score", "train_score")

try:  # parameters searched on a log scale, from the calibration search spaces
    from antares.calibrate import SERVO_SPACE as _SERVO_SPACE
    from antares.calibrate import SPACE as _SPACE

    LOG_PARAMS: frozenset[str] = frozenset(
        n for space in (_SPACE, _SERVO_SPACE) for n, (_, _, s) in space.items() if s == "log"
    )
    del _SPACE, _SERVO_SPACE
except (ImportError, AttributeError):  # pragma: no cover - stubbed sibling module
    LOG_PARAMS = frozenset({"kp", "kv", "damping", "armature", "frictionloss", "servo_gain", "max_velocity"})


def _json_float(v: float) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def _parse_float(v: Any) -> float:
    return float("nan") if v is None else float(v)


def _sim_params_from_dict(d: dict) -> Any:
    try:
        from antares.sim import SimParams
    except ImportError:  # pragma: no cover
        from antares.calibrate import SimParams  # type: ignore
    return SimParams.from_dict(d)


# --------------------------------------------------------------------------
# Trend statistics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Trend:
    """Linear trend of one quantity against episode index.

    ``slope`` is per episode, in log10 units when ``log_space`` is set.
    ``total_change`` is the relative change of the fitted line between the
    first and last window (``0.12`` = +12%). ``ci_lo``/``ci_hi`` bound the
    slope at 95% from a bootstrap over windows.
    """

    slope: float
    intercept: float
    r2: float
    ci_lo: float
    ci_hi: float
    total_change: float
    log_space: bool
    n: int
    drifting: bool

    @property
    def significant(self) -> bool:
        """The bootstrap interval excludes a zero slope."""
        if not (math.isfinite(self.ci_lo) and math.isfinite(self.ci_hi)):
            return False
        return self.ci_lo > 0.0 or self.ci_hi < 0.0

    def to_dict(self) -> dict:
        return {
            "slope": _json_float(self.slope),
            "intercept": _json_float(self.intercept),
            "r2": _json_float(self.r2),
            "ci_lo": _json_float(self.ci_lo),
            "ci_hi": _json_float(self.ci_hi),
            "total_change": _json_float(self.total_change),
            "log_space": bool(self.log_space),
            "n": int(self.n),
            "significant": bool(self.significant),
            "drifting": bool(self.drifting),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trend":
        return cls(
            slope=_parse_float(d.get("slope")),
            intercept=_parse_float(d.get("intercept")),
            r2=_parse_float(d.get("r2")),
            ci_lo=_parse_float(d.get("ci_lo")),
            ci_hi=_parse_float(d.get("ci_hi")),
            total_change=_parse_float(d.get("total_change")),
            log_space=bool(d.get("log_space", False)),
            n=int(d.get("n", 0)),
            drifting=bool(d.get("drifting", False)),
        )


def _nan_trend(log_space: bool, n: int) -> Trend:
    nan = float("nan")
    return Trend(nan, nan, nan, nan, nan, nan, log_space, n, False)


def fit_trend(
    x: Sequence[float],
    y: Sequence[float],
    *,
    log_space: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    n_boot: int = 1000,
    seed: int = 0,
) -> Trend:
    """OLS slope of ``y`` against ``x`` with r², bootstrap CI and drift flag.

    Non-finite pairs (and non-positive ``y`` in log space) are dropped. With
    fewer than two distinct ``x`` the trend is undefined (all NaN, not
    drifting); the bootstrap needs at least three points.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    keep = np.isfinite(x) & np.isfinite(y)
    if log_space:
        keep &= y > 0
    x, y = x[keep], y[keep]
    n = int(len(x))
    if n < 2 or np.ptp(x) == 0:
        return _nan_trend(log_space, n)
    yy = np.log10(y) if log_space else y

    xm, ym = x.mean(), yy.mean()
    slope = float(np.sum((x - xm) * (yy - ym)) / np.sum((x - xm) ** 2))
    intercept = float(ym - slope * xm)
    pred = intercept + slope * x
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ci_lo = ci_hi = float("nan")
    if n >= 3 and n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        xb, yb = x[idx], yy[idx]
        ok = np.ptp(xb, axis=1) > 0
        xb, yb = xb[ok], yb[ok]
        if len(xb):
            xm = xb.mean(axis=1, keepdims=True)
            ym = yb.mean(axis=1, keepdims=True)
            slopes = np.sum((xb - xm) * (yb - ym), axis=1) / np.sum((xb - xm) ** 2, axis=1)
            ci_lo, ci_hi = (float(v) for v in np.percentile(slopes, [2.5, 97.5]))

    span = float(x.max() - x.min())
    if log_space:
        total_change = 10.0 ** (slope * span) - 1.0
    else:
        first = intercept + slope * float(x.min())
        last = intercept + slope * float(x.max())
        denom = abs(first) if abs(first) > 1e-12 else abs(last)
        total_change = (last - first) / denom if denom > 1e-12 else 0.0

    trend = Trend(slope, intercept, r2, ci_lo, ci_hi, total_change, log_space, n, False)
    drifting = (
        trend.significant and math.isfinite(total_change) and abs(total_change) > threshold
    )
    return Trend(slope, intercept, r2, ci_lo, ci_hi, total_change, log_space, n, drifting)


# --------------------------------------------------------------------------
# Windows and the scan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftWindow:
    """One refit on a block of consecutive episodes."""

    episodes: list[int]  # episode indices in the window, ascending
    params: Any  # SimParams fitted on this window
    train_score: float  # fitted params on the window, deg RMSE
    lock_score: float  # the lock's static params on the same window
    per_joint_fit: dict[str, float] = field(default_factory=dict)
    per_joint_lock: dict[str, float] = field(default_factory=dict)
    n_evaluations: int = 0
    elapsed_s: float = 0.0

    @property
    def first(self) -> int:
        return int(self.episodes[0])

    @property
    def last(self) -> int:
        return int(self.episodes[-1])

    @property
    def center(self) -> float:
        """Mean episode index, the x coordinate used for trends."""
        return float(np.mean(self.episodes))

    @property
    def improvement(self) -> float:
        """Fraction of the lock's error on this window the refit removed."""
        if not math.isfinite(self.lock_score) or self.lock_score <= 0:
            return 0.0
        return 1.0 - self.train_score / self.lock_score

    def to_dict(self) -> dict:
        return {
            "episodes": [int(i) for i in self.episodes],
            "first": self.first,
            "last": self.last,
            "center": self.center,
            "params": self.params.to_dict(),
            "train_score": _json_float(self.train_score),
            "lock_score": _json_float(self.lock_score),
            "improvement": _json_float(self.improvement),
            "per_joint_fit": {k: _json_float(v) for k, v in self.per_joint_fit.items()},
            "per_joint_lock": {k: _json_float(v) for k, v in self.per_joint_lock.items()},
            "n_evaluations": int(self.n_evaluations),
            "elapsed_s": float(self.elapsed_s),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DriftWindow":
        return cls(
            episodes=[int(i) for i in d["episodes"]],
            params=_sim_params_from_dict(d["params"]),
            train_score=_parse_float(d.get("train_score")),
            lock_score=_parse_float(d.get("lock_score")),
            per_joint_fit={k: _parse_float(v) for k, v in d.get("per_joint_fit", {}).items()},
            per_joint_lock={k: _parse_float(v) for k, v in d.get("per_joint_lock", {}).items()},
            n_evaluations=int(d.get("n_evaluations", 0)),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
        )


@dataclass(frozen=True)
class DriftProgress:
    """Emitted once per finished window."""

    index: int  # 0-based window number
    n_windows: int
    window: DriftWindow
    elapsed_s: float

    def to_dict(self) -> dict:
        return {
            "index": int(self.index),
            "n_windows": int(self.n_windows),
            "window": self.window.to_dict(),
            "elapsed_s": float(self.elapsed_s),
        }


@dataclass(frozen=True)
class DriftScan:
    lock: str
    dataset: str
    lock_params: Any
    window: int
    step: int
    max_evaluations: int
    seed: int
    threshold: float
    windows: list[DriftWindow]
    trends: dict[str, Trend]
    n_windows_planned: int
    cancelled: bool = False
    elapsed_s: float = 0.0
    created_at: str = field(default_factory=now_iso)
    antares_version: str = __version__
    #: Parameters the lock pins (see :func:`antares.lock.pinned_params`); only
    #: these are announced as drifting. None (scans from before seed spread
    #: was recorded) keeps the ungated verdict.
    pinned: list[str] | None = None

    @property
    def drifting(self) -> dict[str, bool]:
        """Raw trend flags, scores and parameters alike, gated by nothing."""
        return {k: t.drifting for k, t in self.trends.items()}

    @property
    def drifting_keys(self) -> list[str]:
        return [k for k, d in self.drifting.items() if d]

    def is_pinned(self, key: str) -> bool:
        """Scores always count; a parameter counts when the lock pins it."""
        return key in SCORE_KEYS or self.pinned is None or key in self.pinned

    @property
    def announced_keys(self) -> list[str]:
        """Drifting quantities the verdict names: scores, and pinned parameters."""
        return [k for k in self.drifting_keys if self.is_pinned(k)]

    @property
    def unpinned_drifting(self) -> list[str]:
        """Parameters that trend but are not pinned, so not read as wear."""
        return [k for k in self.drifting_keys if not self.is_pinned(k)]

    @property
    def mean_improvement(self) -> float:
        """Average fraction of the lock's error the per-window refits removed.

        When this is small the windows do not constrain the parameters, so a
        parameter trend is more likely scatter than wear.
        """
        if not self.windows:
            return float("nan")
        return float(np.mean([w.improvement for w in self.windows]))

    def verdict(self) -> str:
        """One-line human summary."""
        if len(self.windows) < 3:
            return "too few windows to estimate a trend"
        hits = []
        for k in self.announced_keys:
            t = self.trends[k]
            hits.append(f"{k} {t.total_change:+.0%} over the session (r²={t.r2:.2f})")
        unpinned = self.unpinned_drifting
        unpinned_note = (
            f"{'/'.join(unpinned)} move across seeds at equal error, so their trends are not read as wear"
            if unpinned
            else ""
        )
        if not hits:
            out = (
                "no parameter or score drifts across the session "
                f"(no trend both significant and larger than {self.threshold:.0%})"
            )
            return f"{out}; {unpinned_note}" if unpinned_note else out
        out = "drift: " + "; ".join(hits)
        params_hit = [k for k in self.announced_keys if k not in SCORE_KEYS]
        if params_hit and self.mean_improvement < self.threshold:
            out += (
                f"; caution: refits beat the lock by only {self.mean_improvement:.0%} on average, "
                "so parameter trends are weakly identified"
            )
        if unpinned_note:
            out += f"; {unpinned_note}"
        return out

    def to_dict(self) -> dict:
        return {
            "lock": self.lock,
            "dataset": self.dataset,
            "lock_params": self.lock_params.to_dict(),
            "window": int(self.window),
            "step": int(self.step),
            "max_evaluations": int(self.max_evaluations),
            "seed": int(self.seed),
            "threshold": float(self.threshold),
            "windows": [w.to_dict() for w in self.windows],
            "trends": {k: t.to_dict() for k, t in self.trends.items()},
            "drifting": self.drifting,
            "pinned": None if self.pinned is None else [str(p) for p in self.pinned],
            "announced": self.announced_keys,
            "mean_improvement": _json_float(self.mean_improvement),
            "verdict": self.verdict(),
            "n_windows_planned": int(self.n_windows_planned),
            "cancelled": bool(self.cancelled),
            "elapsed_s": float(self.elapsed_s),
            "created_at": self.created_at,
            "antares_version": self.antares_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DriftScan":
        return cls(
            lock=str(d["lock"]),
            dataset=str(d["dataset"]),
            lock_params=_sim_params_from_dict(d["lock_params"]),
            window=int(d["window"]),
            step=int(d["step"]),
            max_evaluations=int(d.get("max_evaluations", 0)),
            seed=int(d.get("seed", 0)),
            threshold=float(d.get("threshold", DEFAULT_THRESHOLD)),
            windows=[DriftWindow.from_dict(w) for w in d.get("windows", [])],
            trends={k: Trend.from_dict(t) for k, t in d.get("trends", {}).items()},
            n_windows_planned=int(d.get("n_windows_planned", len(d.get("windows", [])))),
            cancelled=bool(d.get("cancelled", False)),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            created_at=str(d.get("created_at", "")),
            antares_version=str(d.get("antares_version", "")),
            pinned=None if d.get("pinned") is None else [str(p) for p in d["pinned"]],
        )

    def default_path(self, drift_dir: Path = DRIFT_DIR) -> Path:
        return Path(drift_dir) / f"{self.lock}{DRIFT_SUFFIX}"

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path is not None else self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "DriftScan":
        return cls.from_dict(json.loads(Path(path).read_text()))


def window_slices(n_episodes: int, window: int, step: int) -> list[range]:
    """Start/stop positions of each window over ``n_episodes`` ordered episodes."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")
    if n_episodes < window:
        raise ValueError(f"need at least {window} episodes for a window of {window}, got {n_episodes}")
    return [range(s, s + window) for s in range(0, n_episodes - window + 1, step)]


def trended_params(params: Any) -> list[str]:
    """Keys of ``params.to_dict()`` a trend can be fitted to: scalar numbers,
    so not ``servo`` (the actuator model's name) nor per-joint lists."""
    return [n for n, v in params.to_dict().items() if isinstance(v, (int, float)) and not isinstance(v, bool)]


def compute_trends(
    windows: Sequence[DriftWindow],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Trend]:
    """Trend of every fitted parameter and of both scores against episode index.

    Only the numeric, shared parameters are trended: ``servo`` names the
    actuator model (a string) and is skipped, as is any per-joint value.
    """
    windows = list(windows)
    if not windows:
        return {}
    x = [w.center for w in windows]
    names = trended_params(windows[0].params)
    out: dict[str, Trend] = {}
    for i, name in enumerate(names):
        y = [float(w.params.to_dict()[name]) for w in windows]
        out[name] = fit_trend(
            x, y, log_space=name in LOG_PARAMS, threshold=threshold, n_boot=n_boot, seed=seed + i
        )
    for j, key in enumerate(SCORE_KEYS):
        y = [getattr(w, key) for w in windows]
        out[key] = fit_trend(
            x, y, log_space=False, threshold=threshold, n_boot=n_boot, seed=seed + len(names) + j
        )
    return out


def drift_scan(
    episodes: Sequence[Episode],
    lock: Any,
    window: int = 10,
    step: int = 5,
    max_evaluations: int = 60,
    seed: int = 0,
    on_progress: Callable[[DriftProgress], None] | None = None,
    *,
    simulate: Callable[..., Any] | None = None,
    sigma0: float = 0.1,
    threshold: float = DEFAULT_THRESHOLD,
    n_boot: int = 1000,
    should_stop: Callable[[], bool] | None = None,
) -> DriftScan:
    """Refit the sim on sliding windows of ``episodes`` starting from ``lock``.

    ``lock`` is a :class:`antares.lock.RobotLock` (anything with ``name``,
    ``dataset`` and ``params`` works). Episodes are ordered by index; each
    window is ``window`` consecutive episodes, advancing by ``step``. The
    lock's static score on a window is the warm start's own training score,
    so it costs nothing extra. ``should_stop`` ends the scan after the
    current window; the trends are then computed on the windows finished.
    The lock's pinned parameters (:func:`antares.lock.pinned_params`; only
    latency for a lock without seed spread) gate which parameter trends the
    verdict announces.
    """
    from antares.calibrate import calibrate
    from antares.lock import pinned_params

    t0 = time.perf_counter()
    ordered = sorted(episodes, key=lambda ep: ep.index)
    slices = window_slices(len(ordered), window, step)
    if simulate is None:
        from antares.calibrate import default_simulate

        simulate = default_simulate()
    stop = should_stop or (lambda: False)

    windows: list[DriftWindow] = []
    cancelled = False
    for i, sl in enumerate(slices):
        if stop():
            cancelled = True
            break
        block = [ordered[k] for k in sl]
        result = calibrate(
            block,
            [],
            simulate=simulate,
            start=lock.params,
            max_evaluations=max_evaluations,
            sigma0=sigma0,
            seed=seed + i,
            should_stop=should_stop,
        )
        w = DriftWindow(
            episodes=[ep.index for ep in block],
            params=result.params,
            train_score=float(result.train_score),
            lock_score=float(result.baseline_train_score),
            per_joint_fit=dict(result.per_joint_after),
            per_joint_lock=dict(result.per_joint_before),
            n_evaluations=int(result.n_evaluations),
            elapsed_s=float(result.elapsed_s),
        )
        windows.append(w)
        if on_progress is not None:
            on_progress(DriftProgress(i, len(slices), w, time.perf_counter() - t0))
        if stop():
            cancelled = True
            break

    return DriftScan(
        lock=str(lock.name),
        dataset=str(lock.dataset),
        lock_params=lock.params,
        window=window,
        step=step,
        max_evaluations=max_evaluations,
        seed=seed,
        threshold=threshold,
        windows=windows,
        trends=compute_trends(windows, threshold=threshold, n_boot=n_boot, seed=seed),
        n_windows_planned=len(slices),
        cancelled=cancelled,
        elapsed_s=time.perf_counter() - t0,
        pinned=pinned_params(lock),
    )


def list_scans(drift_dir: Path = DRIFT_DIR) -> list[DriftScan]:
    """All saved scans under ``drift_dir``, sorted by lock name."""
    drift_dir = Path(drift_dir)
    if not drift_dir.is_dir():
        return []
    out = []
    for path in sorted(drift_dir.glob(f"*{DRIFT_SUFFIX}")):
        try:
            out.append(DriftScan.load(path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(out, key=lambda s: s.lock)


__all__ = [
    "DEFAULT_THRESHOLD",
    "DRIFT_DIR",
    "DRIFT_SUFFIX",
    "LOG_PARAMS",
    "SCORE_KEYS",
    "DriftProgress",
    "DriftScan",
    "DriftWindow",
    "Trend",
    "compute_trends",
    "drift_scan",
    "fit_trend",
    "list_scans",
    "window_slices",
]
