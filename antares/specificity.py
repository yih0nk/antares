"""Is a robot lock specific to the arm it was fitted on?

Two questions, answered with the same injected ``simulate`` callable that
:mod:`antares.calibrate` uses (so tests do not need MuJoCo):

1. **Specificity.** Given a cross matrix (every lock scored on every
   dataset's held-out split, see ``antares cross``), does each dataset's own
   lock beat every other arm's lock? :func:`summarize_cross` answers that
   per dataset with the margin.

2. **Identifiability.** Does the tracking error actually pin each parameter
   down, or can some of them trade off against each other? A fitted lock is
   only evidence about *that robot* if its numbers are determined by the
   data. :func:`identifiability` linearizes the replay residuals around a
   lock (a Gauss-Newton view of the objective) and reports how far each
   parameter can move before the error rises by a set amount, both with the
   others held fixed and with the others free to compensate. A large gap
   between the two means the parameter trades off with the rest.
   :func:`latency_profile`, :func:`seed_restarts`, :func:`pinned_refit` and
   :func:`interpolate` are direct, model-free checks of the same thing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from antares.calibrate import DIMS, SPACE, SimulateFn, _safe_score, decode, encode
from antares.data import Episode

# --------------------------------------------------------------------------
# Specificity from a cross matrix
# --------------------------------------------------------------------------


def summarize_cross(matrix: Mapping[str, Any], lock_datasets: Mapping[str, str]) -> dict:
    """Compare each dataset's own lock against the other datasets' locks.

    ``matrix`` is the dict ``antares.server.cross_matrix`` returns (rows =
    datasets, cols = lock names plus ``"default"``). ``lock_datasets`` maps
    each lock name to the dataset it was fitted on. A row is *specific*
    when its best own lock scores lower than every other arm's lock.
    ``margin`` is ``best_other / own``: how many times worse the best
    foreign lock tracks this arm than its own lock does.
    """
    cols = list(matrix["cols"])
    rows_out = []
    for row, scores in zip(matrix["rows"], matrix["scores"]):
        by_col = dict(zip(cols, scores))
        own = {c: s for c, s in by_col.items() if lock_datasets.get(c) == row}
        other = {
            c: s for c, s in by_col.items() if c != "default" and c in lock_datasets and lock_datasets[c] != row
        }
        own_name = min(own, key=own.get) if own else None
        other_name = min(other, key=other.get) if other else None
        own_s = own[own_name] if own_name else None
        other_s = other[other_name] if other_name else None
        specific = None
        margin = None
        if own_s is not None and other_s is not None:
            specific = bool(own_s < other_s)
            margin = float(other_s / own_s) if own_s > 0 else math.inf
        rows_out.append(
            {
                "dataset": row,
                "own_lock": own_name,
                "own_score": own_s,
                "best_other_lock": other_name,
                "best_other_score": other_s,
                "default_score": by_col.get("default"),
                "margin": margin,
                "specific": specific,
            }
        )
    decided = [r["specific"] for r in rows_out if r["specific"] is not None]
    return {"rows": rows_out, "all_specific": bool(decided) and all(decided)}


# --------------------------------------------------------------------------
# Identifiability: linearize the residuals around a lock
# --------------------------------------------------------------------------

#: Parameters the local analysis perturbs. Latency is an integer number of
#: frames, so it has no derivative; :func:`latency_profile` covers it.
CONTINUOUS: tuple[str, ...] = tuple(n for n in DIMS if SPACE[n][2] != "int")

#: Gripper column, excluded from the score like in :mod:`antares.gap`.
_GRIPPER = 5


def residuals(params, episodes: Sequence[Episode], simulate: SimulateFn, exclude: Sequence[int] = (_GRIPPER,)):
    """Sim minus real, in degrees, for every frame and scored joint, flattened.

    ``sqrt(mean(r**2))`` equals the ``overall_deg`` the calibrator minimizes
    (frame-weighted RMS over the scored joints).
    """
    parts = []
    for ep in episodes:
        real_q, sim_q = simulate(params, ep)
        err = np.degrees(np.asarray(sim_q, dtype=np.float64) - np.asarray(real_q, dtype=np.float64))
        keep = [j for j in range(err.shape[1]) if j not in set(exclude)]
        parts.append(err[:, keep].reshape(-1))
    r = np.concatenate(parts) if parts else np.zeros(0)
    if not np.all(np.isfinite(r)):
        raise FloatingPointError("simulation diverged")
    return r


def jacobian(
    params,
    episodes: Sequence[Episode],
    simulate: SimulateFn,
    step: float = 0.01,
    names: Sequence[str] = CONTINUOUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Residuals at ``params`` and their Jacobian in the calibrator's unit space.

    Columns follow ``names``; a unit step is the whole search range of that
    parameter (for log-scaled ones, the full span in decades). Central
    differences of size ``step``, one-sided where the box would clip.
    Returns ``(r0, J)`` with ``J`` of shape ``(len(r0), len(names))``.
    """
    x0 = encode(params)
    r0 = residuals(params, episodes, simulate)
    J = np.empty((r0.size, len(names)), dtype=np.float64)
    for k, name in enumerate(names):
        i = DIMS.index(name)
        hi = min(x0[i] + step, 1.0)
        lo = max(x0[i] - step, 0.0)
        xh, xl = x0.copy(), x0.copy()
        xh[i], xl[i] = hi, lo
        rh = residuals(decode(xh), episodes, simulate) if hi > x0[i] else r0
        rl = residuals(decode(xl), episodes, simulate) if lo < x0[i] else r0
        J[:, k] = (rh - rl) / (hi - lo)
    return r0, J


def _factor(name: str, delta_unit: float) -> float:
    """Multiplicative change a unit-space step means for a log-scaled param."""
    lo, hi, _ = SPACE[name]
    decades = delta_unit * (math.log10(hi) - math.log10(lo))
    if not math.isfinite(decades) or decades > 300:
        return math.inf
    return float(10**decades)


@dataclass(frozen=True)
class Identifiability:
    """Local, linearized view of how well the data pins each parameter.

    ``tolerance_fixed[name]`` is how far (unit space) ``name`` can move with
    every other parameter held at the lock before the overall error rises
    by ``delta_deg``. ``tolerance_free[name]`` is the same when the other
    parameters re-adjust to compensate. ``factor_*`` give those as a
    multiplicative change of the parameter (x1.1 = tightly pinned, x100 =
    barely constrained). ``tradeoff`` is ``free / fixed`` (>= 1); a large
    value means the error alone cannot separate this parameter from the
    others. ``sloppiest`` is the parameter combination the data constrains
    least (unit eigenvector of the Gauss-Newton Hessian).
    """

    names: tuple[str, ...]
    rms_deg: float
    n_residuals: int
    delta_deg: float
    tolerance_fixed: dict[str, float]
    tolerance_free: dict[str, float]
    factor_fixed: dict[str, float]
    factor_free: dict[str, float]
    tradeoff: dict[str, float]
    correlation: list[list[float]]
    eigenvalues: list[float]
    condition_number: float
    sloppiest: dict[str, float]

    def to_dict(self) -> dict:
        def j(v: float) -> float | None:
            return float(v) if math.isfinite(v) else None

        def jd(d: dict[str, float]) -> dict[str, float | None]:
            return {k: j(v) for k, v in d.items()}

        return {
            "names": list(self.names),
            "rms_deg": j(self.rms_deg),
            "n_residuals": int(self.n_residuals),
            "delta_deg": float(self.delta_deg),
            "tolerance_fixed": jd(self.tolerance_fixed),
            "tolerance_free": jd(self.tolerance_free),
            "factor_fixed": jd(self.factor_fixed),
            "factor_free": jd(self.factor_free),
            "tradeoff": jd(self.tradeoff),
            "correlation": [[j(v) for v in row] for row in self.correlation],
            "eigenvalues": [j(v) for v in self.eigenvalues],
            "condition_number": j(self.condition_number),
            "sloppiest": jd(self.sloppiest),
        }


def identifiability(
    params,
    episodes: Sequence[Episode],
    simulate: SimulateFn,
    delta_deg: float = 0.1,
    step: float = 0.01,
    names: Sequence[str] = CONTINUOUS,
) -> Identifiability:
    """Gauss-Newton identifiability of ``params`` on ``episodes``.

    With residuals ``r`` (N of them) and Jacobian ``J``, ``H = J^T J`` and
    a step ``d`` raises the mean squared error by about ``d^T H d / N``.
    Moving one parameter alone by ``t`` costs ``t^2 H_ii / N``; letting the
    others compensate costs ``t^2 / (N (H^-1)_ii)``. Solving for the step
    that raises the RMS from ``rms`` to ``rms + delta_deg`` gives the two
    tolerances. This assumes ``params`` is at (or near) a minimum and is
    only as good as the linearization over that range.
    """
    names = tuple(names)
    r0, J = jacobian(params, episodes, simulate, step=step, names=names)
    n = r0.size
    rms = float(np.sqrt(np.mean(r0**2)))
    budget = ((rms + delta_deg) ** 2 - rms**2) * n  # allowed rise in sum of squares

    H = J.T @ J
    diag = np.diag(H)
    # A tiny ridge keeps an exactly degenerate H invertible; the parameters in
    # its null space then get enormous (but finite) free tolerances.
    ridge = 1e-12 * max(float(np.trace(H)), 1e-300)
    Hinv = np.linalg.inv(H + ridge * np.eye(len(names)))
    hinv_diag = np.diag(Hinv)
    tol_fixed, tol_free = {}, {}
    for k, name in enumerate(names):
        tol_fixed[name] = math.sqrt(budget / diag[k]) if diag[k] > 0 else math.inf
        tol_free[name] = math.sqrt(budget * hinv_diag[k]) if hinv_diag[k] > 0 and diag[k] > 0 else math.inf

    d = np.sqrt(np.clip(hinv_diag, 1e-300, None))
    corr = Hinv / np.outer(d, d)
    w, v = np.linalg.eigh(H)
    cond = float(w[-1] / w[0]) if w[0] > 0 else math.inf
    slop = v[:, 0] * np.sign(v[np.argmax(np.abs(v[:, 0])), 0])

    return Identifiability(
        names=names,
        rms_deg=rms,
        n_residuals=n,
        delta_deg=float(delta_deg),
        tolerance_fixed=tol_fixed,
        tolerance_free=tol_free,
        factor_fixed={k: _factor(k, t) for k, t in tol_fixed.items()},
        factor_free={k: _factor(k, t) for k, t in tol_free.items()},
        tradeoff={k: tol_free[k] / tol_fixed[k] if tol_fixed[k] > 0 else math.inf for k in names},
        correlation=corr.tolist(),
        eigenvalues=w.tolist(),
        condition_number=cond,
        sloppiest={name: float(slop[k]) for k, name in enumerate(names)},
    )


# --------------------------------------------------------------------------
# Direct checks (no linearization)
# --------------------------------------------------------------------------


def latency_profile(
    params, episodes: Sequence[Episode], simulate: SimulateFn, frames: Sequence[int] | None = None
) -> dict[int, float]:
    """Overall error with every latency in ``frames`` (default: the search range)."""
    if frames is None:
        lo, hi, _ = SPACE["latency_frames"]
        frames = range(int(lo), int(hi) + 1)
    return {int(f): _safe_score(params.replace(latency_frames=int(f)), episodes, simulate)[1] for f in frames}


__all__ = [
    "CONTINUOUS",
    "latency_profile",
    "Identifiability",
    "identifiability",
    "jacobian",
    "residuals",
    "summarize_cross",
]
