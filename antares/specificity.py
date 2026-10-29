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

from antares.calibrate import (
    DIMS,
    SERVO_SPACE,
    SPACE,
    SearchSpace,
    SimulateFn,
    _safe_score,
    calibrate,
    decode,
    encode,
    robot_of,
    scoring_names,
    split_episodes,
)
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


def bounds_for(params) -> dict[str, tuple[float, float, str]]:
    """The search space a lock was fitted in: :data:`SERVO_SPACE` for a
    servo-model lock (its kp/kv are derived from the servo gain and cannot
    move on their own), :data:`SPACE` otherwise."""
    return SERVO_SPACE if getattr(params, "uses_servo_model", False) else SPACE


def continuous_names(params) -> tuple[str, ...]:
    """The non-integer dimensions of ``params``' search space."""
    b = bounds_for(params)
    return tuple(n for n in b if b[n][2] != "int")


def _check_names(params, names: Sequence[str]) -> tuple[str, ...]:
    """``names`` as a tuple, each a continuous dimension of the space
    ``params`` was fitted in; anything else is a ValueError that names that
    space (a servo-model lock's kp/kv are derived, so :data:`CONTINUOUS`
    does not fit it)."""
    names = tuple(names)
    space = bounds_for(params)
    label = "servo-model" if getattr(params, "uses_servo_model", False) else "position-actuator"
    allowed = continuous_names(params)
    bad = [n for n in names if n not in space]
    if bad:
        raise ValueError(
            f"{', '.join(map(repr, bad))}: not in the lock's {label} search space; "
            f"its continuous dimensions are {', '.join(allowed)}"
        )
    ints = [n for n in names if space[n][2] == "int"]
    if ints:
        raise ValueError(
            f"{', '.join(map(repr, ints))}: an integer dimension has no derivative "
            f"(see latency_profile); choose from {', '.join(allowed)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate names in {names}")
    return names

def _unscored(simulate: SimulateFn) -> tuple[int, ...]:
    """Gripper columns of the robot ``simulate`` runs (index 5 on the SO
    arms), from the same names the calibrator scores with."""
    names = scoring_names(simulate)
    return tuple(names["joint_names"].index(g) for g in names["exclude"])


def residuals(params, episodes: Sequence[Episode], simulate: SimulateFn, exclude: Sequence[int] | None = None):
    """Sim minus real, in degrees, for every frame and scored joint, flattened.

    ``sqrt(mean(r**2))`` equals the ``overall_deg`` the calibrator minimizes
    (frame-weighted RMS over the scored joints). ``exclude`` defaults to the
    gripper columns of the robot ``simulate`` is tagged with.
    """
    if not episodes:
        raise ValueError("no episodes to score")
    if exclude is None:
        exclude = _unscored(simulate)
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
    names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Residuals at ``params`` and their Jacobian in the calibrator's unit space.

    Columns follow ``names`` (default :func:`continuous_names`); a unit step
    is the whole search range of that parameter (for log-scaled ones, the
    full span in decades) in the space the lock was fitted in
    (:func:`bounds_for`). Central differences of size ``step``, one-sided
    where the box would clip. Returns ``(r0, J)`` with ``J`` of shape
    ``(len(r0), len(names))``.
    """
    _require_shared(params)
    names = continuous_names(params) if names is None else _check_names(params, names)
    space = SearchSpace(bounds_for(params), per_joint=False)
    x0 = space.encode(params)
    r0 = residuals(params, episodes, simulate)
    J = np.empty((r0.size, len(names)), dtype=np.float64)
    for k, name in enumerate(names):
        i = space.fields.index(name)
        hi = min(x0[i] + step, 1.0)
        lo = max(x0[i] - step, 0.0)
        xh, xl = x0.copy(), x0.copy()
        xh[i], xl[i] = hi, lo
        rh = residuals(space.decode(xh, params), episodes, simulate) if hi > x0[i] else r0
        rl = residuals(space.decode(xl, params), episodes, simulate) if lo < x0[i] else r0
        J[:, k] = (rh - rl) / (hi - lo)
    return r0, J


def _require_shared(params) -> None:
    """These checks work in the shared search space; a per-joint lock would
    be silently averaged by ``encode``, so refuse it instead."""
    if getattr(params, "has_per_joint", False):
        raise ValueError("specificity checks need a shared (scalar) lock, not a per-joint one")


def _tradeoff(fixed: float, free: float) -> float:
    """``free / fixed``; NaN (``null`` in JSON, ``-`` in the reports) when the
    data does not constrain the parameter at all, so the ratio is undefined."""
    if not math.isfinite(fixed) or fixed <= 0:
        return math.nan
    return free / fixed


def _factor(name: str, delta_unit: float, bounds=SPACE) -> float:
    """Multiplicative change a unit-space step means for a log-scaled param."""
    lo, hi, _ = bounds[name]
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
    names: Sequence[str] | None = None,
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
    names = continuous_names(params) if names is None else _check_names(params, names)
    bounds = bounds_for(params)
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
        factor_fixed={k: _factor(k, t, bounds) for k, t in tol_fixed.items()},
        factor_free={k: _factor(k, t, bounds) for k, t in tol_free.items()},
        tradeoff={k: _tradeoff(tol_fixed[k], tol_free[k]) for k in names},
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


def seed_restarts(
    train: Sequence[Episode],
    eval: Sequence[Episode],
    seeds: Sequence[int],
    simulate: SimulateFn,
    max_evaluations: int = 150,
    start=None,
) -> dict:
    """Re-run the calibration with different CMA-ES seeds on the same split.

    If the data identifies the parameters, every restart lands on about the
    same values. ``spread[name]`` is ``max / min`` over the restarts (for
    latency, ``max - min`` frames).
    """
    runs = []
    for seed in seeds:
        res = calibrate(train, eval, simulate=simulate, start=start, max_evaluations=max_evaluations, seed=seed)
        runs.append(
            {
                "seed": int(seed),
                "params": res.params.to_dict(),
                "train_score": float(res.train_score),
                "eval_score": float(res.eval_score),
            }
        )
    return {"runs": runs, "spread": _spread([r["params"] for r in runs])}


def _spread(param_dicts: Sequence[dict]) -> dict[str, float]:
    if not param_dicts:
        return {}
    out = {}
    for name in DIMS:
        v = np.array([d[name] for d in param_dicts], dtype=np.float64)
        if SPACE[name][2] == "int":
            out[name] = float(v.max() - v.min())
        else:
            out[name] = float(v.max() / v.min()) if v.min() > 0 else math.inf
    return out


def pinned_refit(
    train: Sequence[Episode],
    eval: Sequence[Episode],
    fixed: Mapping[str, float],
    simulate: SimulateFn,
    max_evaluations: int = 150,
    seed: int = 0,
    start=None,
) -> dict:
    """Calibrate with some parameters pinned to given values.

    The usual use is a swap test: pin arm A's fit to arm B's ``kp``/``kv``
    and let the rest re-fit. If the pinned fit tracks A about as well as
    A's own lock, the data does not distinguish the two arms' values of the
    pinned parameters. The pinned dimensions stay in the CMA-ES search but
    have no effect on the objective.
    """
    from antares.calibrate import SimParams

    unknown = set(fixed) - set(DIMS)
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}")
    pins = {k: (int(round(v)) if SPACE[k][2] == "int" else float(v)) for k, v in fixed.items()}

    def pinned(params, ep):
        return simulate(params.replace(**pins), ep)

    start = (start or SimParams()).replace(**pins)
    res = calibrate(train, eval, simulate=pinned, start=start, max_evaluations=max_evaluations, seed=seed)
    return {
        "fixed": dict(pins),
        "params": res.params.replace(**pins).to_dict(),
        "train_score": float(res.train_score),
        "eval_score": float(res.eval_score),
    }


def interpolate(a, b, episodes: Sequence[Episode], simulate: SimulateFn, n: int = 11) -> list[tuple[float, float]]:
    """Overall error along the straight line from lock ``a`` to lock ``b``.

    The line is in the calibrator's unit space (log scale for the gains),
    and latency rounds to whole frames along the way. A low, flat profile
    means the two fits sit in one connected valley of near-equal error;
    a ridge between them means they are genuinely different optima.
    """
    _require_shared(a)
    _require_shared(b)
    xa, xb = encode(a), encode(b)
    out = []
    for t in np.linspace(0.0, 1.0, n):
        out.append((float(t), _safe_score(decode((1 - t) * xa + t * xb, base=a), episodes, simulate)[1]))
    return out


# --------------------------------------------------------------------------
# The whole study
# --------------------------------------------------------------------------


def _resolve_splits(
    datasets: Mapping[str, Sequence[Episode]],
    given: Mapping[str, tuple[Sequence[int], Sequence[int]]],
    n_train: int,
    n_eval: int,
    seed: int,
    log,
) -> dict[str, tuple[list[Episode], list[Episode]]]:
    """The lock's own split where one is given (and still matches the
    dataset), the calibrator's seeded split otherwise."""
    out = {}
    for d, eps in datasets.items():
        by_index = {ep.index: ep for ep in eps}
        if d in given:
            train_i, eval_i = (list(given[d][0]), list(given[d][1]))
            missing = sorted((set(train_i) | set(eval_i)) - set(by_index))
            if train_i and eval_i and not missing:
                out[d] = ([by_index[i] for i in sorted(train_i)], [by_index[i] for i in sorted(eval_i)])
                continue
            log(f"{d}: lock split does not match the dataset (missing episodes {missing}); "
                f"splitting with seed {seed}")
        out[d] = split_episodes(eps, n_train, n_eval, seed=seed)
    return out


def _simulate_for(simulate: SimulateFn | Mapping[str, SimulateFn], dataset: str) -> SimulateFn:
    """``simulate`` itself, or its entry for ``dataset`` when it is a mapping."""
    if not isinstance(simulate, Mapping):
        return simulate
    try:
        return simulate[dataset]
    except KeyError:
        raise ValueError(f"{dataset}: no simulator given for this dataset") from None


def study(
    datasets: Mapping[str, Sequence[Episode]],
    locks: Mapping[str, tuple[str, Any]],
    simulate: SimulateFn | Mapping[str, SimulateFn],
    *,
    n_train: int = 20,
    n_eval: int = 10,
    split_seed: int = 0,
    restart_seeds: Sequence[int] = (1, 2),
    max_evaluations: int = 150,
    delta_deg: float = 0.1,
    truth: Mapping[str, Any] | None = None,
    splits: Mapping[str, tuple[Sequence[int], Sequence[int]]] | None = None,
    log=lambda msg: None,
) -> dict:
    """Run every check for a set of arms and their locks.

    ``datasets`` maps a dataset id to its episodes; ``locks`` maps a lock
    name to ``(dataset id it was fitted on, params)``. ``splits`` maps a
    dataset id to the ``(train, eval)`` episode indices its own lock was
    fitted with (a saved lock records them), so the cross matrix scores
    episodes the lock really never saw; a dataset without one is split
    like ``antares calibrate`` would with ``split_seed``, which only
    matches a lock fitted with the same seed and counts. Every lock must
    be a shared (scalar) one. ``simulate`` is one callable for every
    dataset, or a mapping from dataset id to the callable that replays it
    (each dataset on its own robot model and units, so a degrees recording
    and a normalized one can sit in the same study). For every dataset
    that has both its own lock and a foreign one, the study also runs:
    identifiability and the latency profile at its own lock (train
    split), seed restarts, a same-arm control fitted on different training
    episodes, a swap test that pins the foreign lock's ``kp``/``kv``, and
    the path from the own lock to the foreign one. ``truth`` (synthetic arms) maps dataset ids to the
    parameters that generated them.
    """
    from antares.calibrate import SimParams

    for name, (_, p) in locks.items():
        try:
            _require_shared(p)
        except ValueError as e:
            raise ValueError(f"lock {name}: {e}") from e
    lock_datasets = {name: ds for name, (ds, _) in locks.items()}
    lock_params = {name: p for name, (_, p) in locks.items()}

    splits = _resolve_splits(datasets, splits or {}, n_train, n_eval, split_seed, log)
    for d, (_, eval_) in splits.items():
        if not eval_:
            raise ValueError(f"{d}: no held-out episodes to score (n_train={n_train}, n_eval={n_eval})")
    sims = {d: _simulate_for(simulate, d) for d in datasets}

    cols = list(locks) + ["default"]
    col_params = [lock_params[c] for c in locks] + [SimParams()]
    log("cross matrix")
    scores = [[_safe_score(p, splits[d][1], sims[d])[1] for p in col_params] for d in datasets]
    matrix = {"rows": list(datasets), "cols": cols, "scores": scores}
    summary = summarize_cross(matrix, lock_datasets)

    per_dataset = {}
    for row in summary["rows"]:
        d = row["dataset"]
        if row["own_lock"] is None or row["best_other_lock"] is None:
            continue
        train, eval_ = splits[d]
        simulate_d = sims[d]
        own = lock_params[row["own_lock"]]
        other = lock_params[row["best_other_lock"]]
        out: dict[str, Any] = {"own_lock": row["own_lock"], "other_lock": row["best_other_lock"]}

        log(f"{d}: identifiability")
        rep = identifiability(own, train, simulate_d, delta_deg=delta_deg)
        out["identifiability"] = rep.to_dict()
        out["latency_profile"] = latency_profile(own, train, simulate_d)

        log(f"{d}: seed restarts {list(restart_seeds)}")
        restarts = seed_restarts(train, eval_, restart_seeds, simulate_d, max_evaluations=max_evaluations)
        restarts["runs"].insert(
            0,
            {
                "seed": None,
                "params": own.to_dict(),
                "train_score": rep.rms_deg,  # the residual RMS *is* the train score
                "eval_score": row["own_score"],
            },
        )
        restarts["spread"] = _spread([r["params"] for r in restarts["runs"]])
        out["seed_restarts"] = restarts

        held_out = {ep.index for ep in eval_}
        used = {ep.index for ep in train}
        fresh = [ep for ep in datasets[d] if ep.index not in held_out and ep.index not in used]
        if fresh:
            log(f"{d}: same-arm control on {min(len(fresh), n_train)} fresh episodes")
            ctrl = calibrate(fresh[:n_train], eval_, simulate=simulate_d, max_evaluations=max_evaluations)
            out["same_arm_control"] = {
                "n_train": min(len(fresh), n_train),
                "params": ctrl.params.to_dict(),
                "eval_score": float(ctrl.eval_score),
            }

        log(f"{d}: swap test with {row['best_other_lock']}'s kp/kv")
        out["swap"] = pinned_refit(
            train, eval_, {"kp": other.kp, "kv": other.kv}, simulate_d, max_evaluations=max_evaluations
        )
        out["path_to_other"] = interpolate(own, other, eval_, simulate_d)
        if truth and d in truth:
            out["truth"] = truth[d].to_dict()
            out["truth_eval_score"] = _safe_score(truth[d], eval_, simulate_d)[1]
        per_dataset[d] = out

    return {
        "config": {
            "n_train": n_train,
            "n_eval": n_eval,
            "split_seed": split_seed,
            "restart_seeds": list(restart_seeds),
            "max_evaluations": max_evaluations,
            "delta_deg": delta_deg,
        },
        "robots": {d: getattr(robot_of(sims[d]), "name", None) for d in datasets},
        "locks": {n: {"dataset": ds, "params": p.to_dict()} for n, (ds, p) in locks.items()},
        "splits": {
            d: {"train": [ep.index for ep in tr], "eval": [ep.index for ep in ev]} for d, (tr, ev) in splits.items()
        },
        "matrix": matrix,
        "specificity": summary,
        "datasets": per_dataset,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _deg(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}°"


def _times(v: float | None) -> str:
    if v is None:
        return "-"
    return "unbounded" if not math.isfinite(v) or v > 1e6 else f"×{v:.3g}"


def _fmt_param(name: str, v: float) -> str:
    return str(int(v)) if name == "latency_frames" else f"{v:.4g}"


def render_markdown(results: Mapping[str, Any], title: str = "Robot specificity") -> str:
    """Human-readable report of a :func:`study` result."""
    m = results["matrix"]
    out = [f"## {title}", ""]
    out.append("Held-out tracking error (RMS over the five body joints). Rows are the arms' "
               "datasets, columns are the locks.")
    robots = {d: r for d, r in (results.get("robots") or {}).items() if r}
    if robots:
        out.append("Each dataset replays on its own model: "
                   + ", ".join(f"{d} on `{r}`" for d, r in robots.items()) + ".")
    out.append("")
    out.append(_md_table(["dataset"] + list(m["cols"]),
                         [[r] + [_deg(v) for v in sc] for r, sc in zip(m["rows"], m["scores"])]))
    out.append("")
    rows = [
        [r["dataset"], r["own_lock"] or "-", _deg(r["own_score"]), r["best_other_lock"] or "-",
         _deg(r["best_other_score"]), "-" if r["margin"] is None else f"{r['margin']:.2f}×",
         {True: "yes", False: "**no**", None: "-"}[r["specific"]]]
        for r in results["specificity"]["rows"]
    ]
    out.append(_md_table(["dataset", "own lock", "own", "best other lock", "other", "margin", "own wins"], rows))

    for d, r in results["datasets"].items():
        own_s = next(x["own_score"] for x in results["specificity"]["rows"] if x["dataset"] == d)
        other_s = next(x["best_other_score"] for x in results["specificity"]["rows"] if x["dataset"] == d)
        out += ["", f"### {d}", ""]
        checks = [["own lock", _deg(own_s)]]
        if "same_arm_control" in r:
            c = r["same_arm_control"]
            checks.append([f"same arm, refit on {c['n_train']} other episodes", _deg(c["eval_score"])])
        checks.append([f"own fit with {r['other_lock']}'s kp/kv pinned", _deg(r["swap"]["eval_score"])])
        checks.append([f"{r['other_lock']} lock", _deg(other_s)])
        if "truth_eval_score" in r:
            checks.append(["true parameters (synthetic)", _deg(r["truth_eval_score"])])
        out.append(_md_table(["parameters", "held-out error"], checks))

        runs = r["seed_restarts"]["runs"]
        names = [n for n in DIMS if n in runs[0]["params"]]
        prow = [["lock" if x["seed"] is None else f"seed {x['seed']}"]
                + [_fmt_param(n, x["params"][n]) for n in names] + [_deg(x["eval_score"])] for x in runs]
        if "same_arm_control" in r:
            prow.append(["same-arm control"] + [_fmt_param(n, r["same_arm_control"]["params"][n]) for n in names]
                        + [_deg(r["same_arm_control"]["eval_score"])])
        if "truth" in r:
            prow.append(["truth"] + [_fmt_param(n, r["truth"][n]) for n in names] + [_deg(r["truth_eval_score"])])
        spread = r["seed_restarts"]["spread"]
        prow.append(["spread"] + [f"±{int(spread[n])}" if n == "latency_frames" else _times(spread[n])
                                  for n in names] + [""])
        out += ["", "Refits of the same data:", "", _md_table(["fit"] + names + ["held-out"], prow)]

        idf = r["identifiability"]
        lock_p = runs[0]["params"]
        irow = [[n, _fmt_param(n, lock_p[n]), _times(idf["factor_fixed"][n]), _times(idf["factor_free"][n]),
                 "-" if idf["tradeoff"][n] is None else f"{idf['tradeoff'][n]:.3g}"] for n in idf["names"]]
        slop = sorted(idf["sloppiest"].items(), key=lambda kv: -abs(kv[1] or 0))
        out += ["", f"How far each parameter can move before the training error rises by "
                    f"{idf['delta_deg']:g}° (linearized at the lock):", "",
                _md_table(["param", "lock", "alone", "others refit", "trade-off"], irow), "",
                f"Least constrained direction (condition number {idf['condition_number'] or float('inf'):.3g}): "
                + ", ".join(f"{v:+.2f} {k}" for k, v in slop if v is not None and abs(v) >= 0.1) + "."]
        lat = r["latency_profile"]
        out += ["", "Latency: " + ", ".join(f"{f} frames {_deg(v)}" for f, v in lat.items()) + "."]
        path = r["path_to_other"]
        out += ["", f"Path from own lock to {r['other_lock']} (held-out): "
                + " → ".join(_deg(v) for _, v in path) + "."]
    return "\n".join(out) + "\n"


__all__ = [
    "CONTINUOUS",
    "interpolate",
    "latency_profile",
    "pinned_refit",
    "seed_restarts",
    "study",
    "Identifiability",
    "identifiability",
    "jacobian",
    "render_markdown",
    "residuals",
    "summarize_cross",
]
