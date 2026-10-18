"""Pre-deploy gate: has this robot drifted away from its calibrated sim?

A :class:`~antares.lock.RobotLock` pins the simulator to one arm and records
how well that sim tracked the arm's held-out episodes at calibration time
(``result.eval_score``, degrees RMSE). :func:`check` replays fresh logs from
the same arm through the locked sim and compares. If the new tracking error
is within ``ratio`` of the reference (or under an absolute ``threshold_deg``,
whichever is looser) the robot still behaves like its sim and the check
passes; otherwise something physical changed (servo wear, a loose belt, a
recalibration) and the check fails, like a test suite failing before a
deploy.

Pure bookkeeping on top of :mod:`antares.gap`; the simulator is injected as
a ``simulate(params, episode) -> (real_q, sim_q)`` callable so this module
does not import MuJoCo and the gate can be tested with a toy servo.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from antares.data import Episode

VERDICT_OK = "ok"
VERDICT_DRIFTED = "drifted"
VERDICT_INSUFFICIENT = "insufficient_data"

DEFAULT_RATIO = 1.5
DEFAULT_MIN_EPISODES = 5


def limit_for(reference: float | None, threshold_deg: float | None, ratio: float) -> float | None:
    """The allowed error: ``max(threshold_deg or 0, ratio * reference)``.

    ``None`` when neither a usable reference nor a threshold is available.
    """
    ref_part = ratio * reference if reference is not None and math.isfinite(reference) else None
    thr_part = float(threshold_deg) if threshold_deg is not None else None
    if ref_part is None and thr_part is None:
        return None
    return max(ref_part or 0.0, thr_part or 0.0)


@dataclass(frozen=True)
class JointCheck:
    score: float
    reference: float | None
    limit: float | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": _json_float(self.score),
            "reference": _json_float(self.reference),
            "limit": _json_float(self.limit),
            "passed": bool(self.passed),
        }


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    verdict: str  # ok | drifted | insufficient_data
    score: float | None  # overall tracking RMSE (deg) on the fresh episodes
    reference: float | None  # the lock's held-out score at calibration time
    limit: float | None  # max(threshold_deg, ratio * reference)
    per_joint: dict[str, JointCheck]
    worst_joint: str | None
    n_episodes: int
    n_frames: int
    lock: str
    dataset: str
    threshold_deg: float | None = None
    ratio: float = DEFAULT_RATIO
    min_episodes: int = DEFAULT_MIN_EPISODES
    episodes: list[int] = field(default_factory=list)
    scored_joints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "verdict": self.verdict,
            "score": _json_float(self.score),
            "reference": _json_float(self.reference),
            "limit": _json_float(self.limit),
            "per_joint": {k: v.to_dict() for k, v in self.per_joint.items()},
            "worst_joint": self.worst_joint,
            "n_episodes": int(self.n_episodes),
            "n_frames": int(self.n_frames),
            "lock": self.lock,
            "dataset": self.dataset,
            "threshold_deg": _json_float(self.threshold_deg),
            "ratio": float(self.ratio),
            "min_episodes": int(self.min_episodes),
            "episodes": [int(i) for i in self.episodes],
            "scored_joints": list(self.scored_joints),
        }

    def summary(self) -> str:
        """One line fit for a CI log."""
        if self.verdict == VERDICT_INSUFFICIENT:
            return (
                f"INSUFFICIENT DATA: {self.n_episodes} episode(s) from {self.dataset}, "
                f"need at least {self.min_episodes} to check lock {self.lock}"
            )
        word = "OK" if self.passed else "DRIFTED"
        ref = f"{self.reference:.3f}" if self.reference is not None and math.isfinite(self.reference) else "n/a"
        lim = f"{self.limit:.3f}" if self.limit is not None else "n/a"
        worst = ""
        if self.worst_joint is not None:
            w = self.per_joint[self.worst_joint]
            worst = f"; worst joint {self.worst_joint} {w.score:.3f} deg"
            if w.limit is not None:
                worst += f" (limit {w.limit:.3f})"
        return (
            f"{word}: {self.dataset} vs lock {self.lock}: {self.score:.3f} deg RMSE "
            f"on {self.n_episodes} episodes (reference {ref}, limit {lim}){worst}"
        )


def _json_float(v: float | None) -> float | None:
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _lock_view(lock: Any) -> tuple[str, str, Any, float | None, dict[str, float]]:
    """``(name, dataset, params, eval_score, per_joint_after)`` from a
    :class:`~antares.lock.RobotLock` or its JSON dict."""
    if isinstance(lock, dict):
        from antares.sim import SimParams

        result = lock.get("result") or {}
        params = SimParams.from_dict(lock.get("params") or result.get("params") or {})
        eval_score = result.get("eval_score")
        train_score = result.get("train_score")
        per_joint = dict(result.get("per_joint_after") or {})
        name, dataset = str(lock.get("name", "?")), str(lock.get("dataset", "?"))
    else:
        result = lock.result
        params = lock.params
        eval_score = getattr(result, "eval_score", None)
        train_score = getattr(result, "train_score", None)
        per_joint = dict(getattr(result, "per_joint_after", None) or {})
        name, dataset = str(lock.name), str(lock.dataset)

    reference = _finite_or_none(eval_score)
    if reference is None:
        # A lock fitted without held-out episodes carries the train score in
        # eval_score already (see calibrate); this is only for hand-made locks.
        reference = _finite_or_none(train_score)
    per_joint = {k: v for k, v in ((k, _finite_or_none(v)) for k, v in per_joint.items()) if v is not None}
    return name, dataset, params, reference, per_joint


def lock_robot(lock: Any) -> str | None:
    """Registry name of the model a lock was fitted on (``None``: default)."""
    if isinstance(lock, dict):
        return lock.get("robot") or None
    return getattr(lock, "robot", None) or None


def _finite_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def select_episodes(episodes: Sequence[Episode], last: int | None = None) -> list[Episode]:
    """All episodes in index order, or only the ``last`` of them."""
    eps = sorted(episodes, key=lambda e: e.index)
    if last is not None:
        if last < 1:
            raise ValueError("last must be >= 1")
        eps = eps[-last:]
    return eps


def check(
    episodes: Sequence[Episode],
    lock: Any,
    threshold_deg: float | None = None,
    ratio: float = DEFAULT_RATIO,
    min_episodes: int = DEFAULT_MIN_EPISODES,
    *,
    last: int | None = None,
    simulate: Any | None = None,
    dataset: str | None = None,
) -> CheckResult:
    """Score ``lock``'s parameters on fresh ``episodes`` and compare with the
    lock's own held-out score.

    Pass iff ``overall_deg <= max(threshold_deg or 0, ratio * reference)``
    where ``reference`` is ``lock.result.eval_score``. Each joint is judged
    by the same rule against ``lock.result.per_joint_after``; those are
    informational and do not change ``passed``. ``worst_joint`` is the
    scored joint furthest over (or least under) its limit.

    Fewer than ``min_episodes`` episodes (after ``last`` is applied) is
    ``insufficient_data``: nothing is simulated and ``passed`` is False.

    ``lock`` may be a :class:`~antares.lock.RobotLock` or its JSON dict.
    ``simulate`` defaults to :func:`antares.calibrate.default_simulate`.
    """
    if ratio <= 0:
        raise ValueError("ratio must be > 0")
    if threshold_deg is not None and threshold_deg < 0:
        raise ValueError("threshold_deg must be >= 0")
    if min_episodes < 1:
        raise ValueError("min_episodes must be >= 1")

    name, lock_dataset, params, reference, per_joint_ref = _lock_view(lock)
    dataset = dataset if dataset is not None else lock_dataset
    eps = select_episodes(episodes, last)
    indices = [int(e.index) for e in eps]
    n_frames = sum(len(e) for e in eps)
    limit = limit_for(reference, threshold_deg, ratio)

    if len(eps) < min_episodes:
        return CheckResult(
            passed=False,
            verdict=VERDICT_INSUFFICIENT,
            score=None,
            reference=reference,
            limit=limit,
            per_joint={},
            worst_joint=None,
            n_episodes=len(eps),
            n_frames=n_frames,
            lock=name,
            dataset=dataset,
            threshold_deg=threshold_deg,
            ratio=ratio,
            min_episodes=min_episodes,
            episodes=indices,
        )

    if limit is None:
        raise ValueError(
            f"lock {name!r} has no finite eval score; pass threshold_deg to check against it"
        )

    from antares.gap import aggregate, tracking_error

    if simulate is None:
        from antares.calibrate import default_simulate

        simulate = default_simulate(robot=lock_robot(lock))
    from antares.calibrate import scoring_names

    names = scoring_names(simulate)
    report = aggregate([tracking_error(*simulate(params, ep), **names) for ep in eps])
    overall = float(report.overall_deg)
    scored = tuple(report.scored_joints)

    per_joint: dict[str, JointCheck] = {}
    for joint, value in report.per_joint_deg.items():
        value = float(value)
        ref = per_joint_ref.get(joint)
        jlimit = limit_for(ref, threshold_deg, ratio)
        passed = True if jlimit is None else bool(value <= jlimit)
        per_joint[joint] = JointCheck(score=value, reference=ref, limit=jlimit, passed=passed)

    worst = _worst_joint(per_joint, scored)
    passed = bool(math.isfinite(overall) and overall <= limit)
    return CheckResult(
        passed=passed,
        verdict=VERDICT_OK if passed else VERDICT_DRIFTED,
        score=overall,
        reference=reference,
        limit=limit,
        per_joint=per_joint,
        worst_joint=worst,
        n_episodes=len(eps),
        n_frames=int(report.n_frames),
        lock=name,
        dataset=dataset,
        threshold_deg=threshold_deg,
        ratio=ratio,
        min_episodes=min_episodes,
        episodes=indices,
        scored_joints=scored,
    )


def _worst_joint(per_joint: dict[str, JointCheck], scored: Sequence[str]) -> str | None:
    """Scored joint with the largest ``score / limit`` (``score`` when no limit)."""
    candidates = [j for j in scored if j in per_joint] or list(per_joint)
    if not candidates:
        return None

    def badness(j: str) -> float:
        jc = per_joint[j]
        if not math.isfinite(jc.score):
            return math.inf
        return jc.score / jc.limit if jc.limit else jc.score

    return max(candidates, key=badness)


__all__ = [
    "DEFAULT_MIN_EPISODES",
    "DEFAULT_RATIO",
    "VERDICT_DRIFTED",
    "VERDICT_INSUFFICIENT",
    "VERDICT_OK",
    "CheckResult",
    "JointCheck",
    "check",
    "limit_for",
    "select_episodes",
]
