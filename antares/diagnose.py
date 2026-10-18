"""Name the cause of drift: what changed in the arm since its lock was fitted.

:func:`antares.check.check` says *whether* a robot still matches its lock.
:func:`diagnose` says *what* changed. It refits the lock on fresh episodes,
but instead of searching the parameters from scratch it searches one
*change* per parameter on top of the lock: a factor on each of ``kp``,
``kv``, ``damping``, ``armature`` and ``frictionloss`` (x1/4 to x4,
log-uniform) and a shift of whole frames on ``latency_frames``. A per-joint
lock keeps its per-joint shape; every joint gets the same change. The search
starts at "no change" (after trying each latency shift), so a parameter
only moves when the new data asks for it.

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
from collections.abc import Mapping

import numpy as np

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


__all__ = [
    "FACTOR_RANGE",
    "FIELDS",
    "LATENCY_SHIFT",
    "apply_change",
    "decode_change",
    "describe_change",
    "encode_change",
    "neutral",
]
