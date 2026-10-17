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
from collections.abc import Mapping
from typing import Any

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


__all__ = ["summarize_cross"]
