"""Drift demo: a simulated arm wears week by week; the check catches it and
names what changed.

No physical arm or download needed. The "real" arm is the MuJoCo arm with
the committed SO-101 lock's dynamics. Each week it is recorded again with a
little more wear: the servo gain fades 5% a week, friction builds 20% a
week, and from week 6 the command latency is one frame longer. Every week
``antares.check`` scores the lock on the new recording, and when it fails,
``antares.diagnose`` refits the changes and names the likely cause, which
can be compared with the wear that was actually applied.

    python examples/drift_demo.py            # under a minute
    python examples/drift_demo.py --quick    # fewer episodes and evaluations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from antares.calibrate import default_simulate  # noqa: E402
from antares.check import check  # noqa: E402
from antares.diagnose import describe_change, diagnose  # noqa: E402
from antares.lock import RobotLock  # noqa: E402
from antares.sim import Arm  # noqa: E402
from antares.synth import synthesize, worn  # noqa: E402

WEEKS = (0, 2, 4, 6, 8, 10)


def wear_at(week: int) -> dict[str, float]:
    return {
        "kp": 0.95**week,
        "frictionloss": 1.2**week,
        "latency_frames": 1 if week >= 6 else 0,
    }


def _wear_text(wear: dict[str, float]) -> str:
    parts = [f"{k} {describe_change(k, v)}" for k, v in wear.items()
             if (v != 0 if k == "latency_frames" else abs(v - 1) > 1e-9)]
    return ", ".join(parts) or "none"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lock", default=str(ROOT / "locks" / "svla_so101_pickplace.lock.json"))
    ap.add_argument("--out", default=None, help="write the per-week reports here as JSON")
    args = ap.parse_args(argv)
    n_eps, n_frames, evals = (6, 150, 90) if args.quick else (10, 300, 300)

    lock = RobotLock.load(args.lock)
    arm, simulate = Arm(), default_simulate()
    print(f"lock {lock.name}: held-out {lock.result.eval_score:.3f} deg when fitted\n", flush=True)

    rows, reports = [], []
    for week in WEEKS:
        wear = wear_at(week)
        ds = synthesize(worn(lock.params, wear), n_episodes=n_eps, n_frames=n_frames,
                        seed=week, arm=arm, repo_id=f"synthetic/week{week}")
        res = check(ds.episodes, lock, simulate=simulate, dataset=ds.repo_id)
        found = "-"
        report = {"week": week, "wear": wear, "check": res.to_dict()}
        if not res.passed:
            d = diagnose(ds.episodes, lock, simulate=simulate, max_evaluations=evals,
                         seed=week, dataset=ds.repo_id)
            report["diagnosis"] = d.to_dict()
            if not d.explained:
                found = "not explained by the dynamics"
            else:
                found = ", ".join(f"{c.name} {c.change}" for c in d.causes) or "several together"
        reports.append(report)
        rows.append((week, res.score, res.limit, "ok" if res.passed else "DRIFTED", found, _wear_text(wear)))
        print(f"week {week:>2}: {res.summary()}", flush=True)

    print()
    width = max(len("named cause"), *(len(r[4]) for r in rows))
    print(f"week  error (deg)  limit  verdict  {'named cause':<{width}}  actual wear")
    for week, score, limit, verdict, found, actual in rows:
        print(f"{week:<4}  {score:<11.3f}  {limit:<5.3f}  {verdict:<7}  {found:<{width}}  {actual}")
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=2) + "\n")
        print(f"\nreports -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
