"""Compare the STS3215 servo model against the position actuator on a lock's own split.

For each lock named on the command line (default: the two demo locks) this
replays the lock's held-out episodes through:

* the stock sim (``SimParams()``),
* the lock itself (position actuator, as fitted),
* BAM's STS3215 model with no fitting at all (``SimParams.sts3215()``), with
  only the command latency picked on the training episodes,
* a servo-model fit (``calibrate`` from that prior) and, on the
  same budget, a fresh position-actuator fit, each the best of ``--seeds``
  CMA-ES seeds, so the spread of the parameters across seeds can be
  compared (is the servo gain pinned where ``kp`` was not?).

Needs the datasets, so it runs where the Hugging Face Hub is reachable::

    python scripts/servo_study.py --out results/servo_study.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from antares.calibrate import calibrate, default_simulate, score
from antares.data import load_dataset
from antares.lock import RobotLock
from antares.sim import SimParams

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCKS = ("svla_so101_pickplace", "svla_so100_pickplace")


def held_out(params: SimParams, episodes, sim) -> dict:
    r = score(params, episodes, sim)
    return {"overall_deg": float(r.overall_deg), "per_joint_deg": {k: float(v) for k, v in r.per_joint_deg.items()}}


def fit(train, eval_, start: SimParams, sim, seeds: int, evals: int) -> dict:
    t0 = time.perf_counter()
    res = calibrate(train, eval_, start=start, simulate=sim, max_evaluations=evals, n_seeds=seeds)
    out = {
        "eval_deg": float(res.eval_score),
        "train_deg": float(res.train_score),
        "params": res.params.to_dict(),
        "elapsed_s": time.perf_counter() - t0,
        "seeds": res.seeds,
    }
    if res.params.uses_servo_model:
        lin = res.params.linearized()
        out["pd_equivalent"] = {"kp": lin.kp, "kv": lin.kv, "forcerange": lin.forcerange}
    return out


def study(name: str, seeds: int, evals: int) -> dict:
    lock = RobotLock.load(ROOT / "locks" / f"{name}.lock.json")
    ds = load_dataset(lock.dataset)
    by_index = {ep.index: ep for ep in ds.episodes}
    train = [by_index[i] for i in lock.result.train_episodes]
    eval_ = [by_index[i] for i in lock.result.eval_episodes]
    sim = default_simulate(robot=lock.robot)
    out: dict = {"name": name, "dataset": lock.dataset, "n_train": len(train), "n_eval": len(eval_)}
    out["stock"] = held_out(SimParams(), eval_, sim)
    out["lock"] = held_out(lock.params, eval_, sim)
    out["lock_params"] = lock.params.to_dict()
    print(f"[{name}] stock {out['stock']['overall_deg']:.3f}  lock {out['lock']['overall_deg']:.3f} deg", flush=True)

    # BAM's servo with nothing fitted but the latency (chosen on train).
    train_by_lat = {lat: score(SimParams.sts3215(latency_frames=lat), train, sim).overall_deg for lat in range(6)}
    best_lat = min(train_by_lat, key=train_by_lat.get)
    out["bam_prior"] = {
        "latency_frames": best_lat,
        "train_deg_by_latency": train_by_lat,
        **held_out(SimParams.sts3215(latency_frames=best_lat), eval_, sim),
    }
    print(f"[{name}] BAM prior (latency {best_lat}) {out['bam_prior']['overall_deg']:.3f} deg", flush=True)

    # The fit starts there: latency is a few discrete steps, cheap to grid.
    out["servo_fit"] = fit(train, eval_, SimParams.sts3215(latency_frames=best_lat), sim, seeds, evals)
    print(f"[{name}] servo fit {out['servo_fit']['eval_deg']:.3f} deg  {out['servo_fit']['params']}", flush=True)
    out["position_fit"] = fit(train, eval_, SimParams(), sim, seeds, evals)
    print(f"[{name}] position fit {out['position_fit']['eval_deg']:.3f} deg  {out['position_fit']['params']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("locks", nargs="*", default=list(DEFAULT_LOCKS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--evaluations", type=int, default=150)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "servo_study.json")
    args = ap.parse_args()
    results = [study(n, args.seeds, args.evaluations) for n in args.locks]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(f"wrote {args.out}")
    for r in results:
        print(
            f"{r['name']}: stock {r['stock']['overall_deg']:.3f} | lock {r['lock']['overall_deg']:.3f} | "
            f"BAM prior {r['bam_prior']['overall_deg']:.3f} | servo fit {r['servo_fit']['eval_deg']:.3f} | "
            f"position fit {r['position_fit']['eval_deg']:.3f} deg (held-out)"
        )


if __name__ == "__main__":
    main()
