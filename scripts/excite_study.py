"""Does the calibration dance make the servo parameters identifiable? (sim-only)

Each synthetic arm is the SO-101 MuJoCo model with known parameters (the
two demo locks' shared fits, and the stock MJCF with two frames of
latency). Two recordings are simulated on each, with 0.3 units of sensor
noise like :func:`antares.synthetic.make_dataset`:

* **pick-and-place**: the real commands of ``lerobot/svla_so101_pickplace``
  (50 episodes, the teleoperator's actions), split 20 train / 10 held out
  with the calibrator's seeded split;
* **dance**: one replay of :func:`antares.excite.dance` (about 60 s).

Three training sets are fitted with ``--seeds`` CMA-ES seeds each, at the
calibrator's default budget (150 evaluations) and a larger one: the 20
pick-and-place episodes, the dance alone, and both. Every fit is scored on
the same 10 held-out pick-and-place episodes. Per arm and training set the
report gives the seed spread of each parameter (``seed_restarts``), the
worst error against the true value, the linearized identifiability at the
true parameters, and the held-out error, plus each fit's training error
next to the truth's (a fit that trains worse than the truth is the
optimizer stopping short).

Needs the Hub for the pick-and-place commands::

    python scripts/excite_study.py --out results/excite_study.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from antares import excite
from antares.calibrate import DIMS, SPACE, _safe_score, default_simulate, split_episodes
from antares.data import Episode, load_dataset
from antares.server import params_from_dict
from antares.sim import Arm, SimParams
from antares.specificity import _spread, identifiability, seed_restarts

ROOT = Path(__file__).resolve().parent.parent
PICKPLACE = "lerobot/svla_so101_pickplace"
NOISE_UNITS = 0.3
CONTINUOUS = tuple(n for n in DIMS if SPACE[n][2] != "int")


def truths() -> dict[str, SimParams]:
    out = {}
    for name in ("svla_so101_pickplace", "svla_so100_pickplace"):
        lock = json.loads((ROOT / "locks" / f"{name}.lock.json").read_text())
        out[f"{name.split('_')[1]} lock"] = params_from_dict(lock["params"])
    out["stock MJCF"] = SimParams(latency_frames=2)
    return out


def record(arm: Arm, params: SimParams, actions: list[np.ndarray], fps: float, rng, start: int = 0) -> list[Episode]:
    """What ``params``' arm records when driven by ``actions`` (like make_dataset)."""
    arm.apply(params)
    eps = []
    for i, a in enumerate(actions):
        cmd = Episode(index=start + i, fps=fps, joint_names=arm.robot.joints, state=a, action=a)
        q = arm.replay(cmd)
        state = arm.unit_map.to_units(q) + rng.normal(0.0, NOISE_UNITS, size=q.shape)
        state[0] = a[0]
        eps.append(Episode(index=start + i, fps=fps, joint_names=arm.robot.joints, state=state, action=a))
    return eps


def param_errors(runs: list[dict], truth: SimParams) -> dict:
    """Per parameter: max(fit/true, true/fit) worst and median over the seeds
    (for latency, the largest frame error)."""
    out = {}
    for n in DIMS:
        t = float(getattr(truth, n))
        v = np.array([r["params"][n] for r in runs], dtype=np.float64)
        if n == "latency_frames":
            out[n] = {"worst": float(np.abs(v - t).max()), "median": float(np.median(np.abs(v - t)))}
        else:
            ratio = np.maximum(v / t, t / v)
            out[n] = {"worst": float(ratio.max()), "median": float(np.median(ratio))}
    return out


_SIM = None


def _sim():
    global _SIM
    if _SIM is None:
        _SIM = default_simulate()
    return _SIM


def _fit(job: tuple) -> tuple:
    """One CMA-ES seed (``seed_restarts`` with a single seed), in a worker process."""
    key, train, eval_, seed, evals = job
    t0 = time.perf_counter()
    rs = seed_restarts(train, eval_, [seed], _sim(), max_evaluations=evals)
    return key, rs["runs"][0], time.perf_counter() - t0


def summarize(runs: list[dict], truth: SimParams, truth_train: float) -> dict:
    ev = [r["eval_score"] for r in runs]
    tr = [r["train_score"] for r in runs]
    return {
        "runs": runs,
        "spread": _spread([r["params"] for r in runs]),
        "error_vs_truth": param_errors(runs, truth),
        "held_out": {"best": float(min(ev)), "median": float(np.median(ev)), "worst": float(max(ev))},
        # a fit that trains worse than the truth is the optimizer stopping short,
        # not the data allowing it
        "train": {"truth": float(truth_train), "best": float(min(tr)), "median": float(np.median(tr)),
                  "worst": float(max(tr))},
    }


def study(seeds: list[int], budgets: list[int], split_seed: int = 0, workers: int = 8) -> dict:
    ds = load_dataset(PICKPLACE)
    actions = [ep.action for ep in ds.episodes]
    d = excite.dance(fps=ds.fps)
    safety = excite.check_safety(d.commands, d.fps, d.steps)
    sim = _sim()
    arm = Arm()
    out: dict = {
        "config": {"pickplace": PICKPLACE, "noise_units": NOISE_UNITS, "seeds": seeds, "budgets": budgets,
                   "split_seed": split_seed, "dance": d.config, "dance_seconds": d.duration},
        "dance_safety": safety,
        "arms": {},
    }
    sets: dict = {}
    jobs = []
    for k, (name, truth) in enumerate(truths().items()):
        rng = np.random.default_rng(100 + k)
        pp = record(arm, truth, actions, ds.fps, rng)
        train_pp, eval_pp = split_episodes(pp, 20, 10, seed=split_seed)
        dance_ep = record(arm, truth, [d.commands], d.fps, rng, start=1000)
        # the simulated arm's own path through the dance, checked like the commands
        response = excite.check_safety(dance_ep[0].state, d.fps, d.steps, max_speed=math.inf)
        out["arms"][name] = {
            "truth": truth.to_dict(),
            "truth_held_out": _safe_score(truth, eval_pp, sim)[1],
            "dance_response_safety": {k2: response[k2] for k2 in ("min_tip_z", "min_link_z", "max_abs_units", "ok")},
            "train_sets": {},
        }
        for label, train in (("pick-and-place", train_pp), ("dance", dance_ep),
                             ("dance + pick-and-place", dance_ep + train_pp)):
            sets[(name, label)] = (truth, train, eval_pp)
            idf = identifiability(truth, train, sim, names=CONTINUOUS)
            out["arms"][name]["train_sets"][label] = {
                "n_train_episodes": len(train),
                "n_train_frames": int(sum(len(e) for e in train)),
                "identifiability_at_truth": idf.to_dict(),
                "budgets": {},
            }
            for b in budgets:
                jobs += [((name, label, b), train, eval_pp, s, b) for s in seeds]
    # longest fits first so the pool does not end on one straggler
    jobs.sort(key=lambda j: -j[4] * sum(len(e) for e in j[1]))
    print(f"{len(jobs)} fits on {workers} workers", flush=True)
    runs: dict = {}
    with ProcessPoolExecutor(workers) as pool:
        for i, (key, run, secs) in enumerate(pool.map(_fit, jobs)):
            runs.setdefault(key, []).append(run)
            print(f"  [{i + 1}/{len(jobs)}] {key} seed {run['seed']}: train {run['train_score']:.3f} "
                  f"held-out {run['eval_score']:.3f} ({secs:.0f}s)", flush=True)
    for (name, label, b), rs in runs.items():
        truth, train, _ = sets[(name, label)]
        rs.sort(key=lambda r: r["seed"])
        summ = summarize(rs, truth, _safe_score(truth, train, sim)[1])
        out["arms"][name]["train_sets"][label]["budgets"][str(b)] = summ
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _x(v: float | None) -> str:
    if v is None or not math.isfinite(v) or v > 1e6:
        return "unbounded"
    return f"×{v:.3g}"


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def _spread_cell(n: str, v: float) -> str:
    return f"±{int(v)}" if n == "latency_frames" else _x(v)


def _err_cell(n: str, v: float) -> str:
    return f"{int(v)} fr" if n == "latency_frames" else _x(v)


def render(res: dict) -> str:
    c = res["config"]
    s = res["dance_safety"]
    budgets = [str(b) for b in c["budgets"]]
    labels = list(next(iter(res["arms"].values()))["train_sets"])
    out = ["## Calibration dance (sim-only)", "",
           f"Synthetic SO-101 arms with known parameters and {NOISE_UNITS} units of sensor noise, recorded "
           f"twice: on the 50 real teleoperated pick-and-place command sequences of `{c['pickplace']}` "
           f"(20 train, 10 held out) and on one {c['dance_seconds']:.1f} s calibration dance. Each training "
           f"set is fitted with {len(c['seeds'])} CMA-ES seeds at {' and '.join(budgets)} evaluations; every "
           f"fit is scored on the same 10 held-out pick-and-place episodes. The dance's commanded fingertip "
           f"stays {(s['min_tip_z'] - excite.TABLE_Z) * 100:.0f} cm or more above the table, body joints within "
           f"±{s['max_abs_units']:.0f} units, peak speed outside steps "
           f"{max(list(s['peak_speed_rad_s'].values())[:5]):.1f} rad/s. Nothing here was run on an arm.", ""]
    for name, a in res["arms"].items():
        t = a["truth"]
        ts = a["train_sets"]
        out += [f"### {name}", "",
                "True parameters: " + ", ".join(f"{n} {t[n]:.4g}" for n in DIMS)
                + f". Held-out error at the truth: {a['truth_held_out']:.3f}°.", ""]
        hdr = ["param"] + [f"{lab}, {b}" for b in budgets for lab in labels]
        rows = [[n] + [_spread_cell(n, ts[lab]["budgets"][b]["spread"][n]) for b in budgets for lab in labels]
                for n in DIMS]
        out += ["Seed spread (max/min over the seeds; latency: max-min frames), per training set and budget:", "",
                _table(hdr, rows), ""]
        rows = [[n] + [_err_cell(n, ts[lab]["budgets"][b]["error_vs_truth"][n]["worst"]) for b in budgets
                       for lab in labels] for n in DIMS]
        out += ["Worst error against the truth over the seeds (max of fit/true, true/fit):", "",
                _table(hdr, rows), ""]
        rows = [[n] + [_x(ts[lab]["identifiability_at_truth"]["factor_free"][n]) for lab in labels]
                for n in CONTINUOUS]
        rows.append(["condition number"] + [f"{ts[lab]['identifiability_at_truth']['condition_number']:.3g}"
                                            for lab in labels])
        out += ["Linearized at the truth: how far each parameter can move, the others refit, before the "
                "training error rises 0.1°:", "", _table(["param"] + labels, rows), ""]
        rows = []
        for b in budgets:
            for lab in labels:
                cd = ts[lab]["budgets"][b]
                rows.append([lab, b, ts[lab]["n_train_frames"], f"{cd['train']['truth']:.3f}°",
                             f"{cd['train']['best']:.3f}°", f"{cd['train']['worst']:.3f}°",
                             f"{cd['held_out']['best']:.3f}°", f"{cd['held_out']['median']:.3f}°",
                             f"{cd['held_out']['worst']:.3f}°"])
        out += ["Training error of the fits against the truth's own, and held-out pick-and-place error:", "",
                _table(["trained on", "evals", "frames", "train, truth", "train, best", "train, worst",
                        "held-out best", "median", "worst"], rows), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--evaluations", type=int, nargs="+", default=[150, 600],
                    help="CMA-ES budgets per fit (150 is the calibrator's default)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "excite_study.json")
    args = ap.parse_args()
    res = study(list(range(1, args.seeds + 1)), args.evaluations, workers=args.workers)
    args.out.write_text(json.dumps(res, indent=2, default=float) + "\n")
    args.out.with_suffix(".md").write_text(render(res) + "\n")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
