"""``antares`` command line.

Heavy imports (MuJoCo, FastAPI, CMA-ES) live inside the command functions so
``antares --help`` is instant.
"""

from __future__ import annotations

import argparse
import sys


def _table(headers: list[str], rows: list[list[str]]) -> str:
    cols = [headers] + rows
    widths = [max(len(str(r[i])) for r in cols) for i in range(len(headers))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*(str(c) for c in r)) for r in rows]
    return "\n".join(lines)


def _joint_table(per_joint: dict, overall: float, title: str = "") -> str:
    rows = [[j, f"{v:.3f}"] for j, v in per_joint.items()]
    rows.append(["overall", f"{overall:.3f}"])
    out = _table(["joint", "rmse (deg)"], rows)
    return f"{title}\n{out}" if title else out


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "antares.server:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from antares.data import load_dataset

    ds = load_dataset(args.repo_id)
    frames = sum(len(e) for e in ds.episodes)
    print(
        f"{ds.repo_id}: {ds.robot_type}, {len(ds.episodes)} episodes, "
        f"{frames} frames @ {ds.fps:g} fps"
    )
    print("joints:", ", ".join(ds.joint_names))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from antares import server

    ds = server.get_dataset(args.repo_id)
    params = None
    if args.lock:
        params = server.params_from_dict(server.load_lock_dict(args.lock)["params"])
    out = server.replay_episode(ds, args.episode, params)
    gap = out["gap"]
    label = f"lock {args.lock}" if args.lock else "default params"
    print(f"{ds.repo_id} episode {out['episode']} ({len(out['real_q'])} frames, {label})")
    print(_joint_table(gap["per_joint_deg"], gap["overall_deg"]))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from antares import server

    ds = server.get_dataset(args.repo_id)
    print(
        f"calibrating {ds.repo_id}: {args.n_train} train / {args.n_eval} eval episodes, "
        f"up to {args.max_evaluations} evaluations"
    )
    last = {"n": 0}

    def on_progress(ev) -> None:
        n = ev.evaluation
        if n - last["n"] >= 10 or n <= 1:
            last["n"] = n
            print(
                f"  eval {n:4d}  gen {ev.generation:3d}  "
                f"candidate {ev.candidate_score:7.3f}  best {ev.best_score:7.3f} deg  "
                f"[{ev.elapsed_s:.1f}s]",
                flush=True,
            )

    result, lock, path = server.run_calibration(
        ds,
        n_train=args.n_train,
        n_eval=args.n_eval,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        name=args.name,
        on_progress=on_progress,
    )
    r = result.to_dict()
    before = r.get("per_joint_before") or {}
    after = r.get("per_joint_after") or {}
    rows = [
        [j, f"{before.get(j, float('nan')):.3f}", f"{after.get(j, float('nan')):.3f}"]
        for j in sorted(set(before) | set(after), key=list(before).index)
    ]
    rows.append(
        [
            "eval overall",
            f"{r['baseline_eval_score']:.3f}",
            f"{r['eval_score']:.3f}",
        ]
    )
    print()
    print(_table(["joint", "before (deg)", "after (deg)"], rows))
    print()
    print("params:", r["params"])
    print(f"{r['n_evaluations']} evaluations in {r['elapsed_s']:.1f}s")
    print(f"saved lock {lock['name']} -> {path}")
    return 0


def cmd_cross(args: argparse.Namespace) -> int:
    from antares import server

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    locks = [l.strip() for l in (args.locks or "").split(",") if l.strip()]
    m = server.cross_matrix(
        datasets, locks, n_train=args.n_train, n_eval=args.n_eval, seed=args.seed
    )
    rows = [
        [row] + [f"{s:.3f}" for s in scores]
        for row, scores in zip(m["rows"], m["scores"])
    ]
    print("eval tracking RMSE (deg), rows = datasets, cols = locks")
    print(_table(["dataset"] + m["cols"], rows))
    return 0


def _feasibility_line(f: dict) -> str:
    verdict = "ok" if f.get("ok") else f"FLAGGED {', '.join(f.get('flagged_joints', []))}"
    bits = [f"feasibility (sim-only): {verdict}"]
    real = f.get("real_vmax_deg_s")
    comp = f.get("comp_cmd_vmax_deg_s")
    if isinstance(real, dict) and isinstance(comp, dict):
        ratio = max((comp[j] / real[j]) for j in real if real[j] > 0) if real else float("nan")
    else:
        per = f.get("per_joint") or {}
        ratios = [v["velocity_ratio"] for v in per.values() if v.get("velocity_ratio") is not None]
        ratio = max(ratios) if ratios else float("nan")
    bits.append(f"peak command velocity {ratio:.2f}x the fastest real joint motion (limit {f.get('velocity_factor', 1.5):g}x)")
    bits.append(f"exceeds the original command range by {f.get('max_range_excess_deg', 0.0):.2f} deg")
    bits.append(f"{f.get('clipped_frames', 0)} frames at the ctrl limit (raw: {f.get('raw_clipped_frames', 0)})")
    return "; ".join(bits)


def cmd_compensate(args: argparse.Namespace) -> int:
    from antares import server

    ds = server.get_dataset(args.repo_id)
    lock = server.load_lock_dict(args.lock) if args.lock else None
    params = server.params_from_dict(lock["params"]) if lock else None
    label = f"lock {args.lock}" if args.lock else "default params (no lock: latency 0)"
    print("tracking RMS vs the real trajectory, judged in the calibrated sim (no hardware)")

    if args.heldout:
        if lock is None:
            print("--heldout needs --lock (its held-out split is the eval set)", file=sys.stderr)
            return 2
        out = server.compensate_heldout(ds, lock, method=args.method)
        n = len(out["episodes"])
        print(f"{ds.repo_id}: {n} held-out episodes {out['episodes']} ({out['n_frames']} frames, {label})")
        it = out["ilc_iterations"]
        rows = [
            ["raw", f"{out['rms']['raw']:.3f}", "logged commands (= the lock's gap)"],
            ["shift", f"{out['rms']['shift']:.3f}", f"q_des[t+{out['params']['latency_frames']}]"],
            ["ilc", f"{out['rms']['ilc']:.3f}", f"{min(it)}-{max(it)} iterations/episode"],
        ]
        print(_table(["method", "rms (deg)", "note"], rows))
        print(_feasibility_line(out["feasibility"]))
        return 0

    if args.episode is None:
        print("--episode N (or --heldout with --lock) is required", file=sys.stderr)
        return 2
    out = server.compensate_episode_dict(ds, args.episode, params, method=args.method, lock=args.lock)
    print(f"{ds.repo_id} episode {out['episode']} ({len(out['q_des'])} frames, {label}, method {args.method})")
    rows = [
        ["raw", f"{out['rms']['raw']:.3f}", "logged commands (= the gap)"],
        ["shift", f"{out['rms']['shift']:.3f}", f"q_des[t+{out['params']['latency_frames']}]"],
        ["ilc", f"{out['rms']['ilc']:.3f}", f"{out['ilc_iterations']} iterations"],
    ]
    print(_table(["method", "rms (deg)", "note"], rows))
    print(_feasibility_line(out["feasibility"]))
    if args.out:
        import numpy as np

        um = server.get_arm().unit_map
        cmds = um.to_units(np.radians(np.asarray(out["ctrl_comp"], dtype=np.float64)))
        np.save(args.out, cmds)
        print(f"saved {args.method} commands ({cmds.shape[0]}x{cmds.shape[1]}, dataset units) -> {args.out}")
    return 0


def cmd_locks(args: argparse.Namespace) -> int:
    from antares import server

    locks = server.list_lock_dicts()
    if not locks:
        print(f"no locks in {server.LOCK_DIR}")
        return 0
    rows = []
    for l in locks:
        s = server.lock_summary(l)["scores"]

        def f(v):
            return "-" if v is None else f"{v:.3f}"

        rows.append(
            [
                l.get("name", "?"),
                l.get("dataset", "?"),
                f(s["baseline_eval_score"]),
                f(s["eval_score"]),
                str(l.get("created_at", ""))[:19],
            ]
        )
    print(_table(["name", "dataset", "baseline eval", "calibrated eval", "created"], rows))
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="antares",
        description="Calibrate a simulator to one specific robot from its own logs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the API server for the web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("inspect", help="summarize a LeRobot dataset")
    p.add_argument("repo_id")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("replay", help="replay one episode in sim and print the gap")
    p.add_argument("repo_id")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--lock", default=None, help="use a saved lock's parameters")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("calibrate", help="fit sim parameters to a dataset and save a lock")
    p.add_argument("repo_id")
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--max-evaluations", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="lock name (default: dataset slug)")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("cross", help="score locks across datasets (matrix)")
    p.add_argument("--datasets", required=True, help="comma-separated dataset ids")
    p.add_argument("--locks", default="", help="comma-separated lock names")
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_cross)

    p = sub.add_parser(
        "compensate",
        help="commands that make the calibrated sim track an episode's real trajectory (sim-only)",
    )
    p.add_argument("repo_id")
    p.add_argument("--episode", type=int, default=None, help="episode to compensate")
    p.add_argument("--lock", default=None, help="calibrated lock to compensate against")
    p.add_argument("--method", choices=["shift", "ilc"], default="ilc", help="which commands to save/return")
    p.add_argument("--out", default=None, help="save the compensated commands (T,6) in dataset units as .npy")
    p.add_argument(
        "--heldout", action="store_true", help="instead of one episode, aggregate over the lock's held-out split"
    )
    p.set_defaults(func=cmd_compensate)

    p = sub.add_parser("locks", help="list saved locks")
    p.set_defaults(func=cmd_locks)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
