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


def _pct(v) -> str:
    return "-" if v is None else f"{v:+.0%}"


def _num(v, fmt: str = "{:.3f}") -> str:
    return "-" if v is None else fmt.format(v)


def cmd_drift(args: argparse.Namespace) -> int:
    from antares import server

    ds = server.get_dataset(args.repo_id)
    print(
        f"drift scan of lock {args.lock} over {ds.repo_id}: windows of {args.window} episodes, "
        f"step {args.step}, up to {args.max_evaluations} evaluations each"
    )

    def on_progress(ev) -> None:
        w = ev.window
        print(
            f"  window {ev.index + 1:2d}/{ev.n_windows}  episodes {w.first:3d}-{w.last:<3d}  "
            f"lock {w.lock_score:6.3f}  refit {w.train_score:6.3f} deg  "
            f"({w.improvement:+.1%})  [{ev.elapsed_s:.0f}s]",
            flush=True,
        )

    scan, path = server.run_drift(
        ds,
        args.lock,
        window=args.window,
        step=args.step,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        on_progress=on_progress,
    )
    s = scan.to_dict()
    param_names = list(s["lock_params"])
    rows = []
    for w in s["windows"]:
        p = w["params"]
        rows.append(
            [f"{w['first']}-{w['last']}", _num(w["lock_score"]), _num(w["train_score"]), _pct(w["improvement"])]
            + [_num(p[k], "{:.4g}") for k in param_names]
        )
    rows.append(
        ["lock", "-", "-", "-"] + [_num(s["lock_params"][k], "{:.4g}") for k in param_names]
    )
    print()
    print("per window: the lock's static parameters vs a refit warm-started from them (RMSE, deg)")
    print(_table(["episodes", "lock", "refit", "impr"] + param_names, rows))

    trows = []
    for k, t in s["trends"].items():
        unit = "log10/ep" if t["log_space"] else "/ep"
        trows.append(
            [
                k,
                f"{_num(t['slope'], '{:+.4g}')} {unit}",
                _num(t["r2"], "{:.2f}"),
                f"[{_num(t['ci_lo'], '{:+.3g}')}, {_num(t['ci_hi'], '{:+.3g}')}]",
                _pct(t["total_change"]),
                "yes" if t["drifting"] else ("sig" if t["significant"] else "no"),
            ]
        )
    print()
    print(f"trends over the session (bootstrap 95% CI over {len(s['windows'])} windows)")
    print(_table(["quantity", "slope", "r2", "slope 95% CI", "total", "drift"], trows))
    print()
    print(f"verdict: {s['verdict']}")
    print(f"saved drift scan -> {path}")
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


def cmd_export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from fastapi import HTTPException

    from antares import server
    from antares.export import SCENE_FILENAME, export_scene, sidecar_path
    from antares.lock import RobotLock

    try:
        lock = RobotLock.from_dict(server.load_lock_dict(args.name))
    except HTTPException as e:
        print(f"error: {e.detail}", file=sys.stderr)
        return 1
    out_dir = Path(args.output) if args.output else Path("export") / lock.name
    scene = export_scene(lock, out_dir, bundle_assets=not args.no_bundle)
    arm = out_dir / f"{lock.name}.xml"
    p = lock.params
    print(f"exported lock {lock.name} ({lock.dataset}) -> {out_dir}/")
    print(f"  {arm.name:<40} calibrated arm MJCF")
    print(f"  {sidecar_path(arm).name:<40} params incl. latency_frames={p.latency_frames}, unit map")
    print(f"  {SCENE_FILENAME:<40} arm + floor/light")
    if args.no_bundle:
        print("  meshdir points at the antares package assets (absolute path)")
    else:
        print(f"  {'assets/':<40} STL meshes")
    print(f"view: python -m mujoco.viewer --mjcf {scene}")
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

    p = sub.add_parser("drift", help="refit a lock on sliding episode windows to spot drift")
    p.add_argument("repo_id")
    p.add_argument("--lock", required=True, help="lock whose parameters seed each window")
    p.add_argument("--window", type=int, default=10, help="episodes per window")
    p.add_argument("--step", type=int, default=5, help="episodes between window starts")
    p.add_argument("--max-evaluations", type=int, default=60, help="CMA-ES budget per window")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("locks", help="list saved locks")
    p.set_defaults(func=cmd_locks)

    p = sub.add_parser("export", help="write a lock as a standalone MJCF (+ scene, sidecar, meshes)")
    p.add_argument("name", help="lock name (see `antares locks`)")
    p.add_argument("-o", "--output", default=None, help="output directory (default: export/<name>)")
    p.add_argument(
        "--no-bundle",
        action="store_true",
        help="do not copy the STL meshes; point meshdir at the package assets instead",
    )
    p.set_defaults(func=cmd_export)

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
