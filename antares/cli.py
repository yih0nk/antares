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


def _f(v, fmt: str = "{:.3f}") -> str:
    return "-" if v is None else fmt.format(v)


def _fleet_tables(report: dict) -> str:
    """Human-readable fleet report: robots, parameter spread, outliers, cross matrix."""
    out: list[str] = []
    robots = report.get("robots", [])
    rows = []
    for r in robots:
        p = r.get("params") or {}
        rows.append(
            [
                r["dataset"],
                r.get("robot_type", "?"),
                str(r.get("n_episodes", "?")),
                f"{r.get('fps', 0):g}",
                _f(r.get("baseline_eval")),
                _f(r.get("calibrated_eval")),
                _f(r.get("improvement"), "{:.0%}"),
                f"{p.get('latency_frames', '-')} ({_f(r.get('latency_ms'), '{:.0f}')} ms)",
                _f(p.get("kp"), "{:.0f}"),
            ]
        )
    s = report.get("summary") or {}
    out.append(
        f"{s.get('n_robots', len(robots))} robots calibrated, {s.get('n_failed', 0)} failed, "
        f"{s.get('n_skipped', 0)} skipped, {s.get('n_rejected', 0)} rejected; "
        f"median gap closed {_f(s.get('median_improvement'), '{:.0%}')}"
    )
    if rows:
        out.append("")
        out.append(
            _table(
                ["dataset", "robot", "eps", "fps", "baseline", "calibrated", "closed", "latency", "kp"],
                rows,
            )
        )
    stats = report.get("stats") or {}
    if stats:
        out.append("")
        out.append("fitted parameters across the fleet")
        srows = [
            [p, _f(st.get("median"), "{:.4g}"), _f(st.get("q25"), "{:.4g}"), _f(st.get("q75"), "{:.4g}"),
             _f(st.get("min"), "{:.4g}"), _f(st.get("max"), "{:.4g}"), str(st.get("n", ""))]
            for p, st in stats.items()
        ]
        out.append(_table(["param", "median", "q25", "q75", "min", "max", "n"], srows))
    outliers = report.get("outliers") or []
    if outliers:
        out.append("")
        out.append("outliers (|robust z| > 2.5)")
        out.append(_table(["dataset", "param", "z", "note"], [[o["dataset"], o["param"], f"{o['z']:+.1f}", o["note"]] for o in outliers]))
    cross = report.get("cross")
    if cross:
        out.append("")
        out.append("cross matrix: eval RMSE (deg), rows = robots, cols = locks")
        short = [c.split("__", 1)[-1][:18] for c in cross["cols"]]
        crow = [[row.split("/", 1)[-1][:28]] + [f"{v:.3f}" for v in scores] for row, scores in zip(cross["rows"], cross["scores"])]
        out.append(_table(["robot"] + short, crow))
        cs = cross.get("summary") or {}
        out.append(
            f"own lock {_f(cs.get('mean_diagonal'))} deg vs other robots' locks {_f(cs.get('mean_off_diagonal'))} deg "
            f"({_f(cs.get('off_over_diagonal'), '{:.2f}')}x) vs default {_f(cs.get('mean_default'))} deg "
            f"({_f(cs.get('default_over_diagonal'), '{:.2f}')}x); own lock best for {cs.get('own_lock_is_best', '-')}/{cs.get('n', '-')}"
        )
    failures = report.get("failures") or []
    if failures:
        out.append("")
        out.append("failures")
        out.append(_table(["dataset", "error"], [[f["dataset"], f["error"][:100]] for f in failures]))
    return "\n".join(out)


def cmd_fleet_scan(args: argparse.Namespace) -> int:
    from antares import fleet

    counts = {"ok": 0, "no": 0}

    def on_candidate(c) -> None:
        counts["ok" if c.accepted else "no"] += 1
        mark = "+" if c.accepted else "-"
        detail = f"{c.robot_type}, {c.n_episodes} eps, {c.data_bytes / 2**20:.1f} MB" if c.accepted else c.reason
        print(f"  {mark} {c.repo_id}: {detail}", flush=True)

    print(f"discovering up to {args.limit} SO-100/101 datasets on the Hub", flush=True)
    candidates = fleet.discover(limit=args.limit, max_inspect=args.max_inspect, on_progress=on_candidate)
    print(f"{counts['ok']} accepted, {counts['no']} rejected", flush=True)

    def on_progress(ev: dict) -> None:
        t = ev["type"]
        if t == "start":
            print(f"[{ev['index'] + 1}/{ev['total']}] {ev['dataset']}", flush=True)
        elif t == "fit":
            print(f"    eval {ev['evaluation']:4d}  best {_f(ev['best_score'])} deg", flush=True)
        elif t == "robot":
            e = ev["entry"]
            print(
                f"    {_f(e['baseline_eval'])} -> {_f(e['calibrated_eval'])} deg "
                f"({_f(e['improvement'], '{:.0%}')} closed), latency {e['params']['latency_frames']} frames, "
                f"kp {e['params']['kp']:.0f}  [{e['elapsed_s']:.0f}s]",
                flush=True,
            )
        elif t == "failed":
            print(f"    FAILED {ev['dataset']}: {ev['error']}", flush=True)
        elif t == "skipped":
            print(f"    skipped {ev['dataset']}: {ev['reason']}", flush=True)
        elif t == "done":
            print(f"saved {ev['path']}", flush=True)

    report = fleet.scan(
        candidates,
        n_train=args.n_train,
        n_eval=args.n_eval,
        max_evaluations=args.max_evaluations,
        time_budget_s=args.budget_s,
        on_progress=on_progress,
        lock_dir=args.lock_dir,
        seed=args.seed,
        cross_n=args.cross_n,
    )
    print()
    print(_fleet_tables(report.to_dict()))
    return 0


def cmd_fleet_list(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from antares import fleet

    path = Path(args.path) if args.path else fleet.FLEET_LOCK_DIR / fleet.FLEET_REPORT_NAME
    if not path.is_file():
        print(f"no fleet report at {path}; run `antares fleet scan`")
        return 1
    print(_fleet_tables(json.loads(path.read_text())))
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

    p = sub.add_parser("locks", help="list saved locks")
    p.set_defaults(func=cmd_locks)

    p = sub.add_parser("fleet", help="calibrate every public SO-100/101 arm on the Hub")
    fsub = p.add_subparsers(dest="fleet_cmd", required=True)
    f = fsub.add_parser("scan", help="discover datasets, calibrate each, write locks/fleet/fleet.json")
    f.add_argument("--limit", type=int, default=60, help="max datasets to accept from the Hub")
    f.add_argument("--budget-s", type=float, default=2400.0, help="stop starting new robots after this many seconds")
    f.add_argument("--max-inspect", type=int, default=None, help="max Hub hits to inspect (default 5x limit)")
    f.add_argument("--n-train", type=int, default=20)
    f.add_argument("--n-eval", type=int, default=10)
    f.add_argument("--max-evaluations", type=int, default=150)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--cross-n", type=int, default=6, help="robots in the partial cross matrix (0 = skip)")
    f.add_argument("--lock-dir", default="locks/fleet")
    f.set_defaults(func=cmd_fleet_scan)
    f = fsub.add_parser("list", help="print the fleet table from fleet.json")
    f.add_argument("--path", default=None, help="fleet report (default locks/fleet/fleet.json)")
    f.set_defaults(func=cmd_fleet_list)

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
