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
    return "-" if v is None else f"{100 * v:.0f}%"


def _event_rows(events: list[dict], fps: float) -> list[list[str]]:
    return [
        [
            f"{ev['start']}-{ev['end']}",
            f"{ev['start'] / fps:.2f}-{ev['end'] / fps:.2f}",
            ev["kind"],
            ",".join(ev["joints"]),
            f"{ev['peak_z']:.1f}",
            f"{ev['peak_deg']:+.2f}",
        ]
        for ev in events
    ]


def _print_summary(title: str, s: dict) -> None:
    print(title)
    print(
        f"  {s['n_episodes']} episodes, {s['n_grasps']} grasps, {s['n_events']} events; "
        f"recall {_pct(s['recall'])} ({s['n_grasps_hit']}/{s['n_grasps']} grasps hit), "
        f"precision {_pct(s['precision'])} ({s['n_events_in_grasp']}/{s['n_events']} events in a grasp window), "
        f"near any transition {_pct(s['near_transition'])}, far {_pct(s['far_fraction'])} ({s['n_events_far']})"
    )
    lift = "-" if s["lift"] is None else f"{s['lift']:.2f}x"
    print(
        f"  residual energy inside grasp windows: {_pct(s['energy_in_grasp_fraction'])} "
        f"(windows cover {_pct(s['chance'])} of frames, lift {lift})"
    )


def cmd_events(args: argparse.Namespace) -> int:
    from antares import server

    ds = server.get_dataset(args.repo_id)
    label = f"lock {args.lock}" if args.lock else "default params"
    if args.episode is None and not args.all:
        print("error: give --episode N or --all", file=sys.stderr)
        return 2

    if args.all:
        out = server.events_summary(ds, args.lock, z=args.z, min_frames=args.min_frames)
        bl = out["baseline"]
        print(
            f"{ds.repo_id} ({label}): z >= {args.z:g} for >= {args.min_frames} frames; "
            f"baseline from {bl['n_frames']} free-motion frames of episodes {bl['episodes']}"
        )
        rows = [
            [
                str(r["episode"]) + ("*" if r["held_out"] else ""),
                str(r["n_frames"]),
                str(r["n_grasps"]),
                str(r["n_grasps_hit"]),
                str(r["n_events"]),
                str(r["n_events_in_grasp"]),
                str(r["n_events_far"]),
                _pct(r["energy_in_grasp"] / r["energy_total"] if r["energy_total"] else None),
            ]
            for r in out["episodes"]
        ]
        print(_table(["episode", "frames", "grasps", "hit", "events", "in grasp", "far", "energy in grasp"], rows))
        print("(* = held out of the fit)")
        _print_summary("all episodes:", out["summary"])
        _print_summary("held-out episodes:", out["summary_held_out"])
        return 0

    out = server.residual_episode(ds, args.episode, args.lock, z=args.z, min_frames=args.min_frames)
    fps = out["fps"]
    thr = out["gripper_thresholds"]
    print(
        f"{ds.repo_id} episode {out['episode']} ({len(out['residual'])} frames, {label}); "
        f"gripper closed < {thr['closed_below']:.1f}, open > {thr['open_above']:.1f}"
    )
    print(f"grasps at frames {out['grasps']}, releases at {out['releases']}")
    bl = out["baseline"]
    print(
        "baseline scale (deg): "
        + ", ".join(f"{j} {s:.2f}" for j, s in zip(bl["joints"], bl["scale_deg"]))
    )
    if not out["events"]:
        print("no events")
    else:
        print(_table(["frames", "seconds", "kind", "joints", "peak z", "peak deg"], _event_rows(out["events"], fps)))
    v = out["validation"]
    print(
        f"{v['n_grasps_hit']}/{v['n_grasps']} grasps hit, {v['n_events_in_grasp']}/{v['n_events']} events in a "
        f"grasp window, {v['n_events_far']} far from any gripper transition; "
        f"{_pct(v['energy_in_grasp'] / v['energy_total'] if v['energy_total'] else None)} of residual energy "
        f"in grasp windows ({_pct(v['n_frames_in_grasp'] / v['n_frames'])} of frames)"
    )
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

    p = sub.add_parser("events", help="detect contact events from the real-sim residual")
    p.add_argument("repo_id")
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--lock", default=None, help="use a saved lock's parameters")
    p.add_argument("--all", action="store_true", help="summarize across every episode")
    p.add_argument("--z", type=float, default=4.0, help="robust z-score to enter an event (exit at z/2)")
    p.add_argument("--min-frames", type=int, default=5, help="shortest event kept")
    p.set_defaults(func=cmd_events)

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
