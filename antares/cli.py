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


EXIT_PASS = 0
EXIT_DRIFT = 1
EXIT_USAGE = 2  # bad arguments, missing lock/dataset, or too few episodes


def _fmt(v, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def cmd_check(args: argparse.Namespace) -> int:
    import json
    import os

    from antares import server

    lock: str | dict = args.lock
    if os.path.isfile(args.lock):
        with open(args.lock) as f:
            lock = json.load(f)
    try:
        ds = server.get_dataset(args.dataset)
        res = server.run_check(
            ds,
            lock,
            threshold_deg=args.threshold,
            ratio=args.ratio,
            last=args.last,
            min_episodes=args.min_episodes,
        )
    except Exception as e:  # noqa: BLE001 - HTTPException or loader error
        detail = getattr(e, "detail", None) or f"{type(e).__name__}: {e}"
        print(f"error: {detail}", file=sys.stderr)
        return EXIT_USAGE

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"{res.dataset} vs lock {res.lock}: {res.n_episodes} episodes, {res.n_frames} frames")
        if res.per_joint:
            rows = []
            for j, jc in res.per_joint.items():
                scored = j in res.scored_joints
                status = ("PASS" if jc.passed else "FAIL") if jc.limit is not None else "n/a"
                if not scored:
                    status += " (not scored)"
                if j == res.worst_joint:
                    status += "  <- worst"
                rows.append([j, _fmt(jc.score), _fmt(jc.reference), _fmt(jc.limit), status])
            rows.append(
                ["overall", _fmt(res.score), _fmt(res.reference), _fmt(res.limit),
                 "PASS" if res.passed else "FAIL"]
            )
            print(_table(["joint", "rmse (deg)", "reference", "limit", "status"], rows))
        print(res.summary())

    if res.verdict == "ok":
        return EXIT_PASS
    if res.verdict == "drifted":
        return EXIT_DRIFT
    return EXIT_USAGE


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
        "check",
        help="pre-deploy gate: has this robot drifted from its lock?",
        description=(
            "Replay fresh episodes through a lock's calibrated sim and fail if the "
            "tracking error exceeds max(--threshold, --ratio * the lock's held-out score). "
            "Exit 0 on pass, 1 on drift, 2 on usage errors or too few episodes."
        ),
    )
    p.add_argument("dataset", help="Hub dataset id or a LeRobot dataset directory on disk")
    p.add_argument("--lock", required=True, help="lock name in the lock dir, or a .lock.json path")
    p.add_argument("--threshold", type=float, default=None, metavar="DEG",
                   help="absolute error allowed in degrees (default: none)")
    p.add_argument("--ratio", type=float, default=1.5,
                   help="allowed multiple of the lock's held-out score (default 1.5)")
    p.add_argument("--last", type=int, default=None, metavar="N",
                   help="only check the newest N episodes (default: all)")
    p.add_argument("--min-episodes", type=int, default=5,
                   help="fewer episodes than this is insufficient data, exit 2 (default 5)")
    p.add_argument("--json", action="store_true", help="print the CheckResult as JSON")
    p.set_defaults(func=cmd_check)

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
