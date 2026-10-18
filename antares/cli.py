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


def _fmt(v) -> str:
    return str(v) if isinstance(v, int) else f"{v:.4g}"


def _params_table(params: dict) -> str:
    """Scalar params on one line; per-joint params as a joints x fields table."""
    if not any(isinstance(v, list) for v in params.values()):
        return "params: " + ", ".join(f"{k}={_fmt(v)}" for k, v in params.items())
    from antares.sim import JOINTS

    names = list(params)
    rows = []
    for j, joint in enumerate(JOINTS):
        row = [joint]
        for k in names:
            v = params[k]
            row.append(_fmt(v[j] if isinstance(v, list) else v))
        rows.append(row)
    return "params (per joint):\n" + _table(["joint"] + names, rows)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "antares.server:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


def _detail(e: Exception) -> str:
    """Human text for an HTTPException whose detail may be a dict with a message."""
    d = getattr(e, "detail", None)
    if isinstance(d, dict):
        return str(d.get("message") or d)
    return str(d) if d else f"{type(e).__name__}: {e}"


def _load_dataset(repo_id: str):
    """``server.get_dataset`` (Hub id, ``upload/<name>`` or a local LeRobot
    directory) with a readable error instead of a traceback. None on failure."""
    from fastapi import HTTPException

    from antares import server

    try:
        return server.get_dataset(repo_id)
    except HTTPException as e:
        print(f"error: {_detail(e)}", file=sys.stderr)
        return None


def _print_verdict(v) -> None:
    print(f"verdict: {'ok' if v.ok else 'REJECTED'}" + (f" (family {v.robot_family})" if v.robot_family else ""))
    for r in v.reasons:
        print(f"  reason:  {r}")
    for w in v.warnings:
        print(f"  warning: {w}")


def cmd_inspect(args: argparse.Namespace) -> int:
    """Validate a dataset (Hub id or local directory) and summarize it.

    The info.json verdict comes first and costs one small download; a
    rejected dataset is not loaded. Exit 0 when it can be replayed, 1 if not.
    """
    import os

    from fastapi import HTTPException

    from antares import server
    from antares.robots import model_for_dataset
    from antares.validate import merge_verdicts, validate_dataset

    try:
        pre = server.validate_id(args.repo_id)
    except HTTPException as e:
        print(f"error: {_detail(e)}", file=sys.stderr)
        return 2
    if not pre.ok:
        print(f"{args.repo_id}: {pre.robot_type or 'unknown robot'}, {pre.n_episodes} episodes @ {pre.fps:g} fps")
        _print_verdict(pre)
        return 1

    from antares.data import load_dataset, read_local

    ds = read_local(args.repo_id, repo_id=args.repo_id) if os.path.isdir(args.repo_id) else load_dataset(args.repo_id)
    v = merge_verdicts(pre, validate_dataset(ds, min_episodes=server.min_episodes_for(args.repo_id)))
    frames = sum(len(e) for e in ds.episodes)
    print(
        f"{ds.repo_id}: {ds.robot_type}, {len(ds.episodes)} episodes, "
        f"{frames} frames @ {ds.fps:g} fps"
    )
    print("joints:", ", ".join(ds.joint_names))
    print(f"sim model: {model_for_dataset(ds.robot_type)}")
    _print_verdict(v)
    if getattr(args, "calibration", None) or getattr(args, "robot_id", None):
        if _print_calibration_check(ds, args) is None:
            return 2
    return 0 if v.ok else 1


def _print_calibration_check(ds, args: argparse.Namespace):
    """Model-range unit map vs the arm's LeRobot calibration, per joint.

    Prints the table and a warning per joint over 2 deg. Returns the check
    dict, or None (after printing the error) when the calibration cannot be
    read.
    """
    from fastapi import HTTPException

    from antares import server
    from antares.lerobot_calib import FLAG_MESSAGE, calibration_check

    try:
        calib, path = server.resolve_calibration(args.calibration, args.robot_id, args.robot_type)
        robot = server.check_robot(server.dataset_robot(ds))
        chk = calibration_check(calib, server.get_arm(robot).model)
    except HTTPException as e:
        print(f"error: {_detail(e)}", file=sys.stderr)
        return None
    print()
    print(f"LeRobot calibration {path} vs the {robot} model's joint ranges")
    print("(calibration map - model map, degrees, at -100/0/+100 units; gripper at 0/50/100):")
    rows = []
    for r in chk["joints"]:
        at = list(r["at"].values())
        rows.append([
            r["joint"], str(r["range_min"]), str(r["range_max"]), str(r["homing_offset"]),
            f"{r['calibration_span_deg']:.1f}", f"{r['model_span_deg']:.1f}",
            f"{at[0]:+.2f}", f"{at[1]:+.2f}", f"{at[2]:+.2f}", f"{r['max_deg']:.2f}",
            "FLAG" if r["flagged"] else "",
        ])
    print(_table(["joint", "range_min", "range_max", "homing", "arm span", "model span",
                  "lo", "mid", "hi", "max", ""], rows))
    for r in chk["joints"]:
        if r["flagged"]:
            print(f"  warning: {r['joint']} is off by up to {r['max_deg']:.1f} deg: {FLAG_MESSAGE}")
    if not chk["flagged"]:
        print(f"  all joints within {chk['threshold_deg']:g} deg of the model's ranges")
    return chk


def cmd_replay(args: argparse.Namespace) -> int:
    from antares import server

    ds = _load_dataset(args.repo_id)
    if ds is None:
        return 2
    params = cmap = unit_map = None
    if args.lock:
        lock = server.load_lock_dict(args.lock)
        params = server.params_from_dict(lock["params"])
        cmap = server.command_map_from_lock(lock)
        unit_map = server.unit_map_from_lock(lock)
    out = server.replay_episode(ds, args.episode, params, cmap, robot=args.robot,
                                **({"unit_map": unit_map} if unit_map is not None else {}))
    gap = out["gap"]
    label = f"lock {args.lock}" if args.lock else "default params"
    print(f"{ds.repo_id} episode {out['episode']} ({len(out['real_q'])} frames, {label}, model {out.get('robot', 'so101')})")
    print(_joint_table(gap["per_joint_deg"], gap["overall_deg"]))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from antares import server

    ds = _load_dataset(args.repo_id)
    if ds is None:
        return 2
    stages = f"up to {args.max_evaluations} shared evaluations"
    if args.seeds > 1:
        stages = f"best of {args.seeds} seeds x {stages}"
    if args.per_joint:
        stages += f" + {args.per_joint_evaluations} per-joint evaluations"
    if args.fit_sensor:
        stages += " per round, fitting the sensor calibration"
    if args.calibration:
        stages += ", unit map from the arm's LeRobot calibration"
    print(f"calibrating {ds.repo_id}: {args.n_train} train / {args.n_eval} eval episodes, {stages}")
    last = {"n": 0, "stage": None, "seed": None}

    def on_progress(ev) -> None:
        n = ev.evaluation
        stage = getattr(ev, "stage", "shared")
        seed = getattr(ev, "seed", 0)
        if stage != last["stage"] or seed != last["seed"]:
            last["stage"], last["seed"] = stage, seed
            run = f", seed {seed + 1}/{args.seeds}" if args.seeds > 1 and stage == "shared" else ""
            print(f"  -- {stage} stage{run} --", flush=True)
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
        per_joint=args.per_joint,
        per_joint_max_evaluations=args.per_joint_evaluations,
        per_joint_sigma0=args.per_joint_sigma0,
        robot=args.robot,
        on_progress=on_progress,
        fit_sensor=args.fit_sensor,
        sensor_mode=args.sensor_mode,
        n_seeds=args.seeds,
        calibration_path=args.calibration,
        robot_id=args.robot_id,
        robot_type=args.robot_type,
    )
    r = result.to_dict()
    only = lock.get("dynamics_only")
    before = r.get("per_joint_before") or {}
    after = r.get("per_joint_after") or {}
    mid = (only or {}).get("per_joint_after") or {}
    nan = float("nan")
    rows = [
        [j, f"{before.get(j, nan):.3f}"]
        + ([f"{mid.get(j, nan):.3f}"] if only else [])
        + [f"{after.get(j, nan):.3f}"]
        for j in sorted(set(before) | set(after), key=list(before).index)
    ]
    rows.append(
        ["eval overall", f"{r['baseline_eval_score']:.3f}"]
        + ([f"{only['eval_score']:.3f}"] if only else [])
        + [f"{r['eval_score']:.3f}"]
    )
    headers = ["joint", "before (deg)"] + (["dynamics only"] if only else []) + ["after (deg)"]
    print()
    print(_table(headers, rows))
    print()
    print(_params_table(r["params"]))
    for s in r.get("stages", []):
        run = f" (seed {s['seed']})" if "seed" in s and r.get("seeds") else ""
        print(f"  {s['name']}{run}: {s['n_evaluations']} evaluations, train {s['train_score']:.3f} deg")
    if r.get("seeds"):
        _print_seed_spread(r["seeds"])
    if lock.get("command_map"):
        cm = lock["command_map"]
        joints = list(before) or [str(i) for i in range(len(cm["gain"]))]
        print()
        print("sensor calibration (command = gain * logged + bias, dataset units):")
        print(_table(["joint", "gain", "bias"],
                     [[j, f"{g:.4f}", f"{b:+.3f}"] for j, g, b in zip(joints, cm["gain"], cm["bias"])]))
    if lock.get("calibration"):
        cal = lock["calibration"]
        dis = cal.get("unit_map_disagreement_deg") or {}
        print()
        print(f"unit map from LeRobot calibration {cal.get('source')} (robot id {cal.get('robot_id')})")
        print("  disagreement with the model's ranges (deg): "
              + ", ".join(f"{j}={v:.2f}" for j, v in dis.items()))
    print(f"{r['n_evaluations']} evaluations in {r['elapsed_s']:.1f}s")
    print(f"saved lock {lock['name']} -> {path}")
    if getattr(args, "robot_id", None):
        # One identity per physical robot: the new lock joins the robot's
        # lineage and the registry's rule decides whether it becomes active.
        try:
            reg = server.register_lock(args.robot_id, lock, activate=True if args.activate else None)
        except Exception as e:  # noqa: BLE001 - the lock is saved; say why it was not registered
            print(f"warning: lock saved but not registered: {_detail(e)}", file=sys.stderr)
        else:
            print(reg["summary"])
    return 0


def _print_seed_spread(sd: dict) -> None:
    """The best-of-N summary: which seed won, and which parameters it pins."""
    scores = " / ".join("-" if v is None else f"{v:.3f}" for v in sd["eval_scores"])
    print(f"seeds: kept seed {sd['seeds'][sd['chosen']]} ({sd['chosen'] + 1} of {sd['n']}); held-out {scores} deg")
    spread = []
    for name, sp in sd.get("spread", {}).items():
        if "ratio" in sp:
            spread.append(f"{name} x{_num(sp['ratio'], '{:.3g}')}")
        else:
            spread.append(f"{name} +-{_num(sp.get('range'), '{:.0f}')} frames")
    pinned = ", ".join(sd.get("pinned") or []) or "none"
    print(f"  pinned (agree across seeds): {pinned}")
    print(f"  spread across seeds: {', '.join(spread)}")
    if set(sd.get("pinned") or []) < set(sd.get("spread", {})):
        print("  unpinned parameters move across seeds at equal error; read them as a fit, not a measurement")


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
    lock_datasets = {n: server.load_lock_dict(n).get("dataset", "") for n in locks}
    summary = _print_specificity(m, lock_datasets)
    if args.json:
        _write_json(args.json, {"matrix": m, "specificity": summary})
    return 0


def _pct(v) -> str:
    return "-" if v is None else f"{v:+.0%}"


def _num(v, fmt: str = "{:.3f}") -> str:
    return "-" if v is None else fmt.format(v)


def cmd_drift(args: argparse.Namespace) -> int:
    from antares import server

    ds = _load_dataset(args.repo_id)
    if ds is None:
        return 2
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


def _ranges_table(ranges: dict) -> str:
    """One row per parameter: nominal, lo, hi and the spread relative to nominal."""
    rows = []
    for name, r in ranges["ranges"].items():
        nom, lo, hi = r["nominal"], r["lo"], r["hi"]
        if r["scale"] == "int":
            rows.append([name, str(nom), str(lo), str(hi), f"{lo - nom:+d} .. {hi - nom:+d} frames"])
        else:
            rel = f"{(lo / nom - 1) * 100:+.0f}% .. {(hi / nom - 1) * 100:+.0f}%"
            rows.append([name, f"{nom:.4g}", f"{lo:.4g}", f"{hi:.4g}", rel])
    return _table(["param", "nominal", "lo", "hi", "spread"], rows)


def cmd_dr(args: argparse.Namespace) -> int:
    from antares import server

    ds = _load_dataset(args.repo_id)
    if ds is None:
        return 2
    lock = args.lock or server._slug(ds.repo_id)
    print(
        f"deriving DR ranges for lock {lock} from {ds.repo_id}: {args.n_fits} fits x "
        f"{args.n_train} episodes, up to {args.max_evaluations} evaluations each"
    )

    def on_fit(i, result) -> None:
        p = result.params.to_dict()
        print(
            f"  fit {i + 1}/{args.n_fits}: train {result.train_score:6.3f} deg  "
            f"kp {p['kp']:.1f}  kv {p['kv']:.2f}  latency {p['latency_frames']}",
            flush=True,
        )

    _, ranges, path = server.run_dr(
        ds,
        lock,
        n_fits=args.n_fits,
        n_train=args.n_train,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        name=args.name,
        on_fit=on_fit,
    )
    pct = ranges["percentiles"]
    print()
    print(f"ranges (p{pct[0]:g}..p{pct[1]:g} of the fitted values, widened to the nominal):")
    print(_ranges_table(ranges))
    print(f"saved ranges {ranges['name']} -> {path}")
    return 0


EXIT_PASS = 0
EXIT_DRIFT = 1
EXIT_USAGE = 2  # bad arguments, missing lock/dataset, or too few episodes
MIN_EXPLAIN_EVALUATIONS = 10  # as POST /api/check validates explain_evaluations


def _fmt(v, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _diagnosis_table(d) -> str:
    """What each change found by the refit recovers when applied alone."""
    causes = {c.name for c in d.causes}
    rows = [
        [c.name, c.change, _fmt(c.alone_deg), f"{c.recovers_deg:+.3f}", "<- cause" if c.name in causes else ""]
        for c in d.changes
    ]
    rows.append(["all together", "", _fmt(d.refit_deg), f"{d.gap_deg:+.3f}", ""])
    head = (
        f"what changed (refit on episodes {_episode_list(d.refit_episodes)}, "
        f"scored on {_episode_list(d.held_out_episodes)}; lock alone {d.stale_deg:.3f} deg):"
    )
    return head + "\n" + _table(["param", "change", "alone (deg)", "recovers", ""], rows)


def _episode_list(indices: list[int]) -> str:
    if len(indices) <= 6:
        return ",".join(str(i) for i in indices)
    return f"{indices[0]}..{indices[-1]} ({len(indices)})"


def _notify_check(args: argparse.Namespace, webhook: str, res=None, error: str | None = None) -> None:
    """Post the check's alert to ``webhook``. A dead webhook only warns on
    stderr; it never changes the exit code."""
    from antares import notify

    out = sys.stderr if args.json else None  # keep --json stdout pure under --notify-dry-run
    kw = dict(when=args.notify_when, fmt=args.notify_format, dry_run=args.notify_dry_run, out=out)
    try:
        if res is not None:
            sent = notify.notify(res, webhook, **kw)
        else:
            sent = notify.notify_error(error or "check could not run", webhook,
                                       lock=str(args.lock), dataset=args.dataset, **kw)
    except Exception as e:  # noqa: BLE001 - NotifyError, a bad URL, anything: the verdict stands
        print(f"warning: alert not sent: {e}", file=sys.stderr)
        return
    if sent:
        verb = "rendered" if args.notify_dry_run else "sent"
        print(f"alert {verb} ({notify.resolve_format(webhook, args.notify_format)} webhook)", file=sys.stderr)


def cmd_check(args: argparse.Namespace) -> int:
    import json
    import os

    from antares import server

    webhook = args.notify or os.environ.get("ANTARES_WEBHOOK_URL") or None
    if args.explain and args.explain_evaluations < MIN_EXPLAIN_EVALUATIONS:
        print(f"error: --explain-evaluations must be >= {MIN_EXPLAIN_EVALUATIONS}", file=sys.stderr)
        return EXIT_USAGE
    try:
        args.lock = _resolve_lock(args.lock)  # @<robot> -> its active lock; names and paths pass through
        lock: str | dict = args.lock
        if os.path.isfile(args.lock):
            with open(args.lock) as f:
                lock = json.load(f)
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
        print(f"error: {_detail(e)}", file=sys.stderr)
        if webhook:
            _notify_check(args, webhook, error=_detail(e))
        return EXIT_USAGE

    diagnosis = None
    if args.explain and res.verdict == "drifted":
        if not args.json:
            print("drifted; refitting to find what changed...", file=sys.stderr, flush=True)
        try:
            diagnosis = server.run_diagnose(
                ds, lock, last=args.last, max_evaluations=args.explain_evaluations
            )
        except Exception as e:  # noqa: BLE001 - the verdict still stands
            detail = getattr(e, "detail", None) or f"{type(e).__name__}: {e}"
            print(f"could not explain the drift: {detail}", file=sys.stderr)

    if args.json:
        out = res.to_dict()
        if diagnosis is not None:
            out["diagnosis"] = diagnosis.to_dict()
        print(json.dumps(out, indent=2))
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
        if diagnosis is not None:
            print()
            print(_diagnosis_table(diagnosis))
            print(diagnosis.summary())

    if not args.no_record:
        _record_check(res, args.lock, args.history_dir, server.LOCK_DIR, to_stderr=args.json)
    if webhook:
        _notify_check(args, webhook, res=res)

    if res.verdict == "ok":
        return EXIT_PASS
    if res.verdict == "drifted":
        return EXIT_DRIFT
    return EXIT_USAGE


EXIT_RECORD = 3  # record-check: lerobot-record itself failed; nothing was checked


def _add_check_options(p: argparse.ArgumentParser) -> None:
    """The ``check`` options ``record-check`` forwards unchanged. Keep in step
    with the ``check`` parser; ``_check_kwargs`` reads the same names."""
    p.add_argument("--threshold", type=float, default=None, metavar="DEG",
                   help="absolute error allowed in degrees (default: none)")
    p.add_argument("--ratio", type=float, default=1.5,
                   help="allowed multiple of the lock's held-out score (default 1.5)")
    p.add_argument("--min-episodes", type=int, default=5,
                   help="fewer episodes than this is insufficient data, exit 2 (default 5)")
    p.add_argument("--json", action="store_true", help="print the CheckResult as JSON")


def _check_kwargs(args: argparse.Namespace) -> dict:
    """``server.run_check`` keywords from the options ``_add_check_options`` adds."""
    return {"threshold_deg": args.threshold, "ratio": args.ratio, "min_episodes": args.min_episodes}


def cmd_record_check(args: argparse.Namespace) -> int:
    """Record fresh episodes with ``lerobot-record``, then run ``check`` on them."""
    import json
    import os
    import shlex
    from pathlib import Path

    from antares import recordcheck as rc

    episodes = args.episodes
    if episodes < args.min_episodes:
        print(f"warning: --episodes {episodes} is below --min-episodes {args.min_episodes}; "
              f"recording {args.min_episodes}", file=sys.stderr)
        episodes = args.min_episodes
    # --lock defaults to the robot's active lock in the registry (@<robot-id>).
    try:
        args.lock = _resolve_lock(args.lock if args.lock else f"@{args.robot_id}")
    except Exception as e:  # noqa: BLE001 - HTTPException from the resolver
        hint = "" if args.lock else " (pass --lock, or register the robot with `antares robots register`)"
        print(f"error: {_detail(e)}{hint}", file=sys.stderr)
        return EXIT_USAGE
    opts = rc.RecordOptions(
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        teleop_type=args.teleop_type,
        teleop_port=args.teleop_port,
        teleop_id=args.teleop_id,
        num_episodes=episodes,
        task=args.task,
        root=Path(args.root) if args.root else None,
        episode_time_s=args.episode_time,
        reset_time_s=args.reset_time,
        extra=list(args.extra or []),
        recorder=args.recorder,
    )
    lock: str | dict = args.lock
    if os.path.isfile(args.lock):
        with open(args.lock) as f:
            lock = json.load(f)
    check_kwargs = _check_kwargs(args)

    root = rc.resolve_root(opts)
    if args.dry_run:
        try:
            cmd = rc.build_record_command(opts, root, recorder=rc.find_recorder(args.recorder) or args.recorder or rc.RECORDER)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return EXIT_USAGE
        check_cmd = ["antares", "check", str(root), "--lock", args.lock]
        if args.threshold is not None:
            check_cmd += ["--threshold", f"{args.threshold:g}"]
        check_cmd += ["--ratio", f"{args.ratio:g}", "--min-episodes", str(args.min_episodes)]
        if args.json:
            check_cmd.append("--json")
        if args.skip_record:
            print(f"would skip recording and check the dataset already under {root}")
        else:
            print("would run:")
            print("  " + shlex.join(cmd))
        print("then:")
        print("  " + shlex.join(check_cmd))
        if not args.skip_record and rc.find_recorder(args.recorder) is None:
            print(f"note: {args.recorder or rc.RECORDER} not found; {rc.INSTALL_HINT}", file=sys.stderr)
        return EXIT_PASS

    try:
        res = rc.record_and_check(opts, lock, check_kwargs=check_kwargs, skip_record=args.skip_record)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except rc.RecordError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_RECORD
    except Exception as e:  # noqa: BLE001 - HTTPException or loader error, as in cmd_check
        print(f"error: {_detail(e)}", file=sys.stderr)
        return EXIT_USAGE

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(res.summary())
        print(rc.banner(res, args.robot_id))
    if res.verdict == "ok":
        return EXIT_PASS
    if res.verdict == "drifted":
        return EXIT_DRIFT
    return EXIT_USAGE


def cmd_notify_test(args: argparse.Namespace) -> int:
    import os
    from urllib.parse import urlsplit

    from antares import notify

    url = args.url or os.environ.get("ANTARES_WEBHOOK_URL")
    if not url:
        print("error: pass --url or set ANTARES_WEBHOOK_URL", file=sys.stderr)
        return EXIT_USAGE
    res = notify.example_result(name=args.lock, dataset=args.dataset)
    try:
        notify.notify(res, url, when="always", fmt=args.format, title=args.title, dry_run=args.dry_run)
    except Exception as e:  # noqa: BLE001 - NotifyError or a bad URL/format
        print(f"error: alert not sent: {e}", file=sys.stderr)
        return EXIT_DRIFT
    fmt = notify.resolve_format(url, args.format)
    verb = "rendered" if args.dry_run else "sent"
    # The URL is a secret; name only the host.
    print(f"{verb} a synthetic OK alert as {fmt} to {urlsplit(url).hostname}")
    return EXIT_PASS


def _history_target(lock_arg: str, history_dir: str | None, lock_dir) -> tuple[str, Path]:
    """``(lock name, directory)`` of the history file for a ``--lock`` argument.

    A ``.lock.json`` path keeps its history next to it under the file's name;
    a bare name lives in ``lock_dir``. ``history_dir`` overrides the directory.
    """
    import os
    from pathlib import Path

    from antares.history import LOCK_SUFFIX

    if os.path.isfile(lock_arg):
        base = os.path.basename(lock_arg)
        for suffix in (LOCK_SUFFIX, ".json"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        name, d = base, Path(lock_arg).resolve().parent
    else:
        name, d = lock_arg, Path(lock_dir)
    return name, Path(history_dir) if history_dir else d


def _record_check(res, lock_arg: str, history_dir: str | None, lock_dir, *, to_stderr: bool) -> None:
    """Append ``res`` to the lock's history and print the one-line trend.

    Insufficient-data checks are recorded too: a check that could not run is
    itself a signal. A failure to write never changes the verdict.
    """
    from antares import history

    name, d = _history_target(lock_arg, history_dir, lock_dir)
    try:
        history.append(res, d, name=name)
        line = history.summary_line(history.load(name, d))
    except (OSError, ValueError) as e:
        print(f"warning: could not record the check: {_detail(e)}", file=sys.stderr)
        return
    print(line, file=sys.stderr if to_stderr else sys.stdout)


def cmd_history(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from antares import history, server

    if args.last is not None and args.last < 1:
        print("error: --last must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    if args.robot:
        return _history_for_robot(args)
    if not args.lock:
        print("error: give a lock (name, .lock.json path or @<robot>) or --robot <id>", file=sys.stderr)
        return EXIT_USAGE
    try:
        args.lock = _resolve_lock(args.lock)
    except Exception as e:  # noqa: BLE001 - HTTPException from the resolver
        print(f"error: {_detail(e)}", file=sys.stderr)
        return EXIT_USAGE
    name, d = _history_target(args.lock, args.history_dir, server.LOCK_DIR)
    try:
        records = history.load(name, d)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    lock_file = Path(server.LOCK_DIR) / f"{name}{server.LOCK_SUFFIX}"
    if not records and not lock_file.is_file() and not Path(args.lock).is_file():
        print(f"error: no lock named {name!r} in {server.LOCK_DIR}", file=sys.stderr)
        return EXIT_USAGE
    shown = records[-args.last:] if args.last else records
    if args.json:
        out = {"lock": name, "entries": [r.to_dict() for r in shown], "trend": history.trend(records)}
        print(json.dumps(out, indent=2))
        return EXIT_PASS
    if not records:
        print(f"no checks recorded for lock {name} ({history.history_path(name, d)})")
        return EXIT_PASS
    rows = [
        [r.at[:16].replace("T", " "), r.dataset, r.verdict, _fmt(r.score), _fmt(r.limit), r.worst_joint or "-"]
        for r in shown
    ]
    print(_table(["date (utc)", "dataset", "verdict", "score (deg)", "limit", "worst joint"], rows))
    print(history.summary_line(records))
    return EXIT_PASS


def _history_for_robot(args: argparse.Namespace) -> int:
    """``antares history --robot <id>``: every lock in the robot's lineage, one
    time-ordered history, each entry tagged with its lock."""
    import json
    from pathlib import Path

    from antares import history, server
    from antares.registry import UnknownRobot

    try:
        entry = server.load_registry().get(args.robot)
    except UnknownRobot as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as e:  # noqa: BLE001 - HTTPException on a corrupt registry
        print(f"error: {_detail(e)}", file=sys.stderr)
        return EXIT_USAGE
    d = Path(args.history_dir) if args.history_dir else Path(server.LOCK_DIR)
    # Each entry is tagged with the lineage lock whose history file holds it
    # (a lock checked through a file path records the name inside the file).
    tagged = [(name, r) for name in entry.names for r in history.load(name, d)]
    tagged.sort(key=lambda t: t[1].time)
    records = [r for _, r in tagged]
    shown = tagged[-args.last:] if args.last else tagged
    if args.json:
        out = {"robot": args.robot, "locks": entry.names, "active": entry.active,
               "entries": [{**r.to_dict(), "lock": name} for name, r in shown], "trend": history.trend(records)}
        print(json.dumps(out, indent=2))
        return EXIT_PASS
    if not records:
        print(f"no checks recorded for robot {args.robot} (locks: {', '.join(entry.names) or 'none'})")
        return EXIT_PASS
    rows = [
        [r.at[:16].replace("T", " "), name + (" *" if name == entry.active else ""), r.dataset, r.verdict,
         _fmt(r.score), _fmt(r.limit), r.worst_joint or "-"]
        for name, r in shown
    ]
    print(_table(["date (utc)", "lock", "dataset", "verdict", "score (deg)", "limit", "worst joint"], rows))
    n_locks = len({name for name, _ in tagged})
    print(f"{history.summary_line(records)}; across {n_locks} lock{'s' if n_locks != 1 else ''} of robot {args.robot}"
          f" (* = active)")
    return EXIT_PASS


# --------------------------------------------------------------------------
# robots: the registry (antares.registry)
# --------------------------------------------------------------------------


def _resolve_lock(ref: str) -> str:
    """The one place the CLI turns ``@<robot>`` into the robot's active lock
    name; lock names and ``.lock.json`` paths pass through unchanged."""
    from antares import server

    return server.resolve_lock_ref(ref)


def _pfmt(v) -> str:
    """A parameter value for a table: scalars to 4 significant digits, a
    per-joint list as its range."""
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        xs = [x for x in v if isinstance(x, (int, float))]
        if not xs:
            return "-"
        lo, hi = min(xs), max(xs)
        return _pfmt(lo) if lo == hi else f"{_pfmt(lo)}..{_pfmt(hi)}"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) < 1e6):
        return str(int(v))
    return f"{v:.4g}"


def _score_pair(before, after) -> str:
    b = "-" if before is None else f"{before:.3f}"
    a = "-" if after is None else f"{after:.3f}"
    return f"{b} -> {a}"


def _last_check(name: str | None, lock_dir) -> str:
    """``<verdict> <date>`` of the newest recorded check of ``name``, or ``never``."""
    if not name:
        return "-"
    from antares import history

    try:
        records = history.load(name, lock_dir)
    except ValueError:
        return "-"
    if not records:
        return "never"
    r = records[-1]
    return f"{r.verdict} {r.date}"


def cmd_robots(args: argparse.Namespace) -> int:
    import json

    from antares import server
    from antares.registry import RegistryError, UnknownRobot, param_drift

    sub = getattr(args, "robots_cmd", None) or "list"
    try:
        reg = server.load_registry()
    except Exception as e:  # noqa: BLE001 - HTTPException on a corrupt file
        print(f"error: {_detail(e)}", file=sys.stderr)
        return EXIT_USAGE

    try:
        if sub == "list":
            rows = [reg.summary(rid) for rid in reg.robots()]
            if args.json:
                print(json.dumps({"path": str(reg.path), "robots": rows}, indent=2))
                return EXIT_PASS
            if not rows:
                print(f"no robots registered in {reg.path} "
                      f"(antares robots register <id> --lock <name>, or calibrate ... --robot-id <id>)")
                return EXIT_PASS
            table = []
            for s in rows:
                ver = f"v{s['version']} of {s['n_versions']}" if s["version"] else f"- of {s['n_versions']}"
                table.append([
                    s["robot_id"],
                    s["active"] or "-",
                    ver,
                    "-" if s["eval_score"] is None else f"{s['eval_score']:.3f}",
                    str(s["n_versions"]),
                    str(s["last_calibrated"] or "")[:10] or "-",
                    _last_check(s["active"], server.LOCK_DIR),
                ])
            print(_table(["robot", "active lock", "version", "held-out (deg)", "locks", "last calibrated", "last check"], table))
            return EXIT_PASS

        if sub == "show":
            s = reg.summary(args.robot_id)
            lineage = reg.lineage(args.robot_id)
            drift = param_drift(lineage)
            if args.json:
                print(json.dumps({**s, "lineage": lineage, "param_drift": drift}, indent=2))
                return EXIT_PASS
            typ = f" ({s['robot_type']})" if s["robot_type"] else ""
            active = f"active {s['active']} (v{s['version']} of {s['n_versions']})" if s["active"] else "no active lock"
            print(f"robot {s['robot_id']}{typ}: {s['n_versions']} lock{'s' if s['n_versions'] != 1 else ''}, {active}")
            rows = []
            for row in lineage:
                mark = "*" if row["is_active"] else ""
                missing = "" if row["exists"] else " (file missing)"
                rows.append([
                    f"v{row['version']}{mark}",
                    f"{row['name']}{missing}",
                    str(row["created_at"] or "")[:10] or "-",
                    row["dataset"] or "-",
                    _score_pair(row["baseline_eval_score"], row["eval_score"]),
                    ", ".join(row["pinned"]) if row["pinned"] else ("-" if row["exists"] else "?"),
                    _last_check(row["name"], server.LOCK_DIR),
                    row["note"] or "",
                ])
            print(_table(["version", "lock", "created", "dataset", "held-out before -> after", "pinned", "last check", "note"], rows))
            print("* = active; a parameter is pinned when the fit can read it as a measurement (see `--seeds`)")
            if drift["versions"]:
                print()
                print("parameter drift across versions (pinned in every version only):")
                heads = ["param"] + [f"v{v}" for v in drift["versions"]] + ["change"]
                prow = [[k] + [_pfmt(v) for v in c["values"]] + [c["change"]] for k, c in drift["comparable"].items()]
                if prow:
                    print(_table(heads, prow))
                else:
                    print("  none: no parameter is pinned in every version")
                if drift["not_comparable"]:
                    print("not comparable across fits (unpinned, so each value is a fit, not a measurement): "
                          + ", ".join(f"{k} (v{', v'.join(str(v) for v in vs)})" for k, vs in drift["not_comparable"].items()))
            return EXIT_PASS

        if sub == "register":
            if args.max_regression is not None and args.max_regression < 0:
                print("error: --max-regression must be >= 0", file=sys.stderr)
                return EXIT_USAGE
            res = server.register_lock(
                args.robot_id, args.lock, activate=True if args.activate else None, note=args.note,
                max_regression=args.max_regression,
            )
            print(res["summary"])
            return EXIT_PASS

        if sub == "activate":
            previous = reg.activate(args.robot_id, args.lock)
            reg.save()
            e = reg.get(args.robot_id)
            was = f" (was {previous})" if previous and previous != args.lock else ""
            print(f"robot {args.robot_id}: active lock {args.lock} (v{e.version(args.lock)} of {len(e.locks)}){was}")
            return EXIT_PASS

        if sub == "rollback":
            previous, now = reg.rollback(args.robot_id)
            reg.save()
            e = reg.get(args.robot_id)
            print(f"robot {args.robot_id}: rolled back {previous} -> {now} (v{e.version(now)} of {len(e.locks)})")
            return EXIT_PASS

        if sub == "remove":
            entry = reg.remove(args.robot_id)
            reg.save()
            print(f"removed robot {args.robot_id} from {reg.path}; its {len(entry.locks)} lock file(s) stay: "
                  + (", ".join(entry.names) or "none"))
            return EXIT_PASS
    except UnknownRobot as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except RegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as e:  # noqa: BLE001 - HTTPException from register_lock (missing lock, bad name)
        print(f"error: {_detail(e)}", file=sys.stderr)
        return EXIT_USAGE
    print(f"error: unknown robots command {sub!r}", file=sys.stderr)
    return EXIT_USAGE


def _frac(v) -> str:
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
        f"recall {_frac(s['recall'])} ({s['n_grasps_hit']}/{s['n_grasps']} grasps hit), "
        f"precision {_frac(s['precision'])} ({s['n_events_in_grasp']}/{s['n_events']} events in a grasp window), "
        f"near any transition {_frac(s['near_transition'])}, far {_frac(s['far_fraction'])} ({s['n_events_far']})"
    )
    lift = "-" if s["lift"] is None else f"{s['lift']:.2f}x"
    print(
        f"  residual energy inside grasp windows: {_frac(s['energy_in_grasp_fraction'])} "
        f"(windows cover {_frac(s['chance'])} of frames, lift {lift})"
    )


def cmd_events(args: argparse.Namespace) -> int:
    from antares import server

    ds = _load_dataset(args.repo_id)
    if ds is None:
        return 2
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
                _frac(r["energy_in_grasp"] / r["energy_total"] if r["energy_total"] else None),
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
        f"{_frac(v['energy_in_grasp'] / v['energy_total'] if v['energy_total'] else None)} of residual energy "
        f"in grasp windows ({_frac(v['n_frames_in_grasp'] / v['n_frames'])} of frames)"
    )
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

    ds = _load_dataset(args.repo_id)
    if ds is None:
        return 2
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

        um = server.get_arm(out.get("robot")).unit_map
        cmds = um.to_units(np.radians(np.asarray(out["ctrl_comp"], dtype=np.float64)))
        cmap = server.command_map_from_lock(lock) if lock else None
        if cmap is not None:  # back to the raw units the arm is commanded in
            cmds = cmap.invert(cmds)
        np.save(args.out, cmds)
        print(f"saved {args.method} commands ({cmds.shape[0]}x{cmds.shape[1]}, dataset units) -> {args.out}")
    return 0


def _load_lock_arg(lock: str) -> dict:
    """A lock given on the command line: a file path or a saved lock's name."""
    import json
    import os

    from antares import server

    if os.path.isfile(lock):
        with open(lock) as f:
            return json.load(f)
    return server.load_lock_dict(lock)


def cmd_synth(args: argparse.Namespace) -> int:
    from antares import server, synth

    if args.lock:
        from antares.check import _lock_view

        try:
            params = _lock_view(_load_lock_arg(args.lock))[2]
        except Exception as e:  # noqa: BLE001 - HTTPException, bad JSON or a lock without params
            detail = getattr(e, "detail", None) or f"{type(e).__name__}: {e}"
            print(f"--lock: {detail}", file=sys.stderr)
            return EXIT_USAGE
    else:
        params = server.default_params()
    try:
        params = synth.worn(params, synth.parse_changes(args.wear))
    except ValueError as e:
        print(f"--wear: {e}", file=sys.stderr)
        return EXIT_USAGE
    ds = synth.synthesize(
        params,
        n_episodes=args.episodes,
        n_frames=args.frames,
        fps=args.fps,
        seed=args.seed,
        noise_units=args.noise,
        repo_id=args.out,
    )
    synth.write_local(ds, args.out)
    print(f"wrote {len(ds.episodes)} episodes x {args.frames} frames -> {args.out}")
    print("true params:", params.to_dict())
    return 0




def cmd_transfer_run(args: argparse.Namespace) -> int:
    from antares import transfer

    lock_paths = dict(transfer.DEFAULT_ARM_LOCKS)
    if args.arm:
        lock_paths = {}
        for item in args.arm:
            name, sep, path = item.partition("=")
            if not sep or not name or not path:
                print(f"--arm: expected NAME=LOCK_PATH, got {item!r}", file=sys.stderr)
                return EXIT_USAGE
            lock_paths[name] = path
    try:
        arms = transfer.arms_from_locks(lock_paths)
    except (OSError, KeyError, ValueError) as e:
        print(f"could not read a hidden arm's lock: {e}", file=sys.stderr)
        return EXIT_USAGE
    overrides = {}
    if args.seeds is not None:
        overrides["seeds"] = tuple(range(args.seeds))
    if args.generations is not None:
        overrides["generations"] = args.generations
    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.quick:
        cfg = transfer.TransferConfig.quick_config(arms, **overrides)
    else:
        cfg = transfer.TransferConfig(arms=arms, **overrides)
    report = transfer.run_experiment(cfg, log=lambda line: print(line, flush=True))
    out = args.out
    json_path = out if out.endswith(".json") else out + ".json"
    path = report.save(json_path)
    print(report.to_markdown())
    print(f"wrote {path} and {path.with_suffix('.md')} ({report.elapsed_s / 60:.1f} min)")
    return 0


def cmd_transfer_report(args: argparse.Namespace) -> int:
    from antares.transfer import TransferReport

    try:
        report = TransferReport.load(args.json)
    except (OSError, KeyError, ValueError) as e:
        print(f"could not read {args.json}: {e}", file=sys.stderr)
        return EXIT_USAGE
    md = report.to_markdown()
    if args.md:
        from pathlib import Path

        Path(args.md).write_text(md)
        print(f"wrote {args.md}")
    else:
        print(md)
    return 0


def _print_specificity(m: dict, lock_datasets: dict[str, str]) -> dict:
    """Print whether each dataset's own lock beats every other arm's lock."""
    from antares.specificity import summarize_cross

    s = summarize_cross(m, lock_datasets)
    if not any(r["specific"] is not None for r in s["rows"]):
        return s

    def f(v):
        return "-" if v is None else f"{v:.3f}"

    rows = [
        [
            r["dataset"],
            f"{r['own_lock'] or '-'} {f(r['own_score'])}",
            f"{r['best_other_lock'] or '-'} {f(r['best_other_score'])}",
            "-" if r["margin"] is None else f"{r['margin']:.2f}x",
            {True: "yes", False: "NO", None: "-"}[r["specific"]],
        ]
        for r in s["rows"]
    ]
    print()
    print("robot specificity (own lock vs best lock fitted on another dataset)")
    print(_table(["dataset", "own lock", "best other lock", "margin", "own wins"], rows))
    return s


def _json_safe(obj):
    """NaN/inf are not JSON; write them as null like the lock files do."""
    import math

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write_json(path: str, obj: dict) -> None:
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False) + "\n")
    print(f"wrote {path}")


def cmd_identify(args: argparse.Namespace) -> int:
    from antares import server
    from antares.calibrate import default_simulate, split_episodes
    from antares.specificity import identifiability, latency_profile

    ds = server.get_dataset(args.repo_id)
    params = server.params_from_dict(server.load_lock_dict(args.lock)["params"])
    train, eval_ = split_episodes(ds.episodes, args.n_train, args.n_eval, seed=args.seed)
    episodes = train if args.split == "train" else eval_
    if not episodes:
        raise SystemExit(
            f"{ds.repo_id} has {len(ds.episodes)} episodes; --n-train {args.n_train} --n-eval {args.n_eval} "
            f"leaves no {args.split} split"
        )
    simulate = default_simulate()
    rep = identifiability(params, episodes, simulate, delta_deg=args.delta)
    lat = latency_profile(params, episodes, simulate)
    print(f"lock {args.lock} on {ds.repo_id} ({args.split} split, {len(episodes)} episodes)")
    print(_identifiability_text(rep, lat))
    if args.json:
        _write_json(args.json, {"identifiability": rep.to_dict(), "latency_profile": lat})
    return 0


def _identifiability_text(rep, lat: dict[int, float]) -> str:
    """Tolerance table, least-constrained direction and latency profile."""
    import math

    def x(v: float) -> str:
        return "unbounded" if v == float("inf") or v > 1e6 else f"x{v:.3g}"

    rows = [
        [n, x(rep.factor_fixed[n]), x(rep.factor_free[n]),
         f"{rep.tradeoff[n]:.3g}" if math.isfinite(rep.tradeoff[n]) else "-"]
        for n in rep.names
    ]
    lines = [
        f"overall {rep.rms_deg:.3f} deg; change each parameter can take before it rises "
        f"by {rep.delta_deg:g} deg:",
        _table(["param", "alone", "others refit", "trade-off"], rows),
        "",
        f"condition number {rep.condition_number:.3g}; least constrained direction: "
        + ", ".join(f"{v:+.2f} {k}" for k, v in sorted(rep.sloppiest.items(), key=lambda kv: -abs(kv[1]))
                    if abs(v) >= 0.1),
        "",
        "latency (frames): " + "  ".join(f"{f}: {s:.3f}" for f, s in lat.items()),
    ]
    return "\n".join(lines)


def cmd_specificity(args: argparse.Namespace) -> int:
    from pathlib import Path

    from antares import server
    from antares.calibrate import default_simulate
    from antares.specificity import render_markdown, study

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    lock_names = [l.strip() for l in (args.locks or "").split(",") if l.strip()]
    if lock_names:
        lock_dicts = [server.load_lock_dict(n) for n in lock_names]
    else:
        lock_dicts = _default_locks(server, datasets)
    for d in lock_dicts:
        if _lock_space(d) != "shared":
            raise SystemExit(f"lock {d['name']} is a {_lock_space(d)} lock; the specificity checks need "
                             "a shared one (calibrate without --per-joint or pass --locks)")
    simulate = default_simulate()
    truth = None
    splits = None
    if args.synthetic:
        episodes, locks, truth, splits = _synthetic_arms(datasets, lock_dicts, args, simulate)
    else:
        episodes = {d: server.get_dataset(d).episodes for d in datasets}
        locks = {}
        splits = {}
        for d in lock_dicts:
            locks[d["name"]] = (d["dataset"], server.params_from_dict(d["params"]))
            r = d.get("result") or {}
            if d["dataset"] not in splits and r.get("train_episodes") and r.get("eval_episodes"):
                splits[d["dataset"]] = (list(r["train_episodes"]), list(r["eval_episodes"]))

    results = study(
        episodes,
        locks,
        simulate,
        n_train=args.n_train,
        n_eval=args.n_eval,
        split_seed=args.seed,
        restart_seeds=[int(s) for s in args.restart_seeds.split(",") if s.strip()],
        max_evaluations=args.max_evaluations,
        truth=truth,
        splits=splits,
        log=lambda msg: print(f"  {msg}", flush=True),
    )
    title = "Robot specificity (synthetic arms)" if args.synthetic else "Robot specificity"
    md = render_markdown(results, title=title)
    print()
    print(md)
    out = Path(args.out)
    _write_json(str(out / "robot_specificity.json"), results)
    (out / "robot_specificity.md").write_text(md)
    print(f"wrote {out / 'robot_specificity.md'}")
    return 0


def _lock_space(lock: dict) -> str:
    """``shared`` or ``per_joint``: from ``result.space``, or from a list-valued
    parameter for locks saved before that field existed."""
    space = (lock.get("result") or {}).get("space")
    if space:
        return str(space)
    per_joint = any(isinstance(v, (list, tuple)) for v in (lock.get("params") or {}).values())
    return "per_joint" if per_joint else "shared"


def _default_locks(server, datasets: list[str]) -> list[dict]:
    """The newest saved *shared* lock for each dataset, as lock dicts."""
    by_ds: dict[str, dict] = {}
    for l in server.list_lock_dicts():
        ds = l.get("dataset")
        if ds not in datasets or _lock_space(l) != "shared":
            continue
        if str(l.get("created_at", "")) >= str(by_ds.get(ds, {}).get("created_at", "")):
            by_ds[ds] = l
    missing = [d for d in datasets if d not in by_ds]
    if missing:
        raise SystemExit(f"no saved shared lock for {', '.join(missing)}; run antares calibrate first")
    return [by_ds[d] for d in datasets]


def _synthetic_arms(datasets: list[str], lock_dicts: list[dict], args, simulate):
    """Stand-in arms whose true parameters are the given locks' parameters.

    Each lock becomes a synthetic dataset driven by random commands, and a
    fresh lock is fitted to it exactly like ``antares calibrate`` would.
    Returns the episodes, locks, truth and splits for :func:`study`.
    """
    from antares import server
    from antares.calibrate import calibrate, split_episodes
    from antares.synthetic import make_dataset

    if len(lock_dicts) != len(datasets):
        raise SystemExit(f"--synthetic needs one lock per dataset: got {len(datasets)} datasets "
                         f"and {len(lock_dicts)} locks")
    episodes, locks, truth, splits = {}, {}, {}, {}
    for i, lock in enumerate(lock_dicts):
        n = lock["name"]
        p = server.params_from_dict(lock["params"])
        sid = f"synthetic/{n}"
        print(f"  {sid}: simulating an arm with {n}'s parameters", flush=True)
        ds = make_dataset(sid, p, n_episodes=args.n_train + args.n_eval + args.n_train, seed=i + 1)
        train, eval_ = split_episodes(ds.episodes, args.n_train, args.n_eval, seed=args.seed)
        print(f"  {sid}: calibrating", flush=True)
        res = calibrate(train, eval_, simulate=simulate, max_evaluations=args.max_evaluations, seed=args.seed)
        episodes[sid] = ds.episodes
        locks[f"{sid.split('/')[1]}-fit"] = (sid, res.params)
        truth[sid] = p
        splits[sid] = ([ep.index for ep in train], [ep.index for ep in eval_])
    return episodes, locks, truth, splits


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
    from antares.export_targets import TARGET_LABELS, TargetExportError, dataset_fps, render, resolve_targets
    from antares.lock import RobotLock

    try:
        lock = RobotLock.from_dict(server.load_lock_dict(args.name))
    except HTTPException as e:
        print(f"error: {e.detail}", file=sys.stderr)
        return 1
    try:
        targets = resolve_targets(getattr(args, "format", "mjcf"))
    except TargetExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
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
    fps = getattr(args, "fps", None)
    for target in targets:
        export = render(lock, target, fps=fps, lock_dir=server.LOCK_DIR)
        for path in export.write(out_dir):
            twin = " (JSON twin)" if path.suffix == ".json" and target != "json" else ""
            print(f"  {path.name:<40} {TARGET_LABELS[target]}{twin}")
    if targets:
        hz, source = dataset_fps(lock, fps)
        print(f"latency: {p.latency_frames} frame(s) at {hz:g} Hz (fps {source}); apply as a command delay in your control loop")
    print(f"view: python -m mujoco.viewer --mjcf {scene}")
    return 0


def cmd_camera_fit(args: argparse.Namespace) -> int:
    from antares import server

    try:
        ds = server.get_dataset(args.repo_id)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    cams = {c["name"]: c for c in server.list_camera_infos(args.repo_id)}
    if args.camera not in cams:
        print(f"error: {args.repo_id} has no camera {args.camera!r}; cameras: {sorted(cams)}", file=sys.stderr)
        return 1
    if not cams[args.camera]["fixed"]:
        print(f"error: {args.camera} is a wrist camera; only fixed cameras can be fitted", file=sys.stderr)
        return 1
    lock = args.lock or server._slug(args.repo_id)
    print(f"fitting camera {args.camera} of {args.repo_id} for lock {lock}: "
          f"{args.frames} frames over {args.episodes} episodes, {args.starts} starts")
    last_stage = [None]

    def on_progress(ev) -> None:
        if ev.stage != last_stage[0]:
            last_stage[0] = ev.stage
            if ev.stage == "sample":
                print(f"  sampled {ev.stage_evaluation} cameras: best IoU {ev.stage_best_iou:.3f}  [{ev.elapsed_s:.0f}s]")
            elif ev.stage == "start":
                print(f"  starts: {ev.n_starts} x CMA-ES at low resolution")
            elif ev.stage == "refine":
                print(f"  refining start {ev.start} at full resolution")
        if ev.stage in ("start", "refine") and 12 < ev.stage_evaluation and ev.stage_evaluation % 100 < 12:
            print(f"    {ev.stage} {ev.start}: eval {ev.stage_evaluation} best IoU {ev.stage_best_iou:.3f} "
                  f"(overall {ev.best_iou:.3f})  [{ev.elapsed_s:.0f}s]")

    try:
        fit, path = server.run_camera_fit(
            ds, args.camera, lock, n_frames=args.frames, n_episodes=args.episodes,
            n_starts=args.starts, seed=args.seed, on_progress=on_progress,
        )
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"camera {fit.camera}: fit IoU {fit.fit_iou:.3f} on {len(fit.fit_frames)} frames, "
          f"held-out IoU {fit.heldout_iou:.3f} on {len(fit.heldout_frames)} frames, "
          f"{fit.n_evaluations} evaluations in {fit.elapsed_s:.0f}s"
          + (" (cancelled)" if fit.cancelled else ""))
    print(f"  pos {['%.3f' % v for v in fit.pos]} quat {['%.4f' % v for v in fit.quat]} fovy {fit.fovy:.1f} deg")
    print(f"  starts reached IoU {', '.join('%.2f' % v for v in fit.start_ious)}; best was {fit.best_start}")
    print(f"saved {path}")
    return 0


def cmd_camera_list(args: argparse.Namespace) -> int:
    from antares import server

    try:
        fits = server.load_camera_fit_dicts(args.lock)
    except Exception as e:  # noqa: BLE001
        print(f"error: {getattr(e, 'detail', e)}", file=sys.stderr)
        return 1
    if not fits:
        print(f"no camera fits for lock {args.lock} (run: antares camera-fit <dataset> --camera <name> --lock {args.lock})")
        return 0
    rows = []
    for name, f in sorted(fits.items()):
        rows.append([
            name,
            f.get("key", ""),
            f"{f.get('width', '?')}x{f.get('height', '?')}",
            _f(f.get("fit_iou")),
            _f(f.get("heldout_iou")),
            f"{f.get('fovy', float('nan')):.1f}",
            " ".join(f"{v:.3f}" for v in f.get("pos", [])),
            str(f.get("created_at", ""))[:19],
        ])
    print(_table(["camera", "key", "size", "fit IoU", "held-out IoU", "fovy", "pos (m)", "created"], rows))
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
        from antares.fleet import pinned_across

        out.append("")
        out.append(
            _table(
                ["dataset", "robot", "eps", "fps", "baseline", "calibrated", "closed", "latency", "kp"],
                rows,
            )
        )
        pinned = pinned_across(r.get("pinned") for r in robots)
        n_seeded = sum(1 for r in robots if r.get("pinned") is not None)
        out.append(
            f"pinned across the fleet's locks: {', '.join(pinned) or 'none'} "
            f"({n_seeded}/{len(robots)} locks carry seed spread); "
            "the other parameters are fit outcomes, not measurements of the arm"
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

    dataset_help = "Hub dataset id, upload/<name>, or a LeRobot dataset directory on disk"
    robot_help = "sim model from the registry (so101, so100); default: the dataset's family"

    def add_calibration_args(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("LeRobot calibration (the follower arm's own tick-to-angle map)")
        g.add_argument("--calibration", default=None, metavar="PATH",
                       help="the follower's LeRobot calibration file (<robot.id>.json)")
        g.add_argument("--robot-id", default=None, metavar="ID",
                       help="LeRobot --robot.id of this arm: its calibration ID.json is looked up under "
                       "$HF_LEROBOT_CALIBRATION/robots/<robot-type>/ (inspect: required; calibrate: used "
                       "when present), and calibrate also registers the saved lock for this robot in "
                       "locks/registry.json")
        g.add_argument("--robot-type", default="so101_follower",
                       help="LeRobot robot type for --robot-id (default so101_follower; the so_follower "
                       "directory of LeRobot >= 0.5 is tried too)")

    p = sub.add_parser("inspect", help="validate and summarize a LeRobot dataset")
    p.add_argument("repo_id", help=dataset_help)
    add_calibration_args(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("replay", help="replay one episode in sim and print the gap")
    p.add_argument("repo_id", help=dataset_help)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--lock", default=None, help="use a saved lock's parameters")
    p.add_argument("--robot", default=None, help=robot_help)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("calibrate", help="fit sim parameters to a dataset and save a lock")
    p.add_argument("repo_id", help=dataset_help)
    p.add_argument("--robot", default=None, help=robot_help)
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--max-evaluations", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seeds",
        type=int,
        default=1,
        metavar="N",
        help="run the shared fit from N CMA-ES seeds and keep the best held-out score; "
        "records which parameters agree across seeds (3 for a lock worth keeping; default 1)",
    )
    p.add_argument("--name", default=None, help="lock name (default: dataset slug)")
    p.add_argument(
        "--per-joint",
        action="store_true",
        help="after the shared fit, refine one value per joint (plus force limit)",
    )
    p.add_argument(
        "--per-joint-evaluations",
        type=int,
        default=600,
        help="evaluation budget of the per-joint stage (default 600)",
    )
    p.add_argument(
        "--per-joint-sigma0",
        type=float,
        default=None,
        help="CMA-ES step size of the per-joint stage in unit coordinates "
        "(default 0.05; 0.1 explores more and can win with a budget >= 1200)",
    )
    p.add_argument(
        "--fit-sensor",
        action="store_true",
        help="also fit the per-joint command offset/scale and compare against dynamics only",
    )
    p.add_argument(
        "--sensor-mode",
        choices=("sim", "kinematic"),
        default="sim",
        help="with --fit-sensor: fit the map through the sim, alternating with the dynamics "
        "(sim, default), or pre-fit it from still frames and freeze it (kinematic)",
    )
    add_calibration_args(p)
    p.add_argument("--activate", action="store_true",
                   help="with --robot-id: activate the new lock even if it scores worse")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("cross", help="score locks across datasets (matrix)")
    p.add_argument("--datasets", required=True, help="comma-separated dataset ids")
    p.add_argument("--locks", default="", help="comma-separated lock names")
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="also write the matrix and summary here")
    p.set_defaults(func=cmd_cross)

    p = sub.add_parser("drift", help="refit a lock on sliding episode windows to spot drift")
    p.add_argument("repo_id", help=dataset_help)
    p.add_argument("--lock", required=True, help="lock whose parameters seed each window")
    p.add_argument("--window", type=int, default=10, help="episodes per window")
    p.add_argument("--step", type=int, default=5, help="episodes between window starts")
    p.add_argument("--max-evaluations", type=int, default=60, help="CMA-ES budget per window")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_drift)


    p = sub.add_parser("dr", help="derive domain-randomization ranges from a lock's data")
    p.add_argument("repo_id", help=dataset_help)
    p.add_argument("--lock", default=None, help="lock to start from (default: dataset slug)")
    p.add_argument("--n-fits", type=int, default=8, help="bootstrap refits")
    p.add_argument("--n-train", type=int, default=12, help="episodes per refit")
    p.add_argument("--max-evaluations", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="ranges name (default: lock name)")
    p.set_defaults(func=cmd_dr)


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
    p.add_argument("--lock", required=True,
                   help="lock name in the lock dir, a .lock.json path, or @<robot-id> for a registered robot's active lock")
    p.add_argument("--threshold", type=float, default=None, metavar="DEG",
                   help="absolute error allowed in degrees (default: none)")
    p.add_argument("--ratio", type=float, default=1.5,
                   help="allowed multiple of the lock's held-out score (default 1.5)")
    p.add_argument("--last", type=int, default=None, metavar="N",
                   help="only check the newest N episodes (default: all)")
    p.add_argument("--min-episodes", type=int, default=5,
                   help="fewer episodes than this is insufficient data, exit 2 (default 5)")
    p.add_argument("--json", action="store_true", help="print the CheckResult as JSON")
    p.add_argument("--explain", action="store_true",
                   help="on drift, refit and name the parameter change that explains it")
    p.add_argument("--explain-evaluations", type=int, default=240, metavar="N",
                   help="refit budget for --explain (default 240, at least 10)")
    p.add_argument("--notify", default=None, metavar="URL",
                   help="POST an alert to this webhook (default: $ANTARES_WEBHOOK_URL); "
                        "a failed POST only warns and leaves the exit code alone")
    p.add_argument("--notify-when", choices=["drifted", "failure", "always"], default="drifted",
                   help="alert on: drifted, failure (drifted or too few episodes) or always (default drifted)")
    p.add_argument("--notify-format", choices=["auto", "slack", "json"], default="auto",
                   help="payload: slack blocks, generic json, or auto from the host (default auto)")
    p.add_argument("--notify-dry-run", action="store_true",
                   help="print the alert payload instead of posting it")
    p.add_argument("--no-record", action="store_true",
                   help="do not append this check to the lock's <name>.history.jsonl")
    p.add_argument("--history-dir", default=None, metavar="DIR",
                   help="directory for the history file (default: next to the lock)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("notify-test", help="post a synthetic OK alert to confirm a webhook works")
    p.add_argument("--url", default=None, help="webhook URL (default: $ANTARES_WEBHOOK_URL)")
    p.add_argument("--format", choices=["auto", "slack", "json"], default="auto",
                   help="payload: slack blocks, generic json, or auto from the host (default auto)")
    p.add_argument("--title", default="Antares webhook test", help="alert title / Slack header")
    p.add_argument("--lock", default="example", help="lock name shown in the message")
    p.add_argument("--dataset", default="your-org/so101_arm_a", help="dataset shown in the message")
    p.add_argument("--dry-run", action="store_true", help="print the payload instead of posting it")
    p.set_defaults(func=cmd_notify_test)


    p = sub.add_parser(
        "record-check",
        help="record fresh episodes with lerobot-record, then run check: PASS or FAIL",
        description=(
            "One command at the arm: drive lerobot-record to capture a few episodes into "
            "--root (default recordings/<robot-id>/<timestamp>, never pushed to the Hub), "
            "then run the same gate as `antares check <root> --lock <name>`. "
            "Exit 0 pass, 1 drifted, 2 usage or too few episodes, 3 when recording failed. "
            "Needs LeRobot (pip install 'lerobot[core_scripts,feetech]', Python >= 3.12); "
            "pass --recorder when lerobot-record lives in another environment."
        ),
    )
    p.add_argument("--lock", default=None,
                   help="lock name in the lock dir, a .lock.json path, or @<robot-id>; "
                        "default: the active lock registered for --robot-id")
    p.add_argument("--robot-port", required=True, metavar="PORT", help="follower arm serial port, e.g. /dev/tty.usbmodem...")
    p.add_argument("--robot-id", required=True, metavar="ID", help="the LeRobot id the arm was calibrated under (names the recording and the banner)")
    p.add_argument("--robot-type", default="so101_follower", help="LeRobot robot type (default so101_follower)")
    p.add_argument("--teleop-port", default=None, metavar="PORT", help="leader arm serial port (required for a *_leader teleop)")
    p.add_argument("--teleop-id", default=None, metavar="ID", help="leader arm LeRobot id (default <robot-id>_leader)")
    p.add_argument("--teleop-type", default="so101_leader", help="LeRobot teleop type (default so101_leader)")
    p.add_argument("--episodes", type=int, default=5, metavar="N",
                   help="episodes to record (default 5; raised to --min-episodes when lower)")
    p.add_argument("--root", default=None, metavar="DIR",
                   help="where to write the dataset (default recordings/<robot-id>/<YYYYMMDD-HHMMSS>)")
    p.add_argument("--task", default="antares check", help="--dataset.single_task text (default 'antares check')")
    p.add_argument("--episode-time", type=float, default=None, metavar="S", help="--dataset.episode_time_s (LeRobot default 60)")
    p.add_argument("--reset-time", type=float, default=None, metavar="S", help="--dataset.reset_time_s (LeRobot default 60)")
    p.add_argument("--extra", action="append", default=None, metavar="FLAG",
                   help="extra lerobot-record flag, verbatim; repeatable (e.g. --extra=--robot.cameras=...)")
    p.add_argument("--recorder", default=None, metavar="PATH", help="lerobot-record executable (default: on PATH)")
    p.add_argument("--dry-run", action="store_true", help="print the lerobot-record and check commands, exit 0")
    p.add_argument("--skip-record", action="store_true", help="check the dataset already under --root")
    _add_check_options(p)
    p.set_defaults(func=cmd_record_check)
    p = sub.add_parser(
        "history",
        help="checks recorded for a lock across sessions, with the trend",
        description=(
            "List the checks appended to <lock>.history.jsonl by `antares check` (one per run, "
            "oldest first) and the least-squares slope of their score in degrees per day."
        ),
    )
    p.add_argument("lock", nargs="?", default=None,
                   help="lock name in the lock dir, a .lock.json path, or @<robot-id> (its active lock)")
    p.add_argument("--robot", default=None, metavar="ID",
                   help="instead of one lock: every lock in the registered robot's lineage, in time order")
    p.add_argument("--last", type=int, default=None, metavar="N",
                   help="show only the newest N checks (the trend still uses all of them)")
    p.add_argument("--json", action="store_true",
                   help="print {lock, entries, trend} as GET /api/history/<lock> does")
    p.add_argument("--history-dir", default=None, metavar="DIR",
                   help="directory of the history file (default: next to the lock)")
    p.set_defaults(func=cmd_history)

    p = sub.add_parser(
        "robots",
        help="the registry: one id per physical robot, its lock lineage and the active lock",
        description=(
            "locks/registry.json gives each physical robot (LeRobot's robot id) a versioned lineage "
            "of calibrations and one active lock. `@<id>` then stands for that lock wherever a --lock "
            "is accepted. Without a subcommand, list the robots."
        ),
    )
    p.add_argument("--json", action="store_true", help="print the table as JSON")
    p.set_defaults(func=cmd_robots, robots_cmd="list")
    rsub = p.add_subparsers(dest="robots_cmd")
    r = rsub.add_parser("list", help="one row per robot: active lock, version, held-out score, last check")
    r.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    r.set_defaults(func=cmd_robots)
    r = rsub.add_parser("show", help="a robot's lineage and how its pinned parameters moved between versions")
    r.add_argument("robot_id")
    r.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    r.set_defaults(func=cmd_robots)
    r = rsub.add_parser("register", help="add a saved lock to a robot's lineage (activated by the rule)")
    r.add_argument("robot_id")
    r.add_argument("--lock", required=True, help="lock name in the lock dir, or a .lock.json path")
    r.add_argument("--activate", action="store_true", help="activate even if it scores worse than the active lock")
    r.add_argument("--note", default=None, help="free text kept with the entry (why it was recalibrated)")
    r.add_argument("--max-regression", type=float, default=None, metavar="FRAC",
                   help="activate when the held-out score is at most this fraction worse (default 0.10)")
    r.set_defaults(func=cmd_robots)
    r = rsub.add_parser("activate", help="make one lock of the lineage the active one")
    r.add_argument("robot_id")
    r.add_argument("lock")
    r.set_defaults(func=cmd_robots)
    r = rsub.add_parser("rollback", help="activate the lock before the active one")
    r.add_argument("robot_id")
    r.set_defaults(func=cmd_robots)
    r = rsub.add_parser("remove", help="forget a robot (its lock files stay)")
    r.add_argument("robot_id")
    r.set_defaults(func=cmd_robots)


    p = sub.add_parser("events", help="detect contact events from the real-sim residual")
    p.add_argument("repo_id", help=dataset_help)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--lock", default=None, help="use a saved lock's parameters")
    p.add_argument("--all", action="store_true", help="summarize across every episode")
    p.add_argument("--z", type=float, default=4.0, help="robust z-score to enter an event (exit at z/2)")
    p.add_argument("--min-frames", type=int, default=5, help="shortest event kept")
    p.set_defaults(func=cmd_events)


    p = sub.add_parser(
        "compensate",
        help="commands that make the calibrated sim track an episode's real trajectory (sim-only)",
    )
    p.add_argument("repo_id", help=dataset_help)
    p.add_argument("--episode", type=int, default=None, help="episode to compensate")
    p.add_argument("--lock", default=None, help="calibrated lock to compensate against")
    p.add_argument("--method", choices=["shift", "ilc"], default="ilc", help="which commands to save/return")
    p.add_argument("--out", default=None, help="save the compensated commands (T,6) in dataset units as .npy")
    p.add_argument(
        "--heldout", action="store_true", help="instead of one episode, aggregate over the lock's held-out split"
    )
    p.set_defaults(func=cmd_compensate)

    p = sub.add_parser("identify", help="check which of a lock's parameters the data pins down")
    p.add_argument("repo_id")
    p.add_argument("--lock", required=True, help="saved lock to analyse")
    p.add_argument("--split", choices=("train", "eval"), default="train")
    p.add_argument("--delta", type=float, default=0.1, help="error rise (deg) defining tolerance")
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="also write the report here")
    p.set_defaults(func=cmd_identify)

    p = sub.add_parser("specificity", help="test whether each arm's lock is specific to that arm")
    p.add_argument("--datasets", default="lerobot/svla_so101_pickplace,lerobot/svla_so100_pickplace",
                   help="comma-separated dataset ids (default: the two demo arms)")
    p.add_argument("--locks", default="", help="comma-separated lock names (default: newest per dataset)")
    p.add_argument("--synthetic", action="store_true",
                   help="replace each dataset with a simulated arm that has its lock's parameters")
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restart-seeds", default="1,2", help="extra CMA-ES seeds to refit with")
    p.add_argument("--max-evaluations", type=int, default=150)
    p.add_argument("--out", default="results", help="directory for the .md and .json report")
    p.set_defaults(func=cmd_specificity)

    p = sub.add_parser("locks", help="list saved locks")
    p.set_defaults(func=cmd_locks)

    p = sub.add_parser(
        "synth",
        help="write a synthetic recording of an arm with known, optionally worn, params",
        description=(
            "Simulate what an arm would have recorded and save it as a LeRobot dataset "
            "directory, e.g. to try `antares check --explain` without a physical arm."
        ),
    )
    p.add_argument("out", help="output directory (LeRobot layout)")
    p.add_argument("--lock", default=None,
                   help="start from a lock's params, by name or file path (default: MJCF defaults)")
    p.add_argument("--wear", default="", metavar="CHANGES",
                   help='factors, and a frame shift for latency: "kp=0.7,frictionloss=2,latency_frames=1"')
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--noise", type=float, default=0.2, help="sensor noise, dataset units (default 0.2)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("transfer", help="does calibration help a policy transfer? (sim-to-sim experiment)")
    tsub = p.add_subparsers(dest="transfer_cmd", required=True)
    t = tsub.add_parser(
        "run",
        help="fit locks to hidden synthetic arms, train reaching policies in each sim, score them on the hidden arms",
    )
    t.add_argument("--quick", action="store_true", help="tiny budgets (under a minute); what the tests run")
    t.add_argument("--seeds", type=int, default=None, help="training seeds per variant (default 3; 1 with --quick)")
    t.add_argument("--generations", type=int, default=None, help="CMA-ES generations per policy (default 150)")
    t.add_argument("--workers", type=int, default=None, help="training processes (default: all cores)")
    t.add_argument("--arm", action="append", default=None, metavar="NAME=LOCK_PATH",
                   help="hidden arm whose truth is this lock's params (repeatable; default: the two demo locks)")
    t.add_argument("--out", default="results/policy_transfer", help="output path prefix for .json and .md")
    t.set_defaults(func=cmd_transfer_run)
    t = tsub.add_parser("report", help="render a saved transfer report as markdown")
    t.add_argument("json", help="report written by `antares transfer run`")
    t.add_argument("--md", default=None, help="write the markdown here instead of printing it")
    t.set_defaults(func=cmd_transfer_report)

    p = sub.add_parser("export", help="write a lock as a standalone MJCF (+ scene, sidecar, meshes)")
    p.add_argument("name", help="lock name (see `antares locks`)")
    p.add_argument("-o", "--output", default=None, help="output directory (default: export/<name>)")
    p.add_argument(
        "--no-bundle",
        action="store_true",
        help="do not copy the STL meshes; point meshdir at the package assets instead",
    )
    p.add_argument(
        "--format",
        choices=["mjcf", "isaaclab", "maniskill", "mjlab", "json", "all"],  # antares.export_targets.FORMATS
        default="mjcf",
        help="also write that simulator's actuator config next to the MJCF (default: mjcf only; "
        "all = isaaclab + maniskill + mjlab + json)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="dataset frame rate used to turn latency_frames into seconds "
        "(default: the dataset's meta/info.json in the local HF cache, else 30)",
    )
    p.set_defaults(func=cmd_export)


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

    p = sub.add_parser("camera-fit", help="fit a dataset's fixed camera from its videos and joint logs")
    p.add_argument("repo_id", help="Hub dataset id or a LeRobot dataset directory on disk")
    p.add_argument("--camera", required=True, help="camera name, e.g. up (observation.images.up)")
    p.add_argument("--lock", default=None, help="lock to attach the camera to (default: dataset slug)")
    p.add_argument("--episodes", type=int, default=4, help="episodes the fit frames are spread over")
    p.add_argument("--frames", type=int, default=40, help="frames used in the fit")
    p.add_argument("--starts", type=int, default=8, help="CMA-ES starts kept from the random sample")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_camera_fit)

    p = sub.add_parser("camera-list", help="list the camera fits saved for a lock")
    p.add_argument("lock")
    p.set_defaults(func=cmd_camera_list)

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
