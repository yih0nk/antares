"""``antares`` command line.

Heavy imports (MuJoCo, FastAPI, CMA-ES) live inside the command functions so
``antares --help`` is instant.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _params_table(params: dict, joints: list[str] | None = None) -> str:
    """Scalar params on one line; per-joint params as a joints x fields table."""
    if not any(isinstance(v, list) for v in params.values()):
        return "params: " + ", ".join(f"{k}={_fmt(v)}" for k, v in params.items())
    from antares.sim import JOINTS

    names = list(params)
    rows = []
    for j, joint in enumerate(joints or JOINTS):
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

    if args.offline:
        # An environment variable, so a --reload worker process inherits it.
        os.environ["ANTARES_OFFLINE"] = "1"

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
    from antares.catalog import robot_for_dataset

    print(f"sim model: {robot_for_dataset(args.repo_id) or model_for_dataset(ds.robot_type)}")
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
    from antares import factors, server

    if not _load_factor_files(args.factors_file):
        return EXIT_USAGE
    from antares.calibrate import default_factor_names

    # +/- edits apply to what this fit would search by default: the servo
    # model's fields with --servo, and a per-joint stage derived from the
    # shared selection, as calibrate builds them.
    servo = args.servo != "position"
    try:
        fit = factors.parse_selection(args.factors, "shared", defaults=default_factor_names(servo)[0])
        per = factors.parse_selection(
            args.per_joint_factors, "per_joint", defaults=default_factor_names(servo, fit)[1]
        )
    except (KeyError, ValueError) as e:
        print(f"--factors: {str(e).strip(chr(39))}", file=sys.stderr)
        return EXIT_USAGE
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
    if args.servo != "position":
        stages += f", {args.servo} servo model"
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
        fit_gravity=args.fit_gravity,
        fit_deadband=args.fit_deadband,
        robot=args.robot,
        on_progress=on_progress,
        fit_sensor=args.fit_sensor,
        sensor_mode=args.sensor_mode,
        n_seeds=args.seeds,
        factors=None if fit is None else list(fit),
        per_joint_factors=None if per is None else list(per),
        calibration_path=args.calibration,
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        servo=args.servo,
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
    from antares.robots import get_robot

    print(_params_table(r["params"], list(get_robot(lock["robot"]).joints) if lock.get("robot") else None))
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


def _load_factor_files(paths) -> bool:
    """Register the ``--factors-file`` TOMLs; False (after printing why) on a bad file."""
    from antares import factors

    for path in paths or ():
        try:
            factors.load_factor_file(path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"--factors-file {path}: {e}", file=sys.stderr)
            return False
    return True


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
    # numeric, shared parameters only: not the servo model's name, nor per-joint lists
    param_names = [k for k, v in s["lock_params"].items() if isinstance(v, (int, float))]
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
        p = result.params

        def fmt(v, spec):
            # Deliberately a range, not SimParams.scalar's mean: this line
            # shows how far the joints spread within one fit; the derived DR
            # ranges below are from the joint means.
            return f"{min(v):{spec}}-{max(v):{spec}}" if isinstance(v, tuple) else f"{v:{spec}}"

        print(
            f"  fit {i + 1}/{args.n_fits}: train {result.train_score:6.3f} deg  "
            f"kp {fmt(p.kp, '.1f')}  kv {fmt(p.kv, '.2f')}  latency {fmt(p.latency_frames, '')}",
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
MIN_EXPLAIN_EVALUATIONS = 20  # as POST /api/check validates explain_evaluations


def _fmt(v, nd: int = 3) -> str:
    if isinstance(v, str):
        return v
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


def cmd_noise(args: argparse.Namespace) -> int:
    import math

    from antares import server
    from antares.noise import estimate
    from antares.sim import JOINTS

    ds = server.get_dataset(args.repo_id)
    name = args.name or server._slug(ds.repo_id)
    model = estimate(list(ds.episodes), name=name, dataset=ds.repo_id)
    print(f"{ds.repo_id}: joint-reading noise while each joint is parked")
    deg = lambda v, nd=3: "-" if not math.isfinite(v) else f"{math.degrees(v):.{nd}f}"  # noqa: E731
    rows = [
        [j, str(int(model.n_frames[i])), deg(model.scale[i]),
         "-" if not math.isfinite(model.acf1[i]) else f"{model.acf1[i]:.2f}", deg(model.quantum[i], 4)]
        for i, j in enumerate(JOINTS)
    ]
    print(_table(["joint", "parked frames", "std (deg)", "lag-1 corr", "step (deg)"], rows))
    if not args.no_save:
        path = model.save(args.out or model.default_path(server.LOCK_DIR))
        print(f"saved noise model {model.name} -> {path}")
    return 0


def cmd_camera_delay(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    import numpy as np

    from antares import server
    from antares.video import VideoUnavailable
    from antares.videodelay import camera_delay

    try:
        ds = server.get_dataset(args.repo_id)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    cams = {c["name"] for c in server.list_camera_infos(args.repo_id)}
    if args.camera not in cams:
        print(f"error: {args.repo_id} has no camera {args.camera!r}; cameras: {sorted(cams)}", file=sys.stderr)
        return 1
    n = min(args.episodes, len(ds.episodes))
    picks = [ds.episodes[int(i)].index for i in np.unique(np.linspace(0, len(ds.episodes) - 1, n).round())]
    try:
        res = camera_delay(ds, args.camera, picks, max_lag=args.max_lag)
    except VideoUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    rows = [[str(ep), f"{lag:+.2f}", f"{res.peak_corr[ep]:.2f}"] for ep, lag in res.per_episode.items()]
    print(f"{ds.repo_id} camera {args.camera}: video lag behind the joint log, per episode")
    print(_table(["episode", "lag (frames)", "peak corr"], rows))
    print(f"median: {res.frames:+.2f} frames = {res.ms:+.0f} ms at {res.fps:g} fps (positive: video is late)")
    if args.out:
        Path(args.out).write_text(json.dumps(res.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"saved -> {args.out}")
    return 0


def cmd_task(args: argparse.Namespace) -> int:
    from antares import server
    from antares.task import TaskDistribution

    ds = server.get_dataset(args.repo_id)
    name = args.name or server._slug(ds.repo_id)
    td = TaskDistribution.from_episodes(list(ds.episodes), name=name, dataset=ds.repo_id, held_gap=args.held_gap)
    s = td.summary()
    if not td.held:
        print(
            f"error: {ds.repo_id}: none of the {s['grasps']} closes of the gripper held anything "
            f"(jaw never stopped {args.held_gap:g} units above its empty-closed level); "
            f"nothing to sample from, so no task file was written (try a smaller --held-gap)",
            file=sys.stderr,
        )
        return 1
    print(
        f"{ds.repo_id}: {s['episodes']} episodes, {s['grasps']} closes of the gripper: "
        f"{s['held']} picks ({s['carried']} carried somewhere else), {s['missed']} missed, "
        f"{s['idle']} idle closes after letting go"
    )
    rows = []
    for label, st in (("pick", s["pick"]), ("place", s["place"])):
        if st is None:
            continue
        cm = lambda v, f="+.1f": ", ".join(f"{100 * x:{f}}" for x in v)  # noqa: E731
        rows.append([label, str(st["n"]), cm(st["mean"]), cm(st["std"], ".1f"), cm(st["min"]), cm(st["max"])])
    if rows:
        print("gripper tip in the base frame, cm (x, y, z):")
        print(_table(["", "n", "mean", "std", "min", "max"], rows))
    if s["jaw_deg_median"] is not None:
        print(f"jaw stops at {s['jaw_deg_median']:.1f} deg on a held object (median)")
    if s["missed_episodes"]:
        print(f"episodes with a missed grasp: {_episode_list(s['missed_episodes'])}")
    if s["episodes_without_pick"]:
        print(f"episodes with no pick found: {_episode_list(s['episodes_without_pick'])}")
    if not args.no_save:
        path = td.save(args.out or td.default_path(server.LOCK_DIR))
        print(f"saved task {td.name} -> {path}")
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


def cmd_sag(args: argparse.Namespace) -> int:
    import json

    from antares import server
    from antares.sag import sag_report
    from antares.sim import Arm

    ds = server.get_dataset(args.repo_id)
    lock = server.load_lock_dict(args.lock)
    params = server.params_from_dict(lock["params"])
    wanted = set(lock["result"].get("eval_episodes") or []) if not args.all else set()
    episodes = [ep for ep in ds.episodes if args.all or ep.index in wanted]
    if not episodes:
        print("the lock lists no held-out episodes; pass --all", file=sys.stderr)
        return 2
    rep = sag_report(Arm(params), episodes)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
        return 0
    which = "all" if args.all else "held-out"
    print(
        f"{ds.repo_id} vs lock {args.lock}: {rep.n_episodes} {which} episodes, {rep.n_frames} frames, "
        f"gravity_scale {rep.gravity_scale:.3g}"
    )
    print("residual (sim - real) regressed on the torque that holds each pose against gravity")
    rows = [
        [
            j.joint,
            _fmt(j.rms_deg),
            _fmt(j.rms_corrected_deg),
            f"{j.slope_deg_per_nm:+.3g}",
            f"{100 * j.partial_r2:.0f}%",
            _fmt(j.torque_std_nm),
            "-" if j.implied_gravity_scale is None else _fmt(j.implied_gravity_scale),
        ]
        for j in rep.joints.values()
    ]
    print(
        _table(
            ["joint", "rms (deg)", "w/o sag", "deg per N*m", "explained", "torque std", "implied scale"],
            rows,
        )
    )
    worst = rep.worst_joint
    if worst is not None:
        bins = ", ".join(f"{t:+.2f} N*m: {r:+.2f} deg" for t, r, _ in rep.joints[worst].bins)
        print(f"{worst} mean residual by holding torque: {bins}")
    motion = []
    for j in rep.joints.values():
        m = j.by_motion
        motion.append([j.joint] + ["-" if m[k] is None else f"{m[k]:+.2f}" for k in ("still", "up", "down")])
    print("mean residual (deg) by motion; a jump between columns is deadband or backlash, not sag")
    print(_table(["joint", "still", "moving up", "moving down"], motion))
    if rep.implied_gravity_scale is not None:
        print(f"implied gravity_scale {rep.implied_gravity_scale:.3g}; fit it with calibrate --fit-gravity")
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

    arm = None
    if args.lock:
        from antares.check import _lock_view
        from antares.sim import Arm

        try:
            lock_d = _load_lock_arg(args.lock)
            params = _lock_view(lock_d)[2]
            # The lock's own sim: its sensor and unit maps shape the recording
            # just as they shape the replay `antares check` scores it with.
            arm = Arm(
                robot=server.check_robot(lock_d.get("robot")),
                command_map=server.command_map_from_lock(lock_d),
                **({"unit_map": um} if (um := server.unit_map_from_lock(lock_d)) is not None else {}),
            )
        except Exception as e:  # noqa: BLE001 - HTTPException, bad JSON or a lock without params
            detail = getattr(e, "detail", None) or f"{type(e).__name__}: {e}"
            print(f"--lock: {detail}", file=sys.stderr)
            return EXIT_USAGE
        if arm.n_joints != 6:
            print(f"--lock: synth generates SO-arm motions; {arm.robot.name} has {arm.n_joints} joints",
                  file=sys.stderr)
            return EXIT_USAGE
    else:
        params = server.default_params()
    try:
        from antares.diagnose import apply_change

        # apply_change is worn() plus the servo mapping: on a servo lock a kp
        # factor scales the servo's gain instead of the derived (recomputed) kp
        params = apply_change(params, synth.parse_changes(args.wear))
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
        arm=arm,
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
    lock_d = server.load_lock_dict(args.lock)
    params = server.params_from_dict(lock_d["params"])
    train, eval_ = split_episodes(ds.episodes, args.n_train, args.n_eval, seed=args.seed)
    episodes = train if args.split == "train" else eval_
    if not episodes:
        raise SystemExit(
            f"{ds.repo_id} has {len(ds.episodes)} episodes; --n-train {args.n_train} --n-eval {args.n_eval} "
            f"leaves no {args.split} split"
        )
    # The lock's own sim (robot, sensor and unit maps held fixed), else the dataset's robot.
    simulate = (server.lock_simulate(lock_d) or server.simulate_for(server.dataset_robot(ds))
                or default_simulate())
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
    from antares.export import SCENE_FILENAME, TASK_FILENAME, export_scene, sidecar_path
    from antares.export_targets import (
        TARGET_LABELS,
        TargetExportError,
        dataset_fps,
        render,
        resolve_targets,
        write_unsupported_note,
    )
    from antares.lock import RobotLock
    from antares.task import TASK_SUFFIX, TaskDistribution

    try:
        lock = RobotLock.from_dict(server.load_lock_dict(args.name))
    except HTTPException as e:
        print(f"error: {e.detail}", file=sys.stderr)
        return 1
    task = None
    if args.task:
        task_path = Path(args.task)
        if not task_path.is_file():
            task_path = server.LOCK_DIR / f"{args.task}{TASK_SUFFIX}"
        if not task_path.is_file():
            print(f"error: no task file {args.task} (run `antares task` first)", file=sys.stderr)
            return 1
        task = TaskDistribution.load(task_path)
        if not task.held:
            print(f"error: task {task.name} ({task_path}) has no held grasps: nothing to put in the scene", file=sys.stderr)
            return 1
    try:
        targets = resolve_targets(getattr(args, "format", "mjcf"))
    except TargetExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out_dir = Path(args.output) if args.output else Path("export") / lock.name
    try:
        scene = export_scene(
            lock, out_dir, bundle_assets=not args.no_bundle, task=task, object_size=args.object_size, lock_dir=server.LOCK_DIR
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    arm = out_dir / f"{lock.name}.xml"
    p = lock.params
    print(f"exported lock {lock.name} ({lock.dataset}) -> {out_dir}/")
    print(f"  {arm.name:<40} calibrated arm MJCF")
    print(f"  {sidecar_path(arm).name:<40} params incl. latency_frames={p.latency_frames}, unit map")
    extra = " + cube and place marker" if task else ""
    if lock.scene:
        print(f"  {SCENE_FILENAME:<40} arm + table at z={lock.scene.get('table_z', float('nan')):.3f} m + camera backdrops" + extra)
        if (out_dir / "backdrops").is_dir():
            print(f"  {'backdrops/':<40} camera background images (PNG textures)")
    else:
        print(f"  {SCENE_FILENAME:<40} arm + floor/light" + extra)
    if task is not None:
        print(f"  {TASK_FILENAME:<40} pick/place/start-pose distribution ({task.name})")
    if args.no_bundle:
        print("  meshdir points at the antares package assets (absolute path)")
    else:
        print(f"  {'assets/':<40} STL meshes")
    fps = getattr(args, "fps", None)
    # A lock on another model than the SO-101 gets the MJCF and a note in
    # place of the framework configs, as the zip download does.
    note = write_unsupported_note(lock, out_dir) if targets else None
    if note is not None:
        skipped = ", ".join(TARGET_LABELS[t] for t in targets)
        print(f"  {note.name:<40} why the {skipped} configs were not written")
        print(f"note: {note.read_text().strip()}")
        targets = []
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


def cmd_scene_fit(args: argparse.Namespace) -> int:
    from antares import server

    lock = args.lock or server._slug(args.repo_id)
    try:
        ds = server.get_dataset(args.repo_id)
        fit, path = server.run_scene_fit(ds, lock, args.camera or None)
    except Exception as e:  # noqa: BLE001
        print(f"error: {getattr(e, 'detail', e)}", file=sys.stderr)
        return 1
    t, ws = fit.table, fit.workspace
    fz, ez = t.fingertip_z, t.episode_min_z
    print(f"scene for lock {lock} from {ds.repo_id}: {fit.n_episodes} episodes, {fit.n_frames} frames")
    print(f"  table z {t.z * 1000:.1f} mm{' (horizontal)' if t.horizontal else ''}: fingertips at "
          f"{t.n_on_table} of {t.n_grasps} grasps sat at {t.z_grasp * 1000:.1f} mm "
          f"(p10 {fz.get('p10', float('nan')) * 1000:.1f}, median {fz.get('median', float('nan')) * 1000:.1f}, "
          f"p90 {fz.get('p90', float('nan')) * 1000:.1f}), minus object half-height {t.object_half_height * 1000:.0f} mm")
    print(f"  cross-check: lowest fingertip over all frames {t.min_fingertip_z * 1000:.1f} mm; "
          f"per-episode minimum median {ez.get('median', float('nan')) * 1000:.1f} mm, spread {ez.get('spread_mm', float('nan')):.1f} mm")
    print(f"  plane through the on-table grasps tilts {t.tilt_deg:.2f} deg from horizontal")
    print(f"  workspace {ws.size[0] * 100:.0f} x {ws.size[1] * 100:.0f} x {ws.size[2] * 100:.0f} cm: "
          f"x [{ws.min[0]:.3f}, {ws.max[0]:.3f}] y [{ws.min[1]:.3f}, {ws.max[1]:.3f}] z [{ws.min[2]:.3f}, {ws.max[2]:.3f}] m")
    print(f"  {len(fit.grasp_points)} grasp points, {len(fit.release_points)} release points")
    if fit.backdrops:
        rows = [
            [name, f"{b.plane_in_region:.2f}", f"{b.region_in_plane:.2f}", f"{b.iou:.2f}", f"{b.region_frac:.2f}",
             str(server.LOCK_DIR / b.path)]
            for name, b in sorted(fit.backdrops.items())
        ]
        print("  table plane vs the backdrop's dominant colour region:")
        print("  " + _table(["camera", "plane in region", "region in plane", "IoU", "region/image", "backdrop"], rows).replace("\n", "\n  "))
    else:
        print("  no fitted cameras for this lock, so no backdrops (run camera-fit first)")
    print(f"saved {path}")
    return 0


def _fmt_or_dash(v, digits: int = 2, suffix: str = "") -> str:
    return "-" if v is None else f"{v:.{digits}f}{suffix}"


def _policy_train_table(meta: dict) -> str:
    m = meta["metrics"]
    rows = []
    for label, d in (
        ("policy", m["policy"]),
        ("hold state", m["baselines"]["hold_state"]),
        ("repeat command", m["baselines"]["repeat_command"]),
    ):
        rows.append([
            label,
            f"{d['train']['rmse_deg']:.3f}",
            f"{d['eval']['rmse_deg']:.3f}",
            f"{d['eval']['first_rmse_deg']:.3f}",
            f"{d['eval']['mse_units']:.2f}",
        ])
    return _table(
        ["predictor", "train rmse (deg)", "held-out rmse (deg)", "held-out first action (deg)", "held-out mse (units^2)"],
        rows,
    )


def cmd_policy_train(args: argparse.Namespace) -> int:
    from fastapi import HTTPException

    from antares import server

    ds = _load_dataset(args.repo_id)
    if ds is None:
        return 2

    def progress(p) -> None:
        if p.epoch == 1 or p.epoch % 10 == 0 or p.epoch == p.epochs:
            print(
                f"  epoch {p.epoch}/{p.epochs}: train {p.train_mse:.4f}  eval {p.eval_mse:.4f}"
                f"  (best {p.best_eval_mse:.4f} at {p.best_epoch}, {p.elapsed_s:.0f}s)",
                file=sys.stderr,
            )

    try:
        lock = server.load_lock_dict(args.lock)
        meta = server.run_policy_training(
            ds, lock, epochs=args.epochs, name=args.name, seed=args.seed, on_progress=progress
        )
    except HTTPException as e:
        print(f"error: {_detail(e)}", file=sys.stderr)
        return 2
    m = meta["metrics"]
    feats = ", ".join(k for k, v in meta["features"].items() if v) or "states only"
    print(
        f"{ds.repo_id}: chunked policy (K={meta['k']} states + {feats} -> H={meta['h']} actions, "
        f"{meta['n_params']} params) on lock {lock['name']}'s split: "
        f"{len(meta['train_episodes'])} train / {len(meta['eval_episodes'])} held-out episodes"
        + (f", {len(meta['dropped_episodes'])} without a grasp dropped" if meta["dropped_episodes"] else "")
    )
    print(
        f"{meta['n_train_windows']} / {meta['n_eval_windows']} windows, {m['epochs_run']} epochs "
        f"(best at {m['best_epoch']}), {meta['elapsed_s']:.1f}s, obs noise {meta['obs_noise_units']} units"
    )
    print(_policy_train_table(meta))
    print(
        "held-out: beats hold-state " + ("yes" if m["beats_hold_state"] else "NO")
        + ", beats repeat-command " + ("yes" if m["beats_repeat_command"] else "NO")
        + " (chunk RMS over the body joints; the first-action column is what a receding horizon executes)"
    )
    print(f"saved {meta['path']}")
    return 0


def _shadow_tables(out: dict) -> str:
    s = out["summary"]
    rows = []
    for name, v in s["variants"].items():
        rows.append([
            name,
            _fmt_or_dash(v["sim_vs_real_deg"], 3),
            _fmt_or_dash(v.get("divergence_deg"), 3),
            _fmt_or_dash(v["grasp_error_cm"]),
            _fmt_or_dash(v.get("grasp_divergence_cm")),
            _fmt_or_dash(v["replay_vs_real_deg"], 3),
        ])
    head = _table(
        ["sim", "policy: sim vs real (deg)", "vs default sim (deg)", "grasp-point err (cm)", "grasp vs default (cm)", "logged cmds: sim vs real (deg)"],
        rows,
    )
    names = list(s["variants"])
    ep_rows = []
    for e in out["episodes"]:
        ep_rows.append(
            [str(e["episode"]), str(e["n_frames"]), f"{e['action_error_deg']:.2f}"]
            + [f"{e['sim_vs_real_deg'][n]:.2f}" for n in names]
            + [f"{e['divergence_deg'][n]:.2f}" for n in names if n in e["divergence_deg"]]
        )
    per_ep = _table(
        ["episode", "frames", "policy vs logged (deg)"]
        + [f"{n}: sim vs real" for n in names]
        + [f"{n}: vs default" for n in names if n != out["baseline"]],
        ep_rows,
    )
    worst = []
    for n in names:
        for w in s["variants"][n].get("worst_frames", [])[:3]:
            worst.append(f"{n}: episode {w['episode']} frame {w['frame']} ({w['deg']:.1f} deg)")
    lines = [
        f"shadow (open loop): the policy's commands on the real observations, rolled through each sim; "
        f"{s['n_episodes']} held-out episodes, {s['n_frames']} frames; policy vs logged commands {s['action_error_deg']:.3f} deg",
        head,
        per_ep,
    ]
    if worst:
        lines.append("largest default-vs-calibrated disagreement: " + "; ".join(worst))
    if s.get("skipped_episodes"):
        lines.append(f"skipped (no grasp point): {s['skipped_episodes']}")
    return "\n".join(lines)


def _closedloop_tables(out: dict) -> str:
    s = out["summary"]
    rows = []
    for name, v in s["variants"].items():
        rows.append([
            name,
            f"{v['trajectory_rms_deg']:.2f}",
            _fmt_or_dash(v["approach_rms_deg"]),
            _fmt_or_dash(v["grasp_error_cm"]),
            _fmt_or_dash(v["min_tip_distance_cm"]),
            f"{v['n_success']}/{v['n_judged']}",
            f"{v['n_closed']}/{v['n_judged']}",
            _fmt_or_dash(v["close_frame_offset"], 1),
        ])
    head = _table(
        ["sim", "trajectory rms (deg)", "approach rms (deg)", "grasp-point err (cm)", "closest approach (cm)",
         f"within {out['success_cm']:.0f} cm", "closed gripper", "close offset (frames)"],
        rows,
    )
    names = out["run_names"]
    by_ep: dict[int, dict[str, dict]] = {}
    for r in out["runs"]:
        by_ep.setdefault(r["episode"], {})[r["variant"]] = r
    ep_rows = []
    for ep, runs in sorted(by_ep.items()):
        row = [str(ep)]
        for n in names:
            r = runs.get(n)
            if r is None:
                row.append("-")
            elif r["grasp_error_cm"] is None:
                row.append("never closed")
            else:
                row.append(f"{r['grasp_error_cm']:.1f} cm" + (" ok" if r["success"] else ""))
        ep_rows.append(row)
    per_ep = _table(["episode"] + [f"{n}: grasp err" for n in names], ep_rows)
    lines = [
        f"closed loop: the policy drives each sim from the real initial state, same goal and clock; "
        f"{s['n_episodes']} held-out episodes (success = fingertip within {out['success_cm']:.0f} cm of the real grasp point when the gripper closes)",
        head,
        per_ep,
    ]
    if s.get("skipped_episodes"):
        lines.append(f"skipped (no grasp point): {s['skipped_episodes']}")
    return "\n".join(lines)


def cmd_policy_evaluate(args: argparse.Namespace) -> int:
    from fastapi import HTTPException

    from antares import server

    ds = _load_dataset(args.repo_id)
    if ds is None:
        return 2
    modes = ["shadow", "closedloop"] if args.mode == "both" else [args.mode]
    robots = [r.strip() for r in args.robots.split(",") if r.strip()] if args.robots else None
    try:
        lock = server.load_lock_dict(args.lock)
        for i, mode in enumerate(modes):
            out = server.run_policy_evaluation(
                ds, lock, args.policy, mode, robots, save=not args.no_save,
                on_progress=lambda ev: print(f"  {ev['mode']}: episode {ev['episode']} ({ev['i']}/{ev['n']})", file=sys.stderr),
            )
            if i:
                print()
            print(f"{ds.repo_id}, lock {lock['name']}, policy {out['policy']} ({out['note']})")
            print(_shadow_tables(out) if mode == "shadow" else _closedloop_tables(out))
            if out.get("missing_locks"):
                print(f"not evaluated (no such lock): {', '.join(out['missing_locks'])}")
            if out.get("path"):
                print(f"saved {out['path']}")
    except HTTPException as e:
        print(f"error: {_detail(e)}", file=sys.stderr)
        return 2
    return 0


def _cm(v, fmt="{:+.1f}") -> str:
    return "-" if v is None else ", ".join(fmt.format(100 * x) for x in v)


def _payload_lines(est: dict) -> list[str]:
    """The payload summary as printed by ``objects-fit`` and ``payload``."""
    lines = [f"payload from the residual: {est.get('verdict', '')}"]
    rows = []
    for j in ("shoulder_lift", "elbow_flex"):
        pd, md, mj = est["payload_deg"].get(j), est["motion_deg"].get(j), est["mass_by_joint"].get(j)
        if not pd:
            continue
        step, floor = est.get("mass_per_step_kg", {}).get(j), est.get("mass_noise_floor_kg", {}).get(j)
        rows.append([
            j, f"{est['kp'][j]:.1f}", f"{(est.get('lever_m') or {}).get(j) or float('nan'):.3f}",
            f"{pd['median']:+.3f} ({pd['iqr']:.3f})", f"{md['median']:+.3f} ({md['iqr']:.3f})",
            f"{1000 * mj['median']:+.0f} ({1000 * mj['iqr']:.0f})" if mj else "-",
            f"{1000 * step:.0f}" if step is not None else "-", f"{1000 * floor:.0f}" if floor is not None else "-",
        ])
    if rows:
        lines.append(_table(["joint", "kp (N*m/rad)", "lever (m)", "extra deflection deg (IQR)", "motion term deg (IQR)",
                             "mass g (IQR)", "g per encoder step", "noise floor g"], rows))
    m = est.get("mass")
    if m:
        lines.append(f"combined over {m['n']} picks: median {1000 * m['median']:+.0f} g, IQR {1000 * m['iqr']:.0f} g "
                     f"(q1 {1000 * m['q1']:+.0f}, q3 {1000 * m['q3']:+.0f})")
    return lines


def cmd_objects_fit(args: argparse.Namespace) -> int:
    from antares import server

    lock = args.lock or server._slug(args.repo_id)
    try:
        ds = server.get_dataset(args.repo_id)
    except Exception as e:  # noqa: BLE001
        print(f"error: {getattr(e, 'detail', e)}", file=sys.stderr)
        return 1
    eps = [int(x) for x in args.episodes.split(",")] if args.episodes else None

    def on_progress(ev: dict) -> None:
        if ev.get("stage") == "prior":
            pr = ev["prior"]
            print(f"  object colour from the first frames: rgb {pr['rgb']}, {pr['area_px']:.0f} px at rest ({pr['n_samples']} samples)")
        elif ev.get("stage") == "track":
            t = ev["track"]
            d = t.get("vision_vs_fk_m")
            print(f"  episode {t['episode']:3d}: seen in {t['n_visible']}/{t['n_frames']} frames, picks {t['picks']}, "
                  f"vision vs FK {'-' if d is None else f'{100 * d:.1f} cm'} {t['note']}  [{ev['elapsed_s']:.0f}s]")
        elif ev.get("stage") == "cross":
            print("  cross-camera check done")
        elif ev.get("stage") == "payload" and ev["i"] % 10 == 0:
            print(f"  payload: {ev['i']}/{ev['n']} episodes replayed")

    print(f"tracking the object in {ds.repo_id} for lock {lock}" + (f" (camera {args.camera})" if args.camera else ""))
    try:
        fit, tracks, path = server.run_objects_fit(ds, lock, args.camera or None, episodes=eps, payload=not args.no_payload, on_progress=on_progress)
    except Exception as e:  # noqa: BLE001
        print(f"error: {getattr(e, 'detail', e)}", file=sys.stderr)
        return 1
    print(f"objects for lock {lock} from {ds.repo_id} (camera {fit.camera}): {fit.n_tracked} of {fit.n_episodes} episodes tracked in {fit.elapsed_s:.0f}s")
    print(f"  {fit.description}")
    fp = fit.footprint_m
    if fp and fp["long"] and fp["short"]:
        print(f"  footprint {100 * fp['long']['median']:.1f} x {100 * fp['short']['median']:.1f} cm "
              f"(IQR {100 * fp['long']['iqr']:.1f} / {100 * fp['short']['iqr']:.1f} cm over {fp['long']['n']} episodes; "
              f"height {200 * fit.object_half_height:.1f} cm assumed)")
    rows = []
    for label, c in (("start (vision)", fit.start), ("end (vision)", fit.end)):
        if c:
            rows.append([label, str(c["n"]), _cm(c["mean"]), _cm(c["std"], "{:.1f}"), _cm(c["min"]), _cm(c["max"])])
    if rows:
        print("  object centre on the table plane, base frame, cm (x, y, z):")
        print("  " + _table(["", "n", "mean", "std", "min", "max"], rows).replace("\n", "\n  "))
    v = fit.vision_vs_fk_m
    if v:
        print(f"  vision start pose vs FK fingertip at the grasp: median {100 * v['median']:.1f} cm, "
              f"IQR {100 * v['iqr']:.1f} cm, worst {100 * v['max']:.1f} cm over {v['n']} episodes")
        if fit.vision_vs_fk_offset_m and fit.vision_vs_fk_centred_m:
            print(f"    of which a fixed offset (vision - FK) of {_cm(fit.vision_vs_fk_offset_m)} cm; "
                  f"{100 * fit.vision_vs_fk_centred_m['median']:.1f} cm median spread around it")
    e = fit.end_vs_fk_m
    if e:
        print(f"  vision end pose vs FK fingertip at the release: median {100 * e['median']:.1f} cm (the object drops after the release)")
    cc = fit.cross_camera
    if cc:
        if cc.get("n"):
            t, h = cc["triangulated_vs_plane_m"], cc["triangulated_height_above_plane_m"]
            print(f"  cross-camera ({cc['primary']} x {cc['other']}): triangulated point vs plane intersection median "
                  f"{100 * t['median']:.1f} cm (IQR {100 * t['iqr']:.1f}) over {cc['n']} frames; "
                  f"triangulated height above the plane median {100 * h['median']:+.1f} cm; ray gap median {100 * cc['ray_gap_m']['median']:.1f} cm")
        else:
            print(f"  cross-camera: {cc.get('note', 'not run')}")
    if fit.payload:
        for line in _payload_lines(fit.payload):
            print("  " + line.replace("\n", "\n  "))
    bad = [t["episode"] for t in fit.tracks if t.get("start_pose") is None]
    if bad:
        print(f"  episodes where the object was not found before the grasp: {_episode_list(bad)}")
    print(f"saved {path}")
    return 0


def cmd_payload(args: argparse.Namespace) -> int:
    from antares import server
    from antares.payload import estimate_payload
    from antares.sim import Arm

    lock_name = args.lock or server._slug(args.repo_id)
    try:
        ds = server.get_dataset(args.repo_id)
        lock = server.load_lock(lock_name)
    except Exception as e:  # noqa: BLE001
        print(f"error: {getattr(e, 'detail', e)}", file=sys.stderr)
        return 1
    arm = Arm(lock.params, robot=lock.robot)
    eps = [int(x) for x in args.episodes.split(",")] if args.episodes else None
    est = estimate_payload(ds, arm, lock_name, episodes=eps, window_s=args.window)
    print(f"payload for lock {lock_name} on {ds.repo_id}: {est.n_ok} of {est.n_episodes} carried picks measured "
          f"(windows of {est.window_s:g} s either side of each pick and release)")
    rows = []
    for e in est.episodes:
        sl, el = e.joints.get("shoulder_lift"), e.joints.get("elbow_flex")
        rows.append([
            str(e.episode), f"{e.grasp}-{e.release}",
            f"{sl.payload_deg:+.3f}" if sl else "-", f"{el.payload_deg:+.3f}" if el else "-",
            f"{1000 * sl.mass_kg:+.0f}" if sl and sl.mass_kg == sl.mass_kg else "-",
            f"{1000 * el.mass_kg:+.0f}" if el and el.mass_kg == el.mass_kg else "-",
            f"{1000 * e.mass_kg:+.0f}" if e.ok else e.note,
        ])
    print(_table(["episode", "carry frames", "shoulder deg", "elbow deg", "shoulder g", "elbow g", "mass g"], rows))
    for line in _payload_lines(est.to_dict()):
        print(line)
    print("assumptions: " + "; ".join(est.assumptions))
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(__import__("json").dumps(est.to_dict(), indent=2) + "\n")
        print(f"saved {args.out}")
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
        elif t == "rejected":
            print(f"    rejected {ev['dataset']}: {ev['reason']}", flush=True)
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


def cmd_factors(args: argparse.Namespace) -> int:
    """What the fit can account for, and what it can't yet."""
    from antares import factors

    if not _load_factor_files(args.file):
        return EXIT_USAGE
    d = factors.describe()
    if args.json:
        print(json.dumps(d, indent=2))
        return 0
    rows = []
    for f in d["factors"]:
        if args.stage and args.stage not in f["stages"]:
            continue
        lo, hi = f["bounds"]
        rows.append([
            f["name"], f["label"], ",".join(f["stages"]), f"{lo:g}..{hi:g} {f['unit']}".strip(),
            f["identifiable"], f["origin"],
        ])
    print(f"factors the fit can account for (registry v{d['version']}):")
    print(_table(["name", "what", "stages", "range", "pinned by data", "source"], rows))
    if args.verbose:
        for f in d["factors"]:
            print(f"\n{f['name']}: {f['explains']}\n  needs: {f['needs']}\n  evidence: {f['evidence']}")
    print()
    print("known effects not modelled yet:")
    print(_table(["name", "what", "shows up as"], [[u["name"], u["label"], u["symptom"]] for u in d["not_modelled"]]))
    print()
    print("fit a different set with `antares calibrate --factors +name,-name`; add your own with --factors-file "
          "(see examples/factors.toml)")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from antares.robots import REGISTRY

    rows = []
    for r in REGISTRY.values():
        d = r.to_dict()
        rows.append([r.name, d["label"], str(r.n_joints), ",".join(r.grippers) or "-",
                     ",".join(r.unit_kinds()[:1]) + ("..." if len(set(r.unit_kinds())) > 1 else ""),
                     d["source"], "yes" if d["fetched"] else "no"])
    if args.json:
        print(json.dumps([r.to_dict() for r in REGISTRY.values()], indent=2))
        return 0
    print(_table(["name", "robot", "columns", "grippers", "units", "model", "on disk"], rows))
    return 0


def cmd_fetch_model(args: argparse.Namespace) -> int:
    from antares.menagerie import fetch
    from antares.robots import REGISTRY, get_robot

    names = [n for n, r in REGISTRY.items() if r.menagerie is not None] if args.robot == "all" else [args.robot]
    for name in names:
        try:
            r = get_robot(name)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return EXIT_USAGE
        if r.menagerie is None:
            print(f"{name}: ships with antares ({r.model_path})")
            continue
        path = fetch(r.menagerie, force=args.force)
        print(f"{name}: {path}")
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    from antares.catalog import entries

    rows = entries()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(_table(["dataset", "robot", "episodes", "fps", "units", "checked"],
                 [[e["repo_id"], e["robot"], str(e["episodes"]), _fmt(e["fps"]), e["units"], e["checked"] or "-"]
                  for e in rows]))
    print("\ncalibrate one with `antares calibrate <dataset>`; the catalog picks the model.")
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
    p.add_argument(
        "--offline",
        action="store_true",
        help="never contact the Hugging Face Hub; read datasets and videos from the local cache only",
    )
    p.set_defaults(func=cmd_serve)

    dataset_help = "Hub dataset id, upload/<name>, or a LeRobot dataset directory on disk"
    from antares.robots import REGISTRY

    robot_help = (f"sim model from the registry ({', '.join(REGISTRY)}); "
                  "default: the dataset's catalog entry or family (see `antares robots`)")

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
        "--fit-gravity",
        action="store_true",
        help="also fit gravity_scale, the arm's gravity load (sets how much it sags per pose)",
    )
    p.add_argument(
        "--fit-deadband",
        action="store_true",
        help="also fit the servo deadband (firmware deadzone + gear backlash), per joint with --per-joint",
    )
    p.add_argument(
        "--servo",
        choices=("position", "sts3215"),
        default="position",
        help="actuator model: MuJoCo's position actuator (default) or the STS3215's firmware and "
        "motor with BAM's bench-identified constants, fitting gain ratio, friction, inertia, "
        "rate limit and latency instead of kp/kv",
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
    p.add_argument(
        "--factors",
        default=None,
        metavar="LIST",
        help="factors the shared fit accounts for: '+gravity,-kp' edits the defaults, "
        "'kp,kv,latency_frames' replaces them (see `antares factors`)",
    )
    p.add_argument(
        "--per-joint-factors",
        default=None,
        metavar="LIST",
        help="same, for the --per-joint stage (default: the shared ones plus forcerange)",
    )
    p.add_argument(
        "--factors-file",
        action="append",
        metavar="TOML",
        help="register the [[factor]] entries of a TOML file first; repeatable",
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
    p.add_argument("--explain-evaluations", type=int, default=300, metavar="N",
                   help="refit budget for --explain (default 300, at least 20)")
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


    p = sub.add_parser("noise", help="measure joint-reading noise while the arm is parked")
    p.add_argument("repo_id")
    p.add_argument("--name", default=None, help="noise model name (default: dataset slug)")
    p.add_argument("--out", default=None, help="output path (default: locks/<name>.noise.json)")
    p.add_argument("--no-save", action="store_true")
    p.set_defaults(func=cmd_noise)

    p = sub.add_parser("task", help="where objects were picked and placed, from the gripper and FK")
    p.add_argument("repo_id")
    p.add_argument("--name", default=None, help="task name (default: dataset slug)")
    p.add_argument("--held-gap", type=float, default=3.0,
                   help="jaw this far above an empty close (0..100 units) counts as holding")
    p.add_argument("--out", default=None, help="output path (default: locks/<name>.task.json)")
    p.add_argument("--no-save", action="store_true")
    p.set_defaults(func=cmd_task)

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

    p = sub.add_parser("sag", help="measure how a lock's leftover error depends on gravity load (pose)")
    p.add_argument("repo_id")
    p.add_argument("--lock", required=True)
    p.add_argument("--all", action="store_true", help="use every episode, not just the lock's held-out set")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_sag)
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

    p = sub.add_parser("factors", help="list what the fit can account for, and what it can't yet")
    p.add_argument("--stage", choices=("shared", "per_joint", "sensor"), default=None)
    p.add_argument("--file", action="append", metavar="TOML", help="also load the factors in a TOML file")
    p.add_argument("-v", "--verbose", action="store_true", help="print what each factor explains and the evidence")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_factors)

    p = sub.add_parser("models", help="list the robot models antares can simulate")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("fetch-model", help="download a robot's model from MuJoCo Menagerie (otherwise fetched on first use)")
    p.add_argument("robot", help="registry name, or 'all'")
    p.add_argument("--force", action="store_true", help="download again")
    p.set_defaults(func=cmd_fetch_model)

    p = sub.add_parser("datasets", help="list public datasets of other robots to test on")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_datasets)

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
    p.add_argument("--task", default=None,
                   help="task name or .task.json path (from `antares task`): add a cube and place marker")
    p.add_argument("--object-size", type=float, default=0.03, help="cube edge length in metres (with --task)")
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

    p = sub.add_parser("camera-delay", help="how many frames a camera's video lags the joint log")
    p.add_argument("repo_id", help="Hub dataset id or a LeRobot dataset directory on disk")
    p.add_argument("--camera", required=True, help="camera name, e.g. up (observation.images.up)")
    p.add_argument("--episodes", type=int, default=6, help="episodes measured, spread over the dataset")
    p.add_argument("--max-lag", type=int, default=10, help="largest lag tried either way, frames")
    p.add_argument("--out", default=None, help="also save the result as JSON here")
    p.set_defaults(func=cmd_camera_delay)

    p = sub.add_parser("camera-list", help="list the camera fits saved for a lock")
    p.add_argument("lock")
    p.set_defaults(func=cmd_camera_list)

    p = sub.add_parser("scene-fit", help="recover the table, workspace and camera backdrops from a dataset's logs")
    p.add_argument("repo_id", help="Hub dataset id or a LeRobot dataset directory on disk")
    p.add_argument("--lock", default=None, help="lock to attach the scene to (default: dataset slug)")
    p.add_argument("--camera", action="append", default=[], help="only these fitted cameras (repeatable; default all)")
    p.set_defaults(func=cmd_scene_fit)

    p = sub.add_parser(
        "policy",
        help="a learned policy in the loop: train on a lock's split, judge it in the default and the calibrated sim",
    )
    psub = p.add_subparsers(dest="policy_cmd", required=True)
    t = psub.add_parser("train", help="train the state-based chunked policy on the lock's training episodes (needs torch)")
    t.add_argument("repo_id", help=dataset_help)
    t.add_argument("--lock", required=True, help="lock whose train / held-out split to use")
    t.add_argument("--epochs", type=int, default=200, help="maximum epochs (early-stopped on the held-out split)")
    t.add_argument("--seed", type=int, default=0, help="initialization and batch order")
    t.add_argument("--name", default=None, help="save as locks/<name>.policy.pt (default: the lock's name)")
    t.set_defaults(func=cmd_policy_train)
    e = psub.add_parser("evaluate", help="run a saved policy on the held-out episodes in each sim and print the tables")
    e.add_argument("repo_id", help=dataset_help)
    e.add_argument("--lock", required=True, help="the lock (its held-out split and its calibrated sim)")
    e.add_argument("--policy", default=None, help="saved policy name (default: the lock's)")
    e.add_argument("--mode", choices=["shadow", "closedloop", "both"], default="both")
    e.add_argument(
        "--robots", default=None,
        help="comma-separated sims: default, lock, perjoint (= <lock>_perjoint) or lock names (default: all three)",
    )
    e.add_argument("--no-save", action="store_true", help="do not write locks/<lock>.policy.<mode>.json")
    e.set_defaults(func=cmd_policy_evaluate)


    p = sub.add_parser("objects-fit", help="find and track the manipulated object in a dataset's video; mass from the residual")
    p.add_argument("repo_id", help="Hub dataset id or a LeRobot dataset directory on disk")
    p.add_argument("--lock", default=None, help="lock with a camera fit and a scene fit (default: dataset slug)")
    p.add_argument("--camera", default=None, help="camera to track in (default: the lock's best-fitted camera)")
    p.add_argument("--episodes", default=None, help="comma-separated episode indices (default: all)")
    p.add_argument("--no-payload", action="store_true", help="skip the residual-based mass estimate")
    p.set_defaults(func=cmd_objects_fit)

    p = sub.add_parser("payload", help="payload mass per episode from the residual while the object is carried")
    p.add_argument("repo_id", help="Hub dataset id or a LeRobot dataset directory on disk")
    p.add_argument("--lock", default=None, help="lock whose calibrated sim replays the episodes (default: dataset slug)")
    p.add_argument("--episodes", default=None, help="comma-separated episode indices (default: all)")
    p.add_argument("--window", type=float, default=0.6, help="seconds of residual either side of each pick/release")
    p.add_argument("--out", default=None, help="also save the estimate as JSON here")
    p.set_defaults(func=cmd_payload)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "robot", None) and getattr(args, "repo_id", None):
        # Validate and replay the dataset on the model asked for.
        from antares.catalog import assign

        assign(args.repo_id, args.robot)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
