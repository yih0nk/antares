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
    params = cmap = None
    if args.lock:
        lock = server.load_lock_dict(args.lock)
        params = server.params_from_dict(lock["params"])
        cmap = server.command_map_from_lock(lock)
    out = server.replay_episode(ds, args.episode, params, cmap)
    gap = out["gap"]
    label = f"lock {args.lock}" if args.lock else "default params"
    print(f"{ds.repo_id} episode {out['episode']} ({len(out['real_q'])} frames, {label})")
    print(_joint_table(gap["per_joint_deg"], gap["overall_deg"]))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from antares import server

    ds = server.get_dataset(args.repo_id)
    stages = f"up to {args.max_evaluations} shared evaluations"
    if args.per_joint:
        stages += f" + {args.per_joint_evaluations} per-joint evaluations"
    if args.fit_sensor:
        stages += " per round, fitting the sensor calibration"
    print(f"calibrating {ds.repo_id}: {args.n_train} train / {args.n_eval} eval episodes, {stages}")
    last = {"n": 0, "stage": None}

    def on_progress(ev) -> None:
        n = ev.evaluation
        stage = getattr(ev, "stage", "shared")
        if stage != last["stage"]:
            last["stage"] = stage
            print(f"  -- {stage} stage --", flush=True)
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
        on_progress=on_progress,
        fit_sensor=args.fit_sensor,
        sensor_mode=args.sensor_mode,
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
        print(f"  {s['name']}: {s['n_evaluations']} evaluations, train {s['train_score']:.3f} deg")
    if lock.get("command_map"):
        cm = lock["command_map"]
        joints = list(before) or [str(i) for i in range(len(cm["gain"]))]
        print()
        print("sensor calibration (command = gain * logged + bias, dataset units):")
        print(_table(["joint", "gain", "bias"],
                     [[j, f"{g:.4f}", f"{b:+.3f}"] for j, g, b in zip(joints, cm["gain"], cm["bias"])]))
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

    ds = server.get_dataset(args.repo_id)
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

    if res.verdict == "ok":
        return EXIT_PASS
    if res.verdict == "drifted":
        return EXIT_DRIFT
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
        cmap = server.command_map_from_lock(lock) if lock else None
        if cmap is not None:  # back to the raw units the arm is commanded in
            cmds = cmap.invert(cmds)
        np.save(args.out, cmds)
        print(f"saved {args.method} commands ({cmds.shape[0]}x{cmds.shape[1]}, dataset units) -> {args.out}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from antares import server, synth

    if args.lock:
        params = server.params_from_dict(server.load_lock_dict(args.lock)["params"])
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
    p.add_argument("repo_id")
    p.add_argument("--lock", required=True, help="lock whose parameters seed each window")
    p.add_argument("--window", type=int, default=10, help="episodes per window")
    p.add_argument("--step", type=int, default=5, help="episodes between window starts")
    p.add_argument("--max-evaluations", type=int, default=60, help="CMA-ES budget per window")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_drift)


    p = sub.add_parser("dr", help="derive domain-randomization ranges from a lock's data")
    p.add_argument("repo_id")
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
    p.add_argument("--explain", action="store_true",
                   help="on drift, refit and name the parameter change that explains it")
    p.add_argument("--explain-evaluations", type=int, default=240, metavar="N",
                   help="refit budget for --explain (default 240)")
    p.set_defaults(func=cmd_check)


    p = sub.add_parser("events", help="detect contact events from the real-sim residual")
    p.add_argument("repo_id")
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
    p.add_argument("repo_id")
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
    p.add_argument("--lock", default=None, help="start from a lock's params (default: MJCF defaults)")
    p.add_argument("--wear", default="", metavar="CHANGES",
                   help='factors, and a frame shift for latency: "kp=0.7,frictionloss=2,latency_frames=1"')
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--noise", type=float, default=0.2, help="sensor noise, dataset units (default 0.2)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("export", help="write a lock as a standalone MJCF (+ scene, sidecar, meshes)")
    p.add_argument("name", help="lock name (see `antares locks`)")
    p.add_argument("-o", "--output", default=None, help="output directory (default: export/<name>)")
    p.add_argument(
        "--no-bundle",
        action="store_true",
        help="do not copy the STL meshes; point meshdir at the package assets instead",
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
