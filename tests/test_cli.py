from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from antares import cli
from antares.data import Dataset, Episode

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@pytest.fixture(autouse=True)
def _locks_in_tmp(monkeypatch, tmp_path):
    """`antares check` records into the lock dir; no test may write the repo's locks/."""
    from antares import server

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path / "locks")


def _dataset(repo_id="fake/robot", n=3, T=10):
    eps = []
    for i in range(n):
        a = np.linspace(0, 10, T)[:, None].repeat(6, axis=1)
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=a + 0.5, action=a))
    return Dataset(repo_id=repo_id, robot_type="so101_follower", fps=30.0,
                   joint_names=JOINTS, episodes=tuple(eps))


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("serve", "inspect", "replay", "calibrate", "cross", "drift", "dr", "locks", "export", "check", "events", "compensate", "fleet", "synth"):
        assert cmd in out


def test_help_does_not_import_heavy_modules():
    """``antares --help`` must not pull in uvicorn, MuJoCo, CMA-ES or FastAPI."""
    import subprocess

    code = (
        "import sys, antares.cli as c\n"
        "try:\n    c.main(['--help'])\nexcept SystemExit:\n    pass\n"
        "heavy = {'uvicorn', 'mujoco', 'cma', 'fastapi', 'antares.server', 'antares.sim'}\n"
        "print('HEAVY=' + repr(sorted(heavy & set(sys.modules))))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "HEAVY=[]", out.stdout


def test_no_command_is_usage_error():
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2


def _info(**over):
    names = [f"{j}.pos" for j in JOINTS]
    base = {"codebase_version": "v3.0", "robot_type": "so101_follower", "fps": 30, "total_episodes": 20,
            "features": {"action": {"names": names}, "observation.state": {"names": names}}}
    base.update(over)
    return base


def test_inspect(monkeypatch, capsys):
    import antares.data
    from antares import server

    monkeypatch.setattr(server, "fetch_hub_info", lambda repo_id: _info())
    monkeypatch.setattr(antares.data, "load_dataset", lambda repo_id, cache_dir=None: _dataset(repo_id, n=20))
    assert cli.main(["inspect", "fake/robot"]) == 0
    out = capsys.readouterr().out
    assert "fake/robot: so101_follower, 20 episodes, 200 frames @ 30 fps" in out
    assert "shoulder_pan" in out and "gripper" in out
    assert "sim model: so101" in out and "verdict: ok (family so101)" in out


def test_inspect_rejects_without_loading(monkeypatch, capsys):
    import antares.data
    from antares import server

    monkeypatch.setattr(server, "fetch_hub_info", lambda repo_id: _info(robot_type="koch", fps=60))
    monkeypatch.setattr(antares.data, "load_dataset", lambda *a, **k: pytest.fail("must not download a rejected dataset"))
    assert cli.main(["inspect", "someone/koch_arm"]) == 1
    out = capsys.readouterr().out
    assert "verdict: REJECTED" in out and "reason:  robot_type 'koch'" in out and "fps 60" in out


def test_inspect_warns_but_loads(monkeypatch, capsys):
    import antares.data
    from antares import server

    monkeypatch.setattr(server, "fetch_hub_info", lambda repo_id: _info(total_episodes=8))
    monkeypatch.setattr(antares.data, "load_dataset", lambda repo_id, cache_dir=None: _dataset(repo_id, n=8))
    assert cli.main(["inspect", "fake/robot"]) == 0
    out = capsys.readouterr().out
    assert "warning: only 8" in out


def test_inspect_hub_error_exits_two(monkeypatch, capsys):
    from antares import server

    def boom(repo_id):
        raise RuntimeError("offline")

    monkeypatch.setattr(server, "fetch_hub_info", boom)
    assert cli.main(["inspect", "fake/robot"]) == 2
    assert "error:" in capsys.readouterr().err


def test_replay_rejected_dataset_exits_two(monkeypatch, capsys):
    from antares import server

    monkeypatch.setattr(server, "_DATASETS", {})
    monkeypatch.setattr(server, "fetch_hub_info", lambda repo_id: _info(robot_type="koch"))
    assert cli.main(["replay", "someone/koch", "--episode", "0"]) == 2
    err = capsys.readouterr().err
    assert "cannot be replayed" in err and "koch" in err


def test_locks_empty_and_listing(monkeypatch, tmp_path, capsys):
    from antares import server

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "LOCK_SUFFIX", ".lock.json")
    assert cli.main(["locks"]) == 0
    assert "no locks" in capsys.readouterr().out

    (tmp_path / "unit-a.lock.json").write_text(json.dumps({
        "name": "unit-a", "dataset": "fake/robot", "created_at": "2026-10-17T10:00:00+00:00",
        "params": {"kp": 1}, "result": {"eval_score": 1.5, "baseline_eval_score": 3.25},
    }))
    assert cli.main(["locks"]) == 0
    out = capsys.readouterr().out
    assert "unit-a" in out and "fake/robot" in out
    assert "1.500" in out and "3.250" in out


def test_dr_prints_ranges_table(monkeypatch, capsys, tmp_path):
    from antares import server

    ranges = {
        "name": "robot", "lock": "robot", "percentiles": [5, 95],
        "ranges": {
            "kp": {"nominal": 100.0, "lo": 80.0, "hi": 150.0, "scale": "log", "samples": []},
            "latency_frames": {"nominal": 4, "lo": 3, "hi": 5, "scale": "int", "samples": []},
        },
    }
    seen = {}

    def fake_run_dr(ds, lock, *, on_fit=None, **kw):
        seen.update(lock=lock, **kw)
        if on_fit is not None:
            from antares.calibrate import SimParams

            class R:
                params = SimParams(kp=120.0)
                train_score = 1.25

            on_fit(0, R())
        return None, ranges, tmp_path / "robot.dr.json"

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "run_dr", fake_run_dr)
    assert cli.main(["dr", "fake/robot", "--n-fits", "3", "--seed", "7"]) == 0
    out = capsys.readouterr().out
    assert seen["lock"] == "robot"  # default lock is the dataset slug
    assert seen["n_fits"] == 3 and seen["seed"] == 7 and seen["n_train"] == 12
    assert "fit 1/3" in out and "1.250" in out
    assert "kp" in out and "-20% .. +50%" in out
    assert "latency_frames" in out and "-1 .. +1 frames" in out
    assert "saved ranges robot" in out and "robot.dr.json" in out

    assert cli.main(["dr", "fake/robot", "--lock", "unit-a", "--name", "unit-a-wide"]) == 0
    assert seen["lock"] == "unit-a" and seen["name"] == "unit-a-wide"


class _Result:
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


def _result_dict(params, stages):
    return {
        "params": params,
        "per_joint_before": {j: 3.0 for j in JOINTS},
        "per_joint_after": {j: 1.0 for j in JOINTS},
        "baseline_eval_score": 3.0,
        "eval_score": 1.0,
        "n_evaluations": sum(s["n_evaluations"] for s in stages),
        "elapsed_s": 1.5,
        "stages": stages,
    }


@pytest.mark.parametrize("per_joint", [False, True])
def test_calibrate_cli_per_joint_flag(monkeypatch, capsys, tmp_path, per_joint):
    from antares import server

    calls = []
    scalar = {"kp": 100.0, "kv": 1.0, "damping": 0.1, "armature": 0.01, "frictionloss": 0.01,
              "latency_frames": 2, "forcerange": 2.94}
    stages = [{"name": "shared", "n_evaluations": 5, "train_score": 1.2, "params": scalar}]
    params = scalar
    if per_joint:
        params = {**scalar, "kp": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], "latency_frames": [0, 1, 2, 3, 4, 5]}
        stages.append({"name": "per_joint", "n_evaluations": 7, "train_score": 1.1, "params": params})

    def fake_run(ds, **kw):
        calls.append(kw)
        for i in range(3):
            kw["on_progress"](type("Ev", (), {"evaluation": i, "generation": 1, "candidate_score": 2.0,
                                                "best_score": 2.0, "elapsed_s": 0.1,
                                                "stage": "per_joint" if (per_joint and i == 2) else "shared"})())
        return _Result(_result_dict(params, stages)), {"name": "x"}, tmp_path / "x.lock.json"

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "run_calibration", fake_run)
    argv = ["calibrate", "fake/robot", "--max-evaluations", "5"]
    if per_joint:
        argv += ["--per-joint", "--per-joint-evaluations", "7", "--per-joint-sigma0", "0.1"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert calls[0]["per_joint"] is per_joint
    assert calls[0]["per_joint_max_evaluations"] == (7 if per_joint else 600)
    assert calls[0]["per_joint_sigma0"] == (0.1 if per_joint else None)
    assert calls[0]["n_seeds"] == 1
    assert "-- shared stage --" in out and "seeds:" not in out
    assert "shared: 5 evaluations" in out
    assert "saved lock x" in out
    if per_joint:
        assert "+ 7 per-joint evaluations" in out and "-- per_joint stage --" in out
        assert "per_joint: 7 evaluations" in out
        assert "params (per joint)" in out and "elbow_flex" in out and "30" in out
    else:
        assert "per-joint evaluations" not in out and "-- per_joint stage --" not in out
        assert "params: kp=100" in out


def test_calibrate_cli_seeds_flag(monkeypatch, capsys, tmp_path):
    from antares import server

    calls = []
    scalar = {"kp": 100.0, "kv": 1.0, "damping": 0.1, "armature": 0.01, "frictionloss": 0.01,
              "latency_frames": 2, "forcerange": 2.94}
    stages = [{"name": "shared", "n_evaluations": 5, "train_score": 1.2, "params": scalar, "seed": s}
              for s in (0, 100, 200)]
    seeds = {"n": 3, "seeds": [0, 100, 200], "chosen": 1, "eval_scores": [1.3, 1.1, None],
             "train_scores": [1.2, 1.0, 1.4], "params": [scalar] * 3,
             "spread": {"kp": {"min": 100.0, "max": 2500.0, "ratio": 25.0},
                        "latency_frames": {"min": 2.0, "max": 3.0, "range": 1.0}},
             "pinned": ["latency_frames"]}

    def fake_run(ds, **kw):
        calls.append(kw)
        for i, seed in enumerate((0, 0, 1, 2)):
            kw["on_progress"](type("Ev", (), {"evaluation": i, "generation": 1, "candidate_score": 2.0,
                                                "best_score": 2.0, "elapsed_s": 0.1, "stage": "shared",
                                                "seed": seed})())
        return _Result({**_result_dict(scalar, stages), "seeds": seeds}), {"name": "x"}, tmp_path / "x.lock.json"

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "run_calibration", fake_run)
    assert cli.main(["calibrate", "fake/robot", "--seeds", "3"]) == 0
    out = capsys.readouterr().out
    assert calls[0]["n_seeds"] == 3
    assert "best of 3 seeds x up to 150 shared evaluations" in out
    assert "-- shared stage, seed 1/3 --" in out and "-- shared stage, seed 3/3 --" in out
    assert "shared (seed 100): 5 evaluations" in out
    assert "seeds: kept seed 100 (2 of 3); held-out 1.300 / 1.100 / - deg" in out
    assert "pinned (agree across seeds): latency_frames" in out
    assert "kp x25" in out and "latency_frames +-1 frames" in out
    assert "not a measurement" in out


def _exact_simulate(params, ep):
    return np.radians(ep.state), np.radians(ep.action)


def _check_result(offset_deg, n=6, **kw):
    """A real CheckResult against a lock with held-out score 1.0 (limit 1.5)."""
    from antares.calibrate import CalibrationResult, SimParams
    from antares.check import check
    from antares.lock import RobotLock

    eps = []
    for i in range(n):
        a = np.linspace(0, 10, 20)[:, None].repeat(6, axis=1)
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=a + offset_deg, action=a))
    result = CalibrationResult(
        params=SimParams(), baseline=SimParams(), train_score=0.9, eval_score=1.0,
        baseline_train_score=4.0, baseline_eval_score=4.0,
        per_joint_before={j: 4.0 for j in JOINTS}, per_joint_after={j: 1.0 for j in JOINTS},
        history=[], n_evaluations=0, elapsed_s=0.0,
    )
    lock = RobotLock.create("fake/robot", "so101_follower", eps[:1], result, name="unit-a")
    return check(eps, lock, simulate=_exact_simulate, **kw)


def _patch_check(monkeypatch, result, calls=None):
    from antares import server

    def run_check(ds, lock, **kw):
        if calls is not None:
            calls.append((ds, lock, kw))
        return result

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "run_check", run_check)


def test_check_pass_exits_zero_and_prints_table(monkeypatch, capsys):
    calls = []
    _patch_check(monkeypatch, _check_result(1.2), calls)
    code = cli.main(["check", "fake/robot", "--lock", "unit-a", "--threshold", "2", "--last", "8"])
    assert code == 0
    out = capsys.readouterr().out
    assert "fake/robot vs lock unit-a" in out
    assert "shoulder_pan" in out and "overall" in out and "1.200" in out
    assert "PASS" in out and "FAIL" not in out
    assert "(not scored)" in out  # the gripper row
    lines = out.rstrip().splitlines()
    assert lines[-2].startswith("OK:")
    assert lines[-1].startswith("history: 1 check since ")
    (ds, lock, kw), = calls
    assert ds.repo_id == "fake/robot" and lock == "unit-a"
    assert kw == {"threshold_deg": 2.0, "ratio": 1.5, "last": 8, "min_episodes": 5}


def test_check_drift_exits_one(monkeypatch, capsys):
    _patch_check(monkeypatch, _check_result(3.0))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "<- worst" in out
    assert out.rstrip().splitlines()[-2].startswith("DRIFTED:")


def test_check_insufficient_data_exits_two(monkeypatch, capsys):
    _patch_check(monkeypatch, _check_result(0.5, n=2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a"]) == 2
    out = capsys.readouterr().out
    assert "INSUFFICIENT DATA" in out and "overall" not in out


def test_check_json_output(monkeypatch, capsys):
    _patch_check(monkeypatch, _check_result(1.2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--json"]) == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out)  # the history line goes to stderr so stdout stays JSON
    assert out["verdict"] == "ok" and out["passed"] is True
    assert out["score"] == pytest.approx(1.2) and out["limit"] == pytest.approx(1.5)
    assert set(out["per_joint"]) == set(JOINTS)
    assert captured.err.rstrip().splitlines()[-1].startswith("history: 1 check since ")


def _history_lines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_check_records_one_history_line_per_run(monkeypatch, tmp_path, capsys):
    from antares import server

    hist = tmp_path / "locks" / "unit-a.history.jsonl"
    _patch_check(monkeypatch, _check_result(1.2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a"]) == 0
    assert server.LOCK_DIR == tmp_path / "locks" and hist.is_file()
    (rec,) = _history_lines(hist)
    assert rec["lock"] == "unit-a" and rec["dataset"] == "fake/robot"
    assert rec["verdict"] == "ok" and rec["passed"] is True
    assert rec["score"] == pytest.approx(1.2) and rec["limit"] == pytest.approx(1.5)
    assert rec["n_episodes"] == 6 and rec["episodes"] == [0, 1, 2, 3, 4, 5]
    assert set(rec["per_joint"]) == set(JOINTS) and rec["worst_joint"] in JOINTS
    assert rec["at"].endswith("Z") and rec["antares_version"]
    capsys.readouterr()

    _patch_check(monkeypatch, _check_result(3.0))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a"]) == 1
    recs = _history_lines(hist)
    assert [r["verdict"] for r in recs] == ["ok", "drifted"]
    last = capsys.readouterr().out.rstrip().splitlines()[-1]
    assert last.startswith("history: 2 checks since ") and ", last drifted " in last


def test_check_no_record_writes_nothing(monkeypatch, tmp_path, capsys):
    _patch_check(monkeypatch, _check_result(1.2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--no-record"]) == 0
    assert not (tmp_path / "locks").exists()
    assert "history:" not in capsys.readouterr().out


def test_check_history_dir_override_and_lock_file_path(monkeypatch, tmp_path, capsys):
    _patch_check(monkeypatch, _check_result(1.2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--history-dir", str(tmp_path / "h")]) == 0
    assert (tmp_path / "h" / "unit-a.history.jsonl").is_file()
    assert not (tmp_path / "locks").exists()

    # a .lock.json path keeps its history next to it, under the file's name
    lock_file = tmp_path / "arms" / "my_arm.lock.json"
    lock_file.parent.mkdir()
    lock_file.write_text(json.dumps({"name": "unit-a", "params": {}, "result": {}}))
    assert cli.main(["check", "fake/robot", "--lock", str(lock_file)]) == 0
    assert (tmp_path / "arms" / "my_arm.history.jsonl").is_file()


def test_check_records_insufficient_data_with_null_score(monkeypatch, tmp_path, capsys):
    _patch_check(monkeypatch, _check_result(0.5, n=2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a"]) == 2
    (rec,) = _history_lines(tmp_path / "locks" / "unit-a.history.jsonl")
    assert rec["verdict"] == "insufficient_data" and rec["passed"] is False
    assert rec["score"] is None and rec["per_joint"] == {} and rec["n_episodes"] == 2


def test_check_record_failure_keeps_the_verdict(monkeypatch, tmp_path, capsys):
    from antares import history

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(history, "append", boom)
    _patch_check(monkeypatch, _check_result(1.2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a"]) == 0
    captured = capsys.readouterr()
    assert captured.out.rstrip().splitlines()[-1].startswith("OK:")
    assert "could not record the check" in captured.err and "disk full" in captured.err


def _seed_history(tmp_path, name="unit-a", lock_file=True):
    from antares import history

    d = tmp_path / "locks"
    d.mkdir(exist_ok=True)
    if lock_file:
        (d / f"{name}.lock.json").write_text("{}")
    history.append(_check_result(1.0), d, at="2026-09-30T08:00:00Z", name=name)
    history.append(_check_result(3.0), d, at="2026-10-10T08:00:00Z", name=name)
    history.append(_check_result(1.4), d, at="2026-10-20T08:00:00Z", name=name)
    return d


def test_history_prints_table_and_trend(tmp_path, capsys):
    _seed_history(tmp_path)
    assert cli.main(["history", "unit-a"]) == 0
    out = capsys.readouterr().out
    lines = out.rstrip().splitlines()
    assert lines[0].split()[:2] == ["date", "(utc)"] and "worst joint" in lines[0]
    body = lines[2:-1]
    assert len(body) == 3
    assert body[0].startswith("2026-09-30 08:00") and "fake/robot" in body[0] and " ok " in body[0]
    assert "drifted" in body[1] and "3.000" in body[1] and "1.500" in body[1]
    assert lines[-1].startswith("history: 3 checks since 2026-09-30, +0.020 deg/day (r")
    assert lines[-1].endswith(", last drifted 2026-10-10")


def test_history_last_and_json(tmp_path, capsys):
    _seed_history(tmp_path)
    assert cli.main(["history", "unit-a", "--last", "1"]) == 0
    out = capsys.readouterr().out
    assert out.count("fake/robot") == 1 and "2026-10-20" in out
    assert "history: 3 checks" in out  # the trend is over everything

    assert cli.main(["history", "unit-a", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert set(d) == {"lock", "entries", "trend"} and d["lock"] == "unit-a"
    assert [e["verdict"] for e in d["entries"]] == ["ok", "drifted", "ok"]
    assert set(d["trend"]) == {"n", "first", "last", "slope_deg_per_day", "r2", "latest_verdict", "n_drifted"}
    assert d["trend"]["n"] == 3 and d["trend"]["n_drifted"] == 1 and d["trend"]["latest_verdict"] == "ok"
    assert d["trend"]["slope_deg_per_day"] == pytest.approx(0.02)

    assert cli.main(["history", "unit-a", "--last", "0"]) == 2


def test_history_missing_lock_and_empty_history(tmp_path, capsys):
    assert cli.main(["history", "ghost"]) == 2
    assert "no lock named 'ghost'" in capsys.readouterr().err
    (tmp_path / "locks").mkdir()
    (tmp_path / "locks" / "fresh.lock.json").write_text("{}")
    assert cli.main(["history", "fresh"]) == 0
    assert "no checks recorded for lock fresh" in capsys.readouterr().out
    # the lock is looked up in the lock dir even when the history lives elsewhere
    assert cli.main(["history", "fresh", "--history-dir", str(tmp_path / "elsewhere")]) == 0
    assert str(tmp_path / "elsewhere" / "fresh.history.jsonl") in capsys.readouterr().out
    assert cli.main(["history", "fresh", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["entries"] == [] and d["trend"]["n"] == 0 and d["trend"]["slope_deg_per_day"] is None


def test_history_in_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "history" in capsys.readouterr().out


def _diagnosis():
    from antares.diagnose import FIELDS, Diagnosis, ParamChange

    values = {"latency_frames": 1}
    alone = {"latency_frames": 0.4}
    changes = [ParamChange(f, values.get(f, 0 if f == "latency_frames" else 1.0),
                           alone_deg=alone.get(f, 3.0), stale_deg=3.0) for f in FIELDS]
    return Diagnosis(lock="unit-a", dataset="fake/robot", reference_deg=1.0, stale_deg=3.0,
                     refit_deg=0.4, per_joint_stale={}, per_joint_refit={}, changes=changes,
                     refit_episodes=[0, 2, 4], held_out_episodes=[1, 3, 5],
                     n_evaluations=10, elapsed_s=0.1)


def _patch_diagnose(monkeypatch, calls):
    from antares import server

    def run_diagnose(ds, lock, **kw):
        calls.append((lock, kw))
        return _diagnosis()

    monkeypatch.setattr(server, "run_diagnose", run_diagnose)


def test_check_explain_names_the_cause_on_drift(monkeypatch, capsys):
    calls = []
    _patch_check(monkeypatch, _check_result(3.0))
    _patch_diagnose(monkeypatch, calls)
    code = cli.main(["check", "fake/robot", "--lock", "unit-a", "--explain", "--last", "6",
                     "--explain-evaluations", "50"])
    assert code == 1  # explaining never changes the verdict
    out = capsys.readouterr().out
    assert "what changed" in out and "latency_frames  +1 frame" in out and "<- cause" in out
    assert out.rstrip().splitlines()[-2].endswith("likely cause: latency_frames +1 frame (alone recovers 2.60 deg)")
    assert calls == [("unit-a", {"last": 6, "max_evaluations": 50})]


def test_check_explain_skips_a_passing_robot(monkeypatch, capsys):
    calls = []
    _patch_check(monkeypatch, _check_result(1.2))
    _patch_diagnose(monkeypatch, calls)
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--explain"]) == 0
    assert calls == [] and "what changed" not in capsys.readouterr().out


def test_check_explain_json(monkeypatch, capsys):
    _patch_check(monkeypatch, _check_result(3.0))
    _patch_diagnose(monkeypatch, [])
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--explain", "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "drifted"
    assert out["diagnosis"]["causes"] == ["latency_frames"]


def test_check_missing_lock_or_dataset_exits_two(monkeypatch, capsys):
    from antares import server

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "LOCK_DIR", "/nonexistent-lock-dir")
    assert cli.main(["check", "fake/robot", "--lock", "ghost"]) == 2
    assert "no lock named 'ghost'" in capsys.readouterr().err

    import antares.data

    def no_network(repo_id, cache_dir=None):
        raise RuntimeError("offline")

    monkeypatch.setattr(antares.data, "load_dataset", no_network)
    monkeypatch.setattr(server, "fetch_hub_info", no_network)
    assert cli.main(["check", "nope/none", "--lock", "ghost"]) == 2
    err = capsys.readouterr().err
    assert "nope/none" in err and "offline" in err


def test_check_accepts_local_dir_and_lock_file(monkeypatch, tmp_path, capsys):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from antares import server

    root = tmp_path / "recording"
    names = [f"{j}.pos" for j in JOINTS]
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "robot_type": "so101_follower", "fps": 30,
        "features": {"action": {"names": names}, "observation.state": {"names": names}},
    }))
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [0, 0, 1, 1], "frame_index": [0, 1, 0, 1],
        "observation.state": [[0.0] * 6] * 4, "action": [[1.0] * 6] * 4,
    }), root / "data" / "chunk-000" / "file-000.parquet")
    lock_file = tmp_path / "arm.lock.json"
    lock_file.write_text(json.dumps({"name": "arm", "params": {}, "result": {}}))

    calls = []
    monkeypatch.setattr(server, "_DATASETS", {})
    monkeypatch.setattr(server, "run_check", lambda ds, lock, **kw: calls.append((ds, lock)) or _check_result(1.2))
    assert cli.main(["check", str(root), "--lock", str(lock_file)]) == 0
    (ds, lock), = calls
    assert ds.repo_id == str(root) and len(ds.episodes) == 2
    assert isinstance(lock, dict) and lock["name"] == "arm"


FAKE_FEAS = {
    "velocity_factor": 1.5, "ok": True, "flagged_joints": [], "max_range_excess_deg": 0.52,
    "clipped_frames": 0, "raw_clipped_frames": 3,
    "per_joint": {j: {"velocity_ratio": 0.5 + 0.1 * i} for i, j in enumerate(JOINTS)},
}


def _fake_comp_out(method="ilc", T=10):
    return {
        "episode": 1, "method": method, "params": {"latency_frames": 4},
        "q_des": [[0.0] * 6] * T, "ctrl_comp": [[45.0, 0.0, 0.0, 0.0, 0.0, 10.0]] * T,
        "rms": {"raw": 1.054, "shift": 0.133, "ilc": 0.048}, "ilc_iterations": 4,
        "feasibility": dict(FAKE_FEAS),
    }


def test_compensate_prints_three_numbers_and_saves(monkeypatch, capsys, tmp_path):
    import types

    from antares import server

    calls = {}

    def fake(ds, episode, params=None, method="ilc", lock=None):
        calls.update(episode=episode, params=params, method=method, lock=lock)
        return _fake_comp_out(method)

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "compensate_episode_dict", fake)
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "LOCK_SUFFIX", ".lock.json")
    monkeypatch.setattr(server, "params_from_dict", lambda d: ("params", d["kp"]))
    (tmp_path / "unit-a.lock.json").write_text(json.dumps({"name": "unit-a", "params": {"kp": 93.7},
                                                          "result": {"eval_episodes": [1, 2]}}))
    # the --out path converts degrees back to dataset units through the arm's unit map
    unit_map = types.SimpleNamespace(to_units=lambda q: np.degrees(q) * 2.0)
    monkeypatch.setattr(server, "_ARMS", {"so101": types.SimpleNamespace(unit_map=unit_map)})

    out_file = tmp_path / "commands.npy"
    assert cli.main(["compensate", "fake/robot", "--episode", "1", "--lock", "unit-a", "--out", str(out_file)]) == 0
    out = capsys.readouterr().out
    assert calls == {"episode": 1, "params": ("params", 93.7), "method": "ilc", "lock": "unit-a"}
    assert "episode 1" in out and "lock unit-a" in out and "calibrated sim" in out
    assert "1.054" in out and "0.133" in out and "0.048" in out and "4 iterations" in out
    assert "feasibility (sim-only): ok" in out and "1.00x" in out and "0.52 deg" in out and "raw: 3" in out
    saved = np.load(out_file)
    assert saved.shape == (10, 6)
    np.testing.assert_allclose(saved[0], [90.0, 0, 0, 0, 0, 20.0])

    assert cli.main(["compensate", "fake/robot", "--episode", "1", "--method", "shift"]) == 0
    assert calls["method"] == "shift" and calls["params"] is None and calls["lock"] is None
    assert "method shift" in capsys.readouterr().out

    # --episode is required unless --heldout
    assert cli.main(["compensate", "fake/robot"]) == 2
    assert cli.main(["compensate", "fake/robot", "--heldout"]) == 2  # needs --lock


def test_compensate_heldout_prints_aggregate(monkeypatch, capsys, tmp_path):
    from antares import server

    def fake(ds, lock, method="ilc"):
        return {
            "episodes": [1, 2], "n_frames": 20, "method": method, "params": {"latency_frames": 3},
            "rms": {"raw": 0.791, "shift": 0.477, "ilc": 0.119}, "ilc_iterations": [4, 6],
            "feasibility": {**FAKE_FEAS, "real_vmax_deg_s": {j: 100.0 for j in JOINTS},
                            "comp_cmd_vmax_deg_s": {j: 110.0 for j in JOINTS}, "flagged_joints": ["wrist_roll"], "ok": False},
        }

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "compensate_heldout", fake)
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "LOCK_SUFFIX", ".lock.json")
    monkeypatch.setattr(server, "params_from_dict", lambda d: d)
    (tmp_path / "unit-a.lock.json").write_text(json.dumps({"name": "unit-a", "params": {"kp": 1}}))
    assert cli.main(["compensate", "fake/robot", "--lock", "unit-a", "--heldout"]) == 0
    out = capsys.readouterr().out
    assert "2 held-out episodes [1, 2]" in out
    assert "0.791" in out and "0.477" in out and "0.119" in out and "4-6 iterations" in out
    assert "FLAGGED wrist_roll" in out and "1.10x" in out


def test_replay_prints_table(monkeypatch, capsys):
    from antares import server

    fake_out = {
        "episode": 1, "real_q": [[0] * 6] * 10,
        "gap": {"per_joint_deg": {j: 1.25 for j in JOINTS}, "overall_deg": 1.25},
    }
    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "replay_episode", lambda ds, ep, params=None, cmap=None, robot=None: fake_out)
    assert cli.main(["replay", "fake/robot", "--episode", "1"]) == 0
    out = capsys.readouterr().out
    assert "episode 1" in out and "default params" in out
    assert "shoulder_pan" in out and "overall" in out and "1.250" in out


def _real_lock(name="unit_a"):
    from antares.calibrate import CalibrationResult
    from antares.lock import RobotLock
    from antares.sim import SimParams

    p = SimParams(kp=120.0, kv=1.2, damping=0.1, armature=0.02, frictionloss=0.01, latency_frames=3)
    res = CalibrationResult(
        params=p, baseline=SimParams(), train_score=1.0, eval_score=1.1,
        baseline_train_score=5.0, baseline_eval_score=5.5,
        per_joint_before={j: 5.0 for j in JOINTS}, per_joint_after={j: 1.0 for j in JOINTS},
        history=[(0, 5.0), (1, 1.0)], n_evaluations=1, elapsed_s=0.1,
        train_episodes=[1], eval_episodes=[0],
    )
    return RobotLock(name=name, dataset="fake/robot", robot_type="so101_follower", data_sha256="0" * 64,
                     params=p, result=res, created_at="2026-10-17T00:00:00Z", antares_version="0.0.1")


def test_export_writes_bundle(monkeypatch, tmp_path, capsys):
    import mujoco

    from antares import server

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(server, "LOCK_SUFFIX", ".lock.json")
    _real_lock().save(tmp_path / "locks" / "unit_a.lock.json")

    out = tmp_path / "out"
    assert cli.main(["export", "unit_a", "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "exported lock unit_a" in text and "scene.xml" in text and "--mjcf" in text
    assert (out / "unit_a.xml").is_file()
    assert (out / "scene.xml").is_file()
    assert json.loads((out / "unit_a.antares.json").read_text())["latency_frames"] == 3
    assert any(out.joinpath("assets").glob("*.stl"))
    m = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    assert m.actuator_gainprm[0, 0] == 120.0

    # default output dir is export/<name> relative to cwd; --no-bundle skips the meshes
    monkeypatch.chdir(tmp_path)
    assert cli.main(["export", "unit_a", "--no-bundle"]) == 0
    assert (tmp_path / "export" / "unit_a" / "unit_a.xml").is_file()
    assert not (tmp_path / "export" / "unit_a" / "assets").exists()
    assert "absolute path" in capsys.readouterr().out


def test_export_missing_lock_is_an_error(monkeypatch, tmp_path, capsys):
    from antares import server

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    assert cli.main(["export", "ghost"]) == 1
    assert "ghost" in capsys.readouterr().err

def test_drift_prints_windows_trends_and_verdict(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace

    from antares import server

    params = {"kp": 100.0, "kv": 2.0, "damping": 0.1, "armature": 0.02, "frictionloss": 0.05, "latency_frames": 3}
    windows = []
    for i, (lock, fit) in enumerate([(1.0, 0.9), (1.1, 0.9), (1.3, 0.9)]):
        windows.append({"first": 5 * i, "last": 5 * i + 9, "lock_score": lock, "train_score": fit,
                        "improvement": 1 - fit / lock, "params": {**params, "kp": 100.0 - 10 * i}})
    trend = {"slope": 0.03, "r2": 0.95, "ci_lo": 0.01, "ci_hi": 0.05, "total_change": 0.3,
             "log_space": False, "significant": True, "drifting": True}
    flat = {**trend, "slope": 0.0, "ci_lo": -0.01, "total_change": 0.0, "significant": False, "drifting": False}
    scan = {"lock_params": params, "windows": windows,
            "trends": {"kp": {**flat, "log_space": True}, "lock_score": trend},
            "verdict": "drift: lock_score +30% over the session (r²=0.95)"}
    seen = {}

    def fake_run_drift(ds, lock, *, window, step, max_evaluations, seed, on_progress=None, should_stop=None):
        seen.update(lock=lock, window=window, step=step, max_evaluations=max_evaluations, seed=seed)
        w = SimpleNamespace(first=0, last=9, lock_score=1.0, train_score=0.9, improvement=0.1)
        on_progress(SimpleNamespace(index=0, n_windows=3, window=w, elapsed_s=2.0))
        return SimpleNamespace(to_dict=lambda: scan), tmp_path / "unit-a.drift.json"

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "run_drift", fake_run_drift)
    assert cli.main(["drift", "fake/robot", "--lock", "unit-a", "--window", "10", "--step", "5"]) == 0
    assert seen == {"lock": "unit-a", "window": 10, "step": 5, "max_evaluations": 60, "seed": 0}
    out = capsys.readouterr().out
    assert "window  1/3" in out and "+10.0%" in out  # progress line
    assert "0-9" in out and "10-19" in out and "1.300" in out and "+31%" in out  # window table
    assert "kp" in out and "log10/ep" in out  # trend table, log-space param
    assert "[+0.01, +0.05]" in out and "+30%" in out
    assert "verdict: drift: lock_score +30%" in out
    assert "unit-a.drift.json" in out


def _residual_out(events):
    return {
        "episode": 3, "fps": 30.0, "residual": [[0.0] * 6] * 300,
        "gripper_thresholds": {"closed_below": 8.3, "open_above": 14.9},
        "grasps": [150, 240], "releases": [100, 220],
        "baseline": {"joints": list(JOINTS), "scale_deg": [0.2, 0.3, 0.4, 0.3, 0.3, 0.2],
                     "n_frames": 600, "episodes": [0, 8]},
        "events": events,
        "validation": {"n_grasps": 2, "n_grasps_hit": 1, "n_events": len(events),
                       "n_events_in_grasp": 1, "n_events_far": len(events) - 1,
                       "energy_total": 100.0, "energy_in_grasp": 40.0, "n_frames": 300, "n_frames_in_grasp": 62},
    }


def test_events_single_episode(monkeypatch, capsys):
    from antares import server

    calls = {}

    def fake_residual(ds, episode, lock=None, z=4.0, min_frames=5):
        calls.update(episode=episode, lock=lock, z=z, min_frames=min_frames)
        return _residual_out([
            {"start": 145, "end": 175, "joints": ["elbow_flex"], "peak_z": 12.3, "peak_deg": 4.56, "kind": "contact"},
            {"start": 10, "end": 16, "joints": ["wrist_roll", "wrist_flex"], "peak_z": 5.0, "peak_deg": -1.2, "kind": "anomaly"},
        ])

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "residual_episode", fake_residual)
    assert cli.main(["events", "fake/robot", "--episode", "3", "--lock", "unit-a", "--z", "6", "--min-frames", "8"]) == 0
    assert calls == {"episode": 3, "lock": "unit-a", "z": 6.0, "min_frames": 8}
    out = capsys.readouterr().out
    assert "episode 3" in out and "lock unit-a" in out
    assert "grasps at frames [150, 240]" in out
    assert "145-175" in out and "4.83-5.83" in out and "contact" in out and "elbow_flex" in out
    assert "12.3" in out and "+4.56" in out
    assert "wrist_roll,wrist_flex" in out and "anomaly" in out
    assert "1/2 grasps hit" in out and "1/2 events in a grasp window" in out and "40%" in out


def test_events_no_events_and_missing_episode(monkeypatch, capsys):
    from antares import server

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "residual_episode", lambda ds, episode, lock=None, z=4.0, min_frames=5: _residual_out([]))
    assert cli.main(["events", "fake/robot", "--episode", "0"]) == 0
    out = capsys.readouterr().out
    assert "no events" in out and "default params" in out
    assert cli.main(["events", "fake/robot"]) == 2
    assert "--episode N or --all" in capsys.readouterr().err


def test_events_all_summary(monkeypatch, capsys):
    from antares import server

    summary = {
        "n_episodes": 2, "n_frames": 600, "n_grasps": 4, "n_grasps_hit": 3, "n_events": 5,
        "n_events_in_grasp": 3, "n_events_near_transition": 4, "n_events_far": 1,
        "recall": 0.75, "precision": 0.6, "near_transition": 0.8, "far_fraction": 0.2,
        "energy_in_grasp_fraction": 0.228, "chance": 0.177, "lift": 1.29,
    }
    fake = {
        "baseline": {"n_frames": 679, "episodes": [0, 8]},
        "episodes": [
            {"episode": 0, "held_out": True, "n_frames": 300, "n_grasps": 2, "n_grasps_hit": 2, "n_events": 3,
             "n_events_in_grasp": 2, "n_events_far": 1, "energy_total": 10.0, "energy_in_grasp": 5.0, "events": []},
            {"episode": 1, "held_out": False, "n_frames": 300, "n_grasps": 2, "n_grasps_hit": 1, "n_events": 2,
             "n_events_in_grasp": 1, "n_events_far": 0, "energy_total": 0.0, "energy_in_grasp": 0.0, "events": []},
        ],
        "summary": summary,
        "summary_held_out": dict(summary, n_episodes=1),
    }
    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "events_summary", lambda ds, lock=None, z=4.0, min_frames=5: fake)
    assert cli.main(["events", "fake/robot", "--all", "--lock", "unit-a"]) == 0
    out = capsys.readouterr().out
    assert "lock unit-a" in out and "679 free-motion frames" in out
    assert "0*" in out and "held out" in out
    assert "all episodes:" in out and "held-out episodes:" in out
    assert "recall 75% (3/4 grasps hit)" in out
    assert "precision 60% (3/5 events in a grasp window)" in out
    assert "23%" in out and "18% of frames" in out and "lift 1.29x" in out


def test_cross_prints_specificity_and_writes_json(monkeypatch, tmp_path, capsys):
    from antares import server

    m = {"rows": ["lab/a", "lab/b"], "cols": ["lock-a", "lock-b", "default"],
         "scores": [[1.0, 2.0, 5.0], [3.0, 0.5, 4.0]], "per_joint": [[{}] * 3] * 2}
    monkeypatch.setattr(server, "cross_matrix", lambda *a, **k: m)
    monkeypatch.setattr(server, "load_lock_dict",
                        lambda name: {"dataset": {"lock-a": "lab/a", "lock-b": "lab/b"}[name]})
    out_path = tmp_path / "out" / "cross.json"
    assert cli.main(["cross", "--datasets", "lab/a,lab/b", "--locks", "lock-a,lock-b",
                     "--json", str(out_path)]) == 0
    out = capsys.readouterr().out
    assert "robot specificity" in out
    assert "2.00x" in out and "6.00x" in out
    saved = json.loads(out_path.read_text())
    assert saved["specificity"]["all_specific"] is True
    assert saved["matrix"]["cols"] == m["cols"]


def test_identify_prints_tolerances(monkeypatch, tmp_path, capsys):
    import antares.calibrate
    from antares import server

    T = 40
    t = np.arange(T) / 30.0
    eps = []
    for i in range(4):
        a = np.stack([np.sin(2 * np.pi * (0.5 + 0.1 * j) * t + i) for j in range(6)], axis=1)
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=a, action=a))
    ds = Dataset(repo_id="fake/robot", robot_type="so101_follower", fps=30.0,
                 joint_names=JOINTS, episodes=tuple(eps))

    def simulate(params, ep):  # lag shrinks with kp; kv, damping etc. do nothing
        shift = np.radians(100.0 / params.kp)
        return ep.state, ep.state + shift

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": ds})
    monkeypatch.setattr(server, "load_lock_dict", lambda name: {"params": {"kp": 500.0}})
    monkeypatch.setattr(antares.calibrate, "default_simulate", lambda: simulate)
    out_path = tmp_path / "id.json"
    assert cli.main(["identify", "fake/robot", "--lock", "x", "--n-train", "2", "--n-eval", "2",
                     "--json", str(out_path)]) == 0
    out = capsys.readouterr().out
    assert "train split, 2 episodes" in out
    assert "kp" in out and "unbounded" in out  # kv etc. have no effect at all
    assert "nan" not in out  # an unconstrained parameter has no trade-off ratio, shown as "-"
    assert "latency (frames): 0:" in out
    saved = json.loads(out_path.read_text())
    assert saved["identifiability"]["names"][0] == "kp"
    assert saved["identifiability"]["tradeoff"]["kv"] is None
    # asking for a split the dataset cannot provide is an error, not a NaN report
    with pytest.raises(SystemExit, match="leaves no eval split"):
        cli.main(["identify", "fake/robot", "--lock", "x", "--split", "eval", "--n-train", "4"])


def test_specificity_writes_report(monkeypatch, tmp_path, capsys):
    import antares.calibrate
    import antares.specificity
    from antares import server

    seen = {}

    def fake_study(episodes, locks, simulate, **kw):
        seen.update(episodes=episodes, locks=locks, kw=kw)
        return {"matrix": {}, "datasets": {}}

    monkeypatch.setattr(server, "_DATASETS", {"lab/a": _dataset("lab/a"), "lab/b": _dataset("lab/b")})
    monkeypatch.setattr(server, "list_lock_dicts", lambda: list(SPEC_LOCKS.values()))
    monkeypatch.setattr(server, "load_lock_dict", lambda n: SPEC_LOCKS[n])
    monkeypatch.setattr(antares.calibrate, "default_simulate", lambda: None)
    monkeypatch.setattr(antares.specificity, "study", fake_study)
    monkeypatch.setattr(antares.specificity, "render_markdown", lambda r, title: f"## {title}\n")
    assert cli.main(["specificity", "--datasets", "lab/a,lab/b", "--restart-seeds", "3",
                     "--out", str(tmp_path)]) == 0
    # the newest *shared* lock per dataset: a-perjoint is newer than a-new but per-joint
    assert set(seen["locks"]) == {"a-new", "b"}
    assert seen["locks"]["b"][0] == "lab/b"
    assert seen["kw"]["restart_seeds"] == [3]
    # each dataset is split the way its own lock was fitted
    assert seen["kw"]["splits"] == {"lab/a": ([0, 2], [1]), "lab/b": ([1], [0, 2])}
    assert (tmp_path / "robot_specificity.md").read_text() == "## Robot specificity\n"
    assert json.loads((tmp_path / "robot_specificity.json").read_text()) == {"matrix": {}, "datasets": {}}


def _spec_lock(name, dataset, created, space="shared", train=(0, 2), eval=(1,)):
    return {"name": name, "dataset": dataset, "created_at": created, "params": {"kp": 100.0},
            "result": {"space": space, "train_episodes": list(train), "eval_episodes": list(eval)}}


SPEC_LOCKS = {
    "a-old": _spec_lock("a-old", "lab/a", "2026-01-01"),
    "a-new": _spec_lock("a-new", "lab/a", "2026-02-01"),
    "a-perjoint": _spec_lock("a-perjoint", "lab/a", "2026-03-01", space="per_joint"),
    "b": _spec_lock("b", "lab/b", "2026-01-01", train=(1,), eval=(0, 2)),
}


def test_specificity_needs_a_lock_per_dataset(monkeypatch):
    from antares import server

    monkeypatch.setattr(server, "list_lock_dicts", lambda: [])
    with pytest.raises(SystemExit, match="no saved shared lock for lab/a"):
        cli.main(["specificity", "--datasets", "lab/a"])
    # a per-joint lock does not count
    monkeypatch.setattr(server, "list_lock_dicts", lambda: [SPEC_LOCKS["a-perjoint"]])
    with pytest.raises(SystemExit, match="no saved shared lock for lab/a"):
        cli.main(["specificity", "--datasets", "lab/a"])


def test_specificity_refuses_a_named_per_joint_lock(monkeypatch):
    from antares import server

    monkeypatch.setattr(server, "load_lock_dict", lambda n: SPEC_LOCKS[n])
    with pytest.raises(SystemExit, match="a-perjoint is a per_joint lock"):
        cli.main(["specificity", "--datasets", "lab/a", "--locks", "a-perjoint"])


def test_specificity_synthetic_needs_one_lock_per_dataset(monkeypatch):
    import antares.calibrate
    from antares import server

    monkeypatch.setattr(server, "load_lock_dict", lambda n: SPEC_LOCKS[n])
    monkeypatch.setattr(antares.calibrate, "default_simulate", lambda: None)
    with pytest.raises(SystemExit, match="one lock per dataset: got 2 datasets and 1 locks"):
        cli.main(["specificity", "--synthetic", "--datasets", "lab/a,lab/b", "--locks", "a-new"])


def test_write_json_turns_nan_and_inf_into_null(tmp_path):
    path = tmp_path / "out.json"
    cli._write_json(str(path), {"margin": float("inf"), "rows": [float("nan"), 1.0], "ok": "x"})
    assert json.loads(path.read_text()) == {"margin": None, "rows": [None, 1.0], "ok": "x"}


FLEET_REPORT = {
    "summary": {"n_robots": 2, "n_failed": 1, "n_skipped": 0, "n_rejected": 5, "median_improvement": 0.6},
    "robots": [
        {"dataset": "alice/so101_a", "robot_type": "so101_follower", "n_episodes": 50, "fps": 30.0,
         "n_train": 20, "n_eval": 10, "lock": "alice__so101_a", "baseline_eval": 4.0, "calibrated_eval": 1.5, "improvement": 0.625, "latency_ms": 100.0,
         "params": {"kp": 1200.0, "latency_frames": 3}},
        {"dataset": "bob/so100_b", "robot_type": "so100", "n_episodes": 40, "fps": 30.0,
         "n_train": 20, "n_eval": 10, "lock": "bob__so100_b", "baseline_eval": 3.0, "calibrated_eval": 1.2, "improvement": 0.6, "latency_ms": 66.7,
         "params": {"kp": 800.0, "latency_frames": 2}},
    ],
    "stats": {"kp": {"median": 1000.0, "q25": 900.0, "q75": 1100.0, "min": 800.0, "max": 1200.0, "n": 2}},
    "outliers": [{"dataset": "alice/so101_a", "param": "kp", "z": 2.7, "note": "kp 1.2e+03 is 1.2x the fleet median 1e+03"}],
    "cross": {"rows": ["alice/so101_a", "bob/so100_b"], "cols": ["alice__so101_a", "bob__so100_b", "default"],
              "scores": [[1.5, 2.5, 4.0], [2.0, 1.2, 3.0]],
              "summary": {"n": 2, "mean_diagonal": 1.35, "mean_off_diagonal": 2.25, "mean_default": 3.5,
                          "off_over_diagonal": 1.67, "default_over_diagonal": 2.59, "own_lock_is_best": 2}},
    "failures": [{"dataset": "carol/broken", "error": "ValueError: no parquet data"}],
}


def test_fleet_list(tmp_path, capsys):
    assert cli.main(["fleet", "list", "--path", str(tmp_path / "missing.json")]) == 1
    assert "no fleet report" in capsys.readouterr().out

    path = tmp_path / "fleet.json"
    path.write_text(json.dumps(FLEET_REPORT))
    assert cli.main(["fleet", "list", "--path", str(path)]) == 0
    out = capsys.readouterr().out
    assert "2 robots calibrated, 1 failed" in out and "median gap closed 60%" in out
    assert "alice/so101_a" in out and "4.000" in out and "1.500" in out and "62%" in out
    assert "3 (100 ms)" in out and "1200" in out
    assert "fitted parameters" in out and "kp" in out
    # single-seed locks: only latency is pinned across the fleet
    assert "pinned across the fleet's locks: latency_frames (0/2 locks carry seed spread)" in out
    assert "outliers" in out and "+2.7" in out
    assert "cross matrix" in out and "1.67x" in out and "own lock best for 2/2" in out
    assert "carol/broken" in out


def test_fleet_scan_wires_discover_and_scan(monkeypatch, capsys, tmp_path):
    from antares import fleet

    cands = [fleet.Candidate("alice/so101_a", True, robot_type="so101", n_episodes=50, data_bytes=2**20),
             fleet.Candidate("zed/koch", False, "robot_type 'koch' is not an SO-100/101")]
    seen = {}

    def fake_discover(limit, max_inspect=None, on_progress=None):
        seen["discover"] = (limit, max_inspect)
        for c in cands:
            on_progress(c)
        return cands

    def fake_scan(candidates, n_train, n_eval, max_evaluations, time_budget_s, on_progress, lock_dir, seed, cross_n):
        seen["scan"] = (list(candidates), n_train, n_eval, max_evaluations, time_budget_s, lock_dir, seed, cross_n)
        on_progress({"type": "start", "dataset": "alice/so101_a", "index": 0, "total": 1})
        on_progress({"type": "fit", "dataset": "alice/so101_a", "evaluation": 25, "best_score": 2.0, "elapsed_s": 1.0})
        on_progress({"type": "robot", "dataset": "alice/so101_a", "entry": {**FLEET_REPORT["robots"][0], "elapsed_s": 12.0}})
        on_progress({"type": "failed", "dataset": "bob/x", "error": "boom"})
        on_progress({"type": "skipped", "dataset": "carol/y", "reason": "time budget exhausted"})
        on_progress({"type": "done", "summary": {}, "path": str(tmp_path / "fleet.json")})
        return fleet.FleetReport.from_dict({**FLEET_REPORT, "created_at": "", "antares_version": ""})

    monkeypatch.setattr(fleet, "discover", fake_discover)
    monkeypatch.setattr(fleet, "scan", fake_scan)
    assert cli.main(["fleet", "scan", "--limit", "7", "--budget-s", "90", "--max-evaluations", "5",
                     "--lock-dir", str(tmp_path), "--cross-n", "3"]) == 0
    out = capsys.readouterr().out
    assert seen["discover"] == (7, None)
    assert seen["scan"] == (cands, 20, 10, 5, 90.0, str(tmp_path), 0, 3)
    assert "1 accepted, 1 rejected" in out and "koch" in out
    assert "[1/1] alice/so101_a" in out and "eval   25" in out
    assert "4.000 -> 1.500 deg (62% closed), latency 3 frames, kp 1200" in out
    assert "FAILED bob/x: boom" in out and "skipped carol/y" in out and "saved" in out
    assert "2 robots calibrated" in out  # final tables


def test_synth_rejects_bad_wear(capsys):
    assert cli.main(["synth", "/nonexistent/out", "--wear", "stiffness=2"]) == 2
    assert "unknown parameter 'stiffness'" in capsys.readouterr().err


def test_synth_lock_takes_a_name_or_a_path_and_fails_cleanly(monkeypatch, tmp_path, capsys):
    from antares import server, synth

    seen = {}

    def fake_synthesize(params, **kw):
        seen["kp"] = params.kp
        raise SystemExit(99)  # stop before MuJoCo; the params are what matters

    monkeypatch.setattr(synth, "synthesize", fake_synthesize)
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "LOCK_SUFFIX", ".lock.json")
    path = tmp_path / "unit-a.lock.json"
    path.write_text(json.dumps({"name": "unit-a", "params": {"kp": 123.0}, "result": {}}))
    with pytest.raises(SystemExit, match="99"):
        cli.main(["synth", str(tmp_path / "out"), "--lock", str(path)])
    assert seen["kp"] == 123.0
    with pytest.raises(SystemExit, match="99"):
        cli.main(["synth", str(tmp_path / "out"), "--lock", "unit-a"])
    # a lock without a top-level "params" still works (older locks carry them in result)
    path.write_text(json.dumps({"name": "unit-a", "result": {"params": {"kp": 77.0}}}))
    with pytest.raises(SystemExit, match="99"):
        cli.main(["synth", str(tmp_path / "out"), "--lock", "unit-a"])
    assert seen["kp"] == 77.0
    assert cli.main(["synth", str(tmp_path / "out"), "--lock", "ghost"]) == 2
    assert "no lock named 'ghost'" in capsys.readouterr().err
    assert cli.main(["synth", str(tmp_path / "out"), "--lock", "locks/with/slash.json"]) == 2
    assert "invalid lock name" in capsys.readouterr().err


def test_check_explain_evaluations_has_a_floor(monkeypatch, capsys):
    calls = []
    _patch_check(monkeypatch, _check_result(3.0))
    _patch_diagnose(monkeypatch, calls)
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--explain", "--explain-evaluations", "0"]) == 2
    assert "--explain-evaluations must be >= 10" in capsys.readouterr().err
    assert calls == []


# -- alerts (--notify) ----------------------------------------------------------


class _WebhookServer:
    """A localhost webhook that records POST bodies and answers ``status``."""

    def __init__(self, status=200):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        received = self.received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                received.append({"content_type": self.headers.get("Content-Type"), "body": json.loads(body)})
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b"ok" if status < 400 else b"nope")

            def log_message(self, *a):
                pass

        self._srv = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._srv.server_port}/hook"

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture
def webhook():
    srv = _WebhookServer()
    try:
        yield srv
    finally:
        srv.close()


def _closed_port_url():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}/hook"


def test_check_notify_failed_post_keeps_the_exit_code(monkeypatch, capsys):
    monkeypatch.delenv("ANTARES_WEBHOOK_URL", raising=False)
    dead = _closed_port_url()
    _patch_check(monkeypatch, _check_result(3.0))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--notify", dead]) == 1
    out, err = capsys.readouterr()
    assert out.rstrip().splitlines()[-2].startswith("DRIFTED:")
    assert "warning: alert not sent: could not reach webhook" in err

    _patch_check(monkeypatch, _check_result(1.2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--notify", dead, "--notify-when", "always"]) == 0
    assert "warning: alert not sent" in capsys.readouterr().err

    def boom(url, payload, timeout=10.0):
        raise RuntimeError("kaboom")

    import antares.notify
    monkeypatch.setattr(antares.notify, "send", boom)
    _patch_check(monkeypatch, _check_result(0.5, n=2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--notify", "https://example.com/x",
                     "--notify-when", "failure", "--json"]) == 2
    out, err = capsys.readouterr()
    assert json.loads(out)["verdict"] == "insufficient_data"
    assert "warning: alert not sent: kaboom" in err


def test_check_notify_posts_only_the_verdicts_asked_for(monkeypatch, capsys, webhook):
    monkeypatch.delenv("ANTARES_WEBHOOK_URL", raising=False)
    _patch_check(monkeypatch, _check_result(1.2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--notify", webhook.url]) == 0
    assert webhook.received == []  # ok is quiet by default
    assert "alert" not in capsys.readouterr().err

    _patch_check(monkeypatch, _check_result(3.0))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--notify", webhook.url]) == 1
    (req,) = webhook.received
    assert req["content_type"] == "application/json"
    body = req["body"]
    assert body["event"] == "antares.check" and body["verdict"] == "drifted"
    assert body["text"].startswith("DRIFTED unit-a on fake/robot: 3.00 deg RMSE (limit 1.50)")
    assert body["result"]["lock"] == "unit-a" and body["result"]["n_episodes"] == 6
    assert "alert sent (json webhook)" in capsys.readouterr().err

    # the URL can come from the environment; slack format can be forced
    monkeypatch.setenv("ANTARES_WEBHOOK_URL", webhook.url)
    _patch_check(monkeypatch, _check_result(1.2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--notify-when", "always",
                     "--notify-format", "slack"]) == 0
    assert len(webhook.received) == 2
    slack = webhook.received[1]["body"]
    assert slack["blocks"][0]["type"] == "header" and slack["text"].startswith("OK unit-a")
    assert "alert sent (slack webhook)" in capsys.readouterr().err


def test_check_notify_dry_run_keeps_json_stdout_pure(monkeypatch, capsys, webhook):
    _patch_check(monkeypatch, _check_result(3.0))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--notify", webhook.url,
                     "--notify-dry-run", "--json"]) == 1
    out, err = capsys.readouterr()
    assert json.loads(out)["verdict"] == "drifted"
    assert webhook.received == []
    assert '"event": "antares.check"' in err and "alert rendered" in err

    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--notify", webhook.url, "--notify-dry-run"]) == 1
    out, err = capsys.readouterr()
    assert '"event": "antares.check"' in out and out.rstrip().splitlines()[0].startswith("fake/robot vs lock")


def test_check_notify_reports_a_check_that_could_not_run(monkeypatch, capsys, webhook):
    from antares import server

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "LOCK_DIR", __import__("pathlib").Path("/nonexistent/locks"))
    assert cli.main(["check", "fake/robot", "--lock", "ghost", "--notify", webhook.url]) == 2
    assert webhook.received == []  # drifted-only stays quiet with no verdict
    assert cli.main(["check", "fake/robot", "--lock", "ghost", "--notify", webhook.url,
                     "--notify-when", "failure"]) == 2
    (req,) = webhook.received
    assert req["body"]["verdict"] == "error" and req["body"]["result"] is None
    assert req["body"]["text"].startswith("ERROR ghost on fake/robot:")
    assert "error:" in capsys.readouterr().err


def test_notify_test_sends_a_synthetic_ok(monkeypatch, capsys, webhook):
    monkeypatch.delenv("ANTARES_WEBHOOK_URL", raising=False)
    assert cli.main(["notify-test", "--url", webhook.url]) == 0
    (req,) = webhook.received
    assert req["body"]["verdict"] == "ok" and req["body"]["text"].startswith("Antares webhook test\nOK example on")
    out = capsys.readouterr().out
    assert out.startswith("sent a synthetic OK alert as json to 127.0.0.1")
    assert "/hook" not in out  # the URL is a secret

    assert cli.main(["notify-test", "--url", webhook.url, "--format", "slack", "--title", "Arm A"]) == 0
    assert webhook.received[1]["body"]["blocks"][0]["text"]["text"] == "Arm A"

    assert cli.main(["notify-test", "--url", webhook.url, "--dry-run"]) == 0
    assert len(webhook.received) == 2 and '"event": "antares.check"' in capsys.readouterr().out

    assert cli.main(["notify-test"]) == 2
    assert "pass --url or set ANTARES_WEBHOOK_URL" in capsys.readouterr().err
    assert cli.main(["notify-test", "--url", _closed_port_url()]) == 1
    assert "error: alert not sent: could not reach webhook" in capsys.readouterr().err


@pytest.mark.slow
def test_synth_writes_a_dataset_check_reads(tmp_path, capsys):
    from antares.data import read_local

    out = tmp_path / "worn"
    assert cli.main(["synth", str(out), "--wear", "kp=0.5,latency_frames=2",
                     "--episodes", "2", "--frames", "30"]) == 0
    printed = capsys.readouterr().out
    assert f"wrote 2 episodes x 30 frames -> {out}" in printed
    assert "'latency_frames': 2" in printed and "'kp': 499.11" in printed
    ds = read_local(out)
    assert len(ds.episodes) == 2 and ds.episodes[0].state.shape == (30, 6)


def test_replay_with_lock_passes_command_map(monkeypatch, tmp_path, capsys):
    from antares import server

    cmap = {"gain": [1.0] * 6, "bias": [0.0, -2.0, 1.0, 0.0, 0.0, 0.0]}
    (tmp_path / "unit-a.lock.json").write_text(json.dumps({
        "name": "unit-a", "params": {"kp": 100.0}, "command_map": cmap, "result": {},
    }))
    seen = {}

    def fake_replay(ds, ep, params=None, command_map=None, robot=None):
        seen["cmap"] = command_map
        return {"episode": ep, "real_q": [[0] * 6], "gap": {"per_joint_deg": {}, "overall_deg": 0.5}}

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "LOCK_SUFFIX", ".lock.json")
    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "replay_episode", fake_replay)
    assert cli.main(["replay", "fake/robot", "--episode", "1", "--lock", "unit-a"]) == 0
    assert seen["cmap"].bias.tolist() == cmap["bias"]


def test_calibrate_fit_sensor_prints_comparison(monkeypatch, capsys):
    from antares import server

    per = {j: 1.0 for j in JOINTS}
    result = type("R", (), {"to_dict": lambda self: {
        "per_joint_before": {j: 4.0 for j in JOINTS}, "per_joint_after": per,
        "baseline_eval_score": 4.0, "eval_score": 0.75, "params": {"kp": 1.0},
        "n_evaluations": 10, "elapsed_s": 1.0,
    }})()
    lock = {
        "name": "unit-a",
        "command_map": {"gain": [1.0, 0.98, 1.0, 1.0, 1.0, 1.0], "bias": [0.0, -2.5, 0.0, 0.0, 0.0, 0.0]},
        "dynamics_only": {"eval_score": 1.5, "per_joint_after": {j: 1.5 for j in JOINTS}},
    }
    seen = {}

    def fake_run(ds, **kw):
        seen.update(kw)
        return result, lock, "locks/unit-a.lock.json"

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "run_calibration", fake_run)
    assert cli.main(["calibrate", "fake/robot", "--fit-sensor", "--sensor-mode", "kinematic"]) == 0
    out = capsys.readouterr().out
    assert seen["fit_sensor"] is True and seen["sensor_mode"] == "kinematic"
    assert "dynamics only" in out and "1.500" in out and "0.750" in out
    assert "sensor calibration" in out and "-2.500" in out and "0.9800" in out


# ---------------------------------------------------------------------------
# the follower's LeRobot calibration
# ---------------------------------------------------------------------------


def _calibration_file(path, shift_joint=None, shift=0):
    """A calibration matching the SO-101 model's ranges, one joint optionally shifted."""
    from antares import lerobot_calib as lc
    from antares.sim import load_model

    jr = load_model().jnt_range
    out = {}
    for i, j in enumerate(JOINTS):
        span = int(round((jr[i, 1] - jr[i, 0]) / lc.RAD_PER_TICK))
        lo = 1000 if j == "gripper" else lc.HALF_TURN - span // 2
        if j == shift_joint:
            lo += shift
        out[j] = {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": lo, "range_max": lo + span}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return path


def test_inspect_with_calibration_prints_the_disagreement_table(monkeypatch, capsys, tmp_path):
    import antares.data
    from antares import server

    monkeypatch.setattr(server, "fetch_hub_info", lambda repo_id: _info())
    monkeypatch.setattr(antares.data, "load_dataset", lambda repo_id, cache_dir=None: _dataset(repo_id, n=20))
    path = _calibration_file(tmp_path / "arm.json", shift_joint="wrist_roll", shift=100)  # 8.8 deg
    assert cli.main(["inspect", "fake/robot", "--calibration", str(path)]) == 0
    out = capsys.readouterr().out
    assert f"LeRobot calibration {path} vs the so101 model" in out
    assert "range_min" in out and "FLAG" in out
    assert "warning: wrist_roll is off by up to 8.8 deg: the model's assumed range does not match" in out
    assert "use --calibration when calibrating" in out
    # the untouched joints are not flagged
    assert out.count("FLAG") == 1


def test_inspect_with_robot_id_uses_lerobot_dir(monkeypatch, capsys, tmp_path):
    import antares.data
    from antares import server

    monkeypatch.setattr(server, "fetch_hub_info", lambda repo_id: _info())
    monkeypatch.setattr(antares.data, "load_dataset", lambda repo_id, cache_dir=None: _dataset(repo_id, n=20))
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "cal"))
    _calibration_file(tmp_path / "cal" / "robots" / "so_follower" / "my_arm.json")
    assert cli.main(["inspect", "fake/robot", "--robot-id", "my_arm"]) == 0
    out = capsys.readouterr().out
    assert "all joints within 2 deg" in out and "FLAG" not in out
    assert cli.main(["inspect", "fake/robot", "--robot-id", "nope"]) == 2
    err = capsys.readouterr().err
    assert "no LeRobot calibration named 'nope'" in err and "so101_follower" in err


def test_calibrate_cli_forwards_the_calibration(monkeypatch, capsys, tmp_path):
    from antares import server

    seen = {}
    result = type("R", (), {"to_dict": lambda self: {
        "per_joint_before": {j: 4.0 for j in JOINTS}, "per_joint_after": {j: 1.0 for j in JOINTS},
        "baseline_eval_score": 4.0, "eval_score": 0.75, "params": {"kp": 1.0},
        "n_evaluations": 10, "elapsed_s": 1.0,
    }})()
    lock = {"name": "unit-c", "calibration": {"source": "/x/arm.json", "robot_id": "arm",
                                              "unit_map_disagreement_deg": {"wrist_roll": 8.79, "gripper": 0.0}}}

    def fake_run(ds, **kw):
        seen.update(kw)
        return result, lock, "locks/unit-c.lock.json"

    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "run_calibration", fake_run)
    assert cli.main(["calibrate", "fake/robot", "--calibration", "/x/arm.json"]) == 0
    out = capsys.readouterr().out
    assert seen["calibration_path"] == "/x/arm.json" and seen["robot_id"] is None
    assert seen["robot_type"] == "so101_follower"
    assert "unit map from the arm's LeRobot calibration" in out
    assert "unit map from LeRobot calibration /x/arm.json (robot id arm)" in out and "wrist_roll=8.79" in out
    assert cli.main(["calibrate", "fake/robot", "--robot-id", "arm", "--robot-type", "so100_follower"]) == 0
    assert seen["calibration_path"] is None and seen["robot_id"] == "arm" and seen["robot_type"] == "so100_follower"
