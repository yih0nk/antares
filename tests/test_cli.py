from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from antares import cli
from antares.data import Dataset, Episode

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


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


def test_inspect(monkeypatch, capsys):
    import antares.data

    monkeypatch.setattr(antares.data, "load_dataset", lambda repo_id, cache_dir=None: _dataset(repo_id))
    assert cli.main(["inspect", "fake/robot"]) == 0
    out = capsys.readouterr().out
    assert "fake/robot: so101_follower, 3 episodes, 30 frames @ 30 fps" in out
    assert "shoulder_pan" in out and "gripper" in out


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
    assert "-- shared stage --" in out
    assert "shared: 5 evaluations" in out
    assert "saved lock x" in out
    if per_joint:
        assert "+ 7 per-joint evaluations" in out and "-- per_joint stage --" in out
        assert "per_joint: 7 evaluations" in out
        assert "params (per joint)" in out and "elbow_flex" in out and "30" in out
    else:
        assert "per-joint evaluations" not in out and "-- per_joint stage --" not in out
        assert "params: kp=100" in out


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
    assert out.rstrip().splitlines()[-1].startswith("OK:")
    (ds, lock, kw), = calls
    assert ds.repo_id == "fake/robot" and lock == "unit-a"
    assert kw == {"threshold_deg": 2.0, "ratio": 1.5, "last": 8, "min_episodes": 5}


def test_check_drift_exits_one(monkeypatch, capsys):
    _patch_check(monkeypatch, _check_result(3.0))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "<- worst" in out
    assert out.rstrip().splitlines()[-1].startswith("DRIFTED:")


def test_check_insufficient_data_exits_two(monkeypatch, capsys):
    _patch_check(monkeypatch, _check_result(0.5, n=2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a"]) == 2
    out = capsys.readouterr().out
    assert "INSUFFICIENT DATA" in out and "overall" not in out


def test_check_json_output(monkeypatch, capsys):
    _patch_check(monkeypatch, _check_result(1.2))
    assert cli.main(["check", "fake/robot", "--lock", "unit-a", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "ok" and out["passed"] is True
    assert out["score"] == pytest.approx(1.2) and out["limit"] == pytest.approx(1.5)
    assert set(out["per_joint"]) == set(JOINTS)


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
    assert out.rstrip().splitlines()[-1].endswith("likely cause: latency_frames +1 frame (alone recovers 2.60 deg)")
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
    assert cli.main(["check", "nope/none", "--lock", "ghost"]) == 2
    assert "could not load nope/none" in capsys.readouterr().err


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
    monkeypatch.setattr(server, "_arm", types.SimpleNamespace(unit_map=unit_map))

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
    monkeypatch.setattr(server, "replay_episode", lambda ds, ep, params=None, cmap=None: fake_out)
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

    def fake_replay(ds, ep, params=None, command_map=None):
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
