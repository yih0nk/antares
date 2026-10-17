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
    for cmd in ("serve", "inspect", "replay", "calibrate", "cross", "drift", "dr", "locks", "export"):
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


def test_replay_prints_table(monkeypatch, capsys):
    from antares import server

    fake_out = {
        "episode": 1, "real_q": [[0] * 6] * 10,
        "gap": {"per_joint_deg": {j: 1.25 for j in JOINTS}, "overall_deg": 1.25},
    }
    monkeypatch.setattr(server, "_DATASETS", {"fake/robot": _dataset()})
    monkeypatch.setattr(server, "replay_episode", lambda ds, ep, params=None: fake_out)
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
