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
    for cmd in ("serve", "inspect", "replay", "calibrate", "cross", "locks", "check"):
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
