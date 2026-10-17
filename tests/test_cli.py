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
    for cmd in ("serve", "inspect", "replay", "calibrate", "cross", "locks", "fleet"):
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
