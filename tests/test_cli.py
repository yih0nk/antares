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
    for cmd in ("serve", "inspect", "replay", "calibrate", "cross", "locks"):
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
        argv += ["--per-joint", "--per-joint-evaluations", "7"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert calls[0]["per_joint"] is per_joint
    assert calls[0]["per_joint_max_evaluations"] == (7 if per_joint else 600)
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
