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
    for cmd in ("serve", "inspect", "replay", "calibrate", "cross", "locks", "dr"):
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
