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
    for cmd in ("serve", "inspect", "replay", "calibrate", "cross", "locks", "compensate"):
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
    monkeypatch.setattr(server, "replay_episode", lambda ds, ep, params=None: fake_out)
    assert cli.main(["replay", "fake/robot", "--episode", "1"]) == 0
    out = capsys.readouterr().out
    assert "episode 1" in out and "default params" in out
    assert "shoulder_pan" in out and "overall" in out and "1.250" in out
