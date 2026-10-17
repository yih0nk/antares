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
    for cmd in ("serve", "inspect", "replay", "calibrate", "cross", "locks", "events"):
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
