from __future__ import annotations

import numpy as np
import pytest

from antares import cli, excite


def test_chirp_starts_and_ends_at_zero_and_sweeps_up():
    c = excite.chirp(150, 30.0, 0.3, 2.0)
    assert c[0] == pytest.approx(0.0, abs=1e-9)
    assert abs(c[-1]) < 0.05
    assert np.max(np.abs(c)) <= 1.0
    # zero crossings get closer together as the frequency rises
    z = np.flatnonzero(np.diff(np.sign(c[1:])) != 0)
    gaps = np.diff(z)
    assert gaps[0] > gaps[-1]


def test_dance_is_about_a_minute_and_starts_and_ends_at_home():
    d = excite.dance()
    assert d.commands.shape[1] == 6
    assert 55 <= d.duration <= 65
    np.testing.assert_allclose(d.commands[0], excite.HOME)
    np.testing.assert_allclose(d.commands[-1], excite.HOME)
    # segments tile the trajectory with no gaps
    assert d.segments[0][1] == 0 and d.segments[-1][2] == len(d.commands)
    assert all(a[2] == b[1] for a, b in zip(d.segments, d.segments[1:]))


def test_dance_moves_one_body_joint_at_a_time_in_its_excitation_segments():
    d = excite.dance()
    home = np.asarray(excite.HOME)
    for label, a, b in d.segments:
        if "step" not in label and "chirp" not in label:
            continue
        joint = label.split()[0]
        j = excite.BODY.index(joint)
        moved = np.flatnonzero(np.abs(d.commands[a:b] - home).max(axis=0) > 1e-9)
        assert set(moved) <= {j}, label


def test_dance_scales_with_fps():
    assert len(excite.dance(fps=50.0).commands) == pytest.approx(len(excite.dance(fps=30.0).commands) * 50 / 30, rel=0.02)


@pytest.mark.slow
def test_default_dance_is_safe():
    d = excite.dance()
    s = excite.check_safety(d.commands, d.fps, d.steps)
    assert s["ok"], s
    assert s["min_tip_z"] - excite.TABLE_Z >= excite.CLEARANCE
    assert s["max_abs_units"] <= excite.LIMIT
    assert s["max_step_units"] <= 25


@pytest.mark.slow
def test_safety_flags_a_reach_into_the_table_and_a_fast_sweep():
    # the LeRobot rest pose: folded, fingertip on the table
    rest = np.tile([0.0, -100.0, 100.0, 100.0, 0.0, 0.0], (10, 1))
    s = excite.check_safety(rest, 30.0)
    assert not s["ok"]
    assert s["min_tip_z"] < excite.TABLE_Z + excite.CLEARANCE
    fast = np.tile(np.asarray(excite.HOME), (60, 1))
    fast[:, 0] += 20 * np.sin(2 * np.pi * 5.0 * np.arange(60) / 30.0)
    s = excite.check_safety(fast, 30.0)
    assert not s["ok"] and s["peak_speed_rad_s"]["shoulder_pan"] > excite.MAX_SPEED


@pytest.mark.parametrize("suffix", [".csv", ".json", ".npy"])
def test_write_read_round_trip(tmp_path, suffix):
    d = excite.dance()
    path = excite.write(tmp_path / f"dance{suffix}", d)
    a, fps = excite.read(path)
    np.testing.assert_allclose(a, d.commands, atol=1e-3)
    assert fps == pytest.approx(30.0)


def test_csv_uses_lerobot_action_names(tmp_path):
    path = excite.write(tmp_path / "dance.csv", excite.dance())
    header = path.read_text().splitlines()[0].split(",")
    assert header == ["timestamp", "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                      "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def test_write_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ValueError, match="csv"):
        excite.write(tmp_path / "dance.txt", excite.dance())


def test_episode_replays_the_commands():
    d = excite.dance()
    ep = d.episode()
    assert len(ep) == len(d.commands)
    np.testing.assert_array_equal(ep.action, d.commands)


@pytest.mark.slow
def test_cli_excite_writes_the_dance(tmp_path, capsys):
    out = tmp_path / "dance.json"
    assert cli.main(["excite", "--out", str(out)]) == 0
    assert out.exists()
    text = capsys.readouterr().out
    assert "ok" in text and "sim-only" in text


def test_cli_excite_rejects_unknown_suffix(tmp_path, capsys):
    assert cli.main(["excite", "--out", str(tmp_path / "dance.txt")]) == 2
    assert "expected a .csv" in capsys.readouterr().err
