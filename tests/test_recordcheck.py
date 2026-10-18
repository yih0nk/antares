"""``antares record-check``: drive lerobot-record, then run the drift gate.

No LeRobot and no MuJoCo: the recorder is a fake ``run`` that writes a
minimal LeRobot-format directory, and the check uses the exact toy
``simulate`` from tests/test_check.py (sim == command, so the error is the
offset the episodes carry).
"""

from __future__ import annotations

import json
import os
import shlex
import stat
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from antares import cli
from antares import recordcheck as rc
from antares.calibrate import CalibrationResult, SimParams
from antares.check import check
from antares.data import Episode, read_local
from antares.lock import RobotLock

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NOW = datetime(2026, 10, 17, 9, 30, 5)


def exact_simulate(params, ep):
    """Sim == command (as in tests/test_check.py); the error is the episode's offset."""
    return np.radians(ep.state), np.radians(ep.action)


def make_lock(name="unit-a") -> RobotLock:
    """A lock whose held-out score is 1.0 deg on every joint (limit 1.5 at ratio 1.5)."""
    a = np.linspace(0, 10, 20)[:, None].repeat(6, axis=1)
    eps = [Episode(index=i, fps=30.0, joint_names=JOINTS, state=a + 0.5, action=a) for i in range(2)]
    result = CalibrationResult(
        params=SimParams(kp=1200.0, latency_frames=1), baseline=SimParams(),
        train_score=0.9, eval_score=1.0, baseline_train_score=4.0, baseline_eval_score=4.5,
        per_joint_before={j: 4.0 for j in JOINTS}, per_joint_after={j: 1.0 for j in JOINTS},
        history=[(0, 4.0), (1, 0.9)], n_evaluations=1, elapsed_s=0.1, train_episodes=[0, 1], eval_episodes=[2],
    )
    return RobotLock.create("fake/so101_unit_a", "so101_follower", eps, result, name=name)


def opts(**over) -> rc.RecordOptions:
    base = dict(robot_port="/dev/tty.usbmodem1", robot_id="arm7", teleop_port="/dev/tty.usbmodem2")
    base.update(over)
    return rc.RecordOptions(**base)


def write_dataset(root: Path, n_episodes=6, T=20, offset_deg=1.2) -> Path:
    """A LeRobot v3 layout: meta/info.json + one parquet, state = action + offset."""
    names = [f"{j}.pos" for j in JOINTS]
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "robot_type": "so101_follower", "fps": 30,
        "total_episodes": n_episodes, "total_frames": n_episodes * T,
        "features": {"action": {"names": names}, "observation.state": {"names": names}},
    }))
    (root / "data" / "chunk-000").mkdir(parents=True)
    ep, fr, st, ac = [], [], [], []
    for i in range(n_episodes):
        a = np.linspace(0, 10, T)
        for t in range(T):
            ep.append(i)
            fr.append(t)
            ac.append([float(a[t])] * 6)
            st.append([float(a[t]) + offset_deg] * 6)
    pq.write_table(pa.table({"episode_index": ep, "frame_index": fr, "observation.state": st, "action": ac}),
                   root / "data" / "chunk-000" / "file-000.parquet")
    return root


# -- build_record_command ------------------------------------------------------


def test_build_record_command_defaults():
    cmd = rc.build_record_command(opts(), root=Path("recordings/arm7/x"))
    assert cmd == [
        "lerobot-record",
        "--robot.type=so101_follower",
        "--robot.port=/dev/tty.usbmodem1",
        "--robot.id=arm7",
        "--teleop.type=so101_leader",
        "--teleop.port=/dev/tty.usbmodem2",
        "--teleop.id=arm7_leader",
        "--dataset.repo_id=local/antares-check-arm7",
        "--dataset.root=recordings/arm7/x",
        "--dataset.num_episodes=5",
        "--dataset.single_task=antares check",
        "--dataset.push_to_hub=false",
    ]
    assert not any("push_to_hub=true" in a for a in cmd)


def test_build_record_command_overrides_and_passthrough():
    o = opts(robot_type="so100_follower", teleop_type="so100_leader", teleop_id="lead", num_episodes=8,
             task="wave", repo_id="acme/daily", episode_time_s=20, reset_time_s=5.5,
             extra=["--robot.cameras={}", "--display_data=true"], root=Path("/tmp/r"))
    cmd = rc.build_record_command(o, recorder="/opt/lerobot/bin/lerobot-record")
    assert cmd[0] == "/opt/lerobot/bin/lerobot-record"
    for flag in ("--robot.type=so100_follower", "--teleop.type=so100_leader", "--teleop.id=lead",
                 "--dataset.num_episodes=8", "--dataset.single_task=wave", "--dataset.repo_id=acme/daily",
                 "--dataset.root=/tmp/r", "--dataset.episode_time_s=20", "--dataset.reset_time_s=5.5"):
        assert flag in cmd, flag
    assert cmd[-2:] == ["--robot.cameras={}", "--display_data=true"]


def test_build_record_command_default_root_is_timestamped_per_robot():
    cmd = rc.build_record_command(opts())
    root = next(a for a in cmd if a.startswith("--dataset.root=")).split("=", 1)[1]
    assert root.startswith(os.path.join("recordings", "arm7") + os.sep)
    assert rc.default_root("arm7", NOW) == Path("recordings/arm7/20261017-093005")


def test_build_record_command_validates():
    with pytest.raises(ValueError, match="teleop_port"):
        rc.build_record_command(opts(teleop_port=None))
    # a keyboard teleop has no port
    assert "--teleop.type=keyboard" in rc.build_record_command(opts(teleop_type="keyboard", teleop_port=None))
    with pytest.raises(ValueError, match="num_episodes"):
        rc.build_record_command(opts(num_episodes=0))
    with pytest.raises(ValueError, match="robot_port"):
        rc.build_record_command(opts(robot_port=""))


# -- find_recorder / record ------------------------------------------------------


def test_find_recorder(monkeypatch, tmp_path):
    monkeypatch.setattr(rc.shutil, "which", lambda name: None)
    assert rc.find_recorder() is None
    exe = tmp_path / "lerobot-record"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    assert rc.find_recorder(str(exe)) == str(exe)
    assert rc.find_recorder(str(tmp_path / "missing")) is None
    monkeypatch.setattr(rc.shutil, "which", lambda name: "/usr/bin/" + name)
    assert rc.find_recorder() == "/usr/bin/lerobot-record"


def test_record_without_recorder_raises_with_install_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(rc.shutil, "which", lambda name: None)
    with pytest.raises(rc.RecordError) as e:
        rc.record(opts(root=tmp_path / "r"), run=lambda *a, **k: pytest.fail("must not run"))
    msg = str(e.value)
    assert "lerobot-record: not found" in msg and "pip install 'lerobot[core_scripts,feetech]'" in msg


def test_record_runs_the_command_and_returns_the_root(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rc.shutil, "which", lambda name: "/env/bin/lerobot-record")
    calls = []

    def fake_run(cmd, check=False, **kw):
        calls.append(cmd)
        root = Path(next(a for a in cmd if a.startswith("--dataset.root=")).split("=", 1)[1])
        write_dataset(root, n_episodes=2)
        return type("P", (), {"returncode": 0})()

    root = rc.record(opts(), run=fake_run, now=NOW)
    assert root == Path("recordings/arm7/20261017-093005")
    (cmd,) = calls
    assert cmd[0] == "/env/bin/lerobot-record" and f"--dataset.root={root}" in cmd
    assert "--dataset.push_to_hub=false" in cmd
    assert len(read_local(root).episodes) == 2
    err = capsys.readouterr().err
    assert "recording 5 episode(s)" in err and shlex.join(cmd) in err


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    """Default roots are relative (recordings/...); keep them out of the repo."""
    monkeypatch.chdir(tmp_path)


def test_record_failures_are_record_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(rc.shutil, "which", lambda name: "/env/bin/lerobot-record")
    failing = lambda cmd, **kw: type("P", (), {"returncode": 1})()  # noqa: E731
    with pytest.raises(rc.RecordError, match="exited with status 1"):
        rc.record(opts(root=tmp_path / "a"), run=failing)
    nothing = lambda cmd, **kw: type("P", (), {"returncode": 0})()  # noqa: E731
    with pytest.raises(rc.RecordError, match="wrote no dataset"):
        rc.record(opts(root=tmp_path / "b"), run=nothing)

    def vanished(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    with pytest.raises(rc.RecordError, match="not found"):
        rc.record(opts(root=tmp_path / "c"), run=vanished)
    write_dataset(tmp_path / "d", n_episodes=1)
    with pytest.raises(rc.RecordError, match="already holds a dataset"):
        rc.record(opts(root=tmp_path / "d"), run=nothing)
    assert rc.record(opts(root=tmp_path / "e"), dry_run=True, run=vanished) == tmp_path / "e"


# -- record_and_check ------------------------------------------------------------


def toy_run_check(monkeypatch, calls):
    """``server.run_check`` with the toy simulate: same signature, same CheckResult."""
    from antares import server

    def run_check(ds, lock, **kw):
        calls.append((ds, lock, kw))
        lock_d = lock if isinstance(lock, dict) else server.load_lock_dict(lock)
        return check(ds.episodes, lock_d, simulate=exact_simulate, dataset=ds.repo_id, **kw)

    monkeypatch.setattr(server, "_DATASETS", {})
    monkeypatch.setattr(server, "run_check", run_check)


def test_record_and_check_skip_record_on_a_recorded_directory(monkeypatch, tmp_path):
    root = write_dataset(tmp_path / "rec", n_episodes=6, offset_deg=1.2)
    calls = []
    toy_run_check(monkeypatch, calls)
    lock = make_lock().to_dict()
    res = rc.record_and_check(
        opts(root=root), lock, check_kwargs={"threshold_deg": None, "ratio": 1.5, "min_episodes": 5},
        skip_record=True, run=lambda *a, **k: pytest.fail("must not record"),
    )
    assert res.passed and res.verdict == "ok"
    assert res.score == pytest.approx(1.2) and res.limit == pytest.approx(1.5)
    assert res.n_episodes == 6 and res.dataset == str(root)
    (ds, _, kw), = calls
    assert ds.repo_id == str(root) and kw == {"threshold_deg": None, "ratio": 1.5, "min_episodes": 5}
    assert rc.banner(res, "arm7") == "PASS arm7: 1.200 deg (limit 1.500)"

    drifted = rc.record_and_check(opts(root=write_dataset(tmp_path / "worn", offset_deg=3.0)), lock,
                                  check_kwargs={"min_episodes": 5}, skip_record=True)
    assert drifted.verdict == "drifted"
    assert rc.banner(drifted, "arm7").startswith("FAIL arm7: 3.000 deg (limit 1.500), worst joint ")

    few = rc.record_and_check(opts(root=write_dataset(tmp_path / "few", n_episodes=2)), lock,
                              check_kwargs={"min_episodes": 5}, skip_record=True)
    assert few.verdict == "insufficient_data"
    assert rc.banner(few, "arm7") == "FAIL arm7: insufficient data, 2 episode(s), need 5"


def test_record_and_check_skip_record_needs_a_dataset(tmp_path):
    with pytest.raises(rc.RecordError, match="--skip-record: no LeRobot dataset"):
        rc.record_and_check(opts(root=tmp_path / "empty"), "unit-a", skip_record=True)


def test_record_and_check_records_then_checks(monkeypatch, tmp_path):
    monkeypatch.setattr(rc.shutil, "which", lambda name: "/env/bin/lerobot-record")
    calls = []
    toy_run_check(monkeypatch, calls)

    def fake_run(cmd, check=False, **kw):
        root = Path(next(a for a in cmd if a.startswith("--dataset.root=")).split("=", 1)[1])
        write_dataset(root, n_episodes=5, offset_deg=0.8)
        return type("P", (), {"returncode": 0})()

    lock = make_lock().to_dict()
    res = rc.record_and_check(opts(root=tmp_path / "r"), lock, check_kwargs={"ratio": 1.5}, run=fake_run)
    assert res.passed and res.n_episodes == 5 and res.dataset == str(tmp_path / "r")
    assert rc.record_and_check(opts(root=tmp_path / "q"), lock, dry_run=True) is None


# -- CLI -----------------------------------------------------------------------------


BASE = ["record-check", "--lock", "unit-a", "--robot-port", "/dev/ttyA", "--robot-id", "arm7",
        "--teleop-port", "/dev/ttyB"]


def test_cli_dry_run_prints_both_commands_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(rc.shutil, "which", lambda name: None)
    code = cli.main(BASE + ["--dry-run", "--threshold", "2", "--root", "out/run1", "--extra=--display_data=true"])
    assert code == 0
    cap = capsys.readouterr()
    lines = cap.out.splitlines()
    assert lines[0] == "would run:"
    record_cmd = shlex.split(lines[1])
    assert record_cmd[:4] == ["lerobot-record", "--robot.type=so101_follower", "--robot.port=/dev/ttyA", "--robot.id=arm7"]
    assert "--teleop.port=/dev/ttyB" in record_cmd and "--teleop.id=arm7_leader" in record_cmd
    assert "--dataset.root=out/run1" in record_cmd and "--dataset.num_episodes=5" in record_cmd
    assert "--dataset.push_to_hub=false" in record_cmd and record_cmd[-1] == "--display_data=true"
    assert lines[2] == "then:"
    assert shlex.split(lines[3]) == ["antares", "check", "out/run1", "--lock", "unit-a",
                                     "--threshold", "2", "--ratio", "1.5", "--min-episodes", "5"]
    assert "note: lerobot-record not found" in cap.err and "lerobot[core_scripts,feetech]" in cap.err
    assert not os.path.exists("out")


def test_cli_dry_run_honours_recorder_and_bumps_episodes(monkeypatch, capsys):
    monkeypatch.setattr(rc.shutil, "which", lambda name: None)
    code = cli.main(BASE + ["--dry-run", "--recorder", "/env/bin/lerobot-record", "--episodes", "2", "--min-episodes", "4"])
    assert code == 0
    cap = capsys.readouterr()
    cmd = shlex.split(cap.out.splitlines()[1])
    assert cmd[0] == "/env/bin/lerobot-record" and "--dataset.num_episodes=4" in cmd
    assert "warning: --episodes 2 is below --min-episodes 4" in cap.err
    assert "note: /env/bin/lerobot-record not found" in cap.err


def test_cli_missing_teleop_port_is_usage(capsys):
    code = cli.main(["record-check", "--lock", "unit-a", "--robot-port", "/dev/ttyA", "--robot-id", "arm7", "--dry-run"])
    assert code == 2
    assert "teleop_port is required" in capsys.readouterr().err


def test_cli_skip_record_pass_fail_and_json(monkeypatch, tmp_path, capsys):
    from antares import server

    calls = []
    toy_run_check(monkeypatch, calls)
    monkeypatch.setattr(server, "load_lock_dict", lambda name, lock_dir=None: make_lock(name=name).to_dict())
    good = write_dataset(tmp_path / "good", offset_deg=1.2)
    assert cli.main(BASE + ["--skip-record", "--root", str(good)]) == 0
    out = capsys.readouterr().out.rstrip().splitlines()
    assert out[-2].startswith("OK:") and out[-1] == "PASS arm7: 1.200 deg (limit 1.500)"
    (_, lock, kw), = calls
    assert lock == "unit-a" and kw == {"threshold_deg": None, "ratio": 1.5, "min_episodes": 5}

    bad = write_dataset(tmp_path / "bad", offset_deg=3.0)
    assert cli.main(BASE + ["--skip-record", "--root", str(bad)]) == 1
    out = capsys.readouterr().out.rstrip().splitlines()
    assert out[-2].startswith("DRIFTED:") and out[-1].startswith("FAIL arm7: 3.000 deg (limit 1.500)")

    assert cli.main(BASE + ["--skip-record", "--root", str(bad), "--threshold", "4", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["passed"] is True and d["threshold_deg"] == 4.0 and d["dataset"] == str(bad)

    few = write_dataset(tmp_path / "few", n_episodes=2)
    assert cli.main(BASE + ["--skip-record", "--root", str(few)]) == 2
    assert capsys.readouterr().out.rstrip().splitlines()[-1] == "FAIL arm7: insufficient data, 2 episode(s), need 5"


def test_cli_lock_file_and_recording_failure_exits_three(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rc.shutil, "which", lambda name: None)
    lock_file = tmp_path / "arm.lock.json"
    lock_file.write_text(json.dumps(make_lock().to_dict()))
    code = cli.main(["record-check", "--lock", str(lock_file), "--robot-port", "/dev/ttyA", "--robot-id", "arm7",
                     "--teleop-port", "/dev/ttyB", "--root", str(tmp_path / "r")])
    assert code == 3
    err = capsys.readouterr().err
    assert "lerobot-record: not found" in err and "pip install 'lerobot[core_scripts,feetech]'" in err

    assert cli.main(BASE + ["--skip-record", "--root", str(tmp_path / "nothing")]) == 3
    assert "--skip-record: no LeRobot dataset" in capsys.readouterr().err


def test_help_lists_record_check(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0 and "record-check" in capsys.readouterr().out
