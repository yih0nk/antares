from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from antares import servo_bench as sb
from antares.data import read_local
from antares.sim import Arm

SKETCH = Path(__file__).resolve().parents[1] / "bench" / "servo_log" / "servo_log.ino"


def test_python_program_matches_the_sketch():
    src = SKETCH.read_text()
    steps = re.search(r"STEPS\[\] = \{([^}]*)\}", src).group(1)
    assert tuple(int(v) for v in steps.replace("\n", " ").split(",")) == sb._STEPS
    holds = re.search(r"holds\[5\] = \{([^}]*)\}", src).group(1)
    assert [float(v) for v in holds.split(",")] == [h[2] for h in sb.HOLDS]
    assert "TICK_MS = 20" in src and "LO = 30.0, HI = 150.0" in src


def test_program_is_whole_degrees_inside_the_safe_range():
    cmd = sb.program()
    assert len(cmd) == 3000
    assert np.all(cmd == np.round(cmd))
    assert cmd.min() >= 30 and cmd.max() <= 150
    assert cmd[0] == 30 and cmd[int(9.5 * 50)] == 150


def test_sg92r_model_replays_a_step():
    arm = Arm(robot="sg92r")
    q = arm.rollout(np.radians(np.full((40, 1), 120.0)), np.radians([60.0]), 50.0)
    assert np.degrees(q[-1, 0]) == pytest.approx(120.0, abs=1.0)
    assert np.degrees(q[1, 0]) < 100  # rate-limited, not a jump


def test_simulate_then_import_recovers_the_pot_and_episodes(tmp_path):
    csv = sb.simulate_run(tmp_path / "a.csv", pot_scale=0.3, pot_offset=-50.0, noise_counts=0.5, seed=1)
    run = sb.read_run(csv)
    assert set(run.raw) == {"a"} and len(run.t) == 3000
    out, pot, timing = sb.import_runs([csv], "a", tmp_path / "ds")
    assert pot.scale == pytest.approx(0.3, rel=0.02)
    assert pot.offset == pytest.approx(-50.0, abs=1.5)
    assert timing[0]["late_ticks"] == 0
    ds = read_local(out)
    assert ds.robot_type == sb.ROBOT_TYPE and ds.joint_names == ("shaft",)
    assert len(ds.episodes) == 12 and all(len(e) == 250 for e in ds.episodes)


def test_import_rejects_a_pot_that_does_not_move(tmp_path):
    p = tmp_path / "flat.csv"
    p.write_text("t_ms,cmd_deg,raw_a\n" + "".join(f"{i * 20},{int(sb.program()[i])},500\n" for i in range(3000)))
    with pytest.raises(ValueError, match="barely moves"):
        sb.import_runs([p], "a", tmp_path / "ds")


def test_cross_scores_and_payload_check_tell_servos_apart(tmp_path):
    from types import SimpleNamespace

    base = Arm(robot="sg92r").default_params()
    pa = base.replace(kp=(2.2,), armature=(0.0005,), latency_frames=1)
    pb = base.replace(kp=(1.0,), armature=(0.0002,), kv=(0.03,), latency_frames=1)
    dsa = read_local(sb.import_runs([sb.simulate_run(tmp_path / "a.csv", pa.replace(latency_frames=0))], "a", tmp_path / "a")[0])
    dsb = read_local(sb.import_runs([sb.simulate_run(tmp_path / "b.csv", pb.replace(latency_frames=0))], "a", tmp_path / "b")[0])
    heavy = pa.replace(armature=(0.0005 + 3 * 5.7e-5,), latency_frames=0)
    dsc = read_local(sb.import_runs([sb.simulate_run(tmp_path / "c.csv", heavy, seed=2)], "a", tmp_path / "c")[0])

    def lock(params, name):
        result = SimpleNamespace(eval_score=0.6, train_score=0.6, per_joint_after={})
        return SimpleNamespace(name=name, dataset=name, params=params, result=result, robot=sb.ROBOT)

    locks = {"a": lock(pa, "a"), "b": lock(pb, "b")}
    scores = sb.cross_scores({"a": dsa, "b": dsb}, locks)
    for own, other in (("a", "b"), ("b", "a")):
        assert scores[own][own] < 0.5 * scores[own][other]
        assert scores[own][own] < 0.5 * scores[own]["default"]
    assert sb.payload_check(dsa, locks["a"]).verdict == "ok"
    assert sb.payload_check(dsc, locks["a"]).verdict == "drifted"


@pytest.mark.parametrize("deadband_deg", [0.0, 2.0])
def test_live_step_sim_matches_a_batch_rollout(deadband_deg):
    """Tick by tick through FrameStepper is the same model as Arm.rollout,
    with and without the lock's deadband (the plain mj_step path ignored it)."""
    from antares.servo_live import StepSim

    params = Arm(robot="sg92r").default_params().replace(kp=(2.0,), latency_frames=2, deadband=(np.radians(deadband_deg),))
    cmd = sb.program(400)
    arm = Arm(robot="sg92r")
    arm.apply(params)
    batch = np.degrees(arm.rollout(np.radians(cmd[:, None]), np.radians([cmd[0]]), sb.FPS)[:, 0])
    live = StepSim(params)
    live.reset(cmd[0], cmd[0])
    stepped = [cmd[0]] + [live.step(c) for c in cmd[1:]]
    np.testing.assert_allclose(stepped, batch, atol=1e-6)
    if deadband_deg:
        plain = StepSim(params.replace(deadband=0.0))
        plain.reset(cmd[0], cmd[0])
        assert not np.allclose([plain.step(c) for c in cmd[1:]], batch[1:], atol=1e-3)


def test_live_step_sim_refuses_a_servo_model_lock():
    from antares.servo_live import StepSim
    from antares.sim import SimParams

    with pytest.raises(ValueError, match="servo model"):
        StepSim(SimParams.sts3215())


def test_live_ticks_publish_three_angles(tmp_path):
    from antares import servo_live

    csv = sb.simulate_run(tmp_path / "a.csv", seed=3)
    _, pot, _ = sb.import_runs([csv], "a", tmp_path / "ds")
    run = sb.read_run(csv)
    ticks = list(zip(run.t[:100], run.cmd[:100], run.raw["a"][:100]))
    out = servo_live.Broadcaster()
    q = out.subscribe()
    base = Arm(robot="sg92r").default_params()
    servo_live.run_ticks(iter(ticks), pot, base, base, out, {})
    import json

    msgs = [json.loads(q.get_nowait()) for _ in range(100)]
    assert set(msgs[-1]) >= {"t", "cmd", "real", "lock", "default", "err_lock", "err_default"}
    assert msgs[-1]["lock"] == msgs[-1]["default"]  # same params, same ghost


# --------------------------------------------------------------------------
# Locks, the demo and the live view around the hardware
# --------------------------------------------------------------------------


def _bench_lock(name, dataset, params=None, eval_score=0.6):
    """A real sg92r lock (what calibrate writes), on disk when saved."""
    from antares.calibrate import CalibrationResult
    from antares.lock import RobotLock

    base = Arm(robot="sg92r").default_params()
    p = params if params is not None else base.replace(kp=(2.0,), latency_frames=1)
    res = CalibrationResult(
        params=p, baseline=base, train_score=0.5, eval_score=eval_score, baseline_train_score=3.0,
        baseline_eval_score=3.0, per_joint_before={"shaft": 3.0}, per_joint_after={"shaft": 0.5},
        history=[(0, 3.0), (1, 0.5)], n_evaluations=1, elapsed_s=0.1, train_episodes=[0], eval_episodes=[1],
    )
    return RobotLock(name=name, dataset=dataset, robot_type=sb.ROBOT_TYPE, data_sha256="0" * 64, params=p,
                     result=res, created_at="2026-10-18T00:00:00Z", antares_version="0.0.1", robot=sb.ROBOT)


def _args(**kw):
    from types import SimpleNamespace

    return SimpleNamespace(**kw)


def test_import_counts_episodes_per_run(tmp_path, capsys):
    """Two runs one frame short of 60 s give 11 episodes each: the floor is
    per run, not over the frames of all runs together (which would say 23)."""
    paths = []
    for i in (1, 2):
        full = sb.simulate_run(tmp_path / f"full{i}.csv", seed=i)
        lines = full.read_text().splitlines()
        short = tmp_path / f"run{i}.csv"
        short.write_text("\n".join(lines[:1 + 2999]) + "\n")
        paths.append(str(short))
    assert sb.cmd_import(_args(csv=paths, servo="a", out=str(tmp_path / "ds"), episode_seconds=5.0)) == 0
    out = capsys.readouterr().out
    assert "2 run(s), 22 episodes of 5 s" in out
    assert "2999 frames -> 11 episodes" in out
    assert len(read_local(tmp_path / "ds").episodes) == 22


def test_demo_reports_insufficient_payload_data(tmp_path, capsys):
    """A payload recording of fewer than 5 episodes has no score: the demo
    prints the check's own summary instead of formatting None."""
    csv = sb.simulate_run(tmp_path / "a.csv", seed=1)
    ds, _, _ = sb.import_runs([csv], "a", tmp_path / "ds")
    four, _, _ = sb.import_runs([csv], "a", tmp_path / "four", seconds=15.0)
    assert len(read_local(four).episodes) == 4
    lock_path = _bench_lock("servo_a", str(ds)).save(tmp_path / "locks" / "servo_a.lock.json")
    args = _args(servo=[f"a={ds}"], lock=[f"a={lock_path}"], payload=f"a={four}", fresh=str(four), ratio=1.5)
    assert sb.cmd_demo(args) == 0
    out = capsys.readouterr().out
    assert "INSUFFICIENT DATA: 4 episode(s)" in out and "None" not in out


def test_load_lock_follows_the_lock_dir_and_robot_refs(tmp_path, monkeypatch):
    from antares import server

    lock_dir = tmp_path / "elsewhere"
    monkeypatch.setattr(server, "LOCK_DIR", lock_dir)  # what ANTARES_LOCK_DIR sets at import
    lock = _bench_lock("servo_a", "ds_a")
    lock.save(lock_dir / "servo_a.lock.json")
    assert sb.lock_file("servo_a") == lock_dir / "servo_a.lock.json"
    assert sb.load_lock("servo_a").params == lock.params
    assert sb.load_lock(str(lock_dir / "servo_a.lock.json")).name == "servo_a"
    server.register_lock("bench-a", "servo_a", activate=True)
    assert sb.load_lock("@bench-a").name == "servo_a"
    with pytest.raises(SystemExit, match="no lock named 'nope'"):
        sb.load_lock("nope")
    with pytest.raises(SystemExit, match="@nobody"):
        sb.load_lock("@nobody")
    with pytest.raises(SystemExit, match="no such lock file: other/locks/x.lock.json"):
        sb.load_lock("other/locks/x.lock.json")


def test_bench_parser_does_not_import_servo_bench():
    """Registering `antares bench` must not pull the module (numpy, pyarrow) in."""
    import subprocess
    import sys

    code = (
        "import sys, antares.cli as c\n"
        "c.build_parser()\n"
        "try:\n    c.main(['bench', '--help'])\nexcept SystemExit:\n    pass\n"
        "print('BENCH_LOADED=' + repr('antares.servo_bench' in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "BENCH_LOADED=False", out.stdout


def test_cli_bench_dispatches_lazily(tmp_path, capsys):
    from antares import cli

    assert cli.main(["bench", "simulate", str(tmp_path / "s.csv"), "--set", "kp=2.0", "--seed", "4"]) == 0
    assert "wrote" in capsys.readouterr().out and (tmp_path / "s.csv").is_file()


def test_pot_for_lock_resolves_the_dataset_next_to_the_lock(tmp_path, monkeypatch):
    """calibrate records the dataset as typed (relative to where it ran) and
    saves the lock under locks/ there; the live view finds it from anywhere."""
    from antares import servo_live

    proj = tmp_path / "recordings"
    proj.mkdir()
    csv = sb.simulate_run(proj / "a.csv", seed=1)
    _, pot, _ = sb.import_runs([csv], "a", proj / "ds_a")
    lock_path = _bench_lock("servo_a", "ds_a").save(proj / "locks" / "servo_a.lock.json")
    lock = sb.load_lock(lock_path)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert servo_live.dataset_candidates(lock, lock_path=lock_path) == [proj / "ds_a", elsewhere / "ds_a"]
    assert servo_live.pot_for_lock(lock, lock_path=lock_path) == pot
    with pytest.raises(SystemExit, match="pass --dataset"):
        servo_live.pot_for_lock(lock)  # cwd only: not there
    assert servo_live.pot_for_lock(lock, dataset=str(proj / "ds_a")) == pot

    monkeypatch.chdir(proj)
    assert servo_live.pot_for_lock(lock) == pot  # cwd fallback, as before
    assert servo_live.pot_for_lock(_bench_lock("abs", str(proj / "ds_a")), lock_path=tmp_path / "x.lock.json") == pot


def test_live_rejects_a_missing_replay_before_serving(tmp_path, monkeypatch):
    from antares import cli, servo_live

    csv = sb.simulate_run(tmp_path / "a.csv", seed=1)
    ds, _, _ = sb.import_runs([csv], "a", tmp_path / "ds_a")
    lock_path = _bench_lock("servo_a", str(ds)).save(tmp_path / "locks" / "servo_a.lock.json")

    def no_serve(*a, **k):
        raise AssertionError("serve() must not start")

    monkeypatch.setattr(servo_live, "serve", no_serve)
    with pytest.raises(SystemExit, match="no such replay CSV"):
        cli.main(["bench", "live", "--lock", str(lock_path), "--replay", str(tmp_path / "missing.csv")])
    with pytest.raises(SystemExit, match="no raw_b column"):
        cli.main(["bench", "live", "--lock", str(lock_path), "--replay", str(csv), "--servo", "b"])
    with pytest.raises(SystemExit, match="no lock named"):
        cli.main(["bench", "live", "--lock", "nothing", "--replay", str(csv)])
    # a good replay is read now and only played lazily
    ticks = servo_live.csv_ticks(csv, "a", loop=False)
    assert next(ticks)[1:] == (30.0, pytest.approx(float(sb.read_run(csv).raw["a"][0])))


def _serve_in_thread(ticks, pot, lock):
    import threading
    from antares import servo_live

    result: dict = {}

    def run():
        try:
            result["code"] = servo_live.serve(ticks, pot, lock, port=0)
        except SystemExit as e:
            result["exit"] = e.code

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "serve() did not stop"
    return result


def test_live_worker_failure_stops_the_server(tmp_path, capsys):
    """A tick source that dies (serial timeout, a bad line, SystemExit from
    the reader) takes the server down with its message, instead of leaving
    a page that says live over a stick that stopped."""
    csv = sb.simulate_run(tmp_path / "a.csv", seed=1)
    _, pot, _ = sb.import_runs([csv], "a", tmp_path / "ds_a")
    run = sb.read_run(csv)
    lock = _bench_lock("servo_a", "ds_a")
    good = list(zip(run.t[:3], run.cmd[:3], run.raw["a"][:3]))

    def dying(exc):
        yield from good
        raise exc

    res = _serve_in_thread(dying(RuntimeError("boom")), pot, lock)
    assert "code" not in res and res["exit"] == "error: the ticks stopped after 3 tick(s): RuntimeError: boom"
    res = _serve_in_thread(dying(SystemExit("error: no data from the Arduino for 5 s")), pot, lock)
    assert res["exit"] == "error: the ticks stopped after 3 tick(s): error: no data from the Arduino for 5 s"
    # a stream that simply ends stops the server too, as a normal exit
    assert _serve_in_thread(iter(good), pot, lock) == {"code": 0}
    out = capsys.readouterr().out
    assert out.count("live view of lock servo_a: http://127.0.0.1:") == 3 and "ended after 3 tick(s)" in out


def test_live_info_keeps_json_valid_for_a_nan_score():
    import json

    from antares.servo_live import live_info

    assert json.loads(json.dumps(live_info(_bench_lock("a", "ds", eval_score=0.6)))) == {"lock": "a", "ticks": 0, "lock_score": 0.6}
    text = json.dumps(live_info(_bench_lock("a", "ds", eval_score=float("nan"))))
    assert "NaN" not in text and json.loads(text)["lock_score"] is None
    assert live_info(_bench_lock("a", "ds", eval_score=None))["lock_score"] is None


class _FakeSerial:
    """What pyserial hands back for a scripted sketch."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.written = b""
        self.closed = False

    def write(self, data):
        self.written += data

    def readline(self):
        return (self.lines.pop(0) + "\r\n").encode() if self.lines else b""

    def reset_input_buffer(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


RUN = ["# servo_log ready", "t_ms,cmd_deg,raw_a,raw_b", "0,30,250,400", "20,31,252,401", "#done"]


def test_shared_serial_helper_reads_a_run_and_finds_the_port(monkeypatch):
    from antares import servo_serial as ss

    ser = _FakeSerial(RUN)
    assert list(ss.run_lines(ser)) == RUN[1:4] and ser.written == b"g"
    assert list(ss.run_lines(_FakeSerial(RUN), start=False)) == RUN[1:4]
    with pytest.raises(ss.BenchSerialError, match="no data from the Arduino"):
        list(ss.run_lines(_FakeSerial(RUN[:3])))  # went quiet before #done

    seen = []

    def fake_glob(pattern):
        seen.append(pattern)
        return ["/dev/ttyUSB0"] if pattern == "/dev/ttyUSB*" else []

    monkeypatch.setattr(ss.glob, "glob", fake_glob)
    assert ss.find_port() == "/dev/ttyUSB0" and tuple(seen) == ss.PORT_GLOBS
    monkeypatch.setattr(ss.glob, "glob", lambda pattern: [])
    with pytest.raises(ss.BenchSerialError, match="no Arduino found"):
        ss.find_port()


def test_live_serial_ticks_use_the_shared_reader():
    from antares import servo_live
    from antares.servo_serial import BenchSerialError

    ser = _FakeSerial(RUN)
    assert list(servo_live._serial_ticks(ser, "b", start=False)) == [(0.0, 30.0, 400.0), (0.02, 31.0, 401.0)]
    assert ser.closed
    with pytest.raises(BenchSerialError, match="logs servo\\(s\\) a, b, not c"):
        list(servo_live._serial_ticks(_FakeSerial(RUN), "c", start=False))
    # start=True restarts the program after #done and keeps time running
    ser = _FakeSerial(RUN + RUN[1:])
    ticks = servo_live._serial_ticks(ser, "a", start=True)
    assert [next(ticks) for _ in range(4)] == [(0.0, 30.0, 250.0), (0.02, 31.0, 252.0), (0.04, 30.0, 250.0), (0.06, 31.0, 252.0)]
    assert ser.written == b"gg"


def test_capture_script_uses_the_shared_helper(tmp_path, monkeypatch, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("capture", SKETCH.parents[1] / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    ser = _FakeSerial(RUN)
    monkeypatch.setattr(capture, "find_port", lambda: "/dev/fake")
    monkeypatch.setattr(capture, "open_port", lambda port: ser)
    out = tmp_path / "run.csv"
    monkeypatch.setattr("sys.argv", ["capture.py", str(out)])
    capture.main()
    assert out.read_text().splitlines() == RUN[1:4] and ser.written == b"g"
    assert "Wrote 2 samples" in capsys.readouterr().out
    assert sb.read_run(out).raw["b"].tolist() == [400.0, 401.0]
