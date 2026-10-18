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


def test_live_step_sim_matches_a_batch_rollout():
    from antares.servo_live import StepSim

    params = Arm(robot="sg92r").default_params().replace(kp=(2.0,), latency_frames=2)
    cmd = sb.program(400)
    arm = Arm(robot="sg92r")
    arm.apply(params)
    batch = np.degrees(arm.rollout(np.radians(cmd[:, None]), np.radians([cmd[0]]), sb.FPS)[:, 0])
    live = StepSim(params)
    live.reset(cmd[0], cmd[0])
    stepped = [cmd[0]] + [live.step(c) for c in cmd[1:]]
    np.testing.assert_allclose(stepped, batch, atol=1e-6)


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


def test_viewer_residual_strip_works_on_a_servo(tmp_path):
    """The servo has no gripper; the viewer's residual strip still loads."""
    from antares import server

    csv = sb.simulate_run(tmp_path / "a.csv", seed=1)
    out, _, _ = sb.import_runs([csv], "a", tmp_path / "ds")
    ds = server.get_dataset(str(out))
    res = server.residual_episode(ds, ds.episodes[0].index)
    assert res["dataset"] == ds.repo_id


def test_live_finds_the_pot_next_to_a_pipeline_lock(tmp_path, monkeypatch):
    """pipeline.sh calibrates inside the recordings folder, so the lock names
    its dataset relatively; `bench live --lock <dir>/locks/x.lock.json` run
    from elsewhere still finds the pot conversion."""
    from types import SimpleNamespace

    from antares import servo_live

    csv = sb.simulate_run(tmp_path / "a.csv", pot_scale=0.3, pot_offset=-50.0, seed=1)
    sb.import_runs([csv], "a", tmp_path / "ds_a")
    (tmp_path / "locks").mkdir()
    lock_file = tmp_path / "locks" / "servo_a.lock.json"
    lock_file.write_text("{}")
    monkeypatch.chdir(tmp_path / "locks")
    pot = servo_live.pot_for_lock(SimpleNamespace(dataset="ds_a"), None, lock_file)
    assert pot.scale == pytest.approx(0.3, rel=0.02)
