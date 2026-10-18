"""Tests for antares.transfer: the sim-to-sim policy transfer experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from antares import cli, transfer
from antares.sim import GRIPPER_INDEX, SimParams
from antares.transfer import (
    EvalResult,
    HiddenArmFit,
    Policy,
    Task,
    ToyArm,
    TrainResult,
    TransferConfig,
    TransferReport,
    episode_metrics,
    evaluate,
    rollout,
    sample_tasks,
    train_policy,
)

LOCKS = Path(__file__).resolve().parent.parent / "locks"
SO101 = SimParams(kp=10.54, kv=0.136, damping=0.01226, armature=0.002835, frictionloss=0.0943, latency_frames=4)


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------


def test_sample_tasks_is_deterministic_and_in_range():
    a = sample_tasks(20, seed=3, task_range=70.0)
    b = sample_tasks(20, seed=3, task_range=70.0)
    assert len(a) == 20
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.start, y.start)
        np.testing.assert_array_equal(x.target, y.target)
    starts = np.array([t.start for t in a])
    targets = np.array([t.target for t in a])
    assert np.all(np.abs(starts[:, :5]) <= 70) and np.all(np.abs(targets[:, :5]) <= 70)
    # the gripper is held: start == target, inside its range
    np.testing.assert_array_equal(starts[:, GRIPPER_INDEX], targets[:, GRIPPER_INDEX])
    assert np.all((starts[:, GRIPPER_INDEX] >= 20) & (starts[:, GRIPPER_INDEX] <= 80))
    # a different seed gives different tasks
    c = sample_tasks(20, seed=4)
    assert not np.allclose(starts, np.array([t.start for t in c]))


def test_sample_tasks_prefix_is_stable():
    short = sample_tasks(5, seed=11)
    long = sample_tasks(12, seed=11)
    for x, y in zip(short, long):
        np.testing.assert_array_equal(x.target, y.target)


def test_task_roundtrip():
    t = sample_tasks(1, seed=0)[0]
    back = Task.from_dict(json.loads(json.dumps(t.to_dict())))
    np.testing.assert_array_equal(back.start, t.start)
    np.testing.assert_array_equal(back.target, t.target)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_episode_metrics_success_needs_every_joint_in_tolerance_for_the_last_k():
    task = Task(start=np.zeros(6), target=np.array([10.0, 10, 10, 10, 10, 50]))
    deg = np.ones(6)
    T = 12
    q = np.zeros((T + 1, 6))
    q[:, :5] = 10.0  # perfect from frame 1 on (frame 0 is the start)
    q[0, :5] = 0.0
    ok, final, mean = episode_metrics(q, task, deg, tol_deg=2.0, k_frames=5)
    assert ok and final == 0.0 and mean == pytest.approx(10.0 / (T + 1))

    # one joint 2.5 deg off on the last frame fails; 1.5 deg passes
    q2 = q.copy()
    q2[-1, 2] = 12.5
    ok, final, _ = episode_metrics(q2, task, deg, 2.0, 5)
    assert not ok and final == pytest.approx(2.5 / 25)
    q2[-1, 2] = 11.5
    assert episode_metrics(q2, task, deg, 2.0, 5)[0]

    # a pass-through 6 frames before the end does not count against the last 5
    q3 = q.copy()
    q3[-6, 0] = 30.0
    assert episode_metrics(q3, task, deg, 2.0, 5)[0]
    # degrees per unit scales the error
    assert episode_metrics(q2, task, 2 * deg, 2.0, 5)[0] is False
    with pytest.raises(ValueError):
        episode_metrics(q, task, deg, 2.0, k_frames=0)


def test_bootstrap_ci_brackets_the_mean():
    v = np.random.default_rng(0).normal(5.0, 1.0, size=200)
    lo, hi = transfer.bootstrap_ci(v)
    assert lo < v.mean() < hi and hi - lo < 0.6
    assert transfer.bootstrap_ci(np.array([2.0])) == (2.0, 2.0)
    assert transfer.bootstrap_ci(np.array([0.0, 1.0])) == transfer.bootstrap_ci(np.array([0.0, 1.0]))


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


def test_untrained_policy_commands_the_target():
    p = Policy.zeros(hidden=6)
    assert p.n_params(6) == 111 and p.theta.shape == (111,)
    q = np.array([10.0, -20, 30, 5, 0, 40])
    t = np.array([-5.0, 15, 0, 60, -70, 40])
    np.testing.assert_allclose(p.act(q, np.ones(6), t), t)


def test_policy_proportional_term_and_roundtrip():
    theta = np.zeros(Policy.n_params(3))
    theta[0] = 1.0  # kp on shoulder_pan
    p = Policy(3, theta)
    q = np.zeros(6)
    t = np.array([20.0, 0, 0, 0, 0, 50])
    cmd = p.act(q, np.zeros(6), t)
    assert cmd[0] == pytest.approx(40.0)  # t + kp*(t - q) = 20 + 20
    assert cmd[1:5].tolist() == [0, 0, 0, 0] and cmd[5] == 50
    back = Policy.from_dict(json.loads(json.dumps(p.to_dict())))
    assert back.hidden == 3
    np.testing.assert_array_equal(back.theta, p.theta)
    with pytest.raises(ValueError):
        Policy(3, np.zeros(5))


# --------------------------------------------------------------------------
# toy arm: training helps
# --------------------------------------------------------------------------


def test_toy_arm_rollout_shapes_and_sag():
    toy = ToyArm(rate=0.5, latency=1, sag=1.0)
    task = Task(start=np.zeros(6), target=np.array([10.0, 10, 10, 10, 10, 50]))
    q = rollout(toy, Policy.zeros(2), task, horizon=40)
    assert q.shape == (41, 6)
    # a sagging servo commanded to the target settles short of it: q = u - sag/rate
    np.testing.assert_allclose(q[-1, :5], 8.0, atol=1e-3)


def test_trained_policy_beats_untrained_on_toy_arm():
    toy = ToyArm(rate=0.3, latency=2, sag=1.2)
    tasks = sample_tasks(6, seed=5, task_range=40.0)
    before = evaluate(Policy.zeros(2), toy, tasks, horizon=30, tol_deg=2.0, k_frames=5)
    result = train_policy(toy, tasks, horizon=30, seed=0, generations=100, popsize=8, hidden=2)
    after = evaluate(result.policy, toy, tasks, horizon=30, tol_deg=2.0, k_frames=5)
    assert before.success_rate == 0.0 and before.final_err > 3.0
    assert after.final_err < before.final_err / 2
    assert after.success_rate > before.success_rate
    assert result.n_evaluations == 1 + 100 * 8
    assert [i for i, _ in result.history] == [1 + 8 * (g + 1) for g in range(100)]
    assert result.history[-1][1] <= result.history[0][1]
    back = TrainResult.from_dict(json.loads(json.dumps(result.to_dict())))
    np.testing.assert_array_equal(back.policy.theta, result.policy.theta)


def test_train_policy_never_returns_worse_than_untrained_on_its_tasks():
    toy = ToyArm(rate=0.3, latency=2, sag=0.0)
    tasks = sample_tasks(3, seed=1, task_range=30.0)
    result = train_policy(toy, tasks, horizon=20, seed=0, generations=2, popsize=4, hidden=2)
    cost0 = transfer.episode_cost(toy, Policy.zeros(2), tasks, 20, 2.0, 5, 2.0)
    cost1 = transfer.episode_cost(toy, result.policy, tasks, 20, 2.0, 5, 2.0)
    assert cost1 <= cost0 + 1e-12


# --------------------------------------------------------------------------
# MuJoCo stepping
# --------------------------------------------------------------------------


def test_mujoco_sim_matches_open_loop_rollout():
    from antares.sim import Arm

    arm = Arm()
    sim = transfer.MuJoCoSim(arm=arm)
    rng = np.random.default_rng(0)
    cmds = np.cumsum(rng.normal(0, 2, size=(30, 6)), axis=0)
    q0 = np.zeros(6)
    q0[GRIPPER_INDEX] = 50
    cmds[0] = q0  # at rest at the start pose before the first command arrives
    cmds[:, GRIPPER_INDEX] = 50
    for lat in (4, 1):
        p = SO101.replace(latency_frames=lat)
        sim.set_params(p)
        out = [sim.reset(q0)]
        for c in cmds[:-1]:
            out.append(sim.step(c))
        ref = arm.unit_map.to_units(arm.rollout(arm.unit_map.to_rad(cmds), arm.unit_map.to_rad(q0), 30.0))
        np.testing.assert_allclose(np.array(out), ref, atol=1e-9)
    assert sim.deg_per_unit.shape == (6,) and np.all(sim.deg_per_unit > 0)


def test_untrained_policy_misses_on_a_soft_arm_but_not_the_default():
    sim = transfer.MuJoCoSim()
    tasks = sample_tasks(6, seed=2024)
    sim.set_params(SimParams())
    on_default = evaluate(Policy.zeros(6), sim, tasks, horizon=60)
    sim.set_params(SO101)
    on_soft = evaluate(Policy.zeros(6), sim, tasks, horizon=60)
    assert on_default.success_rate == 1.0 and on_default.final_err < 0.1
    assert on_soft.final_err > 0.3 and on_soft.success_rate < 1.0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _eval(n: int, rate: float, err: float) -> EvalResult:
    s = np.zeros(n, dtype=bool)
    s[: int(round(rate * n))] = True
    return EvalResult(s, np.full(n, err), np.full(n, err + 3), tol_deg=2.0, k_frames=5)


def _fake_report() -> TransferReport:
    cfg = TransferConfig(arms={"a": SO101.to_dict()}, variants=("default", "lock"), seeds=(0, 1), generations=2)
    fit = HiddenArmFit(
        name="a", truth=SO101, fitted=SO101.replace(kp=12.0), fit_eval_score=0.5, default_eval_score=6.0,
        truth_eval_score=0.3, dr={"ranges": {"kp": {"lo": 8.0, "hi": 20.0}}}, pm={}, seed_params=[],
    )
    variants = {}
    for name, hidden_rate in (("default", 0.25), ("lock", 0.75)):
        seeds = [
            transfer.SeedResult(
                seed=s,
                train=TrainResult(Policy.zeros(2), [(1, 3.0), (5, 2.0)], 5, 1.5, s),
                in_train_sim=_eval(8, 1.0, 0.1),
                on_hidden=_eval(8, hidden_rate, 1.0 - hidden_rate),
            )
            for s in (0, 1)
        ]
        variants[name] = transfer.VariantResult(name, SO101, name == "lock_dr", seeds)
    arm = transfer.ArmResult(fit=fit, variants=variants, untrained=_eval(8, 0.0, 2.0))
    return TransferReport(config=cfg, arms={"a": arm}, elapsed_s=12.0)


def test_report_roundtrips_json_and_renders_markdown(tmp_path):
    report = _fake_report()
    path = report.save(tmp_path / "r.json")
    assert path.exists() and (tmp_path / "r.md").exists()
    back = TransferReport.load(path)
    assert back.to_dict() == report.to_dict()
    assert back.config.seeds == (0, 1) and back.config.variants == ("default", "lock")
    v = back.arms["a"].variants["lock"]
    assert v.hidden_success == (0.75, 0.0) and v.train_success == (1.0, 0.0)
    md = back.to_markdown()
    assert "| default sim | a | 100% +- 0 | 25% +- 0 |" in md
    assert "| fitted lock | a | 100% +- 0 | 75% +- 0 |" in md
    assert "Untrained policy" in md and "| truth |" in md and "| DR lo |" in md
    rows = back.headline_rows()
    assert [r["variant"] for r in rows] == ["default", "lock"]


def test_config_roundtrip_and_validation():
    cfg = TransferConfig.quick_config({"a": SO101.to_dict()}, seeds=(0, 1))
    back = TransferConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert back == cfg and back.quick and back.seeds == (0, 1)
    with pytest.raises(ValueError, match="unknown variants"):
        TransferConfig(arms={}, variants=("nope",))
    with pytest.raises(ValueError, match="seeds"):
        TransferConfig(arms={}, seeds=())


def test_variant_sim_picks_the_right_dynamics():
    fit = _fake_report().arms["a"].fit
    assert transfer.variant_sim("default", fit) == (SimParams(), None)
    assert transfer.variant_sim("oracle", fit) == (SO101, None)
    assert transfer.variant_sim("lock", fit) == (fit.fitted, None)
    assert transfer.variant_sim("lock_dr", fit)[1] is fit.dr
    assert transfer.variant_sim("default_pm30", fit)[0] == SimParams()
    with pytest.raises(ValueError):
        transfer.variant_sim("other", fit)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_quick_run_and_report(tmp_path, capsys):
    out = tmp_path / "transfer"
    code = cli.main(
        ["transfer", "run", "--quick", "--arm", f"so101={LOCKS / 'svla_so101_pickplace.lock.json'}", "--out", str(out)]
    )
    assert code == 0
    captured = capsys.readouterr().out
    assert "training 5 policies" in captured and "fitted lock" in captured
    report = TransferReport.load(out.with_suffix(".json"))
    assert report.config.quick and report.config.seeds == (0,)
    assert set(report.arms) == {"so101"}
    arm = report.arms["so101"]
    assert set(arm.variants) == set(transfer.VARIANTS)
    lock_params = json.loads((LOCKS / "svla_so101_pickplace.lock.json").read_text())["params"]
    assert arm.fit.truth == SimParams.from_dict(lock_params)  # the hidden truth is the lock's params
    for v in arm.variants.values():
        assert len(v.seeds) == 1 and v.seeds[0].on_hidden.n_tasks == 8
    assert arm.untrained is not None
    md = out.with_suffix(".md").read_text()
    assert "quick budgets" in md

    assert cli.main(["transfer", "report", str(out.with_suffix(".json"))]) == 0
    assert "oracle (true parameters)" in capsys.readouterr().out
    assert cli.main(["transfer", "report", str(out.with_suffix(".json")), "--md", str(tmp_path / "x.md")]) == 0
    assert (tmp_path / "x.md").read_text() == md


def test_cli_rejects_bad_arm_spec(capsys):
    assert cli.main(["transfer", "run", "--quick", "--arm", "nonsense"]) == 2
    assert cli.main(["transfer", "run", "--quick", "--arm", "a=/nonexistent.lock.json"]) == 2
    assert cli.main(["transfer", "report", "/nonexistent.json"]) == 2
