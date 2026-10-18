"""The state-based chunked policy, the shadow and the closed-loop evaluators.

Training needs torch (the ``[policy]`` extra); the evaluators are exercised
with a policy that replays the logged commands, which must reproduce
``Arm.replay`` exactly, and with scripted policies, so they run without it.
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from antares.data import Episode
from antares.gap import tracking_error
from antares.policy import (
    H_CHUNK,
    K_OBS,
    Features,
    LeRobotPolicy,
    LoggedPolicy,
    Normalizer,
    StepContext,
    action_errors,
    baseline_predictions,
    list_policies,
    load_lerobot_policy,
    load_policy,
    policy_meta,
    save_policy,
    windows,
)
from antares.sim import JOINTS, Arm, CommandMap, SimParams, UnitMap, load_model

FPS = 30.0
SOFT = SimParams(kp=90.0, kv=1.5, damping=0.05, armature=0.02, frictionloss=0.025, latency_frames=4)


def smooth_episode(index: int = 0, n: int = 120, seed: int = 0, close_at: int | None = 50, open_at: int | None = 90) -> Episode:
    """A slow smooth trajectory in dataset units whose gripper command closes
    at ``close_at`` (the real grasp) and opens again at ``open_at``."""
    t = np.linspace(0, 1, n)
    rng = np.random.default_rng(seed)
    # one motion with small per-episode variations, so a few episodes suffice to learn it
    base = np.array([0.0, -40.0, 30.0, 10.0, 0.0, 60.0]) + rng.normal(0, 2.0, 6)
    amp = np.array([25.0, 15.0, 20.0, 15.0, 20.0, 0.0]) * rng.uniform(0.9, 1.1)
    phase = np.array([0.3, 1.1, 2.0, 4.0, 5.2, 0.0]) + rng.normal(0, 0.1, 6)
    freq = np.array([0.5, 0.7, 0.6, 0.8, 0.45, 0.5]) * rng.uniform(0.95, 1.05)
    ramp = 0.5 * (1 - np.cos(np.pi * np.minimum(t / 0.3, 1.0)))
    q = base + amp * np.sin(2 * np.pi * freq * t[:, None] + phase) * ramp[:, None]
    action = np.vstack([q[3:], np.repeat(q[-1:], 3, axis=0)])
    if close_at is not None:
        end = open_at if open_at is not None else n
        action[close_at:end, 5] = 5.0
        # the measured jaw stops on an "object" 10 units above the empty level
        q[close_at + 3 : end, 5] = 15.0
    q[:, 5] = np.clip(q[:, 5], 0, 100)
    action[:, 5] = np.clip(action[:, 5], 0, 100)
    return Episode(index=index, fps=FPS, joint_names=JOINTS, state=q, action=action)


@pytest.fixture(scope="module")
def episodes() -> list[Episode]:
    eps = [smooth_episode(i, seed=i) for i in range(6)]
    # one episode that closes on nothing, so the dataset's empty-jaw level exists
    e = smooth_episode(6, seed=6)
    e.state[53:90, 5] = 2.0
    return eps + [e]


# --------------------------------------------------------------------------
# windows, normalization, metrics
# --------------------------------------------------------------------------


def test_windows_pad_start_and_end():
    ep = smooth_episode(n=20)
    w = windows([ep], k=3, h=4, features=Features(goal=False, time=False))
    obs, tgt, prev = w.obs, w.target, w.prev_action
    assert obs.shape == (20, 3, 6) and tgt.shape == (20, 4, 6) and prev.shape == (20, 6)
    assert w.extras.shape == (20, 0) and np.all(w.episode == 0)
    # frame 0 sees itself three times; the last frame's chunk repeats the last command
    assert np.allclose(obs[0], np.repeat(ep.state[:1], 3, axis=0))
    assert np.allclose(obs[5], ep.state[3:6])
    assert np.allclose(tgt[19], np.repeat(ep.action[-1:], 4, axis=0))
    assert np.allclose(tgt[2], ep.action[2:6])
    assert np.allclose(prev[0], ep.action[0]) and np.allclose(prev[7], ep.action[6])
    with pytest.raises(ValueError):
        windows([ep], k=0, h=1)
    # goal + time features: the goal repeats per frame, time counts seconds
    goal = np.array([0.2, -0.1, 0.05])
    wg = windows([ep], k=2, h=2, features=Features(goal=True, time=True), goals={0: goal})
    assert wg.extras.shape == (20, 4)
    assert np.allclose(wg.extras[:, :3], goal) and wg.extras[6, 3] == pytest.approx(6 / FPS)
    with pytest.raises(ValueError, match="grasp point"):
        windows([ep], k=2, h=2, goals={0: None})


def test_normalizer_floors_degenerate_std():
    ep = smooth_episode(n=30, close_at=None)
    ep.state[:, 5] = 42.0  # constant gripper reading
    norm = Normalizer.from_episodes([ep])
    assert norm.state_std[5] == 1.0 and norm.state_mean[5] == 42.0
    back = Normalizer.from_dict(norm.to_dict())
    assert np.allclose(back.action_mean, norm.action_mean)


def test_action_errors_in_units_and_degrees():
    um = UnitMap.from_model(load_model())
    tgt = np.zeros((5, 3, 6))
    pred = np.zeros((5, 3, 6))
    pred[:, :, 0] = 10.0  # 10 units off on shoulder_pan everywhere
    m = action_errors(pred, tgt, um)
    assert m["mse_units"] == pytest.approx(100.0 / 6)
    deg = np.degrees(abs(um.scale[0])) * 10.0
    assert m["per_joint_deg"]["shoulder_pan"] == pytest.approx(deg)
    assert m["per_joint_deg"]["gripper"] == 0.0
    # RMS over the five body joints, only one of which is off
    assert m["rmse_deg"] == pytest.approx(deg / np.sqrt(5))
    assert m["first_rmse_deg"] == pytest.approx(m["rmse_deg"])
    assert action_errors(tgt, tgt)["rmse_deg"] == 0.0


def test_baselines_hold_state_and_repeat_command():
    ep = smooth_episode(n=15)
    w = windows([ep], k=2, h=3, features=Features(goal=False, time=False))
    b = baseline_predictions(w, 3)
    assert np.allclose(b["hold_state"][4], np.repeat(ep.state[4:5], 3, axis=0))
    assert np.allclose(b["repeat_command"][4], np.repeat(ep.action[3:4], 3, axis=0))


# --------------------------------------------------------------------------
# training (torch)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(episodes):
    torch = pytest.importorskip("torch")
    from antares.policy import train_policy

    progress = []
    res = train_policy(
        episodes[:5], episodes[5:], hidden=64, layers=2, epochs=150, patience=150, batch_size=64, seed=0,
        on_progress=progress.append, meta={"dataset": "fake/ds", "lock": "unit"},
    )
    return res, progress


def test_train_policy_reports_metrics_and_beats_holding_still(trained):
    res, progress = trained
    pol = res.policy
    assert pol.k == K_OBS and pol.h == H_CHUNK
    assert len(progress) == res.epochs_run == 150 and progress[-1].epoch == 150
    m = res.metrics
    for split in ("train", "eval"):
        assert np.isfinite(m["policy"][split]["rmse_deg"])
        for b in ("hold_state", "repeat_command"):
            assert np.isfinite(m["baselines"][b][split]["rmse_deg"])
    # a smooth trajectory is predictable from two states: the policy must beat holding still
    assert m["beats_hold_state"] is True
    assert m["policy"]["eval"]["rmse_deg"] < m["baselines"]["hold_state"]["eval"]["rmse_deg"]
    assert pol.meta["dataset"] == "fake/ds" and pol.meta["lock"] == "unit"
    assert pol.meta["train_episodes"] == [0, 1, 2, 3, 4] and pol.meta["eval_episodes"] == [5, 6]
    assert pol.meta["dropped_episodes"] == [] and pol.needs_goal is True
    ctx = StepContext(0, FPS, np.array([0.2, 0.0, 0.05]))
    a = pol.act(np.repeat(np.zeros((1, 6)), K_OBS, axis=0), ctx)
    assert a.shape == (H_CHUNK, 6)
    with pytest.raises(ValueError):
        pol.act(np.zeros((K_OBS + 1, 6)), ctx)
    with pytest.raises(ValueError, match="goal"):
        pol.act(np.zeros((K_OBS, 6)), StepContext(0, FPS, None))


def test_train_policy_stops_when_asked(episodes):
    pytest.importorskip("torch")
    from antares.policy import train_policy

    res = train_policy(episodes[:3], episodes[3:4], hidden=8, layers=1, epochs=50, should_stop=lambda: True)
    assert res.stopped and res.epochs_run == 1


def test_save_load_round_trip_and_listing(trained, tmp_path):
    res, _ = trained
    path = save_policy(res.policy, "unit", tmp_path)
    assert path == tmp_path / "unit.policy.pt" and (tmp_path / "unit.policy.json").is_file()
    back = load_policy("unit", tmp_path)
    obs = np.stack([np.repeat(e.state[:1], K_OBS, axis=0) for e in [smooth_episode(9, seed=9)]])
    extras = np.array([[0.2, -0.1, 0.05]])
    assert np.allclose(back.act_batch(obs, extras), res.policy.act_batch(obs, extras))
    assert back.features == res.policy.features and back.k == K_OBS
    meta = policy_meta("unit", tmp_path)
    assert meta["name"] == "unit" and meta["metrics"]["policy"]["eval"]["rmse_deg"] > 0
    listed = list_policies(tmp_path)
    assert [m["name"] for m in listed] == ["unit"] and listed[0]["exists"] is True
    assert list_policies(tmp_path, lock="unit")[0]["lock"] == "unit"
    assert list_policies(tmp_path, lock="other") == []
    with pytest.raises(FileNotFoundError):
        load_policy("missing", tmp_path)
    assert policy_meta("missing", tmp_path) is None


# --------------------------------------------------------------------------
# LeRobot wrapper (stubbed)
# --------------------------------------------------------------------------


class _StubLeRobot:
    def __init__(self):
        self.resets = 0
        self.batches = []

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        torch = pytest.importorskip("torch")
        self.batches.append(batch)
        return batch["observation.state"] + 1.0  # (1,6): "move every joint by one unit"

    def eval(self):
        return self

    def to(self, device):
        return self


def test_lerobot_wrapper_feeds_the_last_state_and_returns_one_action():
    pytest.importorskip("torch")
    stub = _StubLeRobot()
    pol = LeRobotPolicy(stub, extra_inputs=lambda ctx: {"frame": ctx.frame})
    assert pol.k == 1 and pol.h == 1 and pol.needs_goal is False
    pol.reset()
    obs = np.array([[1.0, 2, 3, 4, 5, 6], [2.0, 3, 4, 5, 6, 7]])
    a = pol.act(obs, StepContext(0, FPS))
    assert a.shape == (1, 6) and np.allclose(a[0], obs[-1] + 1)
    assert stub.resets == 1 and stub.batches[0]["frame"] == 0
    pol.act(obs, StepContext(1, FPS))
    assert stub.batches[1]["frame"] == 1


def test_load_lerobot_policy_without_lerobot_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot", None)
    monkeypatch.setitem(sys.modules, "lerobot.configs", None)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", None)
    with pytest.raises(ImportError, match="lerobot"):
        load_lerobot_policy("lerobot/act_so101")


def test_load_lerobot_policy_uses_the_checkpoint_config(monkeypatch):
    pytest.importorskip("torch")
    made = {}

    class Cfg:
        type = "act"

        @classmethod
        def from_pretrained(cls, repo_id):
            made["cfg"] = repo_id
            return cls()

    class ActPolicy(_StubLeRobot):
        @classmethod
        def from_pretrained(cls, repo_id, config=None):
            made["policy"] = (repo_id, config.type)
            return cls()

    lerobot = types.ModuleType("lerobot")
    configs = types.ModuleType("lerobot.configs")
    policies_cfg = types.ModuleType("lerobot.configs.policies")
    policies_cfg.PreTrainedConfig = Cfg
    policies = types.ModuleType("lerobot.policies")
    factory = types.ModuleType("lerobot.policies.factory")
    factory.get_policy_class = lambda name: {"act": ActPolicy}[name]
    for name, mod in [
        ("lerobot", lerobot), ("lerobot.configs", configs), ("lerobot.configs.policies", policies_cfg),
        ("lerobot.policies", policies), ("lerobot.policies.factory", factory),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    pol = load_lerobot_policy("someone/act_so101")
    assert isinstance(pol, LeRobotPolicy) and isinstance(pol.policy, ActPolicy)
    assert made == {"cfg": "someone/act_so101", "policy": ("someone/act_so101", "act")}


# --------------------------------------------------------------------------
# closed loop: the stepper reproduces rollout
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def soft_arm() -> Arm:
    return Arm(SOFT)


def test_logged_policy_closed_loop_equals_replay(soft_arm, episodes):
    from antares.closedloop import run_closed_loop

    ep = episodes[0]
    sim_q, actions = run_closed_loop(LoggedPolicy(ep), soft_arm, soft_arm.real_q(ep)[0], len(ep), ep.fps)
    assert np.allclose(actions, ep.action)
    assert np.max(np.abs(sim_q - soft_arm.replay(ep))) < 1e-9


def test_zero_latency_sim_is_stepped_with_one_frame_in_closed_loop(episodes):
    from antares.closedloop import run_closed_loop

    arm = Arm(SimParams.default())
    ep = episodes[1]
    sim_q, _ = run_closed_loop(LoggedPolicy(ep), arm, arm.real_q(ep)[0], len(ep), ep.fps)
    ctrl = arm.unit_map.to_rad(ep.action)
    ref = arm.rollout(ctrl, arm.real_q(ep)[0], ep.fps, latency_frames=1)
    assert np.max(np.abs(sim_q - ref)) < 1e-9
    assert np.max(np.abs(sim_q - arm.replay(ep))) > 1e-6  # and not the zero-latency replay


def test_closed_loop_runs_per_joint_latency_deadband_and_command_map(episodes):
    from antares.closedloop import run_closed_loop

    params = SimParams(
        kp=(90, 80, 70, 60, 50, 40), kv=1.0, damping=0.05, armature=0.01, frictionloss=0.02,
        latency_frames=(2, 3, 4, 3, 2, 1), deadband=0.01,
    )
    arm = Arm(params, command_map=CommandMap(np.full(6, 1.02), np.full(6, 0.5)))
    ep = episodes[2]
    sim_q, actions = run_closed_loop(LoggedPolicy(ep), arm, arm.real_q(ep)[0], len(ep), ep.fps)
    assert np.max(np.abs(sim_q - arm.replay(ep))) < 1e-9
    assert np.allclose(actions, ep.action)


def test_closed_loop_executes_the_shifted_chunk_entry(soft_arm, episodes):
    from antares.closedloop import run_closed_loop

    ep = episodes[0]
    _, actions = run_closed_loop(LoggedPolicy(ep), soft_arm, soft_arm.real_q(ep)[0], len(ep), ep.fps, exec_index=4)
    assert np.allclose(actions[:-4], ep.action[4:])


# --------------------------------------------------------------------------
# shadow
# --------------------------------------------------------------------------


def _lock_dict(name: str, params: SimParams, cmap: CommandMap | None = None) -> dict:
    d = {"name": name, "dataset": "fake/ds", "robot": "so101", "params": params.to_dict()}
    if cmap is not None:
        d["command_map"] = cmap.to_dict()
    return d


def test_variant_from_lock_carries_params_and_command_map():
    from antares.shadow import default_variant, variant_from_lock

    v = variant_from_lock(_lock_dict("unit", SOFT, CommandMap(np.ones(6), np.full(6, 1.0))), "lock")
    assert v.lock == "unit" and v.params == SOFT and v.robot == "so101"
    assert np.array_equal(v.latency_frames, np.full(6, 4))
    arm = v.arm()
    assert np.allclose(arm.command_map.bias, 1.0) and arm.params == SOFT
    d = default_variant().to_dict()
    assert d["name"] == "default" and d["latency_frames"] == [0] * 6 and d["command_map"] is None


def test_shadow_with_logged_policy_reproduces_replay_and_finds_grasps(episodes):
    from antares.shadow import GraspContext, ShadowSummary, default_variant, shadow_episode, variant_from_lock

    ctx = GraspContext.for_dataset(episodes)
    variants = [default_variant(), variant_from_lock(_lock_dict("unit", SOFT), "lock")]
    ep = episodes[0]
    se = shadow_episode(LoggedPolicy(ep), ep, variants, ctx)
    assert np.allclose(se.actions, ep.action)
    assert se.action_gap.overall_deg == pytest.approx(0.0)
    for name in ("default", "lock"):
        assert se.gaps[name].overall_deg == pytest.approx(se.replay_gaps[name].overall_deg)
    assert se.divergence["lock"].overall_deg > 0.1  # a soft laggy servo is not the stock one
    assert len(se.worst_frames["lock"]) == 5 and se.worst_frames["lock"][0]["deg"] >= se.worst_frames["lock"][-1]["deg"]
    assert set(se.per_joint_peak["lock"]) == set(JOINTS)
    assert se.grasp_frames == [50]
    assert len(se.grasp_errors["lock"]) == 1 and len(se.grasp_divergence["lock"]) == 1
    assert se.grasp_errors["default"][0] >= 0 and se.grasp_divergence["lock"][0] > 0
    summary = ShadowSummary("shadow", variants, [se])
    d = summary.to_dict(include_arrays=True)
    json.dumps(d)  # serializable
    agg = d["summary"]
    assert agg["n_episodes"] == 1 and agg["variants"]["lock"]["divergence_deg"] > 0
    assert agg["variants"]["lock"]["n_grasps"] == 1 and "divergence_deg" not in agg["variants"]["default"]
    assert agg["variants"]["lock"]["worst_frames"][0]["episode"] == 0


def test_shadow_episodes_reports_progress_and_can_stop(episodes):
    from antares.shadow import GraspContext, default_variant, shadow_episodes

    ctx = GraspContext.for_dataset(episodes)
    seen = []
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    class Replayer:
        k, h = 1, 1
        features = Features(goal=False, time=False)

        def reset(self):
            pass

        def act(self, obs, ctx):  # not used: a policy with features batches
            raise AssertionError

        def act_batch(self, obs, extras=None):
            assert extras is None
            return np.repeat(obs[:, -1:, :], 1, axis=1)

    s = shadow_episodes(Replayer(), episodes[:4], [default_variant()], ctx, on_progress=lambda i, n, se: seen.append((i, n)), should_stop=stop)
    assert seen == [(1, 4), (2, 4)] and len(s.episodes) == 2


# --------------------------------------------------------------------------
# closed-loop scoring
# --------------------------------------------------------------------------


def test_score_run_grasp_point_and_success(episodes):
    from antares.closedloop import SUCCESS_CM, score_run
    from antares.shadow import GraspContext, default_variant

    ctx = GraspContext.for_dataset(episodes)
    v = default_variant()
    arm = v.arm()
    ep = episodes[0]
    real_q = arm.real_q(ep)
    # the sim went exactly where the arm went and closed when the operator did
    run = score_run(v, arm, ep, real_q.copy(), ep.action.copy(), ctx)
    assert run.gap.overall_deg == pytest.approx(0.0) and run.final_error_deg == pytest.approx(0.0)
    assert run.real_grasp_frame == 50 and run.close_frame == 50
    assert run.grasp_error_cm == pytest.approx(0.0, abs=1e-9) and run.success is True
    assert run.min_tip_distance_cm == pytest.approx(0.0, abs=1e-9)
    # never closing is a failure, with the closest approach still reported
    never = ep.action.copy()
    never[:, 5] = 60.0
    run2 = score_run(v, arm, ep, real_q.copy(), never, ctx)
    assert run2.close_frame is None and run2.grasp_error_cm is None and run2.success is False
    assert run2.min_tip_distance_cm == pytest.approx(0.0, abs=1e-9)
    # closing far from the grasp point fails the 2 cm proxy
    early = ep.action.copy()
    early[:, 5] = 60.0
    early[5:, 5] = 5.0
    run3 = score_run(v, arm, ep, real_q.copy(), early, ctx)
    assert run3.close_frame == 5 and run3.grasp_error_cm > SUCCESS_CM and run3.success is False
    d = run3.to_dict(include_arrays=True)
    json.dumps(d)
    assert d["variant"] == "default" and len(d["sim_q_deg"]) == len(ep)


def test_closed_loop_episodes_runs_every_variant_with_shift(episodes):
    from antares.closedloop import closed_loop_episodes
    from antares.shadow import GraspContext, default_variant, variant_from_lock

    ctx = GraspContext.for_dataset(episodes)
    variants = [default_variant(), variant_from_lock(_lock_dict("unit", SOFT), "lock")]

    class Hold:
        """Commands the current state: the arm stays put (and never closes)."""

        k, h = 2, 10

        def reset(self):
            pass

        def act(self, obs, ctx):
            assert ctx.goal is not None and ctx.goal.shape == (3,)
            return np.repeat(obs[-1:], 10, axis=0)

    seen = []
    s = closed_loop_episodes(Hold(), episodes[:2], variants, ctx, on_progress=lambda i, n, b: seen.append(len(b)))
    assert s.shift_variants == ["lock+shift"] and s.run_names == ["default", "lock", "lock+shift"]
    assert seen == [3, 3] and len(s.runs) == 6
    agg = s.aggregate()
    assert agg["n_episodes"] == 2
    for n in s.run_names:
        row = agg["variants"][n]
        assert row["n_judged"] == 2 and row["n_closed"] == 0 and row["success_rate"] == 0.0
        assert row["grasp_error_cm"] is None and row["min_tip_distance_cm"] > 0
        assert row["trajectory_rms_deg"] > 1.0  # holding still is not the trajectory
    d = s.to_dict()
    json.dumps(d)
    assert d["mode"] == "closedloop" and d["success_cm"] == 2.0
    # the logged commands fed back reach the real grasp point in a sim that
    # follows them well enough
    stiff = variant_from_lock(_lock_dict("stiff", SimParams(latency_frames=1)), "stiff")

    class Logged:
        k, h = 2, 10

        def __init__(self):
            self.inner = None

        def reset(self):
            self.inner.reset()

        def act(self, obs, ctx):
            return self.inner.act(obs, ctx)

    runs = []
    for ep in episodes[:2]:
        pol = Logged()
        pol.inner = LoggedPolicy(ep)
        runs.append(closed_loop_episodes(pol, [ep], [stiff], ctx, shift=False).runs[0])
    assert all(r.close_frame == 50 for r in runs)
    assert all(r.success is True for r in runs), [r.grasp_error_cm for r in runs]
