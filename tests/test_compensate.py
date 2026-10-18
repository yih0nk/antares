"""Lag compensation in the calibrated sim. Everything here is sim-only: the
"desired" trajectory is a synthetic or logged measured trajectory and
tracking is judged against MuJoCo, never hardware."""

from __future__ import annotations

import numpy as np
import pytest

from antares.compensate import (
    METHODS,
    CompensationReport,
    compensate_episode,
    compensate_episodes,
    feasibility,
    held_out_episodes,
    ilc,
    ilc_compensate,
    max_joint_velocity,
    shift_compensate,
)
from antares.data import Dataset, Episode
from antares.gap import tracking_error
from antares.sim import JOINTS, Arm, SimParams

FPS = 30.0
# A "soft, laggy" servo in the spirit of the SO-101 lock.
SOFT = SimParams(kp=90.0, kv=1.5, damping=0.05, armature=0.02, frictionloss=0.025, latency_frames=4)


@pytest.fixture(scope="module")
def arm() -> Arm:
    return Arm()


def smooth_trajectory(n: int = 150, seed: int = 0) -> np.ndarray:
    """A slow, smooth (T,6) trajectory in dataset units that starts gently from rest.

    The first ``latency_frames`` frames of any rollout are uncontrollable
    (nothing sent at t >= 0 arrives before t = L), so a hard velocity onset
    would put an irreducible floor under every method; the cosine ramp
    keeps that floor small so the methods can be told apart.
    """
    t = np.linspace(0, 1, n)
    rng = np.random.default_rng(seed)
    base = np.array([0.0, -40.0, 30.0, 10.0, 0.0, 20.0])
    amp = np.array([30.0, 20.0, 25.0, 20.0, 30.0, 15.0])
    phase = rng.uniform(0, 2 * np.pi, 6)
    freq = rng.uniform(0.5, 1.2, 6)
    ramp = 0.5 * (1 - np.cos(np.pi * np.minimum(t / 0.3, 1.0)))
    q = base + amp * np.sin(2 * np.pi * freq * t[:, None] + phase) * ramp[:, None]
    q[:, 5] = np.clip(q[:, 5], 0, 100)
    return q


def make_episode(index: int = 0, n: int = 150, seed: int = 0) -> Episode:
    q = smooth_trajectory(n, seed)
    # the logged "command" leads the measured state, as a leader arm would
    action = np.vstack([q[3:], np.repeat(q[-1:], 3, axis=0)])
    return Episode(index=index, fps=FPS, joint_names=JOINTS, state=q, action=action)


# -- shift -------------------------------------------------------------------


def test_shift_compensate_leads_and_holds_end():
    q = np.arange(10, dtype=float)[:, None].repeat(6, axis=1)
    out = shift_compensate(q, 3)
    np.testing.assert_array_equal(out[:, 0], [3, 4, 5, 6, 7, 8, 9, 9, 9, 9])
    np.testing.assert_array_equal(shift_compensate(q, 0), q)
    assert shift_compensate(q, 0) is not q  # a copy
    np.testing.assert_array_equal(shift_compensate(q, 100), np.repeat(q[-1:], 10, axis=0))


def test_shift_compensate_rejects_bad_input():
    with pytest.raises(ValueError):
        shift_compensate(np.zeros(6), 1)
    with pytest.raises(ValueError):
        shift_compensate(np.zeros((5, 6)), -1)


def test_shift_cancels_pure_latency_on_stiff_arm(arm):
    """With a stiff servo and only latency, shifting by L recovers the trajectory."""
    arm.apply(SimParams(latency_frames=4))
    q_des = arm.unit_map.to_rad(smooth_trajectory())
    unshifted = arm.rollout(q_des, q_des[0], FPS)
    shifted = arm.rollout(shift_compensate(q_des, 4), q_des[0], FPS)
    e0 = tracking_error(q_des, unshifted).overall_deg
    e1 = tracking_error(q_des, shifted).overall_deg
    assert e1 < 0.25 * e0
    arm.apply(SimParams())


# -- ILC ---------------------------------------------------------------------


def test_ilc_beats_shift_and_stops_when_flat(arm):
    arm.apply(SOFT)
    q_des = arm.unit_map.to_rad(smooth_trajectory())
    res = ilc(arm, q_des, q_des[0], FPS, iters=10, gain=0.5)
    assert res.ctrl.shape == q_des.shape and res.q_sim.shape == q_des.shape
    assert res.history_deg[0] == pytest.approx(
        tracking_error(q_des, arm.rollout(shift_compensate(q_des, 4), q_des[0], FPS)).overall_deg
    )
    assert res.rms_deg < 0.5 * res.history_deg[0]
    assert res.rms_deg < 0.2  # degrees, sim-only
    assert 1 <= res.iterations <= 10
    # history is monotone over the accepted steps, then at most one rejected step
    accepted = res.history_deg[: res.iterations + 1]
    assert all(b < a for a, b in zip(accepted, accepted[1:]))
    assert len(res.history_deg) <= res.iterations + 2
    # the returned commands reproduce the returned trajectory
    np.testing.assert_allclose(arm.rollout(res.ctrl, q_des[0], FPS), res.q_sim)
    # the best error is the last accepted one
    assert res.rms_deg == pytest.approx(res.history_deg[res.iterations])
    arm.apply(SimParams())


def test_ilc_compensate_returns_clipped_ctrl(arm):
    arm.apply(SOFT)
    q_des = arm.unit_map.to_rad(smooth_trajectory())
    lo, hi = arm.ctrl_range
    ctrl = ilc_compensate(arm, q_des, q_des[0], FPS, iters=3)
    assert ctrl.shape == q_des.shape
    assert np.all(ctrl >= lo - 1e-12) and np.all(ctrl <= hi + 1e-12)
    np.testing.assert_array_equal(ctrl, ilc(arm, q_des, q_des[0], FPS, iters=3).ctrl)
    # iters=0 is just the shifted baseline
    res0 = ilc(arm, q_des, q_des[0], FPS, iters=0)
    assert res0.iterations == 0 and len(res0.history_deg) == 1 and not res0.converged
    np.testing.assert_array_equal(res0.ctrl, np.clip(shift_compensate(q_des, 4), lo, hi))
    arm.apply(SimParams())


def test_ilc_explicit_latency_and_validation(arm):
    arm.apply(SOFT)
    q_des = arm.unit_map.to_rad(smooth_trajectory())
    right = ilc(arm, q_des, q_des[0], FPS, iters=2, latency=4)
    wrong = ilc(arm, q_des, q_des[0], FPS, iters=2, latency=0)
    assert right.rms_deg < wrong.rms_deg
    with pytest.raises(ValueError):
        ilc(arm, q_des[:, :5], q_des[0], FPS)
    with pytest.raises(ValueError):
        ilc(arm, q_des, q_des[0], FPS, iters=-1)
    with pytest.raises(ValueError):
        ilc(arm, q_des, q_des[0], FPS, gain=0.0)
    arm.apply(SimParams())


# -- feasibility ---------------------------------------------------------------


def test_max_joint_velocity(arm):
    ep = make_episode()
    v = max_joint_velocity([ep], arm)
    q = np.degrees(arm.real_q(ep))
    np.testing.assert_allclose(v, np.max(np.abs(np.diff(q, axis=0)), axis=0) * FPS)
    assert v.shape == (6,) and np.all(v > 0)
    one = Episode(index=1, fps=FPS, joint_names=JOINTS, state=ep.state[:1], action=ep.action[:1])
    np.testing.assert_array_equal(max_joint_velocity([one], arm), 0.0)


def test_feasibility_flags_fast_and_counts_clipping(arm):
    lo, hi = arm.ctrl_range
    T = 20
    raw = np.zeros((T, 6))
    raw[:, 0] = np.linspace(0, 0.5, T)  # moves at 0.5 rad over 19 frames
    comp = raw.copy()
    comp[:, 1] = np.where(np.arange(T) % 2 == 0, 0.0, 0.3)  # a 0.3 rad zigzag: fast
    comp[5:8, 2] = hi[2] + 1.0  # beyond the limit -> would be clipped by the sim
    comp = np.clip(comp, lo, hi)
    real_vmax = np.full(6, 100.0)  # deg/s
    real_vmax[2] = 1e6  # the elbow step is fast too; only judge its clipping here
    f = feasibility(raw, comp, FPS, real_vmax, (lo, hi), factor=1.5)
    assert f["flagged_joints"] == ["shoulder_lift"]
    assert f["ok"] is False
    assert f["clipped_frames"] == 3 and f["raw_clipped_frames"] == 0
    pj = f["per_joint"]
    assert pj["shoulder_lift"]["flagged"] and pj["shoulder_lift"]["velocity_ratio"] > 1.5
    assert pj["shoulder_pan"]["velocity_ratio"] == pytest.approx(pj["shoulder_pan"]["raw_cmd_vmax_deg_s"] / 100.0)
    assert pj["elbow_flex"]["range_excess_deg"] == pytest.approx(np.degrees(hi[2]))
    assert pj["shoulder_pan"]["range_excess_deg"] == 0.0
    assert f["max_range_excess_deg"] == pytest.approx(max(np.degrees(hi[2]), np.degrees(0.3)))

    ok = feasibility(raw, raw, FPS, real_vmax, (lo, hi))
    assert ok["ok"] is True and ok["flagged_joints"] == [] and ok["max_range_excess_deg"] == 0.0
    with pytest.raises(ValueError):
        feasibility(raw, raw[:-1], FPS, real_vmax, (lo, hi))


# -- episode evaluation ------------------------------------------------------


def test_compensate_episode_raw_reproduces_gap(arm):
    ep = make_episode()
    rep = compensate_episode(arm, ep, SOFT)
    assert isinstance(rep, CompensationReport)
    assert rep.method == "ilc" and rep.params == SOFT and rep.fps == FPS
    # (a) raw commands: identical to replay, so the raw RMS *is* the gap number
    np.testing.assert_array_equal(rep.q_sim_raw, arm.replay(ep))
    assert rep.rms["raw"] == pytest.approx(tracking_error(arm.real_q(ep), arm.replay(ep)).overall_deg)
    # (b) shift and (c) ILC each improve on the previous
    assert rep.rms["shift"] < rep.rms["raw"]
    assert rep.rms["ilc"] < rep.rms["shift"]
    assert rep.ilc_iterations >= 1 and rep.ilc_history_deg[-1] >= 0
    assert rep.ctrl_comp is rep.ctrl_ilc and rep.q_sim_comp is rep.q_sim_ilc
    assert set(rep.feasibility) >= {"ok", "flagged_joints", "per_joint", "clipped_frames", "max_range_excess_deg"}
    for a in (rep.q_des, rep.ctrl_raw, rep.ctrl_shift, rep.ctrl_ilc, rep.q_sim_shift, rep.q_sim_ilc):
        assert a.shape == (len(ep), 6)
    assert arm.params == SOFT

    shift = compensate_episode(arm, ep, SOFT, method="shift")
    assert shift.ctrl_comp is shift.ctrl_shift and shift.rms == rep.rms
    with pytest.raises(ValueError):
        compensate_episode(arm, ep, SOFT, method="pid")
    arm.apply(SimParams())


def test_compensate_episodes_aggregates(arm):
    eps = [make_episode(i, n=120, seed=i) for i in range(3)]
    s = compensate_episodes(arm, eps, SOFT, iters=4)
    assert s.episodes == (0, 1, 2)
    assert s.n_frames == 360
    assert len(s.reports) == 3 and len(s.ilc_iterations) == 3
    assert s.rms["ilc"] < s.rms["shift"] < s.rms["raw"]
    assert set(s.per_joint_deg) == {"raw", "shift", "ilc"}
    assert set(s.per_joint_deg["raw"]) == set(JOINTS)
    f = s.feasibility
    assert set(f) >= {"ok", "flagged_joints", "episodes_flagged", "real_vmax_deg_s", "comp_cmd_vmax_deg_s", "worst_episode"}
    # the shared velocity reference is the max over all the episodes
    np.testing.assert_allclose([f["real_vmax_deg_s"][j] for j in JOINTS], max_joint_velocity(eps, arm))
    d = s.to_dict()
    import json

    json.dumps(d)
    assert d["rms"] == s.rms and d["episodes"] == [0, 1, 2]
    with pytest.raises(ValueError):
        compensate_episodes(arm, [], SOFT)
    arm.apply(SimParams())


def test_held_out_episodes():
    eps = tuple(make_episode(i, n=10) for i in range(4))
    ds = Dataset(repo_id="x/y", robot_type="so101", fps=FPS, joint_names=JOINTS, episodes=eps)
    got = held_out_episodes(ds, [3, 1])
    assert [e.index for e in got] == [3, 1]
    with pytest.raises(ValueError):
        held_out_episodes(ds, [7])


def test_methods_constant():
    assert METHODS == ("shift", "ilc")


def test_compensate_episode_raw_commands_use_the_command_map():
    from antares.sim import CommandMap

    ep = make_episode()
    mapped_arm = Arm(command_map=CommandMap(gain=np.ones(6), bias=[0.0, 4.0, 0.0, 0.0, 0.0, 0.0]))
    rep = compensate_episode(mapped_arm, ep, SOFT, method="shift")
    # the raw commands are the mapped ones, so the raw rollout is the mapped replay
    np.testing.assert_array_equal(rep.q_sim_raw, mapped_arm.replay(ep))
    assert rep.rms["raw"] == pytest.approx(
        tracking_error(mapped_arm.real_q(ep), mapped_arm.replay(ep)).overall_deg)
