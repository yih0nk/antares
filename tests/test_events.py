"""Contact-event detection from the real - sim residual. Pure numpy, no MuJoCo."""

from __future__ import annotations

import numpy as np
import pytest

from antares import events as E
from antares.data import Episode
from antares.sim import JOINTS

RNG = np.random.default_rng(0)


def gripper_profile(T=300, grasp=150, release=220, open_v=22.0, closed_v=1.5):
    """Closed, open at 100, closed again at ``grasp``, open at ``release``, closed at ``release+20``."""
    g = np.full(T, closed_v)
    g[100:grasp] = open_v
    g[release : release + 20] = open_v
    return g


def noise_residual(T=300, sigma=0.3, J=6, rng=RNG):
    return rng.normal(0.0, sigma, size=(T, J))


def noise_baseline(sigma=0.3, J=6):
    return E.ResidualBaseline(
        joints=JOINTS, center=np.zeros(J), scale=np.full(J, sigma), acf1=np.zeros(J), n_frames=1000, n_episodes=1
    )


# --------------------------------------------------------------------------
# residuals
# --------------------------------------------------------------------------


def test_residuals_sign_and_units():
    real = np.zeros((4, 6))
    sim = np.full((4, 6), np.radians(2.0))
    r = E.residuals(real, sim)
    assert r.shape == (4, 6)
    assert np.allclose(r, -2.0)


def test_residuals_rejects_mismatch():
    with pytest.raises(ValueError):
        E.residuals(np.zeros((3, 6)), np.zeros((4, 6)))


# --------------------------------------------------------------------------
# gripper proxy label
# --------------------------------------------------------------------------


def test_gripper_thresholds_split_bimodal_command():
    cmd = np.concatenate([np.full(500, 1.5) + RNG.normal(0, 0.3, 500), np.full(200, 22.0) + RNG.normal(0, 1.0, 200)])
    thr = E.gripper_thresholds(cmd)
    assert 5 < thr.closed_below < thr.open_above < 20
    assert E.GripperThresholds.from_dict(thr.to_dict()) == thr


def test_gripper_thresholds_constant_command():
    thr = E.gripper_thresholds(np.full(50, 3.0))
    assert thr.closed_below == thr.open_above == 3.0


def test_gripper_state_hysteresis():
    thr = E.GripperThresholds(closed_below=8.0, open_above=15.0)
    cmd = np.array([1, 1, 10, 20, 12, 12, 9, 5, 12, 20])
    s = E.gripper_state(cmd, thr)
    assert s.tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
    assert E.gripper_state(np.array([12.0, 1.0]), thr).tolist() == [1, 0]  # start from the midpoint
    assert E.gripper_state(np.zeros(0), thr).shape == (0,)


def test_gripper_transitions_and_grasp_windows():
    thr = E.GripperThresholds(closed_below=8.0, open_above=15.0)
    cmd = gripper_profile()
    grasps, releases = E.gripper_transitions(cmd, thr)
    assert grasps.tolist() == [150, 240]
    assert releases.tolist() == [100, 220]
    assert E.grasp_windows(cmd, thr, pad=15) == [(135, 166), (225, 256)]
    # windows clip to the episode and merge when they touch
    assert E.windows_around(np.array([3, 290]), 300, pad=15) == [(0, 19), (275, 300)]
    assert E.windows_around(np.array([50, 60]), 300, pad=15) == [(35, 76)]


def test_free_motion_mask_excludes_holding_and_transitions():
    thr = E.GripperThresholds(closed_below=8.0, open_above=15.0)
    cmd = gripper_profile()
    m = E.free_motion_mask(cmd, thr, margin=15)
    assert m[:85].all()  # at rest before the first transition
    assert not m[85:116].any()  # around the release at 100
    assert m[116:135].all()  # reaching with an open gripper
    assert not m[150:220].any()  # holding the object
    assert not m[240:].any()  # closed again with no later release: holding to the end
    # with a final release the tail is free again
    cmd2 = cmd.copy()
    cmd2[270:] = 22.0
    m2 = E.free_motion_mask(cmd2, thr, margin=15)
    assert m2[286:].all() and not m2[240:286].any()


# --------------------------------------------------------------------------
# baseline
# --------------------------------------------------------------------------


def test_baseline_from_residuals_recovers_scale_and_center():
    rng = np.random.default_rng(1)
    res = [rng.normal(0.5, 0.4, size=(2000, 6)) for _ in range(3)]
    bl = E.baseline_from_residuals(res, episodes=[0, 1, 2])
    assert np.allclose(bl.center, 0.5, atol=0.05)
    assert np.allclose(bl.scale, 0.4, atol=0.05)
    assert np.all(np.abs(bl.acf1) < 0.1)
    assert bl.n_frames == 6000 and bl.n_episodes == 3 and bl.episodes == (0, 1, 2)
    z = bl.z(np.full((5, 6), 0.9))
    assert np.allclose(z, 1.0, atol=0.2)
    again = E.ResidualBaseline.from_dict(bl.to_dict())
    assert np.allclose(again.scale, bl.scale) and again.joints == bl.joints


def test_baseline_uses_mask_and_measures_autocorrelation():
    T = 4000
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.9 * x[t - 1] + RNG.normal(0, 0.1)
    res = np.tile(x[:, None], (1, 6))
    res[2000:] += 50.0  # a huge "contact" that the mask excludes
    mask = np.arange(T) < 2000
    bl = E.baseline_from_residuals([res], [mask])
    assert bl.free_frames_only and bl.n_frames == 2000
    assert np.all(np.abs(bl.center) < 0.2)
    assert np.all(bl.acf1 > 0.8)
    assert np.all(bl.correlation_frames > 4)


def test_baseline_falls_back_to_all_frames_when_mask_is_tiny():
    res = [RNG.normal(0, 0.2, size=(100, 6))]
    bl = E.baseline_from_residuals(res, [np.arange(100) < 10], min_free_frames=50)
    assert not bl.free_frames_only and bl.n_frames == 100


def test_baseline_scale_floor():
    bl = E.baseline_from_residuals([np.zeros((100, 6))], min_scale_deg=0.1)
    assert np.allclose(bl.scale, 0.1)


class FakeArm:
    """real_q returns the state, replay the action, both as radians."""

    def real_q(self, ep):
        return np.radians(ep.state)

    def replay(self, ep):
        return np.radians(ep.action)


def test_residual_baseline_from_episodes_with_fake_arm():
    eps = []
    for i in range(3):
        T = 300
        action = np.zeros((T, 6))
        action[:, 5] = gripper_profile(T)
        state = action + RNG.normal(0, 0.25, size=(T, 6))
        state[150:220, :5] += 5.0  # the held object loads the arm; not free motion
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=state, action=action))
    bl = E.residual_baseline(eps, FakeArm())
    assert bl.free_frames_only
    assert np.allclose(bl.center[:5], 0.0, atol=0.1)
    assert np.allclose(bl.scale[:5], 0.25, atol=0.08)
    assert bl.episodes == (0, 1, 2)


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def test_no_events_on_pure_noise():
    bl = noise_baseline(0.3)
    r = noise_residual(T=5000, sigma=0.3, rng=np.random.default_rng(2))
    assert E.detect_events(r, bl, z=4.0, min_frames=5) == []


def test_sustained_bump_is_a_contact_on_the_right_joints():
    bl = noise_baseline(0.3)
    r = noise_residual(rng=np.random.default_rng(3))
    r[100:140, 2] += 3.0  # elbow_flex, 10 sigma, one-sided
    r[110:130, 1] -= 2.0  # shoulder_lift too
    evs = E.detect_events(r, bl)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.kind == "contact"
    assert set(ev.joints) == {"shoulder_lift", "elbow_flex"}
    assert 95 <= ev.start <= 100 and 140 <= ev.end <= 145
    assert ev.peak_z == pytest.approx(np.abs(bl.z(r)[100:140, 2]).max())
    assert ev.peak_deg > 2.0
    assert ev.frames == ev.end - ev.start
    d = ev.to_dict()
    assert d["start"] == ev.start and d["kind"] == "contact" and d["joints"] == list(ev.joints)


def test_short_spike_is_dropped_by_min_frames():
    bl = noise_baseline(0.3)
    r = noise_residual(rng=np.random.default_rng(4))
    r[50:53, 0] += 5.0
    assert E.detect_events(r, bl, min_frames=5) == []
    evs = E.detect_events(r, bl, min_frames=2)
    assert len(evs) == 1 and evs[0].kind == "anomaly"


def test_hysteresis_extends_to_half_threshold():
    bl = noise_baseline(1.0)
    r = np.zeros((100, 6))
    r[20:30, 0] = 5.0  # above z=4
    r[30:40, 0] = 3.0  # between z/2 and z: stays active
    r[40:50, 0] = 1.0  # below z/2: exits
    evs = E.detect_events(r, bl, z=4.0, min_frames=5)
    assert len(evs) == 1 and (evs[0].start, evs[0].end) == (20, 40)
    # a plateau that only reaches the exit threshold never enters
    r2 = np.zeros((100, 6))
    r2[20:40, 0] = 3.0
    assert E.detect_events(r2, bl, z=4.0) == []


def test_close_events_merge_and_far_ones_do_not():
    bl = noise_baseline(1.0)
    r = np.zeros((200, 6))
    r[20:30, 1] = 6.0
    r[32:42, 1] = 6.0  # gap of 2 <= merge_frames
    r[100:110, 1] = 6.0  # far away
    evs = E.detect_events(r, bl, merge_frames=3)
    assert [(e.start, e.end) for e in evs] == [(20, 42), (100, 110)]
    evs = E.detect_events(r, bl, merge_frames=0)
    assert [(e.start, e.end) for e in evs] == [(20, 30), (32, 42), (100, 110)]


def test_sign_flipping_deviation_is_an_anomaly():
    bl = noise_baseline(1.0)
    r = np.zeros((100, 6))
    r[20:40, 3] = 6.0 * np.where(np.arange(20) % 2 == 0, 1, -1)
    (ev,) = E.detect_events(r, bl)
    assert ev.kind == "anomaly" and ev.joints == ("wrist_flex",)


def test_gripper_is_not_scored_by_default():
    bl = noise_baseline(1.0)
    r = np.zeros((100, 6))
    r[20:60, 5] = 50.0
    assert E.detect_events(r, bl) == []
    (ev,) = E.detect_events(r, bl, joints=("gripper",))
    assert ev.joints == ("gripper",)


def test_warmup_frames_are_ignored():
    bl = noise_baseline(1.0)
    r = np.zeros((100, 6))
    r[0:8, 4] = 10.0
    assert len(E.detect_events(r, bl)) == 1
    assert E.detect_events(r, bl, warmup=8) == []
    assert E.detect_events(r, bl, warmup=4) == []  # the remaining 4 frames are < min_frames


def test_detect_events_validates_arguments():
    bl = noise_baseline(1.0)
    r = np.zeros((10, 6))
    with pytest.raises(ValueError):
        E.detect_events(r, bl, z=0)
    with pytest.raises(ValueError):
        E.detect_events(r, bl, min_frames=0)
    with pytest.raises(ValueError):
        E.detect_events(np.zeros((10, 5)), bl)
    assert E.detect_events(np.zeros((0, 6)), bl) == []


def test_hysteresis_helper():
    x = np.array([0, 5, 3, 3, 1, 3, 5, 1])
    assert E.hysteresis(x, 4, 2).tolist() == [False, True, True, True, False, False, True, False]
    with pytest.raises(ValueError):
        E.hysteresis(x, 2, 4)


# --------------------------------------------------------------------------
# validation against the proxy label
# --------------------------------------------------------------------------


def test_validate_episode_counts_hits_and_far_events():
    thr = E.GripperThresholds(closed_below=8.0, open_above=15.0)
    cmd = gripper_profile()  # grasps at 150 and 240, releases at 100 and 220
    r = np.zeros((300, 6))
    r[140:160, 2] = 2.0  # energy inside the first grasp window
    r[10:20, 2] = 2.0  # energy far from everything
    events = [
        E.Event(145, 160, ("elbow_flex",), 8.0, 2.0, "contact"),  # hits grasp 1
        E.Event(95, 105, ("elbow_flex",), 5.0, 1.0, "anomaly"),  # near a release only
        E.Event(10, 20, ("elbow_flex",), 5.0, 2.0, "anomaly"),  # far
    ]
    v = E.validate_episode(r, events, cmd, thr, episode=7, pad=15)
    assert v.episode == 7 and v.n_frames == 300
    assert v.n_grasps == 2 and v.n_grasps_hit == 1
    assert v.n_events == 3 and v.n_events_in_grasp == 1
    assert v.n_events_near_transition == 2 and v.n_events_far == 1
    assert v.grasp_windows == ((135, 166), (225, 256))
    assert v.n_frames_in_grasp == 62
    assert v.energy_total == pytest.approx(30 * 4.0)
    assert v.energy_in_grasp == pytest.approx(20 * 4.0)
    d = v.to_dict()
    assert d["grasp_windows"] == [[135, 166], [225, 256]]

    s = E.summarize_validation([v])
    assert s["recall"] == 0.5
    assert s["precision"] == pytest.approx(1 / 3)
    assert s["near_transition"] == pytest.approx(2 / 3)
    assert s["far_fraction"] == pytest.approx(1 / 3)
    assert s["energy_in_grasp_fraction"] == pytest.approx(2 / 3)
    assert s["chance"] == pytest.approx(62 / 300)
    assert s["lift"] == pytest.approx((2 / 3) / (62 / 300))


def test_summarize_validation_handles_empty():
    s = E.summarize_validation([])
    assert s["n_events"] == 0 and s["recall"] is None and s["precision"] is None and s["lift"] is None


def test_validate_episode_rejects_bad_gripper_length():
    thr = E.GripperThresholds(8.0, 15.0)
    with pytest.raises(ValueError):
        E.validate_episode(np.zeros((10, 6)), [], np.zeros(9), thr)
