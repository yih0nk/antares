from __future__ import annotations

import numpy as np
import pytest

from antares.data import Episode
from antares.sag import holding_torque, sag_report
from antares.sim import JOINTS, Arm, SimParams

# A soft servo, like the SO-101 lock (kp pinned near 10), sags by degrees.
SOFT = SimParams(kp=10.5, kv=0.14, damping=0.012, armature=0.0028, frictionloss=0.094, latency_frames=4)


def sweep_episode(arm: Arm, index: int, T: int = 150, seed: int = 0) -> Episode:
    """Slow sinusoidal sweeps of the lift/elbow/wrist; state from ``arm``'s own replay."""
    rng = np.random.default_rng(seed + index)
    t = np.arange(T) / 30.0
    action = np.zeros((T, 6))
    for j in range(1, 4):
        f, ph, amp = rng.uniform(0.15, 0.35), rng.uniform(0, 6), rng.uniform(25, 45)
        action[:, j] = amp * np.sin(2 * np.pi * f * t + ph)
    action[:, 5] = 50.0
    ep = Episode(index=index, fps=30.0, joint_names=JOINTS, state=action, action=action)
    state = arm.unit_map.to_units(arm.replay(ep))
    return Episode(index=index, fps=30.0, joint_names=JOINTS, state=state, action=action)


def test_holding_torque_follows_pose_and_gravity_scale():
    arm = Arm(SOFT)
    stretched = np.zeros(6)
    folded = np.array([0.0, -1.5, 1.5, 0.0, 0.0, 0.0])
    t_s, t_f = holding_torque(arm, stretched), holding_torque(arm, folded)
    assert t_s[0] == pytest.approx(0.0, abs=1e-12)  # the pan axis is vertical
    assert abs(t_s[1]) > 3 * abs(t_f[1])  # the shoulder carries more when stretched out
    both = holding_torque(arm, np.stack([stretched, folded]))
    np.testing.assert_allclose(both, [t_s, t_f])
    arm.apply(SOFT.replace(gravity_scale=0.5))
    np.testing.assert_allclose(holding_torque(arm, stretched), 0.5 * t_s, atol=1e-12)


def test_sag_report_recovers_the_true_gravity_scale():
    truth = Arm(SOFT.replace(gravity_scale=0.4))
    eps = [sweep_episode(truth, i) for i in range(3)]
    rep = sag_report(Arm(SOFT), eps)
    assert rep.n_episodes == 3 and rep.n_frames == 450 and rep.gravity_scale == 1.0
    assert "gripper" not in rep.joints
    assert rep.joints["shoulder_pan"].implied_gravity_scale is None  # no load signal
    elbow = rep.joints["elbow_flex"]
    assert elbow.partial_r2 > 0.5
    assert elbow.rms_corrected_deg < elbow.rms_deg  # only the pose-varying part comes off
    assert elbow.implied_gravity_scale == pytest.approx(0.4, abs=0.05)
    assert rep.implied_gravity_scale == pytest.approx(0.4, abs=0.05)
    # residual mean moves with the holding torque, bin by bin
    means = [b[1] for b in elbow.bins]
    assert len(means) == 5 and np.ptp(means) > 0.5
    # replaying with the true scale leaves nothing for gravity to explain
    matched = sag_report(Arm(SOFT.replace(gravity_scale=0.4)), eps)
    assert matched.joints["elbow_flex"].partial_r2 < 0.05
    assert matched.joints["elbow_flex"].rms_deg < 0.05
    d = rep.to_dict()
    assert d["worst_joint"] in JOINTS and d["joints"]["elbow_flex"]["bins"]


def test_sag_report_uses_the_servo_models_linearized_stiffness():
    # The STS3215 model is as stiff as kp ~9.4; the implied gravity scale must
    # be read with that, not with the stock position actuator's 998 (which
    # would put the implied scale about 100x off).
    servo = SimParams.sts3215()
    assert servo.linearized().kp < 20
    truth = Arm(servo.replace(gravity_scale=0.4))
    eps = [sweep_episode(truth, i) for i in range(3)]
    rep = sag_report(Arm(servo), eps)
    elbow = rep.joints["elbow_flex"]
    assert elbow.partial_r2 > 0.5
    assert elbow.implied_gravity_scale == pytest.approx(0.4, abs=0.1)
    assert rep.implied_gravity_scale == pytest.approx(0.4, abs=0.1)
    # same answer as the explicit PD equivalent, which is what the regression assumes
    lin = sag_report(Arm(servo.linearized()), eps)
    assert lin.joints["elbow_flex"].implied_gravity_scale == pytest.approx(elbow.implied_gravity_scale, abs=0.05)


def test_sag_report_needs_episodes():
    with pytest.raises(ValueError):
        sag_report(Arm(SOFT), [])


def test_deadband_shows_up_as_an_offset_not_a_slope():
    band = float(np.radians(1.0))
    eps = [sweep_episode(Arm(SOFT.replace(deadband=band)), i) for i in range(3)]
    rep = sag_report(Arm(SOFT), eps)
    # Gravity keeps the loaded elbow pressed to one edge of the band: a constant
    # offset of about the band whether still or moving, and no torque slope.
    elbow = rep.joints["elbow_flex"]
    assert set(elbow.by_motion) == {"still", "up", "down"}
    for v in elbow.by_motion.values():
        assert v == pytest.approx(-np.degrees(band), abs=0.15)
    assert elbow.partial_r2 < 0.1
    # The lightly loaded wrist trails its motion instead, so the sign flips.
    wrist = rep.joints["wrist_flex"].by_motion
    assert wrist["up"] > 0.3 and wrist["down"] < -0.3
