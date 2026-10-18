"""Payload mass from the residual, checked on an arm with a known extra sag."""

from __future__ import annotations

import numpy as np
import pytest

from antares.data import Dataset, Episode
from antares.payload import ENCODER_STEP_DEG, LOAD_JOINTS, episode_payload, estimate_payload, fingertip_jz
from antares.sim import JOINTS, STANDARD_GRAVITY, Arm, SimParams

G = STANDARD_GRAVITY
KP = 12.0


@pytest.fixture(scope="module")
def arm():
    return Arm(SimParams(kp=KP))


class Sagging:
    """An arm whose replay differs from the real trajectory only by the sag a
    payload of ``mass`` at the fingertip would cause while it is held, plus
    an optional motion-direction offset on both sides of every event."""

    def __init__(self, arm: Arm, mass: float, holds: dict[int, tuple[int, int]], motion_deg: float = 0.0):
        self.arm = arm
        self.mass = mass
        self.holds = holds
        self.motion = motion_deg
        self.params = arm.params
        self.model = arm.model
        self.unit_map = arm.unit_map

    def real_q(self, ep: Episode) -> np.ndarray:
        return self.arm.real_q(ep)

    def replay(self, ep: Episode) -> np.ndarray:
        real = self.real_q(ep)
        sim = real.copy()
        kp = self.params.per_joint("kp")
        jz = fingertip_jz(self.arm, real)
        g, r = self.holds[ep.index]
        # real = sim - m g Jz / kp while held  ->  sim = real + m g Jz / kp
        sim[g : r + 1] += self.mass * G * jz[g : r + 1] / kp
        # a motion term: the residual steps up by `motion` after both the pick and the release
        if self.motion:
            sim[g + 1 :] -= np.radians(self.motion)
            sim[r + 1 :] -= np.radians(self.motion)
        return sim


def slow_episode(index: int, T: int = 160) -> Episode:
    """The arm sweeping slowly through its workspace, gripper closing at 50 and opening at 110."""
    t = np.linspace(0, 1, T)
    st = np.zeros((T, 6))
    st[:, 1] = -40 + 30 * t
    st[:, 2] = 40 - 20 * t
    st[:, 3] = 10 * np.sin(2 * np.pi * t)
    st[:, 5] = 90.0
    act = st.copy()
    act[50:110, 5] = 5.0
    st[53:110, 5] = 20.0
    act[T - 10 :, 5] = 5.0
    st[T - 8 :, 5] = 2.0
    return Episode(index=index, fps=30.0, joint_names=JOINTS, state=st, action=act)


def test_fingertip_jz_matches_finite_differences(arm):
    q = np.array([0.1, -0.5, 0.8, 0.3, 0.2, 0.1])
    jz = fingertip_jz(arm, q)
    from antares.scene import fingertips

    eps = 1e-6
    for j in range(6):
        dq = np.zeros(6)
        dq[j] = eps
        num = (fingertips(arm, arm.fk(q + dq))[2] - fingertips(arm, arm.fk(q - dq))[2]) / (2 * eps)
        assert jz[j] == pytest.approx(num, abs=1e-5)
    assert abs(jz[JOINTS.index("shoulder_pan")]) < 1e-9  # a vertical axis lifts nothing
    assert fingertip_jz(arm, np.stack([q, q])).shape == (2, 6)


def test_episode_payload_recovers_a_known_mass(arm):
    ep = slow_episode(0)
    fake = Sagging(arm, 0.05, {0: (50, 110)})
    e = episode_payload(ep, fake, 50, 110)
    assert e.ok and set(e.joints) == set(LOAD_JOINTS)
    for j in LOAD_JOINTS:
        assert e.joints[j].mass_kg == pytest.approx(0.05, rel=0.15)
        assert abs(e.joints[j].motion_deg) < 0.05
        assert e.joints[j].kp == KP
    assert e.mass_kg == pytest.approx(0.05, rel=0.15)


def test_motion_term_cancels_in_the_difference(arm):
    ep = slow_episode(0)
    fake = Sagging(arm, 0.04, {0: (50, 110)}, motion_deg=0.8)
    e = episode_payload(ep, fake, 50, 110)
    for j in LOAD_JOINTS:
        assert e.joints[j].motion_deg == pytest.approx(0.8, abs=0.1)
        assert e.joints[j].mass_kg == pytest.approx(0.04, rel=0.2)


def test_short_carry_is_rejected(arm):
    ep = slow_episode(0)
    e = episode_payload(ep, Sagging(arm, 0.05, {0: (50, 60)}), 50, 60)
    assert not e.ok and "carry too short" in e.note and np.isnan(e.mass_kg)


def test_estimate_payload_verdicts(arm):
    eps = [slow_episode(i) for i in range(4)]
    ds = Dataset(repo_id="fake/payload", robot_type="so101", fps=30.0, joint_names=JOINTS, episodes=tuple(eps))
    holds = {i: (50, 110) for i in range(4)}
    seen = []
    est = estimate_payload(ds, Sagging(arm, 0.05, holds), "unit", on_progress=seen.append)
    assert len(seen) == 4 and est.n_ok == 4 and est.n_episodes == 4
    assert est.mass_kg == pytest.approx(0.05, rel=0.15) and est.verdict.startswith("consistent")
    assert est.kp == {j: KP for j in LOAD_JOINTS}
    for j in LOAD_JOINTS:
        assert est.mass_per_step_kg[j] > 0 and est.lever_m[j] > 0.03
    d = est.to_dict()
    assert d["encoder_step_deg"] == ENCODER_STEP_DEG and len(d["assumptions"]) >= 4 and len(d["episodes"]) == 4
    # the picks found from the gripper command agree with the ones handed in
    est2 = estimate_payload(ds, Sagging(arm, 0.05, holds), "unit", picks={i: [(50, 110)] for i in range(4)})
    assert est2.mass_kg == pytest.approx(est.mass_kg)

    # nothing held: the deflection is zero and the verdict says noise
    null = estimate_payload(ds, Sagging(arm, 0.0, holds), "unit")
    assert null.verdict.startswith("below the noise") and abs(null.mass_kg) < 1e-6

    # a wrong-signed sag (the arm rises when loaded) is not reported as a payload
    neg = estimate_payload(ds, Sagging(arm, -0.05, holds), "unit")
    assert "wrong sign" in neg.verdict or "disagree" in neg.verdict
