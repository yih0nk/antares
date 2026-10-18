from __future__ import annotations

import numpy as np
import pytest

from antares import synth
from antares.data import read_local
from antares.sim import SimParams


def test_command_trajectory_is_smooth_and_in_range():
    a = synth.command_trajectory(300, 30.0, np.random.default_rng(0))
    assert a.shape == (300, 6)
    assert np.all(np.abs(a[:, :5]) <= 80)
    assert np.all((a[:, 5] >= 10) & (a[:, 5] <= 90))
    # at most a few units per frame: reaching motions, not noise
    assert np.max(np.abs(np.diff(a, axis=0))) < 10


def test_command_trajectory_is_seeded():
    a = synth.command_trajectory(50, 30.0, np.random.default_rng(3))
    b = synth.command_trajectory(50, 30.0, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_worn_scales_and_shifts():
    p = SimParams(kp=100.0, frictionloss=0.02, latency_frames=2)
    w = synth.worn(p, {"kp": 0.5, "frictionloss": 3, "latency_frames": 1})
    assert w.kp == pytest.approx(50.0)
    assert w.frictionloss == pytest.approx(0.06)
    assert w.latency_frames == 3
    assert w.damping == p.damping


def test_worn_rejects_unknown_parameter():
    with pytest.raises(ValueError, match="unknown parameter"):
        synth.worn(SimParams(), {"stiffness": 2})


def test_parse_changes():
    assert synth.parse_changes("kp=0.7, frictionloss=2,") == {"kp": 0.7, "frictionloss": 2.0}
    assert synth.parse_changes("") == {}
    with pytest.raises(ValueError):
        synth.parse_changes("kp")


@pytest.mark.slow
def test_synthesize_tracks_its_own_params():
    """Replaying a synthetic dataset with its true params leaves only sensor noise."""
    from antares.calibrate import default_simulate, score

    params = SimParams(kp=94.0, kv=1.6, damping=0.05, armature=0.02, frictionloss=0.025, latency_frames=4)
    ds = synth.synthesize(params, n_episodes=2, n_frames=90, seed=1, noise_units=0.0)
    assert len(ds.episodes) == 2 and ds.episodes[0].state.shape == (90, 6)
    simulate = default_simulate()
    assert score(params, ds.episodes, simulate).overall_deg < 1e-6
    assert score(SimParams(), ds.episodes, simulate).overall_deg > 1.0


def test_write_local_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    eps = tuple(
        synth.Episode(index=i, fps=30.0, joint_names=synth.JOINTS,
                      state=rng.normal(size=(5 + i, 6)), action=rng.normal(size=(5 + i, 6)))
        for i in range(3)
    )
    ds = synth.Dataset(repo_id="synthetic/x", robot_type="so101_follower", fps=30.0,
                       joint_names=synth.JOINTS, episodes=eps)
    synth.write_local(ds, tmp_path / "x")
    back = read_local(tmp_path / "x", repo_id="synthetic/x")
    assert back.joint_names == synth.JOINTS and back.fps == 30.0
    assert back.robot_type == "so101_follower"
    assert [e.index for e in back.episodes] == [0, 1, 2]
    for a, b in zip(eps, back.episodes):
        np.testing.assert_allclose(a.state, b.state)
        np.testing.assert_allclose(a.action, b.action)


def test_worn_keeps_per_joint_shape():
    p = SimParams(kp=(100.0, 200.0, 300.0, 400.0, 500.0, 600.0), latency_frames=(0, 1, 2, 3, 4, 0))
    w = synth.worn(p, {"kp": 0.5, "latency_frames": 1})
    assert w.kp == (50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
    assert w.latency_frames == (1, 2, 3, 4, 5, 1)
    # a shift down is the same on every joint, limited so none goes below zero
    assert synth.worn(p, {"latency_frames": -1}).latency_frames == (0, 1, 2, 3, 4, 0)
    assert synth.worn(p.replace(latency_frames=(1, 2, 3, 4, 5, 1)), {"latency_frames": -3}).latency_frames \
        == (0, 1, 2, 3, 4, 0)
