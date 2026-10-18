"""Synthetic datasets: right shape, and the true params replay them exactly."""

from __future__ import annotations

import numpy as np
import pytest

from antares.calibrate import default_simulate, score
from antares.sim import SimParams
from antares.synthetic import commands, make_dataset

pytestmark = pytest.mark.slow

TRUTH = SimParams(kp=300.0, kv=5.0, damping=0.2, armature=0.02, frictionloss=0.03, latency_frames=2)


def test_commands_stay_inside_the_range():
    a = commands(500, 30.0, np.random.default_rng(1))
    assert a.shape == (500, 6)
    assert np.all(np.abs(a[:, :5]) <= 60)
    assert np.all((a[:, 5] >= 10) & (a[:, 5] <= 90))


def test_true_params_track_to_noise_level():
    ds = make_dataset("synthetic/a", TRUTH, n_episodes=3, T=120, noise_units=0.0)
    assert len(ds.episodes) == 3 and ds.episodes[0].state.shape == (120, 6)
    sim = default_simulate()
    assert score(TRUTH, ds.episodes, sim).overall_deg < 1e-6
    assert score(SimParams(), ds.episodes, sim).overall_deg > 0.5


def test_noise_and_seed():
    a = make_dataset("synthetic/a", TRUTH, n_episodes=1, T=60, seed=3)
    b = make_dataset("synthetic/a", TRUTH, n_episodes=1, T=60, seed=3)
    np.testing.assert_array_equal(a.episodes[0].state, b.episodes[0].state)
    sim = default_simulate()
    assert 0.05 < score(TRUTH, a.episodes, sim).overall_deg < 1.0
