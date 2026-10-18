import numpy as np
import pytest

from antares.data import Episode
from antares.noise import NoiseModel, estimate, quantum, still_mask
from antares.sim import JOINTS, UnitMap, load_model

STEP = 200.0 / 4096  # a 12-bit encoder over the -100..100 range, dataset units


def parked_episode(index, rng, sd_units=0.4, rho=0.6, T=400):
    """Commands hold for 100 frames, then jump; readings are AR(1) noise, quantized."""
    levels = rng.uniform(-50, 50, size=(T // 100, 6))
    cmd = np.repeat(levels, 100, axis=0)
    noise = np.zeros((T, 6))
    for t in range(1, T):
        noise[t] = rho * noise[t - 1] + rng.normal(size=6) * sd_units * np.sqrt(1 - rho**2)
    state = np.round((cmd + noise) / STEP) * STEP
    return Episode(index, 30.0, JOINTS, state.astype(np.float32).astype(np.float64), cmd)


def test_still_mask_needs_a_full_quiet_window():
    cmd = np.r_[np.zeros(20), np.ones(20)]
    m = still_mask(cmd, frames=5, tol=0.3)
    assert not m[:4].any() and m[4:20].all()
    assert not m[20:24].any() and m[24:].all()


def test_quantum_ignores_float32_jitter():
    v = (np.arange(50) * STEP).astype(np.float32).astype(np.float64)
    assert quantum(v) == pytest.approx(STEP, rel=1e-3)


def test_estimate_recovers_scale_colour_and_step():
    rng = np.random.default_rng(0)
    um = UnitMap.from_model(load_model())
    model = estimate([parked_episode(i, rng) for i in range(6)], name="toy", unit_map=um)
    sd_rad = 0.4 * np.abs(um.scale)
    np.testing.assert_allclose(model.scale, sd_rad, rtol=0.15)
    np.testing.assert_allclose(model.acf1, 0.6, atol=0.1)
    np.testing.assert_allclose(model.quantum, STEP * np.abs(um.scale), rtol=1e-3)
    assert np.all(model.n_frames > 1000)


def test_sample_apply_and_roundtrip(tmp_path):
    m = NoiseModel(scale=[0.01] * 6, acf1=[0.5] * 6, quantum=[0.002] * 5 + [np.nan], n_frames=[100] * 6, name="toy")
    x = m.sample(20000, np.random.default_rng(1))
    assert x.std(axis=0) == pytest.approx([0.01] * 6, rel=0.05)
    lag1 = np.mean(x[1:] * x[:-1], axis=0) / np.mean(x * x, axis=0)
    assert lag1 == pytest.approx([0.5] * 6, abs=0.03)
    q = m.apply(np.zeros((10, 6)), np.random.default_rng(2))
    np.testing.assert_allclose(q[:, :5] / 0.002, np.round(q[:, :5] / 0.002), atol=1e-9)
    back = NoiseModel.load(m.save(tmp_path / "toy.noise.json"))
    assert back.to_dict() == m.to_dict()
    assert m.default_path(tmp_path).name == "toy.noise.json"
