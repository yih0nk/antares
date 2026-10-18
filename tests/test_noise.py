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


def test_sample_matches_the_recursion():
    """The vectorised scan must reproduce ``out[t] = rho*out[t-1] + eps[t]``
    with the stationary start, for every colour from white to near-unit."""
    m = NoiseModel(scale=[0.01, 0.02, 0.005, 0.01, 0.03, 0.0], acf1=[0.0, 0.3, 0.6, 0.9, 0.999, np.nan],
                   quantum=[np.nan] * 6, n_frames=[1] * 6)
    for T in (1, 2, 3, 50, 257):
        x = m.sample(T, np.random.default_rng(3))
        rng = np.random.default_rng(3)
        rho = np.clip(np.nan_to_num(m.acf1), 0.0, 0.999)
        eps = rng.normal(size=(T, 6)) * m.scale * np.sqrt(1 - rho**2)
        ref = np.empty_like(eps)
        ref[0] = rng.normal(size=6) * m.scale
        for t in range(1, T):
            ref[t] = rho * ref[t - 1] + eps[t]
        np.testing.assert_allclose(x, ref, rtol=0, atol=1e-12)
    assert m.sample(0).shape == (0, 6)


def test_quantize_lands_on_the_real_reading_lattice():
    """With a unit-map offset that is not a multiple of the step, the real
    readings' lattice in radians is shifted off the multiples of ``quantum``;
    quantized values must land on it, not on the multiples."""
    um = UnitMap(scale=[0.01] * 6, offset=[0.0037] * 6)  # 0.0037 / (STEP * 0.01) is not an integer
    rng = np.random.default_rng(5)
    model = estimate([parked_episode(i, rng) for i in range(3)], unit_map=um)
    np.testing.assert_allclose(model.quantum, STEP * 0.01, rtol=1e-3)
    lattice = (model.origin - um.offset) / (um.scale * STEP)
    np.testing.assert_allclose(lattice, np.round(lattice), atol=1e-6)  # origin is a real reading
    q = rng.uniform(-0.8, 0.8, size=(40, 6))
    out = model.quantize(q)
    units = (out - um.offset) / um.scale / STEP
    np.testing.assert_allclose(units, np.round(units), atol=1e-6)  # on the dataset lattice
    assert np.all(np.abs(out - q) <= model.quantum / 2 + 1e-9)  # and the nearest point of it
    assert not np.allclose(out / model.quantum, np.round(out / model.quantum))  # not the plain multiples
    # the origin survives a round trip, and an older file without one quantizes to the multiples
    back = NoiseModel.from_dict(model.to_dict())
    np.testing.assert_allclose(back.origin, model.origin)
    d = model.to_dict()
    del d["origin_rad"]
    old = NoiseModel.from_dict(d)
    assert np.all(old.origin == 0.0)
    np.testing.assert_allclose(old.quantize(q), np.round(q / model.quantum) * model.quantum)


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
    assert m.default_path().parent.name == "locks"
