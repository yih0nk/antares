"""Tests for antares.dr on the toy first-order servo from test_calibrate."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from test_calibrate import TRUE, fake_simulate, make_episodes

from antares.calibrate import DIMS, SPACE, SimParams
from antares.dr import (
    DR_SUFFIX,
    DRRanges,
    FitProgress,
    ParamRange,
    apply,
    bootstrap_fits,
    randomize,
    sample,
    uniform_ranges,
)
from antares.lock import LOCK_DIR

# A plausible lock: close to TRUE, not exactly on it.
START = SimParams(kp=380.0, kv=4.2, damping=1.8, armature=0.11, frictionloss=0.06, latency_frames=2)


@pytest.fixture(scope="module")
def episodes():
    return make_episodes(10, T=60)


@pytest.fixture(scope="module")
def fits(episodes):
    return bootstrap_fits(
        episodes, START, n_fits=4, n_train=3, max_evaluations=12, seed=1, simulate=fake_simulate
    )


# --------------------------------------------------------------------------
# bootstrap_fits
# --------------------------------------------------------------------------


def test_bootstrap_fits_subsets_and_warm_start(episodes, fits):
    assert len(fits) == 4
    subsets = [tuple(f.train_episodes) for f in fits]
    assert len(set(subsets)) > 1  # different random subsets
    for f in fits:
        assert len(f.train_episodes) == 3
        assert len(set(f.train_episodes)) == 3  # drawn without replacement
        assert f.eval_episodes == []  # no held-out scoring: cheap
        assert f.baseline == START  # warm-started from the lock
        assert 1 <= f.n_evaluations <= 12
        assert f.train_score <= f.baseline_train_score  # never worse than the start


def test_bootstrap_fits_accepts_lock_like_and_dict(episodes):
    class LockLike:
        params = START

    kw = dict(n_fits=1, n_train=2, max_evaluations=2, seed=3, simulate=fake_simulate)
    a = bootstrap_fits(episodes, LockLike(), **kw)
    b = bootstrap_fits(episodes, START.to_dict(), **kw)
    c = bootstrap_fits(episodes, START, **kw)
    assert a[0].params == b[0].params == c[0].params
    with pytest.raises(TypeError):
        bootstrap_fits(episodes, "not params", **kw)


def test_bootstrap_fits_deterministic_and_seed_sensitive(episodes):
    kw = dict(n_fits=2, n_train=3, max_evaluations=4, simulate=fake_simulate)
    a = bootstrap_fits(episodes, START, seed=5, **kw)
    b = bootstrap_fits(episodes, START, seed=5, **kw)
    c = bootstrap_fits(episodes, START, seed=6, **kw)
    assert [f.params for f in a] == [f.params for f in b]
    assert [f.train_episodes for f in a] == [f.train_episodes for f in b]
    assert [f.train_episodes for f in a] != [f.train_episodes for f in c]


def test_bootstrap_fits_progress_and_stop(episodes):
    seen: list[FitProgress] = []
    finished: list[tuple] = []  # (fit index, progress events seen so far, result)
    out = bootstrap_fits(
        episodes, START, n_fits=2, n_train=2, max_evaluations=3, simulate=fake_simulate,
        on_progress=seen.append, on_fit=lambda i, r: finished.append((i, len(seen), r)),
    )
    assert {p.fit for p in seen} == {0, 1}
    assert all(p.n_fits == 2 for p in seen)
    d = seen[-1].to_dict()
    assert d["fit"] == 1 and d["n_fits"] == 2 and "best_params" in d and "evaluation" in d
    # on_fit fires once per fit, after that fit's progress and with its result
    assert [f[0] for f in finished] == [0, 1]
    assert finished[0][1] == len([p for p in seen if p.fit == 0])
    assert [f[2] for f in finished] == out

    calls = {"n": 0}

    def stop_after_first_fit() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    partial = bootstrap_fits(
        episodes, START, n_fits=5, n_train=2, max_evaluations=3, simulate=fake_simulate,
        should_stop=stop_after_first_fit,
    )
    assert 1 <= len(partial) < 5


def test_bootstrap_fits_clamps_n_train_and_validates(episodes):
    f = bootstrap_fits(episodes, START, n_fits=1, n_train=99, max_evaluations=1, simulate=fake_simulate)
    assert len(f[0].train_episodes) == len(episodes)
    with pytest.raises(ValueError):
        bootstrap_fits([], START, simulate=fake_simulate)
    with pytest.raises(ValueError):
        bootstrap_fits(episodes, START, n_fits=0, simulate=fake_simulate)


# --------------------------------------------------------------------------
# ParamRange / DRRanges
# --------------------------------------------------------------------------


def test_param_range_validation():
    r = ParamRange(nominal=4.0, lo=2.7, hi=5.2, scale="int")
    assert (r.nominal, r.lo, r.hi) == (4, 3, 5) and isinstance(r.lo, int)
    with pytest.raises(ValueError):
        ParamRange(nominal=1.0, lo=2.0, hi=3.0, scale="log")
    with pytest.raises(ValueError):
        ParamRange(nominal=1.0, lo=0.0, hi=3.0, scale="log")
    with pytest.raises(ValueError):
        ParamRange(nominal=1.0, lo=0.5, hi=3.0, scale="linear")


def test_ranges_from_fits(fits):
    dr = DRRanges.from_fits(fits, START, dataset="org/arm", lock="arm_a", seed=1)
    assert dr.name == "arm_a" and dr.lock == "arm_a" and dr.dataset == "org/arm"
    assert dr.n_fits == 4 and dr.n_train == 3 and dr.seed == 1
    assert dr.nominal == START
    assert list(dr.ranges) == list(DIMS)
    assert len(dr.fit_train_scores) == 4 and len(dr.fit_episodes) == 4
    for dim in DIMS:
        r = dr[dim]
        assert r.scale == SPACE[dim][2]
        assert r.nominal == getattr(START, dim)
        assert r.lo <= r.nominal <= r.hi
        assert r.samples == tuple(getattr(f.params, dim) for f in fits)
        if r.scale == "int":
            assert isinstance(r.lo, int) and r.lo == min(min(r.samples), r.nominal)
            assert r.hi == max(max(r.samples), r.nominal)
        else:
            # percentiles stay inside the sample span, widened only to reach the nominal
            assert min(min(r.samples), r.nominal) <= r.lo * (1 + 1e-12)
            assert r.hi <= max(max(r.samples), r.nominal) * (1 + 1e-12)
    # the toy fits move kp towards TRUE, so the range should be wide enough to hold it
    assert dr["kp"].lo <= max(dr["kp"].samples) and dr["kp"].hi >= min(dr["kp"].samples)


def test_ranges_percentiles_in_log_space(fits):
    full = DRRanges.from_fits(fits, START, dataset="org/arm", percentiles=(0, 100))
    for dim in DIMS:
        r = full[dim]
        assert r.lo == pytest.approx(min(min(r.samples), r.nominal), rel=1e-9)
        assert r.hi == pytest.approx(max(max(r.samples), r.nominal), rel=1e-9)
    narrow = DRRanges.from_fits(fits, START, dataset="org/arm", percentiles=(25, 75))
    for dim in DIMS:
        if SPACE[dim][2] == "log":
            assert narrow[dim].lo >= full[dim].lo and narrow[dim].hi <= full[dim].hi
        else:
            assert (narrow[dim].lo, narrow[dim].hi) == (full[dim].lo, full[dim].hi)  # int: min/max


def test_ranges_default_name_is_slug_and_from_fits_needs_fits(fits):
    assert DRRanges.from_fits(fits, START, dataset="lerobot/svla_so101_pickplace").name == "svla_so101_pickplace"
    with pytest.raises(ValueError):
        DRRanges.from_fits([], START, dataset="x")


def test_ranges_round_trip_and_save(fits, tmp_path, monkeypatch):
    dr = DRRanges.from_fits(fits, START, dataset="org/arm", lock="arm_a")
    d = json.loads(json.dumps(dr.to_dict()))
    back = DRRanges.from_dict(d)
    assert back == dr
    assert back.to_dict() == dr.to_dict()
    assert d["ranges"]["latency_frames"]["lo"] == int(d["ranges"]["latency_frames"]["lo"])
    assert dr.created_at.endswith("Z")

    path = dr.save(tmp_path / "x.dr.json")
    assert DRRanges.load(path) == dr

    monkeypatch.chdir(tmp_path)
    assert dr.save() == LOCK_DIR / f"arm_a{DR_SUFFIX}"
    assert (tmp_path / "locks" / "arm_a.dr.json").is_file()


def test_ranges_nan_scores_serialise_as_null(fits):
    dr = DRRanges.from_fits(fits, START, dataset="org/arm")
    dr = DRRanges(**{**dr.__dict__, "fit_train_scores": (float("nan"), 1.0)})
    d = dr.to_dict()
    assert d["fit_train_scores"] == [None, 1.0]
    back = DRRanges.from_dict(json.loads(json.dumps(d)))
    assert math.isnan(back.fit_train_scores[0]) and back.fit_train_scores[1] == 1.0


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def test_sample_within_ranges_and_reproducible(fits):
    dr = DRRanges.from_fits(fits, START, dataset="org/arm")
    params = sample(dr, 7, 50)
    assert len(params) == 50 and all(isinstance(p, SimParams) for p in params)
    for p in params:
        for dim in DIMS:
            r = dr[dim]
            v = getattr(p, dim)
            assert r.lo <= v <= r.hi
            if r.scale == "int":
                assert isinstance(v, int)
            else:
                assert isinstance(v, float)
    assert sample(dr, 7, 50) == params
    assert sample(dr, 8, 50) != params
    assert sample(dr, np.random.default_rng(7), 3) == params[:3]
    assert sample(dr, None, 0) == []


def test_sample_is_log_uniform():
    r = ParamRange(nominal=10.0, lo=1.0, hi=100.0, scale="log")
    dr = DRRanges(name="t", nominal=SimParams(kp=10.0), ranges={**uniform_ranges(SimParams()).ranges, "kp": r})
    kp = np.array([p.kp for p in sample(dr, 0, 4000)])
    assert np.median(np.log10(kp)) == pytest.approx(1.0, abs=0.05)  # geometric centre, not 50.5
    assert np.mean(kp < 10) == pytest.approx(0.5, abs=0.03)


def test_sample_keeps_the_nominal_unsearched_fields():
    # gravity_scale, deadband and forcerange are not DR dimensions: a draw
    # takes them from the lock (the nominal), not from SimParams' defaults.
    nominal = SimParams(kp=100.0, forcerange=2.5, gravity_scale=1.4, deadband=(0.0, 0.0, 0.01, 0.0, 0.0, 0.0))
    dr = uniform_ranges(nominal, 0.3)
    draws = sample(dr, 0, 5)
    assert len({p.kp for p in draws}) == 5
    for p in draws:
        assert 70.0 <= p.kp <= 130.0
        assert p.forcerange == 2.5 and p.gravity_scale == 1.4
        assert p.per_joint("deadband").tolist() == [0.0, 0.0, 0.01, 0.0, 0.0, 0.0]
    assert sample(uniform_ranges(SimParams(), 0.3), 0, 1)[0].to_dict().keys() == SimParams().to_dict().keys()


def test_uniform_ranges_pm30():
    nominal = SimParams(kp=100.0, kv=2.0, damping=0.5, armature=0.02, frictionloss=0.05, latency_frames=4)
    dr = uniform_ranges(nominal, 0.3)
    assert dr.name == "pm30" and dr.nominal == nominal and dr.n_fits == 0
    assert dr["kp"].lo == pytest.approx(70.0) and dr["kp"].hi == pytest.approx(130.0)
    assert dr["frictionloss"].hi == pytest.approx(0.065)
    assert (dr["latency_frames"].lo, dr["latency_frames"].hi) == (3, 5)
    assert all(r.samples == () for r in dr.ranges.values())
    zero = uniform_ranges(SimParams(latency_frames=0), 0.3)
    assert (zero["latency_frames"].lo, zero["latency_frames"].hi) == (0, 0)
    assert sample(zero, 0, 3)[0].latency_frames == 0
    with pytest.raises(ValueError):
        uniform_ranges(nominal, 1.0)


def test_apply_and_randomize(fits):
    class FakeArm:
        def __init__(self):
            self.params = None

        def apply(self, params):
            self.params = params

    dr = DRRanges.from_fits(fits, START, dataset="org/arm")
    arm = FakeArm()
    p = SimParams(kp=123.0)
    assert apply(arm, p) is p and arm.params is p
    q = randomize(arm, dr, 3)
    assert arm.params == q == sample(dr, 3, 1)[0]
