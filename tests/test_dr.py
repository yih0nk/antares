"""Tests for antares.dr on the toy first-order servo from test_calibrate."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from test_calibrate import TRUE, fake_simulate, make_episodes

from antares.calibrate import DIMS, SPACE, SimParams
from antares.dr import DR_SUFFIX, DRRanges, FitProgress, ParamRange, bootstrap_fits
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
    bootstrap_fits(
        episodes, START, n_fits=2, n_train=2, max_evaluations=3, simulate=fake_simulate,
        on_progress=seen.append,
    )
    assert {p.fit for p in seen} == {0, 1}
    assert all(p.n_fits == 2 for p in seen)
    d = seen[-1].to_dict()
    assert d["fit"] == 1 and d["n_fits"] == 2 and "best_params" in d and "evaluation" in d

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
