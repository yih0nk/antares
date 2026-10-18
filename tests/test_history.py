"""Tests for the cross-session check history (``antares.history``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from antares import __version__, history
from antares.calibrate import CalibrationResult, SimParams
from antares.check import check
from antares.data import Episode
from antares.history import CheckRecord, append, history_path, load, summary_all, summary_line, trend
from antares.lock import RobotLock

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _exact_simulate(params, ep):
    return np.radians(ep.state), np.radians(ep.action)


def _episodes(n=6, offset_deg=1.2):
    eps = []
    for i in range(n):
        a = np.linspace(0, 10, 20)[:, None].repeat(6, axis=1)
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=a + offset_deg, action=a))
    return eps


def _lock(name="unit-a"):
    result = CalibrationResult(
        params=SimParams(), baseline=SimParams(), train_score=0.9, eval_score=1.0,
        baseline_train_score=4.0, baseline_eval_score=4.0,
        per_joint_before={j: 4.0 for j in JOINTS}, per_joint_after={j: 1.0 for j in JOINTS},
        history=[], n_evaluations=0, elapsed_s=0.0,
    )
    return RobotLock.create("fake/robot", "so101_follower", _episodes(1), result, name=name)


def _result(offset_deg=1.2, n=6):
    return check(_episodes(n, offset_deg), _lock(), simulate=_exact_simulate)


def _record(at, score, verdict=None, lock="unit-a"):
    """A synthetic record; ``verdict`` defaults from the score against limit 1.5."""
    if verdict is None:
        verdict = "insufficient_data" if score is None else ("ok" if score <= 1.5 else "drifted")
    return CheckRecord(
        at=at, lock=lock, dataset="fake/robot", verdict=verdict, passed=verdict == "ok",
        score=score, reference=1.0, limit=1.5, n_episodes=6 if score is not None else 2,
        episodes=[0, 1, 2], worst_joint="elbow_flex" if score is not None else None,
        per_joint={"elbow_flex": score}, antares_version="0.0.1",
    )


# -- records -------------------------------------------------------------------


def test_record_from_check_result_follows_the_contract():
    rec = CheckRecord.from_result(_result(1.2), at="2026-10-17T21:44:12Z")
    d = rec.to_dict()
    assert set(d) == {"at", "lock", "dataset", "verdict", "passed", "score", "reference", "limit",
                      "n_episodes", "episodes", "worst_joint", "per_joint", "antares_version"}
    assert d["at"] == "2026-10-17T21:44:12Z" and d["lock"] == "unit-a" and d["dataset"] == "fake/robot"
    assert d["verdict"] == "ok" and d["passed"] is True
    assert d["score"] == pytest.approx(1.2) and d["reference"] == 1.0 and d["limit"] == 1.5
    assert d["n_episodes"] == 6 and d["episodes"] == [0, 1, 2, 3, 4, 5]
    assert d["worst_joint"] in JOINTS
    assert set(d["per_joint"]) == set(JOINTS)
    assert all(isinstance(v, float) for v in d["per_joint"].values())  # scores only, not JointCheck dicts
    assert d["antares_version"] == __version__
    json.dumps(d)


def test_record_from_insufficient_result_has_null_score():
    rec = CheckRecord.from_result(_result(0.5, n=2))
    assert rec.verdict == "insufficient_data" and rec.passed is False
    assert rec.score is None and rec.per_joint == {} and rec.worst_joint is None
    assert rec.n_episodes == 2 and rec.episodes == [0, 1]
    history.parse_time(rec.at)  # defaulted to a parseable UTC timestamp
    assert rec.at.endswith("Z")


def test_record_from_dict_rejects_malformed():
    with pytest.raises(ValueError):
        CheckRecord.from_dict({"lock": "a"})  # no at / verdict
    with pytest.raises(ValueError):
        CheckRecord.from_dict({"at": "not a date", "lock": "a", "verdict": "ok"})
    with pytest.raises(ValueError):
        CheckRecord.from_dict(["not", "an", "object"])


# -- file ----------------------------------------------------------------------


def test_history_path_and_name_validation(tmp_path):
    assert history_path("unit-a", tmp_path) == tmp_path / "unit-a.history.jsonl"
    for bad in ("", "../x", "a/b", ".hidden"):
        with pytest.raises(ValueError):
            history_path(bad, tmp_path)


def test_append_then_load_round_trips_oldest_first(tmp_path):
    p1 = append(_result(1.2), tmp_path, at="2026-10-02T00:00:00Z")
    p2 = append(_result(3.0), tmp_path, at="2026-10-01T00:00:00Z")  # written later, dated earlier
    assert p1 == p2 == tmp_path / "unit-a.history.jsonl"
    lines = p1.read_text().splitlines()
    assert len(lines) == 2 and all(json.loads(l) for l in lines)
    recs = load("unit-a", tmp_path)
    assert [r.at for r in recs] == ["2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z"]
    assert [r.verdict for r in recs] == ["drifted", "ok"]
    assert recs[1].to_dict() == json.loads(lines[0])


def test_append_name_override_and_missing_dir(tmp_path):
    p = append(_result(1.2), tmp_path / "nested", name="renamed")
    assert p == tmp_path / "nested" / "renamed.history.jsonl" and p.is_file()
    assert load("renamed", tmp_path / "nested")[0].lock == "unit-a"  # the record keeps the lock's own name


def test_load_missing_file_is_empty(tmp_path):
    assert load("ghost", tmp_path) == []
    assert trend([]) == {"n": 0, "first": None, "last": None, "slope_deg_per_day": None,
                         "r2": None, "latest_verdict": None, "n_drifted": 0}


def test_load_skips_corrupt_lines_with_a_warning(tmp_path):
    p = history_path("unit-a", tmp_path)
    good = _record("2026-10-01T00:00:00Z", 1.0).to_dict()
    p.write_text(
        json.dumps(good) + "\n"
        + "{not json\n"
        + json.dumps({"at": "2026-10-02T00:00:00Z"}) + "\n"  # missing keys
        + "\n"
        + json.dumps(dict(good, at="2026-10-03T00:00:00Z", score=None, verdict="insufficient_data")) + "\n"
    )
    with pytest.warns(UserWarning, match="corrupt history line") as w:
        recs = load("unit-a", tmp_path)
    assert len(w) == 2
    assert [r.at for r in recs] == ["2026-10-01T00:00:00Z", "2026-10-03T00:00:00Z"]
    assert recs[1].score is None


# -- trend ---------------------------------------------------------------------


def test_trend_fits_slope_per_day_and_r2():
    recs = [
        _record("2026-10-01T00:00:00Z", 1.0),
        _record("2026-10-03T00:00:00Z", 1.2),
        _record("2026-10-05T12:00:00Z", 1.45),
        _record("2026-10-11T00:00:00Z", 2.0),
    ]
    t = trend(recs)
    assert t["n"] == 4 and t["n_drifted"] == 1 and t["latest_verdict"] == "drifted"
    assert t["first"] == "2026-10-01T00:00:00Z" and t["last"] == "2026-10-11T00:00:00Z"
    # exactly 0.1 deg/day
    assert t["slope_deg_per_day"] == pytest.approx(0.1)
    assert t["r2"] == pytest.approx(1.0)


def test_trend_skips_null_scores_but_counts_them():
    recs = [
        _record("2026-10-01T00:00:00Z", 1.0),
        _record("2026-10-02T00:00:00Z", None),
        _record("2026-10-03T00:00:00Z", 1.4),
        _record("2026-10-04T00:00:00Z", None),
    ]
    t = trend(recs)
    assert t["n"] == 4 and t["latest_verdict"] == "insufficient_data" and t["n_drifted"] == 0
    assert t["slope_deg_per_day"] == pytest.approx(0.2)
    assert t["r2"] == pytest.approx(1.0)


def test_trend_needs_two_scored_entries():
    one = [_record("2026-10-01T00:00:00Z", 1.0)]
    t = trend(one)
    assert t["n"] == 1 and t["first"] == t["last"] == "2026-10-01T00:00:00Z"
    assert t["slope_deg_per_day"] is None and t["r2"] is None
    nulls = [_record("2026-10-01T00:00:00Z", None), _record("2026-10-02T00:00:00Z", 1.0)]
    assert trend(nulls)["slope_deg_per_day"] is None
    same_time = [_record("2026-10-01T00:00:00Z", 1.0), _record("2026-10-01T00:00:00Z", 2.0)]
    assert trend(same_time)["slope_deg_per_day"] is None


def test_trend_scatter_has_low_r2_and_flat_has_no_r2():
    scatter = [_record(f"2026-10-{d:02d}T00:00:00Z", s) for d, s in
               [(1, 1.0), (2, 1.4), (3, 1.0), (4, 1.4), (5, 1.0), (6, 1.4)]]
    t = trend(scatter)
    assert t["r2"] is not None and t["r2"] < 0.3
    flat = [_record(f"2026-10-{d:02d}T00:00:00Z", 1.0) for d in (1, 2, 3)]
    t = trend(flat)
    assert t["slope_deg_per_day"] == pytest.approx(0.0) and t["r2"] is None


# -- summaries -----------------------------------------------------------------


def test_summary_line_formats():
    assert summary_line([]) == "history: no checks recorded"
    assert summary_line([_record("2026-09-30T10:00:00Z", 1.0)]) == "history: 1 check since 2026-09-30"
    recs = [
        _record("2026-09-30T00:00:00Z", 1.0),
        _record("2026-10-10T00:00:00Z", 2.0),
        _record("2026-10-20T00:00:00Z", 1.4),
    ]
    line = summary_line(recs)
    assert line.startswith("history: 3 checks since 2026-09-30, +0.020 deg/day (r² ")
    assert line.endswith(", last drifted 2026-10-10")


def test_report_and_summary_all(tmp_path):
    (tmp_path / "unit-a.lock.json").write_text("{}")
    (tmp_path / "unit-b.lock.json").write_text("{}")
    (tmp_path / "other.json").write_text("{}")
    append(_result(1.2), tmp_path, at="2026-10-01T00:00:00Z")
    append(_result(3.0), tmp_path, at="2026-10-02T00:00:00Z")

    rep = history.report("unit-a", tmp_path)
    assert rep["lock"] == "unit-a" and len(rep["entries"]) == 2
    assert rep["entries"][0]["at"] < rep["entries"][1]["at"]
    assert rep["trend"]["n"] == 2 and rep["trend"]["n_drifted"] == 1
    limited = history.report("unit-a", tmp_path, last=1)
    assert len(limited["entries"]) == 1 and limited["entries"][0]["verdict"] == "drifted"
    assert limited["trend"]["n"] == 2  # the trend is over everything

    assert history.lock_names(tmp_path) == ["unit-a", "unit-b"]
    s = summary_all(tmp_path)
    assert [x["lock"] for x in s] == ["unit-a", "unit-b"]
    assert s[0]["trend"]["n"] == 2 and s[0]["latest"]["verdict"] == "drifted"
    assert s[1]["trend"]["n"] == 0 and s[1]["latest"] is None
    assert summary_all(tmp_path / "missing") == []
