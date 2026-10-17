"""Tests for antares.specificity with a toy servo instead of MuJoCo."""

from __future__ import annotations

import math

import pytest

from antares.specificity import summarize_cross

A, B = "lab/arm_a", "lab/arm_b"


def matrix(scores):
    return {"rows": [A, B], "cols": ["lock_a", "lock_b", "default"], "scores": scores}


LOCK_DATASETS = {"lock_a": A, "lock_b": B}


def test_summarize_cross_diagonal_wins():
    s = summarize_cross(matrix([[1.0, 2.5, 6.0], [3.0, 0.8, 2.7]]), LOCK_DATASETS)
    assert s["all_specific"] is True
    a, b = s["rows"]
    assert (a["own_lock"], a["best_other_lock"]) == ("lock_a", "lock_b")
    assert a["margin"] == pytest.approx(2.5)
    assert a["default_score"] == 6.0
    assert b["margin"] == pytest.approx(3.0 / 0.8)


def test_summarize_cross_foreign_lock_better_is_not_specific():
    s = summarize_cross(matrix([[1.0, 0.9, 6.0], [3.0, 0.8, 2.7]]), LOCK_DATASETS)
    assert s["rows"][0]["specific"] is False
    assert s["rows"][1]["specific"] is True
    assert s["all_specific"] is False


def test_summarize_cross_picks_best_of_several_own_locks():
    m = {"rows": [A], "cols": ["a1", "a2", "lock_b", "default"], "scores": [[1.5, 1.0, 2.0, 5.0]]}
    s = summarize_cross(m, {"a1": A, "a2": A, "lock_b": B})
    assert s["rows"][0]["own_lock"] == "a2"
    assert s["rows"][0]["margin"] == pytest.approx(2.0)


def test_summarize_cross_undecided_without_a_foreign_lock():
    m = {"rows": [A], "cols": ["lock_a", "default"], "scores": [[1.0, 5.0]]}
    s = summarize_cross(m, {"lock_a": A})
    assert s["rows"][0]["specific"] is None
    assert s["rows"][0]["margin"] is None
    assert s["all_specific"] is False


def test_summarize_cross_zero_own_score():
    s = summarize_cross(matrix([[0.0, 1.0, 2.0], [1.0, 0.5, 2.0]]), LOCK_DATASETS)
    assert math.isinf(s["rows"][0]["margin"])
