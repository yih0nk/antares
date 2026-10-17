import json

import numpy as np
import pytest

from antares.gap import GapReport, aggregate, tracking_error
from antares.sim import JOINTS


def test_identical_trajectories_have_zero_error():
    q = np.random.default_rng(0).normal(size=(50, 6))
    rep = tracking_error(q, q)
    assert rep.n_frames == 50
    assert rep.overall_deg == 0.0
    assert set(rep.per_joint_deg) == set(JOINTS)
    assert all(v == 0.0 for v in rep.per_joint_deg.values())
    assert rep.scored_joints == JOINTS[:5]
    assert "gripper" not in rep.scored_joints


def test_known_offset_in_degrees():
    real = np.zeros((40, 6))
    sim = real.copy()
    sim[:, 0] += np.radians(2.0)  # 2 deg on shoulder_pan
    sim[:, 5] += np.radians(90.0)  # huge gripper error, must not count
    rep = tracking_error(real, sim)
    assert rep.per_joint_deg["shoulder_pan"] == pytest.approx(2.0)
    assert rep.per_joint_deg["gripper"] == pytest.approx(90.0)
    assert rep.per_joint_deg["elbow_flex"] == 0.0
    # RMS over the 5 scored joints: sqrt(4/5)
    assert rep.overall_deg == pytest.approx(np.sqrt(4.0 / 5.0))


def test_exclude_controls_headline():
    real = np.zeros((10, 6))
    sim = real.copy()
    sim[:, 5] += np.radians(10.0)
    assert tracking_error(real, sim).overall_deg == 0.0
    incl = tracking_error(real, sim, exclude=())
    assert incl.scored_joints == JOINTS
    assert incl.overall_deg == pytest.approx(np.sqrt(100.0 / 6.0))
    only_two = tracking_error(real, sim, exclude=JOINTS[:4])
    assert only_two.scored_joints == ("wrist_roll", "gripper")
    assert only_two.overall_deg == pytest.approx(np.sqrt(100.0 / 2.0))


def test_rmse_not_mean_abs():
    real = np.zeros((2, 6))
    sim = real.copy()
    sim[0, 1] = np.radians(3.0)
    sim[1, 1] = np.radians(-4.0)
    rep = tracking_error(real, sim)
    assert rep.per_joint_deg["shoulder_lift"] == pytest.approx(np.sqrt((9 + 16) / 2))


def test_aggregate_matches_concatenation():
    rng = np.random.default_rng(1)
    reals = [rng.normal(size=(n, 6)) for n in (30, 70, 5)]
    sims = [r + rng.normal(scale=0.05, size=r.shape) for r in reals]
    agg = aggregate([tracking_error(r, s) for r, s in zip(reals, sims)])
    cat = tracking_error(np.concatenate(reals), np.concatenate(sims))
    assert agg.n_frames == cat.n_frames == 105
    assert agg.overall_deg == pytest.approx(cat.overall_deg)
    for name in JOINTS:
        assert agg.per_joint_deg[name] == pytest.approx(cat.per_joint_deg[name])
    # list-of-episodes input takes the same path
    via_lists = tracking_error(reals, sims)
    assert via_lists.overall_deg == pytest.approx(cat.overall_deg)
    assert via_lists.n_frames == 105


def test_aggregate_is_frame_weighted():
    a = GapReport({"j": 1.0}, 1.0, ("j",), n_frames=1)
    b = GapReport({"j": 3.0}, 3.0, ("j",), n_frames=3)
    agg = aggregate([a, b])
    assert agg.overall_deg == pytest.approx(np.sqrt((1 * 1 + 3 * 9) / 4))
    assert agg.per_joint_deg["j"] == pytest.approx(agg.overall_deg)
    assert agg.n_frames == 4


def test_aggregate_rejects_mismatched_reports():
    a = GapReport({"j": 1.0}, 1.0, ("j",), 1)
    b = GapReport({"k": 1.0}, 1.0, ("k",), 1)
    with pytest.raises(ValueError):
        aggregate([a, b])
    with pytest.raises(ValueError):
        aggregate([])


def test_input_validation():
    with pytest.raises(ValueError):
        tracking_error(np.zeros((5, 6)), np.zeros((4, 6)))
    with pytest.raises(ValueError):
        tracking_error(np.zeros((5, 3)), np.zeros((5, 3)))  # 3 cols, 6 names
    with pytest.raises(ValueError):
        tracking_error(np.zeros((0, 6)), np.zeros((0, 6)))
    with pytest.raises(ValueError):
        tracking_error([np.zeros((5, 6))], [])
    with pytest.raises(ValueError):
        tracking_error(np.zeros((5, 6)), np.zeros((5, 6)), exclude=JOINTS)


def test_to_dict_is_json_serializable():
    rep = tracking_error(np.zeros((3, 6)), np.full((3, 6), 0.01))
    d = rep.to_dict()
    s = json.dumps(d)
    back = GapReport.from_dict(json.loads(s))
    assert back.overall_deg == pytest.approx(rep.overall_deg)
    assert back.scored_joints == rep.scored_joints
    assert set(d) == {"per_joint_deg", "overall_deg", "scored_joints", "n_frames"}
