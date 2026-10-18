"""antares.lerobot_calib: LeRobot's per-motor calibration as the arm's unit map."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from antares import lerobot_calib as lc
from antares.sim import JOINTS, Arm, SimParams, UnitMap, load_model

# -- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return load_model()


def matching_calibration(model) -> lc.Calibration:
    """A calibration whose ranges span exactly the MJCF ranges, centred on
    the half turn (gripper: closed at range_min), so both maps agree."""
    jr = model.jnt_range
    out = {}
    for i, j in enumerate(JOINTS):
        span = (jr[i, 1] - jr[i, 0]) / lc.RAD_PER_TICK
        if j == "gripper":
            lo, hi = 1000, 1000 + int(round(span))
        else:
            lo, hi = int(round(lc.HALF_TURN - span / 2)), int(round(lc.HALF_TURN + span / 2))
        out[j] = lc.MotorCalibration(id=i + 1, drive_mode=0, homing_offset=-700 + 100 * i, range_min=lo, range_max=hi)
    return out


def with_joint(calib: lc.Calibration, joint: str, shift: int = 0, scale: float = 1.0) -> lc.Calibration:
    """``calib`` with one joint's range shifted by ``shift`` ticks and its span scaled."""
    m = calib[joint]
    half = m.span * scale / 2
    new = lc.MotorCalibration(m.id, m.drive_mode, m.homing_offset,
                              int(round(m.mid + shift - half)), int(round(m.mid + shift + half)))
    return {**calib, joint: new}


def write_file(path: Path, calib: lc.Calibration) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lc.calibration_to_dict(calib), indent=4))
    return path


# -- parsing ----------------------------------------------------------------


def test_parse_lerobot_schema(tmp_path):
    data = {
        "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1023, "range_min": 741, "range_max": 3352},
        "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 12, "range_min": 2010, "range_max": 3350},
    }
    p = tmp_path / "arm.json"
    p.write_text(json.dumps(data))
    calib = lc.load_calibration(p)
    assert set(calib) == {"shoulder_pan", "gripper"}
    m = calib["shoulder_pan"]
    assert (m.id, m.drive_mode, m.homing_offset, m.range_min, m.range_max) == (1, 0, -1023, 741, 3352)
    assert m.span == 2611 and m.mid == pytest.approx(2046.5)
    assert lc.calibration_to_dict(calib) == data
    assert len(lc.file_sha256(p)) == 64


@pytest.mark.parametrize(
    "bad, match",
    [
        ({}, "non-empty"),
        ([1, 2], "non-empty"),
        ({"gripper": 5}, "expected an object"),
        ({"gripper": {"id": 6, "drive_mode": 0, "range_min": 0, "range_max": 10}}, "missing homing_offset"),
        ({"gripper": {"id": 6, "drive_mode": 0, "homing_offset": 1.5, "range_min": 0, "range_max": 10}}, "integer"),
        ({"gripper": {"id": 6, "drive_mode": 2, "homing_offset": 0, "range_min": 0, "range_max": 10}}, "drive_mode"),
        ({"gripper": {"id": 6, "drive_mode": 0, "homing_offset": 0, "range_min": 10, "range_max": 10}}, "range_max"),
    ],
)
def test_parse_rejects_bad_schema(bad, match):
    with pytest.raises(ValueError, match=match):
        lc.parse_calibration(bad)


def test_load_rejects_bad_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        lc.load_calibration(p)


def test_missing_motor_is_named(model):
    calib = matching_calibration(model)
    del calib["wrist_roll"]
    with pytest.raises(ValueError, match="no motor named wrist_roll"):
        lc.unit_map_from_calibration(calib, model)
    with pytest.raises(ValueError, match="wrist_roll"):
        lc.command_map_prior(calib, model)


# -- discovery --------------------------------------------------------------


def test_calibration_root_follows_lerobot_env(monkeypatch, tmp_path):
    for v in ("HF_LEROBOT_CALIBRATION", "HF_LEROBOT_HOME", "HF_HOME"):
        monkeypatch.delenv(v, raising=False)
    assert lc.calibration_root() == Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert lc.calibration_root() == tmp_path / "hf" / "lerobot" / "calibration"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lr"))
    assert lc.calibration_root() == tmp_path / "lr" / "calibration"
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "cal"))
    assert lc.calibration_root() == tmp_path / "cal"
    assert lc.calibration_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_find_calibration_tries_both_lerobot_layouts(monkeypatch, tmp_path, model):
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    calib = matching_calibration(model)
    assert lc.find_calibration("my_arm") is None
    # LeRobot 0.4.x layout
    old = write_file(tmp_path / "robots" / "so101_follower" / "my_arm.json", calib)
    assert lc.find_calibration("my_arm") == old
    assert lc.find_calibration("my_arm", "so101_follower") == old
    # LeRobot >= 0.5 writes under the merged class name; the type still says so101_follower
    new = write_file(tmp_path / "robots" / "so_follower" / "other_arm.json", calib)
    assert lc.find_calibration("other_arm", "so101_follower") == new
    assert lc.find_calibration("other_arm", "so100_follower") == new
    assert lc.find_calibration("my_arm", "so_follower") == old
    assert lc.find_calibration("my_arm", "koch_follower") is None
    assert [d.name for d in lc.calibration_dirs("so101_follower")] == ["so101_follower", "so_follower"]
    with pytest.raises(ValueError, match="bare name"):
        lc.find_calibration("../etc/passwd")


# -- unit map ---------------------------------------------------------------


def test_matching_calibration_reproduces_the_model_map(model):
    um = lc.unit_map_from_calibration(matching_calibration(model), model)
    mm = UnitMap.from_model(model)
    # ranges are integers in ticks, so agreement is to half a tick (0.044 deg)
    cmp = lc.compare_unit_maps(um, mm)
    assert all(c["max_deg"] < 0.05 for c in cmp.values())
    assert set(cmp["shoulder_pan"]["at"]) == {"-100", "0", "100"} and set(cmp["gripper"]["at"]) == {"0", "50", "100"}
    np.testing.assert_allclose(um.lo, mm.lo, atol=1e-3)
    np.testing.assert_allclose(um.hi, mm.hi, atol=1e-3)


def test_formula_matches_lerobot_normalization(model):
    """rad(n) = mid_mjcf + k * (present(n) - 2047) with LeRobot's inverse normalization."""
    calib = with_joint(matching_calibration(model), "elbow_flex", shift=150, scale=0.8)
    um = lc.unit_map_from_calibration(calib, model)
    m = calib["elbow_flex"]
    i = JOINTS.index("elbow_flex")
    mid_mjcf = float(np.mean(model.jnt_range[i]))
    for n in (-100.0, -37.5, 0.0, 62.0, 100.0):
        present = ((n + 100.0) / 200.0) * (m.range_max - m.range_min) + m.range_min  # motors_bus._unnormalize
        expected = mid_mjcf + lc.RAD_PER_TICK * (present - lc.HALF_TURN)
        assert um.scale[i] * n + um.offset[i] == pytest.approx(expected, abs=1e-12)
    # gripper: n = 0 is the closed stop at the MJCF lo, one unit is span/100 ticks
    g = JOINTS.index("gripper")
    assert um.offset[g] == pytest.approx(model.jnt_range[g, 0])
    assert um.scale[g] == pytest.approx(lc.RAD_PER_TICK * calib["gripper"].span / 100.0)
    # the recorded range of motion bounds the map
    assert um.lo[i] == pytest.approx(um.scale[i] * -100 + um.offset[i])
    assert um.hi[i] == pytest.approx(um.scale[i] * 100 + um.offset[i])


def test_shift_and_scale_show_up_as_offset_and_gain(model):
    calib = matching_calibration(model)
    um0 = lc.unit_map_from_calibration(calib, model)
    shifted = lc.unit_map_from_calibration(with_joint(calib, "wrist_roll", shift=200), model)
    i = JOINTS.index("wrist_roll")
    c = lc.compare_unit_maps(shifted, um0)["wrist_roll"]
    assert c["max_deg"] == pytest.approx(200 * 360 / 4096, abs=1e-9)
    assert c["at"]["0"] == pytest.approx(200 * 360 / 4096, abs=1e-9)
    assert shifted.scale[i] == um0.scale[i]
    scaled = lc.unit_map_from_calibration(with_joint(calib, "wrist_roll", scale=0.5), model)
    assert scaled.scale[i] == pytest.approx(um0.scale[i] * 0.5, rel=1e-3)
    c = lc.compare_unit_maps(scaled, um0)["wrist_roll"]
    assert abs(c["at"]["0"]) < 0.05 and c["at"]["100"] == pytest.approx(-c["at"]["-100"], abs=0.1)
    # untouched joints do not move
    assert lc.compare_unit_maps(shifted, um0)["elbow_flex"]["max_deg"] == 0.0


def test_drive_mode_flips_the_direction(model):
    calib = matching_calibration(model)
    m = calib["shoulder_pan"]
    inv = {**calib, "shoulder_pan": lc.MotorCalibration(m.id, 1, m.homing_offset, m.range_min + 100, m.range_max + 100)}
    fwd = {**calib, "shoulder_pan": lc.MotorCalibration(m.id, 0, m.homing_offset, m.range_min + 100, m.range_max + 100)}
    um_inv = lc.unit_map_from_calibration(inv, model)
    um_fwd = lc.unit_map_from_calibration(fwd, model)
    i = JOINTS.index("shoulder_pan")
    # the dataset's +100 is still the positive MJCF direction (scale > 0) ...
    assert um_inv.scale[i] > 0 and um_inv.scale[i] == um_fwd.scale[i]
    # ... but the half turn's offset from the range centre changes sign
    mid = float(np.mean(model.jnt_range[i]))
    assert um_fwd.offset[i] - mid == pytest.approx(-(um_inv.offset[i] - mid))
    assert inv["shoulder_pan"].homing_pose_units == pytest.approx(-fwd["shoulder_pan"].homing_pose_units)
    # the gripper is anchored at its closed stop either way
    g = calib["gripper"]
    ginv = {**calib, "gripper": lc.MotorCalibration(g.id, 1, g.homing_offset, g.range_min, g.range_max)}
    assert lc.unit_map_from_calibration(ginv, model).offset[JOINTS.index("gripper")] == pytest.approx(model.jnt_range[5, 0])


def test_range_anchor_ignores_the_homing_pose(model):
    calib = with_joint(matching_calibration(model), "shoulder_lift", shift=300)
    um_r = lc.unit_map_from_calibration(calib, model, anchor="range")
    i = JOINTS.index("shoulder_lift")
    assert um_r.offset[i] == pytest.approx(float(np.mean(model.jnt_range[i])))
    um_h = lc.unit_map_from_calibration(calib, model, anchor="homing")
    assert um_h.offset[i] - um_r.offset[i] == pytest.approx(300 * lc.RAD_PER_TICK)
    with pytest.raises(ValueError, match="anchor"):
        lc.unit_map_from_calibration(calib, model, anchor="centre")


def test_degrees_mode(model):
    """LeRobot >= 0.5 default: n = (present - mid) * 360 / 4095 for body joints."""
    calib = with_joint(matching_calibration(model), "elbow_flex", shift=100, scale=0.9)
    um = lc.unit_map_from_calibration(calib, model, norm_mode="degrees")
    i = JOINTS.index("elbow_flex")
    assert um.scale[i] == pytest.approx(math.radians(1.0) * 4095 / 4096)
    m = calib["elbow_flex"]
    for n_deg in (-30.0, 0.0, 45.0):
        present = n_deg * 4095 / 360.0 + m.mid
        expected = float(np.mean(model.jnt_range[i])) + lc.RAD_PER_TICK * (present - lc.HALF_TURN)
        assert um.scale[i] * n_deg + um.offset[i] == pytest.approx(expected, abs=1e-12)
    # the gripper is 0..100 in every mode
    g = JOINTS.index("gripper")
    assert um.scale[g] == pytest.approx(lc.unit_map_from_calibration(calib, model).scale[g])
    with pytest.raises(ValueError, match="norm_mode"):
        lc.unit_map_from_calibration(calib, model, norm_mode="ticks")


def test_command_map_prior_is_identity(model):
    prior = lc.command_map_prior(with_joint(matching_calibration(model), "elbow_flex", shift=400), model)
    assert prior.is_identity() and prior.gain.shape == (6,)


# -- diagnostic, lock record ------------------------------------------------


def test_calibration_check_flags_joints_over_two_degrees(model):
    calib = with_joint(matching_calibration(model), "wrist_roll", shift=60)  # 5.3 deg
    chk = lc.calibration_check(calib, model)
    assert chk["flagged"] == ["wrist_roll"] and chk["message"] == lc.FLAG_MESSAGE
    rows = {r["joint"]: r for r in chk["joints"]}
    assert rows["wrist_roll"]["flagged"] and rows["wrist_roll"]["max_deg"] == pytest.approx(60 * 360 / 4096, abs=0.05)
    assert rows["wrist_roll"]["homing_pose_units"] == pytest.approx(-60 / calib["wrist_roll"].span * 200, abs=1e-3)
    assert not rows["elbow_flex"]["flagged"] and rows["gripper"]["homing_pose_units"] is None
    assert rows["shoulder_pan"]["model_span_deg"] == pytest.approx(math.degrees(2 * 1.91986), abs=1e-3)
    assert rows["shoulder_pan"]["calibration_span_deg"] == pytest.approx(rows["shoulder_pan"]["model_span_deg"], abs=0.1)
    assert rows["shoulder_pan"]["range_min"] == calib["shoulder_pan"].range_min
    clean = lc.calibration_check(matching_calibration(model), model)
    assert clean["flagged"] == [] and clean["message"] is None
    assert UnitMap.from_dict(clean["unit_map"]).scale.shape == (6,)


def test_calibration_record_round_trips_the_unit_map(model, tmp_path):
    calib = with_joint(matching_calibration(model), "shoulder_lift", shift=-80, scale=1.1)
    path = write_file(tmp_path / "arm.json", calib)
    rec = lc.calibration_record(calib, model, path=path, robot_id="arm", robot_type="so101_follower")
    assert rec["source"] == str(path) and rec["sha256"] == lc.file_sha256(path)
    assert rec["robot_id"] == "arm" and rec["norm_mode"] == "range_m100_100" and rec["anchor"] == "homing"
    assert rec["motors"]["shoulder_lift"] == calib["shoulder_lift"].to_dict() and list(rec["motors"]) == list(JOINTS)
    assert rec["unit_map_disagreement_deg"]["shoulder_lift"] > 2 > rec["unit_map_disagreement_deg"]["elbow_flex"]
    json.dumps(rec)  # JSON-clean
    um = lc.unit_map_from_calibration(calib, model)
    stored = lc.unit_map_from_record(rec)
    np.testing.assert_allclose(stored.scale, um.scale)
    np.testing.assert_allclose(stored.offset, um.offset)
    rebuilt = lc.unit_map_from_record({k: v for k, v in rec.items() if k != "unit_map"}, model)
    np.testing.assert_allclose(rebuilt.offset, um.offset)
    assert lc.unit_map_from_record(None) is None and lc.unit_map_from_record({"motors": rec["motors"]}) is None
    nopath = lc.calibration_record(calib, model)
    assert nopath["source"] is None and nopath["sha256"] is None


# -- synthetic validation ---------------------------------------------------


def test_calibration_map_removes_the_physical_bias_of_the_default_replay(model):
    """An arm whose shoulder_lift calibration is shifted 120 ticks (10.5 deg)
    and 10% narrower, recorded through its own map. Replayed with the model's
    map the sim sits 10 deg away from where the arm really was; with the
    calibration map it is where the arm was, with no sensor fit.

    The bias is in physical angle (what FK, the viewer and gravity see): the
    follower closes its loop in its own ticks, so in dataset units command
    and state share the map and the tracking residual barely notices.
    """
    from antares.synthetic import make_dataset

    calib = with_joint(matching_calibration(model), "shoulder_lift", shift=120, scale=0.9)
    calib_map = lc.unit_map_from_calibration(calib, model)
    params = SimParams.default()
    hidden = Arm(params=params, unit_map=calib_map)
    ds = make_dataset("synthetic/shifted", params, n_episodes=2, T=60, noise_units=0.0, arm=hidden)
    j = JOINTS.index("shoulder_lift")
    injected_deg = 120 * 360 / 4096

    default = Arm(params=params)
    calibrated = Arm(params=params, unit_map=lc.unit_map_from_calibration(calib, model))
    for ep in ds.episodes:
        truth = hidden.replay(ep)
        bias_default = np.degrees(np.mean(default.replay(ep) - truth, axis=0))
        bias_calib = np.degrees(np.max(np.abs(calibrated.replay(ep) - truth), axis=0))
        # the default map puts the shifted joint ~10 deg from the truth (shift plus 10% scale), the rest stay put
        assert 8.0 < abs(bias_default[j]) < 13.0
        assert np.all(np.abs(np.delete(bias_default, j)) < 0.5)
        assert np.all(bias_calib < 1e-6)
    c = lc.compare_unit_maps(calib_map, UnitMap.from_model(model))["shoulder_lift"]
    assert c["at"]["0"] == pytest.approx(injected_deg, abs=0.2)
    # the dataset-unit residual is blind to the map: the real arm tracked its own commands either way
    res_units = np.mean(np.abs(ds.episodes[0].state[10:, j] - ds.episodes[0].action[10:, j]))
    assert res_units < 2.0
