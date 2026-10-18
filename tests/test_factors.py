import pytest

from antares import factors


def test_builtin_order_and_shared_space():
    assert factors.names("shared") == ("kp", "kv", "damping", "armature", "frictionloss", "latency_frames")
    assert factors.names("per_joint") == ("forcerange",)
    assert factors.names("sensor") == ("sensor_gain", "sensor_bias")


def test_space_matches_the_old_hand_written_bounds():
    assert factors.space("shared") == {
        "kp": (10.0, 5000.0, "log"),
        "kv": (0.1, 50.0, "log"),
        "damping": (0.01, 10.0, "log"),
        "armature": (1e-4, 0.5, "log"),
        "frictionloss": (1e-3, 2.0, "log"),
        "latency_frames": (0.0, 5.0, "int"),
    }


def test_space_overrides_and_selection():
    s = factors.space("shared", factors=("kp", "latency_frames"), overrides={"kp": (1.0, 2.0, "log")})
    assert s == {"kp": (1.0, 2.0, "log"), "latency_frames": (0.0, 5.0, "int")}


def test_drift_lists():
    assert factors.drift_factors() == {f: 4.0 for f in ("kp", "kv", "damping", "armature", "frictionloss")}
    assert factors.drift_shifts() == {"latency_frames": (-2, 3)}
    assert factors.log_scaled() == {"kp", "kv", "damping", "armature", "frictionloss", "forcerange"}


def test_mjcf_attrs():
    assert factors.mjcf_attrs("joint") == ("damping", "armature", "frictionloss")
    assert factors.mjcf_attrs("position") == ("kp", "kv")


def test_parse_selection():
    assert factors.parse_selection(None) is None
    assert factors.parse_selection("-latency_frames") == ("kp", "kv", "damping", "armature", "frictionloss")
    assert factors.parse_selection("+forcerange,-kv")[-1] == "forcerange"
    assert factors.parse_selection("kp,latency_frames") == ("kp", "latency_frames")
    with pytest.raises(ValueError):
        factors.parse_selection("+kp,kv")
    with pytest.raises(KeyError):
        factors.parse_selection("+nope")


def test_describe_lists_everything_with_identifiability():
    d = factors.describe()
    assert d["version"] == factors.REGISTRY_VERSION
    names = [f["name"] for f in d["factors"]]
    assert names[:6] == list(factors.names("shared"))
    by = {f["name"]: f for f in d["factors"]}
    assert by["latency_frames"]["identifiable"] == "yes"
    assert by["kp"]["identifiable"] == "no"
    assert all(f["explains"] and f["needs"] for f in d["factors"])
    assert {u["name"] for u in d["not_modelled"]} >= {"contact", "temperature", "backlash"}


def test_register_from_config_and_unregister():
    f, fn = factors.factor_from_config(
        {"name": "gravity_scale", "target": "opt.gravity", "bounds": [0.5, 1.5], "neutral": 1.0,
         "explains": "gravity stronger or weaker than modelled"}
    )
    assert f.origin == "model" and f.stages == ("shared",)
    factors.register(f, fn)
    try:
        assert factors.is_extra("gravity_scale")
        assert "gravity_scale" in factors.names("shared")
        with pytest.raises(ValueError):
            factors.register(f, fn)
    finally:
        factors.unregister("gravity_scale")
    assert "gravity_scale" not in factors.REGISTRY


def test_config_errors():
    with pytest.raises(ValueError):
        factors.factor_from_config({"name": "x", "bounds": [0, 1]})
    with pytest.raises(ValueError):
        factors.factor_from_config({"name": "x", "bounds": [0, 1], "target": "dof_damping", "colour": 1})
    with pytest.raises(ValueError):
        factors.register(factors.get("kp"), lambda a, v: None)
