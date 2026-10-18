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
        {"name": "link_mass", "target": "body_mass", "bounds": [0.5, 1.5], "neutral": 1.0,
         "explains": "links heavier or lighter than the CAD model"}
    )
    assert f.origin == "model" and f.stages == ("shared",)
    factors.register(f, fn)
    try:
        assert factors.is_extra("link_mass")
        assert "link_mass" in factors.names("shared")
        with pytest.raises(ValueError):
            factors.register(f, fn)
    finally:
        factors.unregister("link_mass")
    assert "link_mass" not in factors.REGISTRY


def test_config_errors():
    with pytest.raises(ValueError):
        factors.factor_from_config({"name": "x", "bounds": [0, 1]})
    with pytest.raises(ValueError):
        factors.factor_from_config({"name": "x", "bounds": [0, 1], "target": "dof_damping", "colour": 1})
    with pytest.raises(ValueError):
        factors.register(factors.get("kp"), lambda a, v: None)


def test_selection_rejects_what_the_stage_cannot_fit():
    # sensor-only built-ins have no SimParams field and no apply hook
    with pytest.raises(ValueError, match="sensor"):
        factors.names("shared", include=["sensor_gain"])
    with pytest.raises(ValueError, match="sensor"):
        factors.parse_selection("kp,sensor_bias")
    with pytest.raises(ValueError, match="sensor"):
        factors.space("shared", factors=("kp", "sensor_gain"))
    # gravity is one value for the whole arm
    with pytest.raises(ValueError, match="per joint"):
        factors.names("per_joint", include=["gravity_scale"])
    assert "gravity_scale" in factors.names("shared", include=["gravity_scale"])
    assert factors.names(include=["sensor_gain"])  # no stage: just the registry


def test_empty_selection_is_an_error():
    from antares.calibrate import calibrate
    from test_calibrate import fake_simulate, make_episodes

    everything = ",".join("-" + n for n in factors.names("shared"))
    with pytest.raises(ValueError, match="nothing to fit"):
        factors.parse_selection(everything)
    with pytest.raises(ValueError, match="nothing to fit"):
        factors.space("shared", factors=())
    eps = make_episodes(2)
    with pytest.raises(ValueError, match="empty"):
        calibrate(eps[:1], eps[1:], simulate=fake_simulate, max_evaluations=4, fit_factors=())
    with pytest.raises(ValueError, match="empty"):
        calibrate(eps[:1], eps[1:], simulate=fake_simulate, max_evaluations=4, per_joint_factors=[])


def test_lock_extras_need_their_factor_file():
    from antares.sim import SimParams

    f, fn = factors.factor_from_config({"name": "link_mass", "target": "body_mass", "bounds": [0.5, 1.5]})
    factors.register(f, fn)
    try:
        d = SimParams(extra={"link_mass": 1.2}).to_dict()
        assert d["link_mass"] == 1.2 and SimParams.from_dict(d).link_mass == 1.2
    finally:
        factors.unregister("link_mass")
    with pytest.raises(ValueError, match="link_mass.*--factors-file"):
        SimParams.from_dict(d)
