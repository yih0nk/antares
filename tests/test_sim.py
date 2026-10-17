import time

import mujoco
import numpy as np
import pytest

from antares.data import Episode
from antares.sim import (
    JOINTS,
    MESH_DIR,
    MODEL_PATH,
    Arm,
    SimParams,
    UnitMap,
    model_geometry,
)

FPS = 30.0


def make_episode(state: np.ndarray, action: np.ndarray) -> Episode:
    return Episode(
        index=0,
        fps=FPS,
        joint_names=JOINTS,
        state=np.asarray(state, dtype=np.float64),
        action=np.asarray(action, dtype=np.float64),
    )


def constant_episode(start_units, target_units, n: int) -> Episode:
    state = np.tile(np.asarray(start_units, dtype=np.float64), (n, 1))
    action = np.tile(np.asarray(target_units, dtype=np.float64), (n, 1))
    return make_episode(state, action)


@pytest.fixture(scope="module")
def arm() -> Arm:
    return Arm()


# -- SimParams ---------------------------------------------------------------


def test_simparams_defaults_match_mjcf(arm):
    p = SimParams.default()
    assert p == SimParams()
    assert p.kp == pytest.approx(arm.model.actuator_gainprm[0, 0])
    assert p.kv == pytest.approx(-arm.model.actuator_biasprm[0, 2])
    assert p.latency_frames == 0


def test_simparams_dict_roundtrip():
    p = SimParams(kp=500.0, kv=1.0, damping=0.1, armature=0.01, frictionloss=0.0, latency_frames=3)
    d = p.to_dict()
    assert d["latency_frames"] == 3
    assert SimParams.from_dict(d) == p
    # unknown keys ignored, missing keys default, floats coerced
    q = SimParams.from_dict({"kp": "600", "latency_frames": 2.0, "bogus": 1})
    assert q.kp == 600.0 and q.latency_frames == 2 and q.kv == SimParams().kv


def test_simparams_negative_latency_rejected():
    with pytest.raises(ValueError):
        SimParams(latency_frames=-1)


def test_simparams_from_model_reads_back_applied_values(arm):
    assert SimParams.from_model(arm.model) == SimParams()
    p = SimParams(kp=123.5, kv=0.75, damping=0.3, armature=0.011, frictionloss=0.002, latency_frames=2)
    arm.apply(p)
    try:
        assert SimParams.from_model(arm.model) == p.replace(latency_frames=0)
        assert SimParams.from_model(arm.model, latency_frames=2) == p
    finally:
        arm.apply(SimParams())


# -- UnitMap -----------------------------------------------------------------


def test_unitmap_endpoints(arm):
    um = arm.unit_map
    lo = arm.model.jnt_range[:, 0]
    hi = arm.model.jnt_range[:, 1]
    low_units = np.array([-100.0] * 5 + [0.0])
    high_units = np.array([100.0] * 6)
    np.testing.assert_allclose(um.to_rad(low_units), lo, atol=1e-12)
    np.testing.assert_allclose(um.to_rad(high_units), hi, atol=1e-12)
    # body joints are centred: 0 units -> middle of range
    mid = um.to_rad(np.zeros(6))
    np.testing.assert_allclose(mid[:5], (lo[:5] + hi[:5]) / 2, atol=1e-12)
    assert mid[5] == pytest.approx(lo[5])


def test_unitmap_roundtrip_and_shapes(arm):
    um = arm.unit_map
    rng = np.random.default_rng(0)
    u = rng.uniform(-100, 100, size=(20, 6))
    u[:, 5] = rng.uniform(0, 100, size=20)
    q = um.to_rad(u)
    assert q.shape == (20, 6)
    np.testing.assert_allclose(um.to_units(q), u, atol=1e-9)
    assert um.to_rad(u[0]).shape == (6,)


def test_unitmap_clips_to_joint_range(arm):
    um = arm.unit_map
    q = um.to_rad(np.array([1000.0, -1000.0, 1000.0, -1000.0, 1000.0, 1000.0]))
    np.testing.assert_allclose(q[[0, 2, 4, 5]], arm.model.jnt_range[[0, 2, 4, 5], 1])
    np.testing.assert_allclose(q[[1, 3]], arm.model.jnt_range[[1, 3], 0])


def test_unitmap_dict_roundtrip(arm):
    um = arm.unit_map
    um2 = UnitMap.from_dict(um.to_dict())
    np.testing.assert_allclose(um2.scale, um.scale)
    np.testing.assert_allclose(um2.offset, um.offset)
    np.testing.assert_allclose(um2.lo, um.lo)
    np.testing.assert_allclose(um2.hi, um.hi)
    bare = UnitMap.from_dict({"scale": [1] * 6, "offset": [0] * 6})
    assert bare.lo is None
    np.testing.assert_allclose(bare.to_rad(np.full(6, 7.0)), 7.0)  # no clipping without range


# -- Arm -----------------------------------------------------------------------


def test_arm_model_facts(arm):
    assert MODEL_PATH.exists()
    assert arm.model.nq == 6 and arm.model.nu == 6
    assert arm.body_names[0] == "world"
    assert len(arm.body_names) == arm.model.nbody == 9
    assert "gripper" in arm.body_names


def test_fk_shapes_and_base_at_origin(arm):
    q = np.zeros(6)
    poses = arm.fk(q)
    assert poses.shape == (arm.model.nbody, 7)
    np.testing.assert_allclose(poses[0, :3], 0.0)  # world
    np.testing.assert_allclose(poses[1, :3], 0.0)  # base body is at the origin
    np.testing.assert_allclose(poses[1, 3:], [1, 0, 0, 0])
    np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-9)

    rng = np.random.default_rng(1)
    qs = rng.uniform(-1, 1, size=(5, 6))
    many = arm.fk(qs)
    assert many.shape == (5, arm.model.nbody, 7)
    np.testing.assert_allclose(many[:, 1, :3], 0.0)
    np.testing.assert_allclose(many[2], arm.fk(qs[2]))
    # moving a joint moves downstream bodies but not the base
    a = arm.fk(np.zeros(6))
    b = arm.fk(np.array([0.5, 0, 0, 0, 0, 0]))
    assert not np.allclose(a[6, :3], b[6, :3])
    np.testing.assert_allclose(a[1], b[1])


def test_replay_shape_and_initial_state(arm):
    start = np.array([10.0, -50.0, 40.0, 20.0, -10.0, 5.0])
    target = np.array([30.0, -40.0, 50.0, 10.0, 0.0, 20.0])
    ep = constant_episode(start, target, 60)
    sim_q = arm.replay(ep)
    assert sim_q.shape == (60, 6)
    np.testing.assert_allclose(sim_q[0], arm.real_q(ep)[0])
    np.testing.assert_allclose(sim_q[0], arm.unit_map.to_rad(start))
    assert np.all(np.isfinite(sim_q))
    # timestep restored after replay
    assert arm.model.opt.timestep == pytest.approx(0.005)


def test_replay_converges_to_constant_target(arm):
    arm.apply(SimParams())
    start = np.array([0.0, -50.0, 40.0, 20.0, 0.0, 5.0])
    target = np.array([40.0, -20.0, 60.0, 0.0, 30.0, 40.0])
    ep = constant_episode(start, target, 90)  # 3 s
    sim_q = arm.replay(ep)
    tgt = arm.unit_map.to_rad(target)
    err0 = np.abs(sim_q[0] - tgt)
    err_end = np.abs(sim_q[-1] - tgt)
    assert np.all(err_end < 0.05 * err0)
    assert np.all(np.degrees(err_end) < 1.0)


def test_replay_matches_actuator_ctrl_range(arm):
    # a wildly out-of-range command is clipped, not propagated
    start = np.zeros(6)
    target = np.array([1e6, 0, 0, 0, 0, 50.0])
    ep = constant_episode(start, target, 30)
    sim_q = arm.replay(ep)
    assert sim_q[-1, 0] <= arm.model.jnt_range[0, 1] + 1e-6


def test_latency_shifts_response_later(arm):
    n, t_step = 60, 10
    start = np.array([0.0, -50.0, 40.0, 20.0, 0.0, 5.0])
    target = start.copy()
    target[0] = 40.0  # step on shoulder_pan
    action = np.tile(start, (n, 1))
    action[t_step:] = target
    ep = make_episode(np.tile(start, (n, 1)), action)

    arm.apply(SimParams(latency_frames=0))
    s0 = arm.replay(ep)
    arm.apply(SimParams(latency_frames=5))
    s5 = arm.replay(ep)
    arm.apply(SimParams())

    q_start = arm.unit_map.to_rad(start)[0]
    q_tgt = arm.unit_map.to_rad(target)[0]
    half = (q_start + q_tgt) / 2

    def crossing(s):
        return int(np.argmax(s[:, 0] > half))

    assert crossing(s0) >= t_step
    assert crossing(s5) >= crossing(s0) + 4
    # with 5 frames of latency nothing has moved yet at t_step + 5 - 1
    assert abs(s5[t_step + 4, 0] - q_start) < 1e-3
    assert abs(s0[t_step + 4, 0] - q_start) > 1e-2


def test_apply_changes_model_arrays(arm):
    p = SimParams(kp=300.0, kv=1.5, damping=0.2, armature=0.01, frictionloss=0.0, latency_frames=1)
    arm.apply(p)
    m = arm.model
    np.testing.assert_allclose(m.actuator_gainprm[:, 0], 300.0)
    np.testing.assert_allclose(m.actuator_biasprm[:, 1], -300.0)
    np.testing.assert_allclose(m.actuator_biasprm[:, 2], -1.5)
    np.testing.assert_allclose(m.dof_damping, 0.2)
    np.testing.assert_allclose(m.dof_armature, 0.01)
    np.testing.assert_allclose(m.dof_frictionloss, 0.0)
    assert arm.params == p
    arm.apply(p)  # idempotent
    np.testing.assert_allclose(m.actuator_gainprm[:, 0], 300.0)
    arm.apply(SimParams())
    np.testing.assert_allclose(m.actuator_gainprm[:, 0], 998.22)
    np.testing.assert_allclose(m.dof_damping, 0.6)


def test_softer_gain_tracks_worse(arm):
    start = np.array([0.0, -50.0, 40.0, 20.0, 0.0, 5.0])
    target = np.array([40.0, -20.0, 60.0, 0.0, 30.0, 40.0])
    ep = constant_episode(start, target, 15)
    tgt = arm.unit_map.to_rad(target)
    arm.apply(SimParams(kp=998.22))
    stiff = np.abs(arm.replay(ep)[5] - tgt).sum()
    arm.apply(SimParams(kp=50.0))
    soft = np.abs(arm.replay(ep)[5] - tgt).sum()
    arm.apply(SimParams())
    assert soft > stiff


def test_replay_is_fast(arm):
    rng = np.random.default_rng(2)
    n = 300
    state = np.cumsum(rng.normal(0, 0.5, size=(n, 6)), axis=0)
    state[:, 5] = np.abs(state[:, 5])
    ep = make_episode(state, state + 1.0)
    arm.replay(ep)  # warm up
    best = min(_timed(arm, ep) for _ in range(5))
    assert best < 0.1, f"replay took {best*1e3:.1f} ms"


def _timed(arm, ep):
    t = time.perf_counter()
    arm.replay(ep)
    return time.perf_counter() - t


def test_contacts_disabled_by_default():
    a = Arm()
    assert a.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    b = Arm(contacts=True)
    assert not (b.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))


def test_arm_default_params_come_from_model():
    a = Arm()
    assert a.model_path == MODEL_PATH
    assert a.params == SimParams()


def test_arm_model_path(tmp_path):
    # a copy of the MJCF with different sts3215 values, next to the same meshes
    xml = MODEL_PATH.read_text().replace(
        'damping="0.60" frictionloss="0.052" armature="0.028"',
        'damping="0.3" frictionloss="0.01" armature="0.02"',
    ).replace('kp="998.22" kv="2.731"', 'kp="200" kv="1.5"')
    assert xml != MODEL_PATH.read_text()
    xml = xml.replace('meshdir="assets"', f'meshdir="{MESH_DIR}"')
    path = tmp_path / "custom.xml"
    path.write_text(xml)

    a = Arm(model_path=path)
    assert a.model_path == path
    assert a.params == SimParams(kp=200.0, kv=1.5, damping=0.3, armature=0.02, frictionloss=0.01)
    np.testing.assert_allclose(a.model.actuator_gainprm[:, 0], 200.0)
    np.testing.assert_allclose(a.model.dof_damping, 0.3)
    # explicit params still win over the file
    b = Arm(params=SimParams(kp=50.0), model_path=path)
    assert b.params.kp == 50.0
    np.testing.assert_allclose(b.model.actuator_gainprm[:, 0], 50.0)
    with pytest.raises(ValueError):
        Arm(model_path=tmp_path / "missing.xml")


# -- model_geometry ------------------------------------------------------------


def test_model_geometry(arm):
    geo = model_geometry(arm.model)
    assert [j["name"] for j in geo["joints"]] == list(JOINTS)
    assert len(geo["bodies"]) == arm.model.nbody
    assert geo["bodies"][0] == {"id": 0, "name": "world", "parent": 0}
    assert geo["bodies"][1]["parent"] == 0
    assert geo["geoms"], "expected visual mesh geoms"
    body_ids = {b["id"] for b in geo["bodies"]}
    for g in geo["geoms"]:
        assert g["body"] in body_ids
        assert (MESH_DIR / g["mesh"]).is_file(), g["mesh"]
        assert len(g["pos"]) == 3 and len(g["quat"]) == 4 and len(g["rgba"]) == 4
        assert np.isclose(np.linalg.norm(g["quat"]), 1.0)
    for j in geo["joints"]:
        assert j["body"] in body_ids
        assert j["range"][0] < j["range"][1]
    # default-argument path loads the model itself
    assert model_geometry()["joints"] == geo["joints"]
