"""Antares on arms other than the SO-100/101: profiles, units, fit, export.

The Menagerie models are fetched on first use; without network (and with
nothing cached) these tests skip rather than fail.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from antares import catalog, factors
from antares.calibrate import calibrate, default_simulate, score
from antares.data import Episode
from antares.robots import REGISTRY, get_robot
from antares.sim import Arm, SimParams, UnitMap

MENAGERIE = [n for n, r in REGISTRY.items() if r.menagerie is not None and n in ("aloha", "trossen_ai", "piper", "koch")]


def _arm(name: str, **kw) -> Arm:
    try:
        return Arm(robot=name, **kw)
    except OSError as e:  # no network and not cached
        pytest.skip(f"cannot fetch {name}: {e}")


def _episodes(arm: Arm, truth: SimParams, n: int = 3, T: int = 120, seed: int = 0) -> list[Episode]:
    """Smooth random commands inside the joint ranges, measured on ``truth``."""
    rng = np.random.default_rng(seed)
    lo, hi = arm.unit_map.lo, arm.unit_map.hi
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    arm.apply(truth)
    out = []
    for i in range(n):
        w = rng.uniform(0.5, 2.0, arm.n_joints)
        ph = rng.uniform(0, 2 * np.pi, arm.n_joints)
        t = np.arange(T)[:, None] / 30.0
        q_cmd = mid + 0.4 * half * np.sin(w * t + ph)
        action = np.asarray(arm.unit_map.to_units(q_cmd))
        ep = Episode(index=i, fps=30.0, joint_names=tuple(arm.robot.joints), state=action, action=action)
        q = np.asarray(arm.replay(ep))
        state = np.asarray(arm.unit_map.to_units(q))
        out.append(dataclasses.replace(ep, state=state))
    return out


@pytest.mark.parametrize("name", MENAGERIE)
def test_menagerie_robot_maps_every_column(name):
    arm = _arm(name)
    r = arm.robot
    assert arm.n_joints == len(r.joints) == len(r.joint_names)
    assert len(set(arm.actuator_ids.tolist())) == arm.n_joints
    p = arm.default_params(from_model=True)
    assert len(p.per_joint("kp", arm.n_joints)) == arm.n_joints
    eps = _episodes(arm, p, n=1, T=40)
    q = np.asarray(arm.replay(eps[0]))
    assert q.shape == (40, arm.n_joints) and np.all(np.isfinite(q))


def test_unit_kinds():
    lo = np.array([-np.pi / 2, 0.0, -np.pi / 4, -1.0])
    hi = np.array([np.pi / 2, np.pi / 3, np.pi / 4, 1.0])
    um = UnitMap.from_kinds(lo, hi, ("range_m100_100", "range_0_100", "degrees", "radians"))
    rad = um.to_rad(np.array([100.0, 50.0, 10.0, 0.3]))
    assert rad[0] == pytest.approx(np.radians(90))
    assert rad[1] == pytest.approx(np.radians(30))
    assert rad[2] == pytest.approx(np.radians(10))  # degrees are about the middle of the range
    assert rad[3] == pytest.approx(0.3)


def test_aloha_fit_recovers_latency():
    arm = _arm("aloha")
    start = arm.default_params(from_model=True)
    truth = start.replace(latency_frames=3)
    eps = _episodes(arm, truth, n=3, T=150)
    sim = default_simulate(robot="aloha")
    res = calibrate(eps[:2], eps[2:], simulate=sim, start=start, max_evaluations=40, seed=0,
                    fit_factors=("latency_frames", "damping"))
    assert res.params.latency_frames == 3
    assert res.eval_score < res.baseline_eval_score
    assert set(res.per_joint_after) == set(arm.robot.joints)


def test_added_factor_is_fitted():
    """A user factor from TOML-style config moves toward the value the data was made with."""
    f, fn = factors.factor_from_config(
        {"name": "test_gravity", "target": "opt.gravity", "mode": "scale", "bounds": [0.5, 1.5]}
    )
    factors.register(f, fn, replace_existing=True)
    try:
        arm = Arm()
        truth = SimParams().replace(test_gravity=1.3)
        eps = _episodes(arm, truth, n=3, T=150, seed=1)
        sim = default_simulate()
        assert score(SimParams(), eps, sim).overall_deg > score(truth, eps, sim).overall_deg
        res = calibrate(eps[:2], eps[2:], simulate=sim, start=SimParams(), max_evaluations=60, seed=0,
                        fit_factors=("test_gravity",))
        assert res.params.test_gravity == pytest.approx(1.3, abs=0.1)
        assert "test_gravity" in res.params.to_dict()
    finally:
        factors.unregister("test_gravity")


def test_generic_export_replays_identically(tmp_path):
    from antares.export import export_mjcf, load_exported_arm, read_sidecar
    from antares.lock import RobotLock
    from tests.test_export import make_lock

    arm = _arm("piper")
    p = arm.default_params(from_model=True)
    p = p.replace(kp=tuple(np.asarray(p.per_joint("kp", arm.n_joints)) * 1.5), latency_frames=2)
    lock: RobotLock = dataclasses.replace(make_lock(p), robot="piper")
    out = export_mjcf(lock, tmp_path / "piper.xml")
    side = read_sidecar(out)
    assert side["robot"] == "piper" and side["joints"] == list(arm.robot.joints)
    assert (tmp_path / "assets").is_dir()
    exported = load_exported_arm(out)
    ep = _episodes(arm, p, n=1, T=60)[0]
    arm.apply(p)
    np.testing.assert_allclose(exported.replay(ep), arm.replay(ep), atol=1e-6)


def test_catalog_assign_overrides_and_resets():
    repo = "someone/unlisted_dataset"
    assert catalog.robot_for_dataset(repo) is None
    catalog.assign(repo, "koch")
    try:
        assert catalog.robot_for_dataset(repo) == "koch"
    finally:
        catalog.assign(repo, None)
    assert catalog.robot_for_dataset(repo) is None
    for e in catalog.entries():
        assert e["robot"] in REGISTRY, e


def test_cli_lists(capsys):
    from antares.cli import main

    assert main(["factors", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert [f["name"] for f in d["factors"]][:6] == list(factors.names("shared"))[:6]
    assert d["not_modelled"]
    assert main(["models"]) == 0
    assert "aloha" in capsys.readouterr().out
    assert main(["datasets"]) == 0
    assert "lerobot/aloha_static_coffee" in capsys.readouterr().out


def test_cli_rejects_unknown_factor(monkeypatch, capsys):
    """The selection is passed through as typed and resolved once, by
    run_calibration; its 400 is the CLI's usage error."""
    from antares import cli
    from antares.cli import main
    from antares.synthetic import make_dataset

    ds = make_dataset("x/y", SimParams(), n_episodes=2, T=20)
    monkeypatch.setattr(cli, "_load_dataset", lambda repo_id: ds)
    assert main(["calibrate", "x/y", "--factors", "+nope"]) == 2
    assert "nope" in capsys.readouterr().err
    assert main(["calibrate", "x/y", "--factors", "+sensor_gain"]) == 2
    assert "sensor" in capsys.readouterr().err
    everything = ",".join("-" + n for n in factors.names("shared"))
    assert main(["calibrate", "x/y", f"--factors={everything}"]) == 2
    assert "nothing to fit" in capsys.readouterr().err
    assert main(["calibrate", "x/y", "--per-joint", "--per-joint-factors", "+nope"]) == 2
    assert "nope" in capsys.readouterr().err


def test_robot_profile_from_toml(tmp_path):
    from antares.robots import load_robot_file

    so = get_robot("so101")
    toml = tmp_path / "robots.toml"
    toml.write_text(
        f"""
[[robot]]
name = "my_so101"
label = "My SO-101"
model = "{so.model_path}"
joints = {json.dumps(list(so.joints))}
mjcf_joints = {json.dumps(list(so.joint_names))}
grippers = ["gripper"]
units = ["range_m100_100", "range_m100_100", "range_m100_100", "range_m100_100", "range_m100_100", "range_0_100"]
"""
    )
    try:
        (r,) = load_robot_file(toml)
        arm = Arm(robot=r.name)
        assert arm.n_joints == 6
    finally:
        REGISTRY.pop("my_so101", None)


def test_viewer_geometry_and_meshes_for_a_fetched_robot():
    from fastapi.testclient import TestClient

    from antares import server

    _arm("trossen_ai")
    client = TestClient(server.app)
    geo = client.get("/api/model", params={"robot": "trossen_ai"}).json()
    assert geo["robot"] == "trossen_ai" and geo["geoms"]
    g = geo["geoms"][0]
    assert g["mesh"].endswith(".obj") and len(g["scale"]) == 3
    r = client.get(f"/api/model/mesh/{g['mesh']}", params={"robot": "trossen_ai"})
    assert r.status_code == 200 and r.content
    assert client.get("/api/model/mesh/../robots.py", params={"robot": "trossen_ai"}).status_code in (400, 404)
    # the SO arms keep their flat STL names
    so = client.get("/api/model").json()
    assert all("/" not in x["mesh"] and "scale" not in x for x in so["geoms"])


def test_factors_endpoint():
    from fastapi.testclient import TestClient

    from antares import server

    d = TestClient(server.app).get("/api/factors").json()
    assert {"kp", "latency_frames", "forcerange", "sensor_bias"} <= {f["name"] for f in d["factors"]}
    assert TestClient(server.app).get("/api/datasets/catalog").json()


# --------------------------------------------------------------------------
# a small synthetic arm: two hinges and a parallel gripper whose second
# finger follows the first through a joint equality
# --------------------------------------------------------------------------

_MINI_XML = """<mujoco model="mini">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <default>
    <joint damping="0.1" armature="0.01" frictionloss="0.01"/>
  </default>
  <worldbody>
    <body name="link1">
      <joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.2"/>
      <body name="link2" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.2"/>
        <body name="fl" pos="0 0 0.2">
          <joint name="left_finger" type="slide" axis="0 1 0" range="0 0.04" ref="0.01"/>
          <geom type="box" size="0.01 0.01 0.02" mass="0.02"/>
        </body>
        <body name="fr" pos="0 0 0.2">
          <joint name="right_finger" type="slide" axis="0 -1 0" range="-0.05 0.05" ref="0.005"/>
          <geom type="box" size="0.01 0.01 0.02" mass="0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="{j1}" joint2="{j2}" polycoef="0.002 1.5 0 0 0"/>
  </equality>
  <actuator>
    <position name="a1" joint="j1" kp="50" kv="2" forcerange="-10 10"/>
    <position name="a2" joint="j2" kp="50" kv="2"/>
    <position name="af" joint="left_finger" kp="20" kv="1" forcerange="-5 5"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def mini_robot(tmp_path, request):
    """Register ``mini`` from a TOML next to its MJCF; the equality's
    orientation (which side the mapped finger is on) comes from the param."""
    from antares.robots import load_robot_file

    mapped_first = getattr(request, "param", True)
    j1, j2 = ("left_finger", "right_finger") if mapped_first else ("right_finger", "left_finger")
    (tmp_path / "mini.xml").write_text(_MINI_XML.format(j1=j1, j2=j2))
    (tmp_path / "scene.xml").write_text('<mujoco><include file="mini.xml"/></mujoco>')
    toml = tmp_path / "robots.toml"
    toml.write_text(
        """
[[robot]]
name = "mini"
model = "mini.xml"
scene = "scene.xml"
joints = ["j1", "j2", "gripper"]
mjcf_joints = ["j1", "j2", "left_finger"]
grippers = ["gripper"]
units = ["radians", "radians", "native"]
"""
    )
    try:
        (r,) = load_robot_file(toml)
        yield r
    finally:
        REGISTRY.pop("mini", None)


def test_robot_toml_resolves_scene_next_to_the_file(mini_robot, tmp_path):
    assert mini_robot.model_path == (tmp_path / "mini.xml").resolve()
    assert mini_robot.scene_path == (tmp_path / "scene.xml").resolve()
    assert mini_robot.mesh_dir == (tmp_path / "assets").resolve()


@pytest.mark.parametrize("mini_robot", [True, False], indirect=True, ids=["mapped=joint1", "mapped=joint2"])
def test_follower_joint_obeys_the_equality_in_either_orientation(mini_robot):
    import mujoco

    arm = Arm(robot="mini")
    m = arm.model
    assert len(arm._followers) == 1
    f_adr, drv_adr, _, _ = arm._followers[0]
    assert drv_adr == m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")]
    assert f_adr == m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "right_finger")]
    # MuJoCo's constraint: q1 - q1_0 = c0 + c1 * (q2 - q2_0)
    a1, a2 = int(m.jnt_qposadr[m.eq_obj1id[0]]), int(m.jnt_qposadr[m.eq_obj2id[0]])
    c0, c1 = m.eq_data[0, 0], m.eq_data[0, 1]
    d = mujoco.MjData(m)
    for g in (0.0, 0.01, 0.03):
        arm._set_q(d, np.array([0.3, -0.2, g]))
        assert d.qpos[drv_adr] == pytest.approx(g)
        assert d.qpos[a1] - m.qpos0[a1] == pytest.approx(c0 + c1 * (d.qpos[a2] - m.qpos0[a2]), abs=1e-12)
    # the follower moves with the finger, in fk too
    poses = arm.fk(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.03]]))
    fr = arm.body_names.index("fr")
    left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")
    gain = c1 if m.eq_obj2id[0] == left else 1.0 / c1  # follower travel per unit of finger travel
    assert np.linalg.norm(poses[1, fr, :3] - poses[0, fr, :3]) == pytest.approx(0.03 * gain, abs=1e-9)


def test_default_params_survive_a_lock_being_applied(mini_robot):
    arm = Arm(robot="mini")
    d0 = arm.default_params(from_model=True)
    assert d0.per_joint("kp", 3).tolist() == [50.0, 50.0, 20.0]
    arm.apply(d0.replace(kp=999.0, damping=0.7, forcerange=3.0))
    assert arm.default_params(from_model=True) == d0
    assert arm.default_params() == d0  # "model" robots: the MJCF either way
    # the SO arm likewise keeps its compiled values after a replay with a lock
    so = Arm()
    so_d0 = so.default_params(from_model=True)
    so.apply(SimParams(kp=5.0, kv=0.1))
    assert so.default_params(from_model=True) == so_d0 and so.default_params() == SimParams()


def test_unlimited_actuator_starts_inside_the_forcerange_bounds(mini_robot):
    from antares.calibrate import spaces_for

    arm = Arm(robot="mini")
    d0 = arm.default_params(from_model=True)
    top = arm.forcerange_limit()
    assert top == factors.get("forcerange").space_entry(per_joint=True)[1]
    assert d0.per_joint("forcerange", 3).tolist() == [10.0, top, 5.0]  # a2 has no forcerange
    # the per-joint stage can start exactly at the model's values
    _, per_joint = spaces_for(mini_robot)
    back = per_joint.decode(per_joint.encode(d0), d0)
    np.testing.assert_allclose(back.per_joint("forcerange"), d0.per_joint("forcerange"))
    # a robot's own bound override moves the limit
    wide = dataclasses.replace(mini_robot, bounds={"forcerange": (0.5, 500.0, "log")})
    REGISTRY["mini"] = wide
    assert Arm(robot="mini").forcerange_limit() == 500.0


def test_generic_export_of_a_toml_robot_compiles_one_arm(mini_robot, tmp_path, monkeypatch):
    from antares import export
    from antares.export import export_mjcf, load_exported_arm, read_sidecar
    from tests.test_export import make_lock

    arm = Arm(robot="mini")
    p = arm.default_params(from_model=True).replace(kp=(60.0, 40.0, 25.0), latency_frames=1)
    lock = dataclasses.replace(make_lock(p), robot="mini")
    built = []
    real_arm = export.Arm

    class CountingArm(real_arm):
        def __init__(self, *a, **kw):
            built.append(kw.get("robot"))
            super().__init__(*a, **kw)

    monkeypatch.setattr(export, "Arm", CountingArm)
    out = export_mjcf(lock, tmp_path / "out" / "mini.xml")
    assert built == ["mini"]  # the spec's values and the sidecar's unit map from one compile
    side = read_sidecar(out)
    assert side["robot"] == "mini" and side["joints"] == ["j1", "j2", "gripper"]
    assert side["unit_map"]["scale"] == arm.unit_map.to_dict()["scale"]
    exported = load_exported_arm(out)
    ep = _episodes(arm, p, n=1, T=40)[0]
    arm.apply(p)
    np.testing.assert_allclose(exported.replay(ep), arm.replay(ep), atol=1e-6)


def test_check_and_drift_run_on_piper(tmp_path, monkeypatch):
    """The drift gate scores a non-SO arm on its own joints, and a drift scan
    runs on it (it used to pass ``simulate`` twice)."""
    from antares import server
    from antares.data import Dataset
    from tests.test_export import make_lock

    arm = _arm("piper")
    p = arm.default_params(from_model=True)
    eps = _episodes(arm, p, n=6, T=40)
    ds = Dataset(repo_id="local/piper", robot_type="piper", fps=30.0,
                 joint_names=tuple(arm.robot.joints), episodes=tuple(eps))
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "dataset_robot", lambda _ds: "piper")
    lock = dataclasses.replace(make_lock(p), robot="piper", name="piper_unit")
    lock.save(tmp_path / "piper_unit.lock.json")

    result = server.run_check(ds, lock.to_dict(), min_episodes=1)
    assert np.isfinite(result.score) and set(result.per_joint) <= set(arm.robot.joints)
    scan, path = server.run_drift(ds, "piper_unit", window=3, step=3, max_evaluations=4)
    assert path.is_file() and len(scan.windows) == 2


def test_so100_lock_exports_the_so100_model(tmp_path):
    """A lock on the SO-ARM100 model exports that model, not the SO-101 file,
    and the exported arm replays like the sim it was fitted in."""
    from antares.export import export_mjcf, load_exported_arm
    from tests.test_export import make_lock

    arm = Arm(robot="so100")
    p = SimParams(kp=40.0, latency_frames=2)
    lock = dataclasses.replace(make_lock(p), robot="so100")
    out = export_mjcf(lock, tmp_path / "so100.xml")
    exported = load_exported_arm(out)
    assert exported.robot.name == "so100"
    assert "shoulder_pan" not in out.read_text()
    ep = _episodes(arm, p, n=1, T=40)[0]
    arm.apply(p)
    np.testing.assert_allclose(exported.replay(ep), arm.replay(ep), atol=1e-6)


def test_framework_configs_refuse_non_so101_locks(tmp_path):
    """Isaac Lab/ManiSkill/mjlab configs name the SO-101's joints; a lock on
    another model is refused with a reason instead of mislabelled, and its
    zip still ships the MJCF."""
    import io
    import zipfile

    from antares.export import export_zip
    from antares.export_targets import TargetExportError, render
    from tests.test_export import make_lock

    _arm("piper")
    lock = dataclasses.replace(make_lock(SimParams(kp=40.0)), robot="piper", name="piper_unit")
    with pytest.raises(TargetExportError, match="SO-101 model only"):
        render(lock, "isaaclab")
    names = zipfile.ZipFile(io.BytesIO(export_zip(lock, targets=["isaaclab"]))).namelist()
    assert "piper_unit/TARGETS-NOT-EXPORTED.txt" in names
    assert any(n.endswith(".xml") for n in names)
