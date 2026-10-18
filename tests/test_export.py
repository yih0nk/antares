"""Exporting a lock as a standalone MJCF (+ sidecar, scene, zip)."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import pytest

from antares.calibrate import CalibrationResult
from antares.data import Episode
from antares.export import (
    ASSETS_DIRNAME,
    BAKED_PARAMS,
    EXPORT_FORMAT,
    SCENE_FILENAME,
    SIDECAR_SUFFIX,
    ExportError,
    bake_params,
    export_mjcf,
    export_scene,
    export_zip,
    load_exported_arm,
    read_sidecar,
    referenced_meshes,
    set_meshdir,
    sidecar_path,
)
from antares.lock import RobotLock
from antares.sim import JOINTS, MESH_DIR, MODEL_PATH, Arm, SimParams, UnitMap

STOCK_XML = MODEL_PATH.read_text()
LOCK_FILE = Path(__file__).resolve().parent.parent / "locks" / "svla_so101_pickplace.lock.json"

FITTED = SimParams(
    kp=93.70021916179945,
    kv=1.5718591906130155,
    damping=0.052719117670612985,
    armature=0.019703166866811193,
    frictionloss=0.025181795953895247,
    latency_frames=4,
)


def make_lock(params: SimParams = FITTED, name: str = "unit_a", **over) -> RobotLock:
    kw = dict(
        params=params,
        baseline=SimParams(),
        train_score=1.06,
        eval_score=1.054,
        baseline_train_score=6.2,
        baseline_eval_score=6.007,
        per_joint_before={j: 5.0 for j in JOINTS},
        per_joint_after={j: 1.0 for j in JOINTS},
        history=[(0, 6.2), (1, 1.06)],
        n_evaluations=1,
        elapsed_s=1.0,
        train_episodes=[1, 2],
        eval_episodes=[0, 3, 4],
    )
    kw.update(over)
    result = CalibrationResult(**kw)
    return RobotLock(
        name=name,
        dataset="lerobot/svla_so101_pickplace",
        robot_type="so101_follower",
        data_sha256="ab" * 32,
        params=params,
        result=result,
        created_at="2026-10-17T20:35:05Z",
        antares_version="0.0.1",
    )


def synthetic_episode(T: int = 90, seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed)
    action = np.cumsum(rng.normal(0, 1.5, size=(T, 6)), axis=0)
    action[:, 5] = np.abs(action[:, 5])
    state = action + rng.normal(0, 0.5, size=(T, 6))
    state[:, 5] = np.abs(state[:, 5])
    return Episode(index=0, fps=30.0, joint_names=JOINTS, state=state, action=action)


def servo_block(xml: str) -> str:
    return re.search(r'<default class="sts3215">.*?</default>', xml, re.DOTALL).group(0)


def load_with_meshes(tmp_path: Path, xml: str, name: str = "m.xml") -> mujoco.MjModel:
    p = tmp_path / name
    p.write_text(set_meshdir(xml, MESH_DIR))
    return mujoco.MjModel.from_xml_path(str(p))


# -- text editing ------------------------------------------------------------


def test_bake_scalars_only_touches_the_servo_class(tmp_path):
    out = bake_params(STOCK_XML, FITTED.to_dict())
    blk = servo_block(out)
    assert 'damping="0.052719117670612985"' in blk
    assert 'frictionloss="0.025181795953895247"' in blk
    assert 'armature="0.019703166866811193"' in blk
    assert 'kp="93.70021916179945" kv="1.5718591906130155" forcerange="-2.94 2.94"' in blk
    # everything outside the two edited lines is byte-identical
    changed = [(a, b) for a, b in zip(STOCK_XML.splitlines(), out.splitlines()) if a != b]
    assert len(changed) == 2
    assert all("<joint" in a or "<position" in a for a, _ in changed)
    assert "latency" not in out.lower()  # not an MJCF concept; stays out of the model

    m = load_with_meshes(tmp_path, out)
    assert SimParams.from_model(m) == FITTED.replace(latency_frames=0)
    np.testing.assert_array_equal(m.dof_damping, FITTED.damping)
    np.testing.assert_array_equal(m.actuator_gainprm[:, 0], FITTED.kp)
    np.testing.assert_array_equal(m.actuator_biasprm[:, 1], -FITTED.kp)
    np.testing.assert_array_equal(m.actuator_biasprm[:, 2], -FITTED.kv)


def test_bake_per_joint_values_go_on_each_element(tmp_path):
    kp = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    damping = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    out = bake_params(STOCK_XML, {"kp": kp, "damping": np.array(damping), "kv": 2.0})
    # the class keeps the stock joint damping (per-joint values override it per element)
    assert 'damping="0.60"' in servo_block(out)
    assert 'kv="2.0"' in servo_block(out)
    for j, k, d in zip(JOINTS, kp, damping):
        assert re.search(r'<joint\b[^>]*name="%s"[^>]*damping="%s"' % (j, d), out), j
        assert re.search(r'<position\b[^>]*name="%s"[^>]*kp="%s"' % (j, k), out), j
    m = load_with_meshes(tmp_path, out)
    np.testing.assert_array_equal(m.dof_damping, damping)
    np.testing.assert_array_equal(m.actuator_gainprm[:, 0], kp)
    np.testing.assert_array_equal(m.actuator_biasprm[:, 1], [-v for v in kp])
    np.testing.assert_array_equal(m.actuator_biasprm[:, 2], -2.0)
    np.testing.assert_array_equal(m.dof_armature, 0.028)  # untouched stock value


def test_bake_rejects_bad_input():
    with pytest.raises(ExportError, match="per-joint"):
        bake_params(STOCK_XML, {"kp": [1.0, 2.0]})
    with pytest.raises(ExportError, match="non-finite"):
        bake_params(STOCK_XML, {"kp": float("nan")})
    with pytest.raises(ExportError, match="sts3215"):
        bake_params(STOCK_XML.replace('class="sts3215">', 'class="other">'), {"kp": 1.0})
    assert bake_params(STOCK_XML, {"latency_frames": 4, "bogus": 1}) == STOCK_XML


def test_set_meshdir_and_referenced_meshes():
    out = set_meshdir(STOCK_XML, "/abs/dir")
    assert '<compiler angle="radian" meshdir="/abs/dir" autolimits="true"/>' in out
    # adding an attribute that is not there yet
    out = set_meshdir(STOCK_XML.replace(' meshdir="assets"', ""), "x")
    assert re.search(r'<compiler angle="radian" autolimits="true" meshdir="x"/>', out)
    meshes = referenced_meshes(STOCK_XML)
    assert len(meshes) == len(set(meshes)) == 18
    assert all(m.endswith(".stl") and (MESH_DIR / m).is_file() for m in meshes)


# -- export_mjcf ---------------------------------------------------------------


def test_export_mjcf_bundled(tmp_path):
    lock = make_lock()
    out = export_mjcf(lock, tmp_path / "arm.xml")
    assert out == tmp_path / "arm.xml" and out.is_file()
    xml = out.read_text()

    # header with the lock facts, right inside the root element
    head = xml.split("<option", 1)[0]
    assert head.startswith('<mujoco model="so101">\n  <!--')
    for s in ("unit_a", "lerobot/svla_so101_pickplace", "2026-10-17T20:35:05Z", "6.007", "1.054", "latency_frames: 4",
              "gravity_scale=1.0", "Base model: MuJoCo Menagerie"):
        assert s in head, s
    assert "--" not in head.split("<!--", 1)[1].rsplit("-->", 1)[0]

    # meshes bundled, meshdir relative, model loads from its new home
    assert 'meshdir="assets"' in xml
    assets = tmp_path / ASSETS_DIRNAME
    assert sorted(p.name for p in assets.iterdir()) == sorted(referenced_meshes(xml))
    m = mujoco.MjModel.from_xml_path(str(out))
    assert SimParams.from_model(m) == FITTED.replace(latency_frames=0)
    assert m.nq == 6 and m.nu == 6

    # sidecar
    sc_path = sidecar_path(out)
    assert sc_path == tmp_path / f"arm{SIDECAR_SUFFIX}" and sc_path.is_file()
    sc = read_sidecar(out)
    assert sc["format"] == EXPORT_FORMAT
    assert sc["model"] == "arm.xml" and sc["meshdir"] == "assets"
    assert sc["params"] == FITTED.to_dict()
    assert sc["latency_frames"] == 4
    assert sc["baked_params"] == list(BAKED_PARAMS)
    assert sc["lock"]["name"] == "unit_a" and sc["lock"]["dataset"] == lock.dataset
    assert sc["lock"]["created_at"] == lock.created_at
    assert sc["joints"] == list(JOINTS)
    um = UnitMap.from_dict(sc["unit_map"])
    ref = UnitMap.from_model(m)
    np.testing.assert_allclose(um.scale, ref.scale)
    np.testing.assert_allclose(um.offset, ref.offset)
    np.testing.assert_allclose(um.lo, ref.lo)
    np.testing.assert_allclose(um.hi, ref.hi)
    assert sc["scores"]["baseline_eval_score"] == 6.007 and sc["scores"]["eval_score"] == 1.054
    assert sc["scores"]["eval_episodes"] == [0, 3, 4]
    json.dumps(sc)


def test_export_mjcf_unbundled_points_at_package_assets(tmp_path):
    out = export_mjcf(make_lock(), tmp_path / "arm.xml", bundle_assets=False)
    xml = out.read_text()
    assert f'meshdir="{MESH_DIR.resolve()}"' in xml
    assert not (tmp_path / ASSETS_DIRNAME).exists()
    mujoco.MjModel.from_xml_path(str(out))
    assert read_sidecar(out)["meshdir"] == str(MESH_DIR.resolve())


def test_export_mjcf_directory_argument_uses_lock_name(tmp_path):
    out = export_mjcf(make_lock(name="robot_7"), tmp_path / "exports", bundle_assets=False)
    assert out == tmp_path / "exports" / "robot_7.xml"
    assert out.is_file() and sidecar_path(out).is_file()


def test_export_nan_scores_are_null(tmp_path):
    lock = make_lock(eval_score=float("nan"), baseline_eval_score=float("inf"))
    out = export_mjcf(lock, tmp_path / "a.xml", bundle_assets=False)
    assert "n/a before -> n/a after" in out.read_text()
    sc = read_sidecar(out)
    assert sc["scores"]["eval_score"] is None and sc["scores"]["baseline_eval_score"] is None


# -- export_scene / export_zip ------------------------------------------------


def test_export_scene(tmp_path):
    lock = make_lock()
    scene = export_scene(lock, tmp_path / "out")
    assert scene == tmp_path / "out" / SCENE_FILENAME
    xml = scene.read_text()
    assert '<include file="unit_a.xml"/>' in xml
    assert 'name="floor"' in xml and "<light" in xml
    assert (tmp_path / "out" / "unit_a.xml").is_file()
    assert (tmp_path / "out" / f"unit_a{SIDECAR_SUFFIX}").is_file()
    m = mujoco.MjModel.from_xml_path(str(scene))
    assert m.nq == 6 and m.nu == 6
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    assert m.nlight == 1
    assert SimParams.from_model(m) == FITTED.replace(latency_frames=0)


def test_export_zip(tmp_path):
    data = export_zip(make_lock())
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    assert "unit_a/unit_a.xml" in names
    assert f"unit_a/unit_a{SIDECAR_SUFFIX}" in names
    assert f"unit_a/{SCENE_FILENAME}" in names
    stl = sorted(n for n in names if n.endswith(".stl"))
    assert len(stl) == 18 and all(n.startswith("unit_a/assets/") for n in stl)
    assert zf.testzip() is None
    zf.extractall(tmp_path)
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "unit_a" / SCENE_FILENAME))
    assert SimParams.from_model(m) == FITTED.replace(latency_frames=0)


# -- replaying through the export ---------------------------------------------


def test_load_exported_arm_reads_latency_from_sidecar(tmp_path):
    out = export_mjcf(make_lock(), tmp_path / "a.xml", bundle_assets=False)
    arm = load_exported_arm(out)
    assert arm.params == FITTED
    assert arm.model_path == out
    sidecar_path(out).unlink()
    assert load_exported_arm(out).params == FITTED.replace(latency_frames=0)


def test_exported_model_replays_identically_synthetic(tmp_path):
    lock = make_lock()
    out = export_mjcf(lock, tmp_path / "a.xml")
    ep = synthetic_episode()
    ref = Arm(params=lock.params).replay(ep)
    got = load_exported_arm(out).replay(ep)
    assert got.shape == ref.shape == (len(ep), 6)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)
    # and it is not a trivially equal trajectory: the stock model differs
    assert not np.allclose(Arm().replay(ep), ref, atol=1e-3)


@pytest.mark.slow
def test_exported_model_replays_identically_real_dataset(tmp_path):
    """Episode 0 of the lock's dataset, stock Arm with the lock's params vs an
    Arm built from the exported XML: bit-for-bit the same physics."""
    from antares.data import load_dataset

    lock = RobotLock.load(LOCK_FILE)
    ds = load_dataset(lock.dataset)
    ep = next(e for e in ds.episodes if e.index == 0)
    out = export_mjcf(lock, tmp_path / f"{lock.name}.xml")

    ref = Arm(params=lock.params).replay(ep)
    arm = load_exported_arm(out)
    assert arm.params == lock.params
    got = arm.replay(ep)
    assert got.shape == ref.shape == (len(ep), 6)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)


def test_gravity_scale_is_baked_into_option_and_replays_identically(tmp_path):
    xml = MODEL_PATH.read_text()
    assert bake_params(xml, {"kp": 50.0}).count("gravity=") == xml.count("gravity=")
    baked = bake_params(xml, {"gravity_scale": 0.4})
    assert re.search(r'<option\b[^>]*gravity="0 0 -3\.924', baked)
    with pytest.raises(ExportError):
        bake_params(xml, {"gravity_scale": [0.4, 1, 1, 1, 1, 1]})

    lock = make_lock(params=FITTED.replace(gravity_scale=0.4))
    out = export_mjcf(lock, tmp_path / "g.xml")
    ep = synthetic_episode()
    arm = load_exported_arm(out)
    assert arm.params.gravity_scale == pytest.approx(0.4)
    assert "gravity_scale=0.4 (scales option gravity)" in out.read_text().split("<option", 1)[0]
    assert "gravity_scale" in read_sidecar(out)["baked_params"]
    ref = Arm(params=lock.params).replay(ep)
    np.testing.assert_allclose(arm.replay(ep), ref, rtol=0, atol=1e-9)
    assert not np.allclose(Arm(params=FITTED).replay(ep), ref, atol=1e-4)


def test_deadband_travels_in_the_sidecar(tmp_path):
    lock = make_lock(params=FITTED.replace(deadband=[0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))
    out = export_mjcf(lock, tmp_path / "d.xml")
    arm = load_exported_arm(out)
    assert arm.params.per_joint("deadband").tolist() == [0.0, 0.0, 0.01, 0.0, 0.0, 0.0]
    sc = read_sidecar(out)
    assert "deadband" in sc["sidecar_params"] and "deadband" not in sc["baked_params"]
    assert sc["params"]["deadband"] == [0.0, 0.0, 0.01, 0.0, 0.0, 0.0] and "deadband_note" in sc
    head = out.read_text().split("<option", 1)[0]
    assert "deadband: 0.0 0.0 0.01 0.0 0.0 0.0 rad" in head
    assert "deadband" not in export_mjcf(make_lock(), tmp_path / "p.xml").read_text().split("<option", 1)[0]
    assert bake_params(STOCK_XML, {"deadband": 0.01, "latency_frames": 3}) == STOCK_XML
    ep = synthetic_episode()
    np.testing.assert_allclose(arm.replay(ep), Arm(params=lock.params).replay(ep), rtol=0, atol=1e-9)


def test_export_carries_the_command_map(tmp_path):
    from dataclasses import replace

    from antares.sim import CommandMap

    cmap = CommandMap(gain=[1.0, 0.97, 1.03, 1.0, 1.0, 1.0], bias=[0.0, -3.0, 2.0, 0.0, 1.0, 0.0])
    lock = replace(make_lock(), command_map=cmap.to_dict())
    out = export_mjcf(lock, tmp_path / "a.xml", bundle_assets=False)
    sc = read_sidecar(out)
    assert sc["command_map"]["bias"] == cmap.bias.tolist() and "note" in sc["command_map"]
    ep = synthetic_episode()
    ref = Arm(params=lock.params, command_map=cmap).replay(ep)
    np.testing.assert_allclose(load_exported_arm(out).replay(ep), ref, rtol=0, atol=1e-9)
    # a lock without one exports identity
    plain = read_sidecar(export_mjcf(make_lock(), tmp_path / "b.xml", bundle_assets=False))
    assert CommandMap.from_dict(plain["command_map"]).is_identity()


def test_export_scene_with_task(tmp_path):
    from test_task import pick_place_episode

    from antares.task import OBJECT_BODY, OBJECT_JOINT, PLACE_SITE, TaskDistribution, set_task

    td = TaskDistribution.from_episodes([pick_place_episode(0), pick_place_episode(1)], name="toy")
    scene = export_scene(make_lock(), tmp_path / "out", task=td, object_size=0.04)
    assert (tmp_path / "out" / "task.json").is_file()
    m = mujoco.MjModel.from_xml_path(str(scene))
    assert m.nq == 6 + 7  # arm first, then the cube's free joint
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, PLACE_SITE)
    assert body >= 0 and site >= 0
    np.testing.assert_allclose(m.body_pos[body][:2], td.picks.mean(axis=0)[:2], atol=1e-5)
    np.testing.assert_allclose(m.site_pos[site], td.places.mean(axis=0), atol=1e-5)

    d = mujoco.MjData(m)
    inst = {"pick": [0.25, 0.05, 0.0], "place": [0.2, -0.1, 0.02], "start_q": td.start_q[0].tolist()}
    set_task(m, d, inst, size=0.04)
    a = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT)]
    np.testing.assert_allclose(d.qpos[a : a + 3], [0.25, 0.05, 0.02])  # lifted to rest on the floor
    np.testing.assert_allclose(d.qpos[:6], td.start_q[0])
    np.testing.assert_allclose(m.site_pos[site], inst["place"])
    for _ in range(200):
        mujoco.mj_step(m, d)
    assert d.qpos[a + 2] == pytest.approx(0.02, abs=2e-3)  # sits on the floor, does not fall through


def test_scene_xml_needs_a_held_grasp():
    from test_task import SHUT, pick_place_episode

    from antares.task import TaskDistribution, scene_xml

    td = TaskDistribution.from_episodes([pick_place_episode(0, held_jaw=SHUT)], name="toy")
    with pytest.raises(ValueError):
        scene_xml(td)
