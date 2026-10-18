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
    assert main(["robots"]) == 0
    assert "aloha" in capsys.readouterr().out
    assert main(["datasets"]) == 0
    assert "lerobot/aloha_static_coffee" in capsys.readouterr().out


def test_cli_rejects_unknown_factor(capsys):
    from antares.cli import main

    assert main(["calibrate", "x/y", "--factors", "+nope"]) == 2
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
