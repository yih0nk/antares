"""Exporting a lock's fitted parameters for Isaac Lab, ManiSkill, mjlab and JSON."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from antares import __version__, cli, server
from antares.export import export_mjcf, export_zip, load_exported_arm, read_sidecar
from antares.export_targets import (
    DEFAULT_FPS,
    FORMATS,
    MJCF_TIMESTEP,
    TARGETS,
    TargetExport,
    TargetExportError,
    dataset_fps,
    export_targets,
    joint_limits,
    latency_info,
    mjlab_delay_steps,
    render,
    resolve_targets,
)
from antares.lock import RobotLock, pinned_params
from antares.sim import JOINTS, CommandMap

ROOT = Path(__file__).resolve().parent.parent
LOCK_DIR = ROOT / "locks"
SCALAR = "svla_so101_pickplace"  # has locks/<name>.dr.json
PER_JOINT = "svla_so101_pickplace_perjoint"  # no DR ranges
LOCKS = {n: RobotLock.load(LOCK_DIR / f"{n}.lock.json") for n in (SCALAR, PER_JOINT)}


def _lock(name: str) -> RobotLock:
    return LOCKS[name]


def _py_files(ex: TargetExport) -> dict[str, str]:
    return {k: v for k, v in ex.files.items() if k.endswith(".py")}


def _json_files(ex: TargetExport) -> dict[str, dict]:
    return {k: json.loads(v) for k, v in ex.files.items() if k.endswith(".json")}


def _expected_damping(lock: RobotLock):
    p = lock.params
    if p.is_per_joint("kv") or p.is_per_joint("damping"):
        return dict(zip(JOINTS, (p.per_joint("kv") + p.per_joint("damping")).tolist()))
    return p.kv + p.damping


def _as_mapped(lock: RobotLock, name: str):
    v = getattr(lock.params, name)
    return dict(zip(JOINTS, v)) if isinstance(v, tuple) else v


# -- rendering ----------------------------------------------------------------


@pytest.mark.parametrize("name", [SCALAR, PER_JOINT])
@pytest.mark.parametrize("target", TARGETS)
def test_every_target_renders_for_the_demo_locks(name, target):
    lock = _lock(name)
    ex = render(lock, target, lock_dir=LOCK_DIR)
    assert ex.target == target and ex.files and ex.notes
    assert ex.main == next(iter(ex.files))
    assert all(lock.name in rel for rel in ex.files)
    for rel, text in ex.files.items():
        assert text.strip(), rel
        # header facts in every file
        for s in (__version__, lock.name, lock.dataset, lock.created_at, "pinned params", "N*m/rad"):
            assert s in text, (rel, s)
        assert "6.007" in text and ("0.736" in text or "0.656" in text)  # held-out before -> after
        if rel.endswith(".py"):
            compile(text, rel, "exec")
        else:
            json.loads(text)
    per_joint_text = "per-joint" if lock.params.has_per_joint else "scalar (shared"
    assert per_joint_text in ex.files[ex.main]


def test_python_snippets_import_without_the_simulators(tmp_path):
    """The generated modules execute (the sim imports are optional) and expose the constants."""
    import runpy

    for name in (SCALAR, PER_JOINT):
        lock = _lock(name)
        for t in ("isaaclab", "maniskill", "mjlab"):
            [path] = [p for p in render(lock, t, lock_dir=LOCK_DIR).write(tmp_path / name) if p.suffix == ".py"]
            ns = runpy.run_path(str(path))
            assert ns["SO101_JOINTS"] == list(JOINTS)
            assert ns["CONTROL_HZ"] == 30.0
            frames = lock.params.latency_frames
            assert ns["COMMAND_DELAY_STEPS"] == (tuple(frames) if isinstance(frames, tuple) else frames)
            assert ns["STIFFNESS"] == (
                dict(zip(JOINTS, lock.params.kp)) if lock.params.is_per_joint("kp") and t != "maniskill"
                else list(lock.params.kp) if lock.params.is_per_joint("kp") else lock.params.kp
            )


@pytest.mark.parametrize("name", [SCALAR, PER_JOINT])
def test_isaaclab_json_numbers_equal_the_lock(name):
    lock = _lock(name)
    ex = render(lock, "isaaclab", lock_dir=LOCK_DIR)
    assert set(ex.files) == {f"{name}_isaaclab.py", f"{name}_isaaclab.json"}
    data = _json_files(ex)[f"{name}_isaaclab.json"]
    act = data["actuator"]
    assert act["class"] == "isaaclab.actuators.ImplicitActuatorCfg"
    assert act["joint_names_expr"] == list(JOINTS)
    assert act["stiffness"] == _as_mapped(lock, "kp")
    assert act["damping"] == pytest.approx(_expected_damping(lock))
    assert act["armature"] == _as_mapped(lock, "armature")
    assert act["friction"] == _as_mapped(lock, "frictionloss")
    assert act["effort_limit_sim"] == _as_mapped(lock, "forcerange")
    assert act["velocity_limit_sim"] is None
    assert data["mujoco_params"] == lock.params.to_dict()
    assert data["provenance"]["pinned_params"] == pinned_params(lock) == ["latency_frames"]
    assert data["provenance"]["per_joint"] is lock.params.has_per_joint
    assert "friction" in data["units"] and "4.5" in data["units"]["friction"]
    py = _py_files(ex)[f"{name}_isaaclab.py"]
    for s in ("SO101_ACTUATOR_CFG = ImplicitActuatorCfg(", "SO101_ARTICULATION_CFG = ArticulationCfg(",
              "COMMAND_DELAY_STEPS", "CONTROL_HZ = 30.0", "SO101_USD_PATH", "effort_limit_sim=EFFORT_LIMIT"):
        assert s in py, s
    if lock.params.has_per_joint:
        assert "'elbow_flex': %r" % lock.params.kp[2] in py
    else:
        assert f"STIFFNESS = {lock.params.kp!r}" in py


@pytest.mark.parametrize("name", [SCALAR, PER_JOINT])
def test_maniskill_json_numbers_equal_the_lock(name):
    lock = _lock(name)
    ex = render(lock, "maniskill", lock_dir=LOCK_DIR)
    data = _json_files(ex)[f"{name}_maniskill.json"]
    c = data["controller"]
    assert c["class"].endswith("PDJointPosControllerConfig")
    assert c["joint_names"] == list(JOINTS)
    lo, hi = joint_limits()
    assert c["lower"] == lo and c["upper"] == hi
    assert lo[0] == pytest.approx(-1.91986) and hi[5] == pytest.approx(1.7453292)
    assert c["stiffness"] == _as_mapped(lock, "kp")
    assert c["damping"] == pytest.approx(_expected_damping(lock))
    assert c["force_limit"] == _as_mapped(lock, "forcerange")
    assert c["friction"] == 0.0 and c["normalize_action"] is False
    assert data["joint_frictionloss_nm"] == _as_mapped(lock, "frictionloss")
    assert data["joint_armature"] == _as_mapped(lock, "armature")
    assert data["delta_controller"]["lower"] == [-0.05] * 5 + [-0.2]
    py = _py_files(ex)[f"{name}_maniskill.py"]
    assert "pd_joint_pos = PDJointPosControllerConfig(" in py and "use_delta=True" in py
    assert "FRICTION = 0.0" in py and "JOINT_FRICTIONLOSS_NM" in py
    if lock.params.has_per_joint:
        assert f"STIFFNESS = {list(lock.params.kp)!r}" in py
    else:
        assert f"STIFFNESS = {lock.params.kp!r}" in py


@pytest.mark.parametrize("name", [SCALAR, PER_JOINT])
def test_mjlab_snippet_loads_the_exported_mjcf(name):
    lock = _lock(name)
    ex = render(lock, "mjlab", lock_dir=LOCK_DIR)
    assert list(ex.files) == [f"{name}_mjlab.py"]
    py = ex.files[ex.main]
    assert f"with_name({name + '.xml'!r})" in py
    assert "mujoco.MjSpec.from_file" in py and "XmlActuatorCfg(" in py and "EntityCfg(spec_fn=get_spec" in py
    lo, hi = mjlab_delay_steps(latency_info(lock.params, 30.0))
    assert f"COMMAND_DELAY_PHYSICS_STEPS = ({lo}, {hi})" in py
    frames = np.asarray(lock.params.per_joint("latency_frames"))
    assert (lo, hi) == (round(frames.min() / 30 / MJCF_TIMESTEP), round(frames.max() / 30 / MJCF_TIMESTEP))
    assert f"PHYSICS_DT = {MJCF_TIMESTEP!r}" in py


def test_json_target_reuses_the_sidecar():
    lock = _lock(SCALAR)
    ex = render(lock, "json", lock_dir=LOCK_DIR)
    assert list(ex.files) == [f"{SCALAR}.params.json"]
    data = json.loads(ex.files[ex.main])
    # sidecar fields
    assert data["format"] == "antares.export/1" and data["model"] == f"{SCALAR}.xml"
    assert data["params"] == lock.params.to_dict()
    assert data["latency_frames"] == lock.params.latency_frames
    assert "scale" in data["unit_map"] and data["joints"] == list(JOINTS)
    assert CommandMap.from_dict(data["command_map"]).is_identity()
    # target block
    t = data["targets"]
    assert t["isaaclab"]["stiffness"] == lock.params.kp
    assert t["isaaclab"]["damping"] == pytest.approx(lock.params.kv + lock.params.damping)
    assert t["maniskill"]["force_limit"] == lock.params.forcerange
    assert t["mjlab"]["mjcf"] == f"{SCALAR}.xml"
    assert data["pinned_params"] == ["latency_frames"] and data["param_spread"] is None
    assert data["latency"]["seconds"] == pytest.approx(lock.params.latency_frames / 30)
    assert data["dr"] is not None and data["header"][0].startswith("Antares calibrated")


# -- latency ------------------------------------------------------------------


def test_latency_seconds_are_frames_over_fps():
    scalar = _lock(SCALAR)
    lat = latency_info(scalar.params, 50.0)
    assert lat == {
        **lat,
        "control_hz": 50.0,
        "frames": scalar.params.latency_frames,
        "seconds": scalar.params.latency_frames / 50.0,
        "frames_min": scalar.params.latency_frames,
        "frames_max": scalar.params.latency_frames,
        "per_joint": False,
    }
    assert "control steps" in lat["note"]
    pj = _lock(PER_JOINT)
    lat = latency_info(pj.params, 30.0)
    assert lat["per_joint"] and lat["frames"] == list(pj.params.latency_frames)
    assert lat["seconds"] == pytest.approx([f / 30.0 for f in pj.params.latency_frames])
    assert (lat["frames_min"], lat["frames_max"]) == (min(pj.params.latency_frames), max(pj.params.latency_frames))

    # the fps flows into every target
    for t in ("isaaclab", "maniskill"):
        ex = render(scalar, t, fps=60.0, lock_dir=LOCK_DIR)
        py = ex.files[ex.main]
        assert "CONTROL_HZ = 60.0" in py
        assert f"COMMAND_DELAY_S = {scalar.params.latency_frames / 60.0!r}" in py
        [data] = _json_files(ex).values()
        assert data["latency"]["seconds"] == pytest.approx(scalar.params.latency_frames / 60.0)
        assert data["provenance"]["fps_source"] == "given"


def test_dataset_fps_sources():
    lock = _lock(SCALAR)
    assert dataset_fps(lock, 25.0) == (25.0, "given")
    hz, src = dataset_fps(lock)  # local HF cache if present, else the default
    assert hz == 30.0 and ("local cache" in src or "assumed" in src)
    ghost = replace(lock, dataset="nobody/never-downloaded-dataset-xyz")
    assert dataset_fps(ghost) == (DEFAULT_FPS, "assumed (dataset not in the local cache)")
    assert dataset_fps(ghost, default=20.0)[0] == 20.0
    with pytest.raises(TargetExportError, match="fps"):
        dataset_fps(lock, 0.0)


# -- DR ranges and pinned params -----------------------------------------------


def test_dr_ranges_are_emitted_when_the_dr_json_exists(tmp_path):
    from antares.dr import DRRanges

    lock = _lock(SCALAR)
    dr = DRRanges.load(LOCK_DIR / f"{SCALAR}.dr.json")
    ex = render(lock, "isaaclab", lock_dir=LOCK_DIR)
    py = ex.files[ex.main]
    data = _json_files(ex)[f"{SCALAR}_isaaclab.json"]
    assert "mdp.randomize_actuator_gains" in py and "mdp.randomize_joint_parameters" in py
    assert '"distribution": "log_uniform"' in py and "Pinned parameters\n# (latency_frames)" in py
    m = data["dr"]["mapped"]
    assert m["stiffness"] == [dr["kp"].lo, dr["kp"].hi]
    assert m["damping"] == pytest.approx([dr["kv"].lo + dr["damping"].lo, dr["kv"].hi + dr["damping"].hi])
    assert m["friction"] == [dr["frictionloss"].lo, dr["frictionloss"].hi]
    assert m["armature"] == [dr["armature"].lo, dr["armature"].hi]
    assert m["latency_frames"] == [3, 4]
    assert data["dr"]["mujoco"]["kp"] == {"lo": dr["kp"].lo, "hi": dr["kp"].hi}
    assert f'"stiffness_distribution_params": ({dr["kp"].lo!r}, {dr["kp"].hi!r})' in py
    # the committed ranges were rederived from the current lock: no staleness note
    assert data["dr"]["nominal_matches_lock"] is True and "rerun `antares dr`" not in py

    # ranges derived around a different fit of the lock are flagged and the
    # export says to rerun `antares dr`
    import json
    import shutil

    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    shutil.copy(LOCK_DIR / f"{SCALAR}.lock.json", stale_dir / f"{SCALAR}.lock.json")
    stale = json.loads((LOCK_DIR / f"{SCALAR}.dr.json").read_text())
    stale["nominal"]["kp"] = stale["nominal"]["kp"] * 9.0
    (stale_dir / f"{SCALAR}.dr.json").write_text(json.dumps(stale))
    ex2 = render(lock, "isaaclab", lock_dir=stale_dir)
    data2 = _json_files(ex2)[f"{SCALAR}_isaaclab.json"]
    assert data2["dr"]["nominal_matches_lock"] is False and "rerun `antares dr`" in ex2.files[ex2.main]
    assert "DR_RANGES = {" in py
    # every other target carries the same ranges
    for t in ("maniskill", "json"):
        [d] = _json_files(render(lock, t, lock_dir=LOCK_DIR)).values()
        assert d["dr"]["mapped"] == m
    assert "DR_RANGES = {" in render(lock, "mjlab", lock_dir=LOCK_DIR).files[f"{SCALAR}_mjlab.py"]

    # a lock whose .dr.json is missing (or another lock dir) gets none
    for lk, ld in ((_lock(PER_JOINT), LOCK_DIR), (lock, tmp_path)):
        ex = render(lk, "isaaclab", lock_dir=ld)
        py = ex.files[ex.main]
        assert "randomize_actuator_gains" not in py and "DR_RANGES = None" in py
        [d] = _json_files(ex).values()
        assert d["dr"] is None
        assert any("no DR ranges" in n for n in ex.notes)

    # explicit ranges win over the file lookup
    ex = render(_lock(PER_JOINT), "json", dr=dr, lock_dir=tmp_path)
    assert json.loads(ex.files[ex.main])["dr"]["mapped"] == m


def test_command_map_is_noted_in_every_target():
    cmap = CommandMap(gain=[1.0, 0.97, 1.03, 1.0, 1.0, 1.0], bias=[0.0, -3.0, 2.0, 0.0, 1.0, 0.0])
    lock = replace(_lock(SCALAR), command_map=cmap.to_dict())
    for t in TARGETS:
        ex = render(lock, t, lock_dir=LOCK_DIR)
        for rel, text in ex.files.items():
            assert "command map" in text or "command_map" in text, rel
        for d in _json_files(ex).values():
            assert d["command_map"]["gain"] == cmap.gain.tolist()
            assert d["command_map"]["bias"] == cmap.bias.tolist()
            assert d["command_map"]["identity"] is False
    plain = render(_lock(SCALAR), "isaaclab", lock_dir=LOCK_DIR)
    assert "NOTE: this lock carries a fitted command map" not in plain.files[plain.main]
    assert _json_files(plain)[f"{SCALAR}_isaaclab.json"]["command_map"]["identity"] is True


# -- helpers ------------------------------------------------------------------


def test_resolve_targets_and_unknown_target():
    assert FORMATS == ("mjcf",) + TARGETS + ("all",)
    assert resolve_targets("all") == list(TARGETS)
    assert resolve_targets("mjcf") == []
    assert resolve_targets("maniskill") == ["maniskill"]
    with pytest.raises(TargetExportError, match="unknown export format"):
        resolve_targets("urdf")
    with pytest.raises(TargetExportError, match="unknown export target"):
        render(_lock(SCALAR), "urdf")


def test_export_targets_writes_files(tmp_path):
    paths = export_targets(_lock(PER_JOINT), tmp_path / "out", TARGETS, lock_dir=LOCK_DIR)
    assert sorted(p.name for p in paths) == sorted(
        [f"{PER_JOINT}_isaaclab.py", f"{PER_JOINT}_isaaclab.json", f"{PER_JOINT}_maniskill.py",
         f"{PER_JOINT}_maniskill.json", f"{PER_JOINT}_mjlab.py", f"{PER_JOINT}.params.json"]
    )
    assert all(p.is_file() and p.parent == tmp_path / "out" for p in paths)


def test_mjcf_export_of_a_per_joint_lock_round_trips(tmp_path):
    """Regression: the MJCF header and sidecar used to assume scalar params."""
    lock = _lock(PER_JOINT)
    out = export_mjcf(lock, tmp_path / "a.xml", bundle_assets=False)
    head = out.read_text().split("<option", 1)[0]
    assert "kp=[65.50882844603456 " in head and "Base model: MuJoCo Menagerie" in head
    assert read_sidecar(out)["latency_frames"] == list(lock.params.latency_frames)
    assert load_exported_arm(out).params == lock.params


# -- CLI ----------------------------------------------------------------------


def _install_locks(monkeypatch, tmp_path, names=(SCALAR, PER_JOINT)) -> Path:
    import shutil

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    for n in names:
        shutil.copy(LOCK_DIR / f"{n}.lock.json", lock_dir)
    shutil.copy(LOCK_DIR / f"{SCALAR}.dr.json", lock_dir)
    monkeypatch.setattr(server, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(server, "LOCK_SUFFIX", ".lock.json")
    return lock_dir


def test_cli_format_all_writes_every_file(monkeypatch, tmp_path, capsys):
    _install_locks(monkeypatch, tmp_path)
    out = tmp_path / "out"
    assert cli.main(["export", SCALAR, "-o", str(out), "--no-bundle", "--format", "all"]) == 0
    text = capsys.readouterr().out
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted([
        f"{SCALAR}.xml", f"{SCALAR}.antares.json", "scene.xml",
        f"{SCALAR}_isaaclab.py", f"{SCALAR}_isaaclab.json", f"{SCALAR}_maniskill.py",
        f"{SCALAR}_maniskill.json", f"{SCALAR}_mjlab.py", f"{SCALAR}.params.json",
    ])
    for n in names:
        assert n in text
    assert "Isaac Lab" in text and "ManiSkill" in text and "mjlab" in text and "latency: 4 frame(s) at 30 Hz" in text
    assert "randomize_actuator_gains" in (out / f"{SCALAR}_isaaclab.py").read_text()  # dr.json found via LOCK_DIR
    compile((out / f"{SCALAR}_isaaclab.py").read_text(), "x", "exec")

    # one target only, with an explicit fps
    out2 = tmp_path / "out2"
    assert cli.main(["export", PER_JOINT, "-o", str(out2), "--no-bundle", "--format", "maniskill", "--fps", "25"]) == 0
    text = capsys.readouterr().out
    assert sorted(p.name for p in out2.iterdir()) == sorted(
        [f"{PER_JOINT}.xml", f"{PER_JOINT}.antares.json", "scene.xml", f"{PER_JOINT}_maniskill.py", f"{PER_JOINT}_maniskill.json"]
    )
    assert "25 Hz (fps given)" in text
    assert "CONTROL_HZ = 25.0" in (out2 / f"{PER_JOINT}_maniskill.py").read_text()

    # default format is unchanged: MJCF only
    out3 = tmp_path / "out3"
    assert cli.main(["export", SCALAR, "-o", str(out3), "--no-bundle"]) == 0
    assert sorted(p.name for p in out3.iterdir()) == sorted([f"{SCALAR}.xml", f"{SCALAR}.antares.json", "scene.xml"])
    assert "latency:" not in capsys.readouterr().out


def test_cli_rejects_unknown_format(monkeypatch, tmp_path, capsys):
    _install_locks(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.main(["export", SCALAR, "--format", "urdf"])
    assert e.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


# -- API ----------------------------------------------------------------------


def test_api_export_format_route(monkeypatch, tmp_path):
    _install_locks(monkeypatch, tmp_path)
    client = TestClient(server.app)

    r = client.get(f"/api/locks/{SCALAR}/export")  # unchanged default
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/xml")
    assert r.headers["content-disposition"] == f'attachment; filename="{SCALAR}.xml"'

    r = client.get(f"/api/locks/{SCALAR}/export", params={"format": "isaaclab"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/x-python")
    assert r.headers["content-disposition"] == f'attachment; filename="{SCALAR}_isaaclab.py"'
    assert "SO101_ACTUATOR_CFG = ImplicitActuatorCfg(" in r.text and "randomize_actuator_gains" in r.text
    compile(r.text, "x", "exec")

    r = client.get(f"/api/locks/{PER_JOINT}/export", params={"format": "maniskill", "fps": 25})
    assert r.status_code == 200 and "CONTROL_HZ = 25.0" in r.text
    assert "DR_RANGES = None" in r.text

    r = client.get(f"/api/locks/{SCALAR}/export", params={"format": "mjlab"})
    assert r.status_code == 200 and "XmlActuatorCfg(" in r.text

    r = client.get(f"/api/locks/{SCALAR}/export", params={"format": "json"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.headers["content-disposition"] == f'attachment; filename="{SCALAR}.params.json"'
    data = r.json()
    assert data["params"] == _lock(SCALAR).params.to_dict() and data["targets"]["isaaclab"]["stiffness"] == _lock(SCALAR).params.kp

    assert client.get(f"/api/locks/{SCALAR}/export", params={"format": "urdf"}).status_code == 422
    assert client.get(f"/api/locks/{SCALAR}/export", params={"format": "all"}).status_code == 422
    assert client.get(f"/api/locks/{SCALAR}/export", params={"format": "json", "fps": 0}).status_code == 422
    assert client.get("/api/locks/ghost/export", params={"format": "isaaclab"}).status_code == 404


def test_api_zip_includes_targets_on_request(monkeypatch, tmp_path):
    _install_locks(monkeypatch, tmp_path)
    client = TestClient(server.app)
    base = {f"{SCALAR}/{SCALAR}.xml", f"{SCALAR}/{SCALAR}.antares.json", f"{SCALAR}/scene.xml"}
    extra = {
        f"{SCALAR}/{SCALAR}_isaaclab.py", f"{SCALAR}/{SCALAR}_isaaclab.json", f"{SCALAR}/{SCALAR}_maniskill.py",
        f"{SCALAR}/{SCALAR}_maniskill.json", f"{SCALAR}/{SCALAR}_mjlab.py", f"{SCALAR}/{SCALAR}.params.json",
    }

    r = client.get(f"/api/locks/{SCALAR}/export.zip")
    assert r.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert base <= names and not (extra & names)

    r = client.get(f"/api/locks/{SCALAR}/export.zip", params={"targets": "all"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert base | extra <= names
    assert sum(n.endswith(".stl") for n in names) == 18
    py = zf.read(f"{SCALAR}/{SCALAR}_mjlab.py").decode()
    assert f"with_name({SCALAR + '.xml'!r})" in py  # the MJCF it loads sits next to it in the zip
    assert "randomize_actuator_gains" in zf.read(f"{SCALAR}/{SCALAR}_isaaclab.py").decode()

    r = client.get(f"/api/locks/{SCALAR}/export.zip", params={"targets": "mjlab,json"})
    names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert names & extra == {f"{SCALAR}/{SCALAR}_mjlab.py", f"{SCALAR}/{SCALAR}.params.json"}

    assert client.get(f"/api/locks/{SCALAR}/export.zip", params={"targets": "urdf"}).status_code == 400

    # the library function behind the route
    names = set(zipfile.ZipFile(io.BytesIO(export_zip(_lock(PER_JOINT), targets=["json"]))).namelist())
    assert f"{PER_JOINT}/{PER_JOINT}.params.json" in names
