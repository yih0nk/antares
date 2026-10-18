"""The objects routes, CLI and export, with the video / tracking work stubbed out."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from antares import cli, server
from antares.data import Dataset, Episode
from antares.objects import save_object_fit
from tests.test_objects_export import PAYLOAD_OK, PRIOR, _fit

pytest.importorskip("cv2")

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
T = 60


def _dataset(repo_id="fake/objects", n=3):
    eps = []
    for i in range(n):
        state = np.zeros((T, 6))
        state[:, 0] = np.linspace(-30, 30, T) + 5 * i
        action = state.copy()
        action[:, 5] = 90.0
        action[20:40, 5] = 5.0
        state[23:40, 5] = 20.0
        action[T - 10 :, 5] = 5.0
        state[T - 8 :, 5] = 2.0
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=state, action=action))
    return Dataset(repo_id=repo_id, robot_type="so101", fps=30.0, joint_names=JOINTS, episodes=tuple(eps))


@pytest.fixture
def client(tmp_path, monkeypatch):
    from antares.lock import RobotLock
    from tests.test_lock import fake_result

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "_TRACKS", {})
    server.put_dataset(_dataset())
    RobotLock.create("fake/objects", "so101", [], fake_result(), name="unit").save(tmp_path / "unit.lock.json")
    return TestClient(server.app)


def test_objects_get_is_null_before_a_fit(client):
    r = client.get("/api/objects", params={"lock": "unit"})
    assert r.status_code == 200 and r.json() == {"lock": "unit", "objects": None}
    assert client.get("/api/objects", params={"lock": "../x"}).status_code == 400
    r = client.get("/api/objects/track", params={"dataset": "fake/objects", "episode": 1, "lock": "unit"})
    assert r.status_code == 404 and "objects fit" in r.json()["detail"]


def test_objects_fit_needs_camera_and_scene(client):
    with pytest.raises(Exception) as e:
        server.run_objects_fit(server.get_dataset("fake/objects"), "unit")
    assert "scene fit" in str(getattr(e.value, "detail", e.value))


def test_saved_fit_is_served_with_its_tracks(client, tmp_path):
    fit, tracks = _fit(PAYLOAD_OK)
    save_object_fit(fit, "unit", tmp_path, tracks)
    r = client.get("/api/objects", params={"lock": "unit"})
    assert r.status_code == 200
    d = r.json()["objects"]
    assert d["description"].startswith("a red object") and len(d["tracks"]) == 3 and d["payload"]["mass_kg"] == 0.04
    lock = json.loads((tmp_path / "unit.lock.json").read_text())
    assert lock["objects"]["mass_kg"] == 0.04 and lock["objects"]["footprint_m"] == [0.038, 0.018]
    r = client.get("/api/objects/track", params={"dataset": "fake/objects", "episode": 1, "lock": "unit"})
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["camera"] == "up" and t["rgb"] == [200, 102, 97] and t["footprint_m"] == [0.038, 0.018]
    assert len(t["xyz"]) == T and t["source"][25] == "held" and t["source"][0] == "vision"
    assert t["bbox"][0] == [308, 380, 24, 40] and t["bbox"][25] is None and t["centroid"][0] == [320.0, 400.0]
    assert t["picks"] == [[20, 40]] and t["start_pose"] is not None
    assert client.get("/api/objects/track", params={"dataset": "fake/objects", "episode": 9, "lock": "unit"}).status_code == 404


def test_objects_fit_job_streams_progress_and_done(client, tmp_path, monkeypatch):
    fit, tracks = _fit(None)

    def fake_run(ds, lock_name, camera=None, *, episodes=None, payload=True, on_progress=None, should_stop=None):
        on_progress({"stage": "prior", "episode": None, "i": 0, "n": 3, "prior": PRIOR.to_dict(), "elapsed_s": 0.1})
        for i, t in enumerate(tracks):
            on_progress({"stage": "track", "episode": t.episode, "i": i + 1, "n": 3, "track": t.summary(), "elapsed_s": 0.2})
        path = save_object_fit(fit, lock_name, tmp_path, tracks)
        return fit, tracks, path

    monkeypatch.setattr(server, "run_objects_fit", fake_run)
    r = client.post("/api/objects/fit", json={"dataset": "fake/objects", "lock": "unit", "camera": "up"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
            if events and events[-1]["type"] in ("done", "error"):
                break
    assert [e.get("stage") for e in events if e["type"] == "progress"] == ["prior", "track", "track", "track"]
    assert events[-1]["type"] == "done" and events[-1]["fit"]["description"].startswith("a red object")
    assert "tracks" not in events[-1]["fit"] and events[-1]["fit"]["path"].endswith("unit.objects.json")
    assert client.get("/api/jobs/" + job_id).json()["status"] == "done"
    assert client.post("/api/objects/fit", json={"dataset": "fake/objects", "lock": "../etc"}).status_code == 400


def test_cli_objects_fit_prints_the_numbers(monkeypatch, capsys, tmp_path):
    fit, tracks = _fit(PAYLOAD_OK)
    fit.cross_camera = {"n": 12, "primary": "up", "other": "side", "triangulated_vs_plane_m": {"n": 12, "median": 0.03, "q1": 0.029, "q3": 0.031, "iqr": 0.002, "max": 0.031},
                        "triangulated_height_above_plane_m": {"n": 12, "median": -0.026, "q1": -0.026, "q3": -0.026, "iqr": 0.0, "max": -0.026},
                        "ray_gap_m": {"n": 12, "median": 0.005, "q1": 0.004, "q3": 0.005, "iqr": 0.001, "max": 0.009},
                        "other_plane_vs_primary_plane_xy_m": {"n": 12, "median": 0.064, "q1": 0.064, "q3": 0.065, "iqr": 0.001, "max": 0.07}}
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "get_dataset", lambda repo: _dataset(repo))

    def fake_run(ds, lock_name, camera=None, *, episodes=None, payload=True, on_progress=None, should_stop=None):
        on_progress({"stage": "prior", "episode": None, "i": 0, "n": 3, "prior": PRIOR.to_dict(), "elapsed_s": 0.1})
        on_progress({"stage": "track", "episode": 0, "i": 1, "n": 3, "track": tracks[0].summary(), "elapsed_s": 0.2})
        return fit, tracks, save_object_fit(fit, lock_name, tmp_path, tracks)

    monkeypatch.setattr(server, "run_objects_fit", fake_run)
    assert cli.main(["objects-fit", "fake/objects", "--lock", "unit", "--camera", "up"]) == 0
    out = capsys.readouterr().out
    assert "a red object" in out and "footprint 3.8 x 1.8 cm" in out and "3 of 3 episodes tracked" in out
    assert "vision start pose vs FK fingertip at the grasp: median 1.2 cm" in out and "fixed offset" in out
    assert "cross-camera (up x side)" in out and "triangulated height above the plane median -2.6 cm" in out
    assert "payload from the residual: consistent" in out and "shoulder_lift" in out and "g per encoder step" in out
    assert "unit.objects.json" in out

    def boom(ds, lock_name, camera=None, **kw):
        raise ValueError("no object found")

    monkeypatch.setattr(server, "run_objects_fit", boom)
    assert cli.main(["objects-fit", "fake/objects"]) == 1


def test_cli_payload_prints_a_table(monkeypatch, capsys, tmp_path):
    from antares.lock import RobotLock
    from antares.payload import EpisodePayload, JointPayload, PayloadEstimate, Step
    from tests.test_lock import fake_result

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "get_dataset", lambda repo: _dataset(repo))
    RobotLock.create("fake/objects", "so101", [], fake_result(), name="unit").save(tmp_path / "unit.lock.json")
    step = Step(0.1, 0.6, -0.13, 18, 18)
    jp = JointPayload("shoulder_lift", step, Step(0.6, 0.2, -0.12, 18, 18), 12.0, 0.45, 0.05, 0.094, 0.04)
    est = PayloadEstimate(
        lock="unit", dataset="fake/objects", kp={"shoulder_lift": 12.0, "elbow_flex": 12.0},
        episodes=[EpisodePayload(0, 20, 40, {"shoulder_lift": jp}, 0.04, True), EpisodePayload(1, 20, 25, {}, float("nan"), False, "carry too short (5 frames)")],
        mass_kg=0.04, mass_iqr_kg=0.0, mass={"n": 1, "median": 0.04, "q1": 0.04, "q3": 0.04, "iqr": 0.0, "std": 0.0},
        mass_by_joint={"shoulder_lift": {"n": 1, "median": 0.04, "q1": 0.04, "q3": 0.04, "iqr": 0.0, "std": 0.0}, "elbow_flex": None},
        payload_deg={"shoulder_lift": {"n": 1, "median": 0.45, "q1": 0.45, "q3": 0.45, "iqr": 0.0, "std": 0.0}, "elbow_flex": None},
        motion_deg={"shoulder_lift": {"n": 1, "median": 0.05, "q1": 0.05, "q3": 0.05, "iqr": 0.0, "std": 0.0}, "elbow_flex": None},
        n_ok=1, n_episodes=2, mass_per_step_kg={"shoulder_lift": 0.013, "elbow_flex": None}, mass_noise_floor_kg={"shoulder_lift": 0.0, "elbow_flex": None},
        lever_m={"shoulder_lift": 0.13, "elbow_flex": None}, verdict="consistent: median 40 g, IQR 0 g over 1 picks", assumptions=["quasi-static"],
    )
    import antares.payload as payload_mod

    monkeypatch.setattr(payload_mod, "estimate_payload", lambda *a, **k: est)
    out_json = tmp_path / "payload.json"
    assert cli.main(["payload", "fake/objects", "--lock", "unit", "--out", str(out_json)]) == 0
    out = capsys.readouterr().out
    assert "1 of 2 carried picks measured" in out and "carry too short" in out and "+40" in out
    assert "consistent: median 40 g" in out and "assumptions: quasi-static" in out
    assert json.loads(out_json.read_text())["mass_kg"] == 0.04
    assert cli.main(["payload", "fake/objects", "--lock", "ghost"]) == 1
