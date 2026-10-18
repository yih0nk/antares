"""The scene routes and CLI, with the video work stubbed out."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from antares import cli, server
from antares.data import Dataset, Episode
from antares.project import Camera
from antares.scene import SceneFit, fit_scene, save_scene_fit

pytest.importorskip("cv2")

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAM = Camera.look_at([0.45, 0.0, 0.5], [0.15, 0.0, 0.0], 50.0, 320, 240)
FIT = {**CAM.to_dict(), "camera": "up", "key": "observation.images.up", "dataset": "fake/scene", "fit_iou": 0.7,
       "heldout_iou": 0.65, "appearance": {"sign": -1}}


def _dataset(repo_id="fake/scene", n=3, T=60):
    eps = []
    for i in range(n):
        state = np.zeros((T, 6))
        state[:, 0] = np.linspace(-30, 30, T) + 5 * i
        state[:, 1] = -20.0
        state[:, 2] = 30.0
        action = state.copy()
        action[:, 5] = 90.0
        action[T // 3 : 2 * T // 3, 5] = 5.0
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=state, action=action))
    return Dataset(repo_id=repo_id, robot_type="so101", fps=30.0, joint_names=JOINTS, episodes=tuple(eps))


def _bg():
    img = np.full((240, 320, 3), 150, np.uint8)
    img[80:] = 235
    return img


class _FakeBackground:
    bg = _bg()

    def mask(self, frame):
        return (frame[:, :, 0] < 100).astype(np.uint8)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from antares.lock import RobotLock
    from tests.test_lock import fake_result

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    server.put_dataset(_dataset())
    lock = RobotLock.create("fake/scene", "so101", [], fake_result(), name="unit").replace(cameras={"up": FIT})
    lock.save(tmp_path / "unit.lock.json")
    (tmp_path / "unit.camera.json").write_text(json.dumps({"lock": "unit", "cameras": {"up": FIT}}))
    monkeypatch.setattr(server, "camera_background", lambda ds, camera: _FakeBackground())
    return TestClient(server.app)


def test_scene_get_is_null_before_a_fit(client):
    r = client.get("/api/scene", params={"lock": "unit"})
    assert r.status_code == 200 and r.json() == {"lock": "unit", "scene": None}
    assert client.get("/api/scene", params={"lock": "../x"}).status_code == 400


def test_scene_fit_saves_and_returns_the_fit(client, tmp_path):
    r = client.post("/api/scene/fit", json={"dataset": "fake/scene", "lock": "unit"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["lock"] == "unit" and d["table"]["n_grasps"] == 3 and d["table"]["horizontal"]
    assert set(d["backdrops"]) == {"up"} and d["backdrops"]["up"]["path"] == "scene/unit_up_backdrop.jpg"
    assert d["path"].endswith("unit.scene.json")
    assert (tmp_path / "scene" / "unit_up_backdrop.jpg").is_file()
    saved = client.get("/api/scene", params={"lock": "unit"}).json()["scene"]
    assert saved["table"]["z"] == d["table"]["z"] and len(saved["grasp_points"]) == 3
    lock = json.loads((tmp_path / "unit.lock.json").read_text())
    assert lock["scene"]["table_z"] == d["table"]["z"] and lock["scene"]["backdrops"] == {"up": "scene/unit_up_backdrop.jpg"}
    # the lock's export now carries the table and the backdrop texture
    z = client.get("/api/locks/unit/export.zip")
    assert z.status_code == 200
    import io
    import zipfile

    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert "unit/backdrops/up.png" in names


def test_scene_fit_rejects_bad_lock_or_missing_lock(client):
    assert client.post("/api/scene/fit", json={"dataset": "fake/scene", "lock": "../etc"}).status_code == 400
    assert client.post("/api/scene/fit", json={"dataset": "fake/scene", "lock": "ghost"}).status_code == 404


def test_scene_view_composites_real_and_sim(client, tmp_path, monkeypatch):
    from antares import video

    frame = _bg()
    frame[100:140, 150:190] = 30  # a dark blob standing in for the arm
    monkeypatch.setattr(video, "frames", lambda repo, camera, ep, idx, ds=None, cache=True: np.stack([frame] * len(idx)))
    # no scene fit yet
    r = client.get("/api/scene/view", params={"dataset": "fake/scene", "camera": "up", "episode": 1, "frame": 4, "lock": "unit"})
    assert r.status_code == 404 and "scene" in r.json()["detail"]
    assert client.post("/api/scene/fit", json={"dataset": "fake/scene", "lock": "unit"}).status_code == 200
    r = client.get("/api/scene/view", params={"dataset": "fake/scene", "camera": "up", "episode": 1, "frame": 4, "lock": "unit"})
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["real_image"].startswith("data:image/jpeg;base64,") and v["sim_image"].startswith("data:image/jpeg;base64,")
    assert (v["width"], v["height"]) == (320, 240) and v["n_geoms"] > 5
    assert 0.0 <= v["iou"] <= 1.0 and len(v["table_polygon"]) >= 3 and v["table_z"] < 0.5
    assert client.get("/api/scene/view", params={"dataset": "fake/scene", "camera": "up", "episode": 1, "frame": 999, "lock": "unit"}).status_code == 404
    assert client.get("/api/scene/view", params={"dataset": "fake/scene", "camera": "side", "episode": 1, "frame": 4, "lock": "unit"}).status_code == 404


def test_cli_scene_fit_prints_the_numbers(monkeypatch, capsys, tmp_path):
    from antares.sim import Arm

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "get_dataset", lambda repo: _dataset(repo))

    def fake_run(ds, lock_name, cameras=None):
        fit = fit_scene(ds, Arm(), lock_name, {"up": CAM}, {"up": _bg()})
        return fit, save_scene_fit(fit, lock_name, tmp_path, {"up": _bg()})

    monkeypatch.setattr(server, "run_scene_fit", fake_run)
    assert cli.main(["scene-fit", "fake/scene", "--lock", "unit"]) == 0
    out = capsys.readouterr().out
    assert "table z" in out and "3 of 3 grasps" in out and "workspace" in out
    assert "3 grasp points, 3 release points" in out and "plane in region" in out
    assert "unit.scene.json" in out
    assert SceneFit.from_dict(json.loads((tmp_path / "unit.scene.json").read_text())).lock == "unit"

    def boom(ds, lock_name, cameras=None):
        raise ValueError("no grasp instants")

    monkeypatch.setattr(server, "run_scene_fit", boom)
    assert cli.main(["scene-fit", "fake/scene"]) == 1
