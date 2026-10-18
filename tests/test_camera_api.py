"""The camera routes and CLI, with the video / fit work stubbed out."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from antares import cli, server
from antares.data import Dataset, Episode

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FIT = {
    "camera": "up",
    "key": "observation.images.up",
    "dataset": "fake/ds",
    "pos": [0.5, 0.1, 0.5],
    "rotvec": [0.1, 0.2, 0.3],
    "quat": [1.0, 0.0, 0.0, 0.0],
    "fovy": 50.0,
    "width": 640,
    "height": 480,
    "fit_iou": 0.7,
    "heldout_iou": 0.65,
    "fit_frames": [[0, 1]],
    "heldout_frames": [[2, 3]],
    "fit_ious": [0.7],
    "heldout_ious": [0.65],
    "n_starts": 8,
    "best_start": 2,
    "start_ious": [0.3] * 8,
    "n_evaluations": 100,
    "elapsed_s": 1.0,
    "cancelled": False,
    "created_at": "2026-10-17T00:00:00Z",
    "appearance": {"sign": -1},
}


def _dataset(repo_id="fake/ds"):
    eps = []
    for i in range(3):
        a = np.linspace(0, 10, 20)[:, None].repeat(6, axis=1)
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=a, action=a))
    return Dataset(repo_id=repo_id, robot_type="so101", fps=30.0, joint_names=JOINTS, episodes=tuple(eps))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    server.put_dataset(_dataset())
    (tmp_path / "unit.camera.json").write_text(json.dumps({"lock": "unit", "cameras": {"up": FIT}}))
    return TestClient(server.app)


def test_camera_cameras_lists_dataset_streams(client, monkeypatch):
    monkeypatch.setattr(
        server, "list_camera_infos",
        lambda repo: [{"name": "up", "key": "observation.images.up", "width": 640, "height": 480, "fps": 30.0, "fixed": True}],
    )
    r = client.get("/api/camera/cameras", params={"dataset": "fake/ds"})
    assert r.status_code == 200 and r.json()[0]["name"] == "up"


def test_camera_fits_for_lock_omit_per_frame_arrays(client):
    r = client.get("/api/camera", params={"lock": "unit"})
    assert r.status_code == 200
    cams = r.json()["cameras"]
    assert set(cams) == {"up"}
    assert cams["up"]["fit_iou"] == 0.7 and "fit_ious" not in cams["up"] and "appearance" not in cams["up"]
    assert client.get("/api/camera", params={"lock": "nope"}).json()["cameras"] == {}
    assert client.get("/api/camera", params={"lock": "../x"}).status_code == 400


def test_camera_fit_job_streams_progress_and_saves(client, monkeypatch, tmp_path):
    from antares.camera import CameraFit, ProgressEvent

    def fake_fit(ds, camera, lock_name, *, on_progress=None, should_stop=None, **kw):
        assert ds.repo_id == "fake/ds" and camera == "side" and lock_name == "unit"
        for k in range(3):
            on_progress(ProgressEvent("start", k, 3, k * 10, 10, 0.2, 0.3, 0.3 + 0.1 * k, 1.0))
        fit = CameraFit.from_dict({**FIT, "camera": "side", "key": "observation.images.side"})
        from antares.camera import save_camera_fit

        return fit, save_camera_fit(fit, lock_name, server.LOCK_DIR)

    monkeypatch.setattr(server, "run_camera_fit", fake_fit)
    r = client.post("/api/camera/fit", json={"dataset": "fake/ds", "camera": "side", "lock": "unit"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    for _ in range(50):
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] != "running":
            break
        time.sleep(0.05)
    assert st["status"] == "done" and st["kind"] == "camera"
    assert st["result"]["camera"] == "side" and st["result"]["lock"] == "unit"
    events = [json.loads(l[6:]) for l in client.get(f"/api/jobs/{job_id}/events").text.splitlines() if l.startswith("data:")]
    assert [e["type"] for e in events] == ["progress"] * 3 + ["done"]
    assert events[2]["best_iou"] == pytest.approx(0.5) and events[2]["stage"] == "start"
    assert events[-1]["fit"]["path"].endswith("unit.camera.json")
    saved = json.loads((tmp_path / "unit.camera.json").read_text())["cameras"]
    assert set(saved) == {"up", "side"}


def test_camera_fit_rejects_bad_lock_name(client):
    r = client.post("/api/camera/fit", json={"dataset": "fake/ds", "camera": "up", "lock": "../etc"})
    assert r.status_code == 400


def test_camera_overlay_draws_sim_polygons(client, monkeypatch):
    from antares import video

    frame = np.full((480, 640, 3), 220, np.uint8)
    monkeypatch.setattr(video, "frames", lambda repo, camera, ep, idx, ds=None, cache=True: np.stack([frame] * len(idx)))
    r = client.get("/api/camera/overlay", params={"dataset": "fake/ds", "camera": "up", "episode": 1, "frame": 4, "lock": "unit", "mask": False})
    assert r.status_code == 200, r.text
    o = r.json()
    assert o["image"].startswith("data:image/jpeg;base64,")
    assert (o["width"], o["height"]) == (640, 480)
    assert o["sim_polygons"] and all(len(p) >= 3 and len(p[0]) == 2 for p in o["sim_polygons"])
    assert len(o["sim_bodies"]) == len(o["sim_polygons"])
    assert o["fit"]["fit_iou"] == 0.7 and o["iou"] is None and "real_mask_polygon" not in o
    # Without a lock there is no camera, so no outline, but the frame still comes back.
    o2 = client.get("/api/camera/overlay", params={"dataset": "fake/ds", "camera": "up", "episode": 1, "frame": 4, "mask": False}).json()
    assert o2["sim_polygons"] == [] and o2["fit"] is None
    assert client.get("/api/camera/overlay", params={"dataset": "fake/ds", "camera": "up", "episode": 1, "frame": 99, "mask": False}).status_code == 404


def test_camera_overlay_reports_missing_video_support(client, monkeypatch):
    from antares import video

    def boom(*a, **k):
        raise video.VideoUnavailable("video decoding needs PyAV")

    monkeypatch.setattr(video, "frames", boom)
    r = client.get("/api/camera/overlay", params={"dataset": "fake/ds", "camera": "up", "episode": 0, "frame": 0})
    assert r.status_code == 501 and "PyAV" in r.json()["detail"]


def test_cli_camera_list(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    assert cli.main(["camera-list", "unit"]) == 0
    assert "no camera fits for lock unit" in capsys.readouterr().out
    (tmp_path / "unit.camera.json").write_text(json.dumps({"lock": "unit", "cameras": {"up": FIT}}))
    assert cli.main(["camera-list", "unit"]) == 0
    out = capsys.readouterr().out
    assert "observation.images.up" in out and "0.700" in out and "0.650" in out and "640x480" in out
    assert cli.main(["camera-list", "../bad"]) == 1


def test_cli_camera_fit_prints_result(monkeypatch, capsys, tmp_path):
    from antares.camera import CameraFit

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "get_dataset", lambda repo: _dataset(repo))
    monkeypatch.setattr(
        server, "list_camera_infos",
        lambda repo: [
            {"name": "up", "key": "observation.images.up", "width": 640, "height": 480, "fps": 30.0, "fixed": True},
            {"name": "wrist", "key": "observation.images.wrist", "width": 640, "height": 480, "fps": 30.0, "fixed": False},
        ],
    )

    def fake_fit(ds, camera, lock_name, *, on_progress=None, **kw):
        assert kw["n_frames"] == 24 and kw["n_episodes"] == 3
        return CameraFit.from_dict(FIT), tmp_path / f"{lock_name}.camera.json"

    monkeypatch.setattr(server, "run_camera_fit", fake_fit)
    assert cli.main(["camera-fit", "fake/ds", "--camera", "up", "--frames", "24", "--episodes", "3"]) == 0
    out = capsys.readouterr().out
    assert "fit IoU 0.700" in out and "held-out IoU 0.650" in out and "ds.camera.json" in out
    assert cli.main(["camera-fit", "fake/ds", "--camera", "wrist"]) == 1
    assert cli.main(["camera-fit", "fake/ds", "--camera", "nope"]) == 1


def test_camera_clip_serves_the_episode_mp4(client, monkeypatch, tmp_path):
    from antares import video

    clip = tmp_path / "ep.mp4"
    clip.write_bytes(b"\x00" * 1000)
    seen = {}

    def fake_clip(repo, camera, episode, ds=None, fmt="mp4"):
        seen.update(repo=repo, camera=camera, episode=episode, fmt=fmt)
        return clip

    monkeypatch.setattr(video, "episode_clip", fake_clip)
    r = client.get("/api/camera/clip", params={"dataset": "fake/ds", "camera": "up", "episode": 2})
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and len(r.content) == 1000
    assert seen == {"repo": "fake/ds", "camera": "up", "episode": 2, "fmt": "mp4"}
    r = client.get("/api/camera/clip", params={"dataset": "fake/ds", "camera": "up", "episode": 2, "format": "webm"})
    assert r.headers["content-type"] == "video/webm" and seen["fmt"] == "webm"
    assert client.get("/api/camera/clip", params={"dataset": "fake/ds", "camera": "up", "episode": 2, "format": "gif"}).status_code == 422
    # Seeking in the browser needs byte ranges.
    r = client.get("/api/camera/clip", params={"dataset": "fake/ds", "camera": "up", "episode": 2}, headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and len(r.content) == 10

    def missing(*a, **k):
        raise video.VideoUnavailable("fake/ds has no camera 'side'")

    monkeypatch.setattr(video, "episode_clip", missing)
    assert client.get("/api/camera/clip", params={"dataset": "fake/ds", "camera": "side", "episode": 0}).status_code == 400


def test_camera_fits_for_dataset_take_the_best_fit_over_its_locks(client, tmp_path):
    def lock(name, dataset, created):
        (tmp_path / f"{name}.lock.json").write_text(json.dumps({"name": name, "dataset": dataset, "created_at": created}))

    def fits(name, **cams):
        (tmp_path / f"{name}.camera.json").write_text(json.dumps({"lock": name, "cameras": cams}))

    lock("fresh", "fake/ds", "2026-10-18T09:00:00Z")  # newest, no camera fit
    lock("base", "fake/ds", "2026-10-17T00:00:00Z")
    lock("perjoint", "fake/ds", "2026-10-17T01:00:00Z")
    lock("other", "other/ds", "2026-10-18T10:00:00Z")
    fits("base", up={**FIT, "heldout_iou": 0.6}, side={**FIT, "camera": "side", "heldout_iou": 0.5})
    fits("perjoint", up={**FIT, "heldout_iou": 0.8})
    fits("other", up={**FIT, "dataset": "other/ds", "heldout_iou": 0.99}, front={**FIT, "dataset": "other/ds"})

    cams = client.get("/api/camera/dataset", params={"dataset": "fake/ds"}).json()["cameras"]
    assert set(cams) == {"up", "side"}
    assert cams["up"]["lock"] == "perjoint" and cams["up"]["heldout_iou"] == 0.8
    assert cams["side"]["lock"] == "base"
    assert "fit_ious" not in cams["up"] and "appearance" not in cams["up"]
    assert client.get("/api/camera/dataset", params={"dataset": "nobody/ds"}).json()["cameras"] == {}


def _sibling_setup(tmp_path, fovy=43.0, size=(640, 480)):
    (tmp_path / "base.lock.json").write_text(json.dumps({"name": "base", "dataset": "fake/ds", "created_at": "2026-10-17T00:00:00Z"}))
    up = {**FIT, "fovy": fovy, "width": size[0], "height": size[1], "heldout_iou": 0.83}
    (tmp_path / "base.camera.json").write_text(json.dumps({"lock": "base", "cameras": {"up": up}}))


def _fake_fit_camera(calls, heldout):
    from antares.camera import CameraFit

    def fake(ds, camera, *, fovy_range=None, **kw):
        calls.append(fovy_range)
        h = heldout(fovy_range)
        return CameraFit.from_dict({**FIT, "camera": camera, "key": f"observation.images.{camera}", "fovy": 44.0 if fovy_range else 26.0, "heldout_iou": h})

    return fake


@pytest.mark.parametrize("banded_better", [True, False])
def test_camera_fit_tries_a_sibling_cameras_fov(client, monkeypatch, tmp_path, banded_better):
    import antares.camera as camera_mod

    _sibling_setup(tmp_path)
    calls, passes = [], []
    monkeypatch.setattr(camera_mod, "fit_camera", _fake_fit_camera(calls, lambda band: (0.78 if banded_better else 0.6) if band else 0.72))
    fit, path = server.run_camera_fit(server.get_dataset("fake/ds"), "side", "unit", on_second_pass=passes.append)
    assert calls == [None, (41.0, 45.0)] and passes == ["up"]
    assert fit.heldout_iou == (0.78 if banded_better else 0.72)
    saved = json.loads(path.read_text())["cameras"]["side"]
    assert saved["fovy"] == (44.0 if banded_better else 26.0)


def test_camera_fit_skips_the_fov_pass_without_a_sibling_of_the_same_size(client, monkeypatch, tmp_path):
    import antares.camera as camera_mod

    calls = []
    monkeypatch.setattr(camera_mod, "fit_camera", _fake_fit_camera(calls, lambda band: 0.7))
    server.run_camera_fit(server.get_dataset("fake/ds"), "side", "unit")
    assert calls == [None]  # no other camera fitted
    _sibling_setup(tmp_path, size=(1280, 720))
    calls.clear()
    server.run_camera_fit(server.get_dataset("fake/ds"), "side", "unit")
    assert calls == [None]  # the fitted camera has another image size
    calls.clear()
    server.run_camera_fit(server.get_dataset("fake/ds"), "up", "unit")
    assert calls == [None]  # a camera is not its own sibling
