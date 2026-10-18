"""Bring-your-own-dataset API: validation pre-check, zip/parquet upload,
listing with source + warnings, delete. Real parquet/zip handling, no
network (the Hub fetch is stubbed), no MuJoCo except where marked."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from antares import server

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NAMES = [f"{j}.pos" for j in JOINTS]


def info(n_episodes=6, **over):
    base = {
        "codebase_version": "v3.0",
        "robot_type": "so101_follower",
        "fps": 30,
        "total_episodes": n_episodes,
        "total_frames": n_episodes * 10,
        "features": {"action": {"names": list(NAMES)}, "observation.state": {"names": list(NAMES)}},
    }
    base.update(over)
    return base


def parquet_bytes(n_episodes=6, T=10, scale=1.0, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    ep, fr, st, ac = [], [], [], []
    for i in range(n_episodes):
        a = rng.uniform(-80, 80, size=(T, 6)) * scale
        a[:, 5] = rng.uniform(0, 100, size=T) * scale
        s = a + rng.normal(0, 0.3, size=a.shape)
        ep += [i] * T
        fr += list(range(T))
        st += s.tolist()
        ac += a.tolist()
    table = pa.table({"episode_index": ep, "frame_index": fr, "observation.state": st, "action": ac})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def make_zip(prefix="my_arm/", n_episodes=6, info_over=None, data=None, extra=None, with_video=True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}meta/info.json", json.dumps(info(n_episodes, **(info_over or {}))))
        zf.writestr(f"{prefix}meta/episodes.jsonl", "{}\n")
        zf.writestr(f"{prefix}data/chunk-000/file-000.parquet", data if data is not None else parquet_bytes(n_episodes))
        if with_video:
            zf.writestr(f"{prefix}videos/observation.images.top/chunk-000/file-000.mp4", b"\x00" * 64)
        for name, content in (extra or {}).items():
            zf.writestr(name, content)
    return buf.getvalue()


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_DATASETS", {})
    monkeypatch.setattr(server, "_DATASET_META", {})
    monkeypatch.setattr(server, "_BASELINES", {})
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")

    def offline(repo_id):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(server, "fetch_hub_info", offline)
    import antares.data

    monkeypatch.setattr(antares.data, "load_dataset", lambda *a, **k: offline("x"))
    return tmp_path


@pytest.fixture
def client():
    return TestClient(server.app)


def upload(client, content: bytes, filename="my_arm.zip", name=None, info_json: bytes | None = None):
    files = {"file": (filename, content, "application/octet-stream")}
    if info_json is not None:
        files["info"] = ("info.json", info_json, "application/json")
    data = {"name": name} if name else None
    return client.post("/api/datasets/upload", files=files, data=data)


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------


def test_zip_upload_registers_loads_and_lists(client, app):
    r = upload(client, make_zip())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "upload/my_arm" and body["source"] == "upload" and body["loaded"] is True
    assert body["n_episodes"] == 6 and body["joints"] == list(JOINTS) and body["fps"] == 30
    assert body["robot_family"] == "so101" and body["robot"] == "so101"
    assert body["verdict"]["ok"] is True
    assert any("only 6" in w for w in body["warnings"])  # < 15 episodes
    assert body["warnings"] == body["verdict"]["warnings"]

    # extracted under UPLOAD_DIR/<slug>-<hash>, without the video
    dirs = list((app / "uploads").iterdir())
    assert len(dirs) == 1 and dirs[0].name.startswith("my_arm-")
    assert (dirs[0] / "meta" / "info.json").is_file()
    assert (dirs[0] / "data" / "chunk-000" / "file-000.parquet").is_file()
    assert not (dirs[0] / "videos").exists()
    marker = json.loads((dirs[0] / server.UPLOAD_MARKER).read_text())
    assert marker["id"] == "upload/my_arm" and len(marker["sha256"]) == 64

    listing = client.get("/api/datasets").json()
    ids = [d["id"] for d in listing]
    assert ids[: len(server.DEMO_DATASETS)] == server.DEMO_DATASETS and "upload/my_arm" in ids
    demo = listing[0]
    assert demo["source"] == "demo" and demo["loaded"] is False and demo["warnings"] == []
    assert demo["robot_family"] == "so101" and demo["robot"] == "so101"
    mine = next(d for d in listing if d["id"] == "upload/my_arm")
    assert mine["source"] == "upload" and mine["warnings"]

    eps = client.get("/api/episodes", params={"dataset": "upload/my_arm"}).json()
    assert len(eps) == 6 and eps[0]["frames"] == 10
    v = client.get("/api/datasets/validate", params={"dataset": "upload/my_arm"}).json()
    assert v["ok"] and v["loaded"] is True and v["source"] == "upload"


def test_zip_at_archive_root_and_explicit_name(client, app):
    r = upload(client, make_zip(prefix=""), filename="whatever.zip", name="Arm B (lab)")
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "upload/arm_b_lab"


def test_reupload_same_data_is_idempotent_and_conflict_on_different_data(client, app):
    z = make_zip()
    assert upload(client, z).status_code == 200
    again = upload(client, z)
    assert again.status_code == 200 and again.json()["id"] == "upload/my_arm"
    other = upload(client, make_zip(data=parquet_bytes(seed=1)))
    assert other.status_code == 409
    assert len(list((app / "uploads").iterdir())) == 1


def test_delete_upload_removes_files_and_registration(client, app):
    assert upload(client, make_zip()).status_code == 200
    r = client.delete("/api/datasets", params={"dataset": "upload/my_arm"})
    assert r.status_code == 200 and r.json()["deleted"] == "upload/my_arm"
    assert list((app / "uploads").iterdir()) == []
    assert "upload/my_arm" not in [d["id"] for d in client.get("/api/datasets").json()]
    assert client.delete("/api/datasets", params={"dataset": "upload/my_arm"}).status_code == 404
    assert client.get("/api/datasets/validate", params={"dataset": "upload/my_arm"}).status_code == 404
    assert client.delete("/api/datasets", params={"dataset": "never/loaded"}).status_code == 404


def test_upload_survives_a_restart_via_the_marker(client, app):
    assert upload(client, make_zip()).status_code == 200
    server._DATASETS.clear()
    server._DATASET_META.clear()
    assert client.post("/api/datasets/load", params={"dataset": "upload/my_arm"}).json()["loaded"] is True


def test_parquet_plus_info_upload(client, app):
    r = upload(client, parquet_bytes(), filename="episodes.parquet", info_json=json.dumps(info()).encode())
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "upload/episodes" and r.json()["n_episodes"] == 6
    r = upload(client, parquet_bytes(), filename="bare.parquet")
    assert r.status_code == 400 and "info.json" in r.json()["detail"]
    r = upload(client, parquet_bytes(), filename="bad.parquet", info_json=b"{not json")
    assert r.status_code == 400
    r = upload(client, b"hello world, not an archive", filename="notes.txt")
    assert r.status_code == 400 and "zip" in r.json()["detail"]


@pytest.mark.parametrize(
    "member",
    ["../escape.txt", "my_arm/../../escape.txt", "/abs/escape.txt", "my_arm/data/../../escape.parquet"],
)
def test_zip_traversal_is_rejected(client, app, member, tmp_path):
    r = upload(client, make_zip(extra={member: b"evil"}))
    assert r.status_code == 400, r.text
    assert "escape" in r.json()["detail"] or "absolute" in r.json()["detail"]
    assert not (tmp_path / "escape.txt").exists() and not (app / "escape.txt").exists()
    assert not (app / "uploads").exists() or list((app / "uploads").iterdir()) == []


def test_zip_without_dataset_layout_is_rejected(client, app):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no dataset here")
    r = upload(client, buf.getvalue())
    assert r.status_code == 400 and "meta/info.json" in r.json()["detail"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x/meta/info.json", json.dumps(info()))
    r = upload(client, buf.getvalue())
    assert r.status_code == 400 and "parquet" in r.json()["detail"]


def test_upload_validation_failure_is_400_with_verdict_and_cleans_up(client, app):
    r = upload(client, make_zip(info_over={"robot_type": "koch"}))
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "cannot be replayed" in detail["message"]
    assert any("koch" in reason for reason in detail["verdict"]["reasons"])
    assert not (app / "uploads").exists() or list((app / "uploads").iterdir()) == []
    # too few episodes for an upload
    r = upload(client, make_zip(n_episodes=3))
    assert r.status_code == 400 and "3 episodes < 5" in r.json()["detail"]["message"]
    # unit sanity is a warning, not a rejection
    r = upload(client, make_zip(data=parquet_bytes(scale=3.0)), name="degrees")
    assert r.status_code == 200
    assert any("look like degrees" in w for w in r.json()["warnings"])


def test_upload_size_cap_is_413(client, app, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 2000)
    r = upload(client, make_zip())
    assert r.status_code == 413
    assert not (app / "uploads").exists() or list((app / "uploads").iterdir()) == []


# --------------------------------------------------------------------------
# hub ids: pre-check and gated load
# --------------------------------------------------------------------------


def test_validate_endpoint_for_hub_ids(client, app, monkeypatch):
    monkeypatch.setattr(server, "fetch_hub_info", lambda rid: info(50, robot_type="so100"))
    v = client.get("/api/datasets/validate", params={"dataset": "someone/so100_arm"}).json()
    assert v["ok"] and v["robot_family"] == "so100" and v["loaded"] is False and v["source"] == "hub"
    monkeypatch.setattr(server, "fetch_hub_info", lambda rid: info(50, robot_type="bi_so100_follower"))
    v = client.get("/api/datasets/validate", params={"dataset": "someone/bimanual"}).json()
    assert not v["ok"] and any("bimanual" in r for r in v["reasons"])

    class RepositoryNotFoundError(Exception):
        pass

    def missing(rid):
        raise RepositoryNotFoundError(rid)

    monkeypatch.setattr(server, "fetch_hub_info", missing)
    assert client.get("/api/datasets/validate", params={"dataset": "nobody/nothing"}).status_code == 404
    monkeypatch.setattr(server, "fetch_hub_info", lambda rid: (_ for _ in ()).throw(OSError("offline")))
    assert client.get("/api/datasets/validate", params={"dataset": "a/b"}).status_code == 502


def test_hub_load_is_gated_by_the_verdict(client, app, monkeypatch):
    monkeypatch.setattr(server, "fetch_hub_info", lambda rid: info(50, fps=60))
    import antares.data

    monkeypatch.setattr(antares.data, "load_dataset", lambda *a, **k: pytest.fail("rejected datasets must not download"))
    r = client.post("/api/datasets/load", params={"dataset": "someone/fast_arm"})
    assert r.status_code == 400
    assert "fps 60" in r.json()["detail"]["message"]
    assert r.json()["detail"]["verdict"]["ok"] is False


def test_robots_endpoint_lists_the_registry(client):
    out = client.get("/api/robots").json()
    assert {r["name"] for r in out} >= {"so101", "so100"}
    assert all(r["joints"] == list(JOINTS) for r in out)


def test_model_rejects_unknown_robot(client):
    assert client.get("/api/model", params={"robot": "koch"}).status_code == 400
    assert client.get("/api/model/mesh/Base.stl", params={"robot": "koch"}).status_code == 400


@pytest.mark.slow
def test_model_and_meshes_per_robot(client, monkeypatch):
    monkeypatch.setattr(server, "_ARMS", {})
    monkeypatch.setattr(server, "MESH_DIR", None)
    so101 = client.get("/api/model").json()
    so100 = client.get("/api/model", params={"robot": "so100"}).json()
    assert so101["robot"] == "so101" and so100["robot"] == "so100"
    assert {g["mesh"] for g in so101["geoms"]} != {g["mesh"] for g in so100["geoms"]}
    mesh = so100["geoms"][0]["mesh"]
    assert client.get(f"/api/model/mesh/{mesh}", params={"robot": "so100"}).status_code == 200
    assert client.get(f"/api/model/mesh/{mesh}").status_code == 404  # not an SO-101 mesh
