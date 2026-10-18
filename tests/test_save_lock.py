"""POST /api/locks: hand-tuned parameters saved as a lock (real MuJoCo, no network)."""

from __future__ import annotations

import dataclasses
import json

from fastapi.testclient import TestClient

from antares import server
from antares.lock import LOCK_SUFFIX
from antares.sim import SimParams
from antares.synthetic import make_dataset

DS = "synthetic/tuned_arm"


def _setup(monkeypatch, tmp_path):
    d = SimParams.default()
    truth = dataclasses.replace(d, kp=d.kp * 0.3, latency_frames=3)
    ds = make_dataset(DS, truth, n_episodes=4, T=60, seed=2)
    monkeypatch.setattr(server, "_DATASETS", {DS: ds})
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    return truth


def test_save_tuned_lock_scores_and_reloads(monkeypatch, tmp_path):
    truth = _setup(monkeypatch, tmp_path)
    client = TestClient(server.app)
    body = {"dataset": DS, "name": "tuned-a", "params": truth.to_dict(), "n_train": 2, "n_eval": 2}

    r = client.post("/api/locks", json=body)
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["name"] == "tuned-a" and summary["dataset"] == DS
    scores = summary["scores"]
    # the true parameters beat the defaults on held-out episodes
    assert scores["eval_score"] < scores["baseline_eval_score"]

    lock = client.get("/api/locks/tuned-a").json()
    assert lock["params"]["latency_frames"] == 3
    assert lock["result"]["n_evaluations"] == 0
    assert [s["name"] for s in lock["result"]["stages"]] == ["manual"]
    assert (tmp_path / f"tuned-a{LOCK_SUFFIX}").is_file()

    # replaying the saved lock gives the score the panel showed for those values
    by_lock = client.get("/api/replay", params={"dataset": DS, "episode": 0, "lock": "tuned-a"}).json()
    by_params = client.post("/api/replay", json={"dataset": DS, "episode": 0, "params": truth.to_dict()}).json()
    assert by_lock["gap"]["overall_deg"] == by_params["gap"]["overall_deg"]


def test_save_tuned_lock_refuses_overwrite_and_bad_input(monkeypatch, tmp_path):
    truth = _setup(monkeypatch, tmp_path)
    client = TestClient(server.app)
    body = {"dataset": DS, "name": "tuned-b", "params": truth.to_dict(), "n_train": 2, "n_eval": 1}
    assert client.post("/api/locks", json=body).status_code == 200

    again = client.post("/api/locks", json=body)
    assert again.status_code == 409
    assert client.post("/api/locks", json={**body, "name": "../escape"}).status_code == 400
    assert client.post("/api/locks", json={**body, "name": "tuned-c", "params": {"kp": "x"}}).status_code in (400, 422)


def test_save_tuned_lock_keeps_source_lock_setup(monkeypatch, tmp_path):
    truth = _setup(monkeypatch, tmp_path)
    client = TestClient(server.app)
    base = {"dataset": DS, "params": truth.to_dict(), "n_train": 2, "n_eval": 1}
    assert client.post("/api/locks", json={**base, "name": "src"}).status_code == 200
    src_path = tmp_path / f"src{LOCK_SUFFIX}"
    src = json.loads(src_path.read_text())
    cmap = {"gain": [1.0] * 6, "bias": [0.5, 0, 0, 0, 0, 0], "mode": "sim"}
    src["command_map"] = cmap
    src["cameras"] = {"top": {"note": "fitted"}}
    src["dynamics_only"] = {"eval_score": 9.0}  # the source's own fit, not this lock's
    src_path.write_text(json.dumps(src))
    # the scene and object fits are sidecars looked up by lock name
    (tmp_path / "src.scene.json").write_text(json.dumps({"lock": "src", "table": {"z": 0.01}}))
    (tmp_path / "src.objects.json").write_text(json.dumps({"lock": "src", "footprint": [1, 2]}))

    r = client.post("/api/locks", json={**base, "name": "from-src", "from_lock": "src"})
    assert r.status_code == 200, r.text
    lock = client.get("/api/locks/from-src").json()
    assert lock["command_map"] == cmap
    assert lock["cameras"] == {"top": {"note": "fitted"}}
    assert "dynamics_only" not in lock
    assert json.loads((tmp_path / "from-src.scene.json").read_text()) == {"lock": "from-src", "table": {"z": 0.01}}
    assert json.loads((tmp_path / "from-src.objects.json").read_text())["lock"] == "from-src"


def test_save_tuned_lock_refuses_a_source_for_another_robot_or_width(monkeypatch, tmp_path):
    truth = _setup(monkeypatch, tmp_path)
    client = TestClient(server.app)
    base = {"dataset": DS, "params": truth.to_dict(), "n_train": 2, "n_eval": 1}
    assert client.post("/api/locks", json={**base, "name": "src"}).status_code == 200
    src_path = tmp_path / f"src{LOCK_SUFFIX}"
    src = json.loads(src_path.read_text())
    src["command_map"] = {"gain": [1.0] * 3, "bias": [0.0] * 3}
    src_path.write_text(json.dumps(src))
    r = client.post("/api/locks", json={**base, "name": "narrow", "from_lock": "src"})
    assert r.status_code == 400 and "3 joints" in r.json()["detail"]
    assert not (tmp_path / f"narrow{LOCK_SUFFIX}").exists()


def test_save_tuned_lock_reports_unstable_parameters(monkeypatch, tmp_path):
    import antares.calibrate as cal

    truth = _setup(monkeypatch, tmp_path)
    client = TestClient(server.app)
    monkeypatch.setattr(cal, "_safe_score", lambda p, eps, sim: (None, cal.PENALTY))
    r = client.post("/api/locks", json={"dataset": DS, "name": "boom", "params": truth.to_dict(),
                                        "n_train": 2, "n_eval": 1})
    assert r.status_code == 400 and "unstable" in r.json()["detail"]
    assert not (tmp_path / f"boom{LOCK_SUFFIX}").exists()
