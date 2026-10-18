"""POST /api/locks: hand-tuned parameters saved as a lock (real MuJoCo, no network)."""

from __future__ import annotations

import dataclasses

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
    monkeypatch.setattr(server, "LOCK_SUFFIX", LOCK_SUFFIX)
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
    import json

    src = json.loads(src_path.read_text())
    cmap = {"gain": [1.0] * 6, "bias": [0.5, 0, 0, 0, 0, 0], "mode": "sim"}
    src["command_map"] = cmap
    src["cameras"] = {"top": {"note": "fitted"}}
    src_path.write_text(json.dumps(src))

    r = client.post("/api/locks", json={**base, "name": "from-src", "from_lock": "src"})
    assert r.status_code == 200, r.text
    lock = client.get("/api/locks/from-src").json()
    assert lock["command_map"] == cmap
    assert lock["cameras"] == {"top": {"note": "fitted"}}
