"""The policy routes and CLI on a tiny synthetic dataset: train as a job,
list, evaluate in both modes as jobs, saved results, the viewer's poses."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from antares import cli, server
from antares.calibrate import CalibrationResult
from antares.data import Dataset
from antares.lock import RobotLock
from antares.sim import JOINTS, SimParams
from tests.test_policy import SOFT, smooth_episode

pytest.importorskip("torch")

DS = "fake/policy_ds"
TRAIN = [0, 1, 2, 3, 4]
EVAL = [5, 6]


def _dataset() -> Dataset:
    eps = [smooth_episode(i, seed=i) for i in range(7)]
    eps[6].state[53:90, 5] = 2.0  # one empty close, so the dataset has an empty-jaw level
    return Dataset(repo_id=DS, robot_type="so101", fps=30.0, joint_names=JOINTS, episodes=tuple(eps))


def _result(params: SimParams, space: str = "shared") -> CalibrationResult:
    return CalibrationResult(
        params=params, baseline=SimParams.default(), train_score=1.0, eval_score=1.2,
        baseline_train_score=3.0, baseline_eval_score=3.3,
        per_joint_before={j: 3.0 for j in JOINTS}, per_joint_after={j: 1.0 for j in JOINTS},
        history=[(1, 3.0)], n_evaluations=1, elapsed_s=0.1, train_episodes=TRAIN, eval_episodes=EVAL, space=space,
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    server.put_dataset(_dataset())
    RobotLock.create(DS, "so101", [], _result(SOFT), name="unit").save(tmp_path / "unit.lock.json")
    pj = SimParams(kp=(90, 80, 70, 60, 50, 40), kv=1.5, damping=0.05, armature=0.02, frictionloss=0.025, latency_frames=(4, 4, 3, 3, 4, 4))
    RobotLock.create(DS, "so101", [], _result(pj, "per_joint"), name="unit_perjoint").save(tmp_path / "unit_perjoint.lock.json")
    return TestClient(server.app)


def _wait(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _events(client: TestClient, job_id: str) -> list[dict]:
    body = client.get(f"/api/jobs/{job_id}/events").text
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_policy_list_is_empty_before_training(client):
    r = client.get("/api/policy", params={"lock": "unit"})
    assert r.status_code == 200 and r.json() == {"lock": "unit", "policies": []}
    assert client.get("/api/policy", params={"lock": "../x"}).status_code == 400
    r = client.get("/api/policy/results", params={"lock": "unit"})
    assert r.json() == {"lock": "unit", "shadow": None, "closedloop": None}
    assert client.get("/api/policy/results", params={"lock": "unit", "mode": "nope"}).status_code == 400


def test_train_evaluate_and_view(client, tmp_path):
    # ---- train as a job --------------------------------------------------
    assert client.post("/api/policy/train", json={"dataset": DS, "lock": "missing"}).status_code == 404
    r = client.post("/api/policy/train", json={"dataset": DS, "lock": "unit", "epochs": 30})
    assert r.status_code == 200
    st = _wait(client, r.json()["job_id"])
    assert st["status"] == "done" and st["kind"] == "policy_train", st
    res = st["result"]
    assert res["name"] == "unit" and res["lock"] == "unit" and res["dataset"] == DS
    assert res["train_episodes"] == TRAIN and res["eval_episodes"] == EVAL
    assert (tmp_path / "unit.policy.pt").is_file() and (tmp_path / "unit.policy.json").is_file()
    evs = _events(client, st["job_id"])
    assert [e["type"] for e in evs][-1] == "done" and sum(e["type"] == "progress" for e in evs) == 30
    assert evs[0]["epoch"] == 1 and "eval_mse" in evs[0]
    listed = client.get("/api/policy", params={"lock": "unit"}).json()["policies"]
    assert [p["name"] for p in listed] == ["unit"] and listed[0]["metrics"]["policy"]["eval"]["rmse_deg"] > 0

    # ---- shadow ----------------------------------------------------------
    r = client.post("/api/policy/evaluate", json={"dataset": DS, "lock": "unit", "mode": "shadow"})
    st = _wait(client, r.json()["job_id"])
    assert st["status"] == "done", st
    out = st["result"]
    assert out["mode"] == "shadow" and out["policy"] == "unit" and out["episodes_evaluated"] == EVAL
    assert [v["name"] for v in out["variants"]] == ["default", "lock", "perjoint"] and out["missing_locks"] == []
    s = out["summary"]
    assert s["n_episodes"] == 2 and s["variants"]["lock"]["divergence_deg"] > 0
    assert len(out["episodes"]) == 2 and "actions" not in out["episodes"][0]
    assert (tmp_path / "unit.policy.shadow.json").is_file()
    saved = client.get("/api/policy/results", params={"lock": "unit", "mode": "shadow"}).json()
    assert saved["shadow"]["summary"] == s
    evs = _events(client, st["job_id"])
    assert evs[0]["type"] == "progress" and evs[0]["i"] == 1 and evs[0]["n"] == 2 and evs[0]["row"]["episode"] == 5
    assert evs[-1]["type"] == "done" and evs[-1]["mode"] == "shadow"

    # ---- closed loop, a subset of sims -----------------------------------
    r = client.post(
        "/api/policy/evaluate", json={"dataset": DS, "lock": "unit", "mode": "closedloop", "robots": ["default", "lock"]}
    )
    st = _wait(client, r.json()["job_id"])
    assert st["status"] == "done", st
    out = st["result"]
    assert out["run_names"] == ["default", "lock", "lock+shift"]
    row = out["summary"]["variants"]["lock"]
    assert row["n_judged"] == 2 and 0.0 <= row["success_rate"] <= 1.0 and row["approach_rms_deg"] is not None
    assert len(out["runs"]) == 6 and out["runs"][0]["variant"] == "default"
    both = client.get("/api/policy/results", params={"lock": "unit"}).json()
    assert both["shadow"]["mode"] == "shadow" and both["closedloop"]["mode"] == "closedloop"

    # ---- viewer: poses per sim for one held-out episode ------------------
    r = client.get("/api/policy/viewer", params={"dataset": DS, "lock": "unit", "episode": 5, "mode": "shadow"})
    assert r.status_code == 200, r.text
    v = r.json()
    T = len(_dataset().episodes[5])
    assert v["episode"] == 5 and len(v["real_poses"]) == T and set(v["sims"]) == {"default", "lock", "perjoint"}
    assert np.asarray(v["sims"]["lock"]["poses"]).shape == (T, len(v["body_names"]), 7)
    assert v["sims"]["lock"]["divergence_deg"] > 0 and "divergence_deg" not in v["sims"]["default"]
    assert v["goal"] is not None and len(v["goal"]) == 3
    r = client.get(
        "/api/policy/viewer",
        params={"dataset": DS, "lock": "unit", "episode": 6, "mode": "closedloop", "robots": "default,lock"},
    )
    assert r.status_code == 200, r.text
    v = r.json()
    assert set(v["sims"]) == {"default", "lock", "lock+shift"} and v["sims"]["lock"]["success"] in (True, False)
    assert len(v["sims"]["lock+shift"]["q"]) == len(_dataset().episodes[6])
    # a training episode is not held out
    assert client.get("/api/policy/viewer", params={"dataset": DS, "lock": "unit", "episode": 0, "mode": "shadow"}).status_code == 400
    # unknown lock names in robots fail at submission
    r = client.post("/api/policy/evaluate", json={"dataset": DS, "lock": "unit", "mode": "shadow", "robots": ["nope"]})
    assert r.status_code == 404
    assert client.post("/api/policy/evaluate", json={"dataset": DS, "lock": "unit", "mode": "sideways"}).status_code == 422

    # ---- CLI -------------------------------------------------------------
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["policy", "evaluate", DS, "--lock", "unit", "--mode", "both", "--no-save"]) == 0
    text = buf.getvalue()
    assert "shadow" in text and "closed loop" in text and "default" in text and "lock+shift" in text


def test_evaluate_without_a_policy_is_404(client):
    r = client.post("/api/policy/evaluate", json={"dataset": DS, "lock": "unit", "mode": "shadow"})
    st = _wait(client, r.json()["job_id"])
    assert st["status"] == "error" and "no policy named" in st["error"]


def test_cli_train_prints_the_baseline_table(client, tmp_path, capsys):
    assert cli.main(["policy", "train", DS, "--lock", "unit", "--epochs", "5", "--name", "unit_cli"]) == 0
    out = capsys.readouterr().out
    assert "hold state" in out and "repeat command" in out and "unit_cli.policy.pt" in out
    assert (tmp_path / "unit_cli.policy.pt").is_file()
    listed = client.get("/api/policy", params={"lock": "unit"}).json()["policies"]
    assert [p["name"] for p in listed] == ["unit_cli"]
