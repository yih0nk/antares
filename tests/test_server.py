"""API tests. Everything heavy (MuJoCo, CMA-ES, the Hub) is stubbed through
``sys.modules`` so these run without the network or the sibling modules."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pytest
from fastapi.testclient import TestClient

from antares import server
from antares.data import Dataset, Episode

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BODIES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]
FAKE_DS = "fake/so101_unit_a"
FAKE_DS2 = "fake/so101_unit_b"
LOCK_SUFFIX = ".lock.json"


# --------------------------------------------------------------------------
# fakes matching the module contract
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeParams:
    kp: float = 10.0
    kv: float = 1.0
    damping: float = 0.1
    armature: float = 0.01
    frictionloss: float = 0.05
    latency_frames: int = 0

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def default(cls):
        return cls()


class FakeArm:
    body_names = BODIES

    def __init__(self, params=None, unit_map=None):
        self.params = params or FakeParams()
        self.applied: list = []

    def apply(self, params):
        self.params = params
        self.applied.append(params)

    def real_q(self, ep):
        return np.radians(ep.state)

    def replay(self, ep):
        # error that shrinks as kp grows, so calibrated params "help"
        return np.radians(ep.action) + 0.1 / self.params.kp

    def fk(self, q):
        T = len(q)
        poses = np.zeros((T, len(BODIES), 7))
        poses[:, :, 0] = q[:, :1]
        poses[:, :, 3] = 1.0
        return poses


@dataclass(frozen=True)
class FakeGap:
    per_joint_deg: dict
    overall_deg: float
    scored_joints: tuple
    n_frames: int

    def to_dict(self):
        return {
            "per_joint_deg": self.per_joint_deg,
            "overall_deg": self.overall_deg,
            "scored_joints": list(self.scored_joints),
            "n_frames": self.n_frames,
        }


def fake_tracking_error(real_q, sim_q, joint_names=JOINTS, exclude=("gripper",)):
    err = np.degrees(np.sqrt(np.mean((real_q - sim_q) ** 2, axis=0)))
    per = {j: float(e) for j, e in zip(joint_names, err)}
    scored = tuple(j for j in joint_names if j not in exclude)
    return FakeGap(per, float(np.mean([per[j] for j in scored])), scored, len(real_q))


def fake_aggregate(reports):
    joints = reports[0].per_joint_deg.keys()
    per = {j: float(np.mean([r.per_joint_deg[j] for r in reports])) for j in joints}
    scored = reports[0].scored_joints
    return FakeGap(per, float(np.mean([per[j] for j in scored])), scored, sum(r.n_frames for r in reports))


@dataclass
class FakeProgress:
    evaluation: int
    generation: int
    candidate_score: float
    best_score: float
    best_params: FakeParams
    best_per_joint_deg: dict
    elapsed_s: float

    def to_dict(self):
        d = self.__dict__.copy()
        d["best_params"] = self.best_params.to_dict()
        return d


@dataclass
class FakeResult:
    params: FakeParams
    baseline: FakeParams
    train_score: float = 1.0
    eval_score: float = 1.2
    baseline_train_score: float = 3.0
    baseline_eval_score: float = 3.3
    per_joint_before: dict = field(default_factory=lambda: {j: 3.0 for j in JOINTS})
    per_joint_after: dict = field(default_factory=lambda: {j: 1.0 for j in JOINTS})
    history: list = field(default_factory=lambda: [(1, 3.0), (2, 2.0), (3, 1.0)])
    n_evaluations: int = 3
    elapsed_s: float = 0.5
    train_episodes: list = field(default_factory=list)
    eval_episodes: list = field(default_factory=list)

    def to_dict(self):
        d = self.__dict__.copy()
        d["params"] = self.params.to_dict()
        d["baseline"] = self.baseline.to_dict()
        d["history"] = [list(h) for h in self.history]
        return d


def fake_split(episodes, n_train, n_eval, seed=0):
    eps = list(episodes)
    return eps[:n_train], eps[n_train : n_train + n_eval]


def fake_calibrate(train, eval, simulate=None, start=None, max_evaluations=150,
                   popsize=None, sigma0=0.3, seed=0, on_progress=None, should_stop=None):
    best = FakeParams(kp=50.0)
    for i in range(1, 4):
        if should_stop and should_stop():
            break
        if on_progress:
            on_progress(FakeProgress(i, 1, 4.0 - i, 4.0 - i, best, {j: 4.0 - i for j in JOINTS}, 0.1 * i))
    return FakeResult(
        params=best, baseline=start or FakeParams(),
        train_episodes=[e.index for e in train], eval_episodes=[e.index for e in eval],
    )


@dataclass(frozen=True)
class FakeLock:
    name: str
    dataset: str
    robot_type: str
    data_sha256: str
    params: FakeParams
    result: FakeResult
    created_at: str
    antares_version: str

    @classmethod
    def create(cls, dataset, robot_type, train_episodes, result, name=None):
        return cls(name or dataset.replace("/", "__"), dataset, robot_type,
                   f"digest-of-{len(list(train_episodes))}", result.params, result,
                   "2026-10-17T12:00:00Z", "0.0.1")

    def to_dict(self):
        return {
            "name": self.name, "dataset": self.dataset, "robot_type": self.robot_type,
            "data_sha256": self.data_sha256, "params": self.params.to_dict(),
            "result": self.result.to_dict(), "created_at": self.created_at,
            "antares_version": self.antares_version,
        }

    def save(self, path=None):
        path = path or (server.LOCK_DIR / f"{self.name}{LOCK_SUFFIX}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, d):
        params = FakeParams.from_dict(d["params"])
        return cls(d["name"], d["dataset"], d.get("robot_type", "?"), d["data_sha256"], params,
                   FakeResult(params, FakeParams()), d["created_at"], d.get("antares_version", ""))


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def make_dataset(repo_id: str, n_episodes: int = 6, T: int = 20, offset: float = 0.0) -> Dataset:
    rng = np.random.default_rng(hash(repo_id) % 1000)
    eps = []
    for i in range(n_episodes):
        action = rng.uniform(-50, 50, size=(T, 6))
        state = action + offset + rng.normal(0, 0.5, size=(T, 6))
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=state, action=action))
    return Dataset(repo_id=repo_id, robot_type="so101_follower", fps=30.0,
                   joint_names=JOINTS, episodes=tuple(eps))


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """Stub the sibling modules, inject a dataset, and point locks at tmp."""
    sim = _module("antares.sim", JOINTS=JOINTS, SimParams=FakeParams, Arm=FakeArm,
                  MODEL_PATH=tmp_path / "so101.xml",
                  model_geometry=lambda model=None: {"bodies": [], "geoms": [], "joints": []})
    gap = _module("antares.gap", GapReport=FakeGap, tracking_error=fake_tracking_error,
                  aggregate=fake_aggregate)
    cal = _module("antares.calibrate", calibrate=fake_calibrate, split_episodes=fake_split,
                  default_simulate=lambda: (lambda params, ep: (ep.state, ep.action)))
    lock = _module("antares.lock", RobotLock=FakeLock, dataset_digest=lambda eps: "deadbeef",
                   slug=lambda r: r.replace("/", "__"), LOCK_DIR=tmp_path / "locks",
                   LOCK_SUFFIX=LOCK_SUFFIX)
    for m in (sim, gap, cal, lock):
        monkeypatch.setitem(sys.modules, m.__name__, m)

    monkeypatch.setattr(server, "_arm", FakeArm())
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(server, "LOCK_SUFFIX", LOCK_SUFFIX)
    monkeypatch.setattr(server, "MESH_DIR", None)
    monkeypatch.setattr(server, "_DATASETS", {FAKE_DS: make_dataset(FAKE_DS),
                                              FAKE_DS2: make_dataset(FAKE_DS2, offset=-2.0)})
    monkeypatch.setattr(server, "_JOBS", {})

    def no_network(repo_id, cache_dir=None):
        raise RuntimeError("network disabled in tests")

    import antares.data
    monkeypatch.setattr(antares.data, "load_dataset", no_network)
    return tmp_path


@pytest.fixture
def client():
    return TestClient(server.app)


def write_lock(tmp_path, name, kp=50.0, dataset=FAKE_DS):
    d = tmp_path / "locks"
    d.mkdir(exist_ok=True)
    lock = FakeLock(name, dataset, "so101_follower", "abc", FakeParams(kp=kp),
                    FakeResult(FakeParams(kp=kp), FakeParams()), "2026-10-17T00:00:00+00:00", "0.0.1")
    (d / f"{name}{LOCK_SUFFIX}").write_text(json.dumps(lock.to_dict()))
    return lock


def sse_events(resp) -> list[dict]:
    out = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "version" in r.json()


def test_datasets_listing(client, fakes):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    body = r.json()
    ids = [d["id"] for d in body]
    for demo in server.DEMO_DATASETS:
        assert demo in ids
    demo = next(d for d in body if d["id"] == server.DEMO_DATASETS[0])
    assert demo["loaded"] is False
    loaded = next(d for d in body if d["id"] == FAKE_DS)
    assert loaded["loaded"] is True
    assert loaded["n_episodes"] == 6
    assert loaded["joints"] == list(JOINTS)
    assert loaded["robot_type"] == "so101_follower"


def test_datasets_load_returns_summary_and_load_failure_is_502(client, fakes):
    r = client.post("/api/datasets/load", params={"dataset": FAKE_DS})
    assert r.status_code == 200
    assert r.json()["loaded"] is True
    r = client.post("/api/datasets/load", params={"dataset": "nope/none"})
    assert r.status_code == 502


def test_episodes(client, fakes):
    r = client.get("/api/episodes", params={"dataset": FAKE_DS})
    assert r.status_code == 200
    eps = r.json()
    assert len(eps) == 6
    assert eps[0] == {"index": 0, "frames": 20, "duration_s": pytest.approx(20 / 30, abs=1e-3)}


@pytest.mark.parametrize("bad", ["../x.stl", "..%2Fx.stl", "x.obj", "x.xml", ".stl", "a%5Cb.stl"])
def test_mesh_route_rejects_bad_names(client, fakes, bad):
    r = client.get(f"/api/model/mesh/{bad}")
    assert r.status_code in (400, 404)
    if r.status_code == 404:
        # a 404 is only acceptable as "not a route", never a served file
        assert r.headers.get("content-type") != "model/stl"


def test_mesh_route_serves_stl(client, fakes, monkeypatch):
    mesh_dir = fakes / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "base.stl").write_bytes(b"solid fake\nendsolid fake\n")
    (mesh_dir / "secret.txt").write_text("nope")
    monkeypatch.setattr(server, "MESH_DIR", mesh_dir)
    r = client.get("/api/model/mesh/base.stl")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("model/stl")
    assert r.content.startswith(b"solid fake")
    assert client.get("/api/model/mesh/missing.stl").status_code == 404
    assert client.get("/api/model/mesh/secret.txt").status_code == 400


def test_model_geometry_stubbed(client, fakes):
    r = client.get("/api/model")
    assert r.status_code == 200
    assert set(r.json()) >= {"bodies", "geoms", "joints"}


def test_model_geometry_real():
    sim = pytest.importorskip("antares.sim")
    if not hasattr(sim, "model_geometry"):
        pytest.skip("antares.sim has no model_geometry yet")
    client = TestClient(server.app)
    r = client.get("/api/model")
    assert r.status_code == 200
    geo = r.json()
    assert geo["bodies"] and geo["geoms"]
    mesh = geo["geoms"][0]["mesh"]
    assert mesh.endswith(".stl")
    r = client.get(f"/api/model/mesh/{mesh}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("model/stl")
    assert len(r.content) > 84  # binary STL header + count


def test_replay_real_sim_integration(monkeypatch):
    """Real MuJoCo Arm + real gap on a synthetic episode (no network)."""
    sim = pytest.importorskip("antares.sim")
    pytest.importorskip("antares.gap")
    if not hasattr(sim, "Arm"):
        pytest.skip("antares.sim has no Arm yet")
    T = 12
    rng = np.random.default_rng(0)
    action = np.cumsum(rng.normal(0, 1.0, size=(T, 6)), axis=0)
    action[:, 5] = np.abs(action[:, 5])  # gripper lives in [0, 100]
    ep = Episode(index=0, fps=30.0, joint_names=JOINTS, state=action, action=action)
    ds = Dataset(repo_id="synthetic/arm", robot_type="so101_follower", fps=30.0,
                 joint_names=JOINTS, episodes=(ep,))
    monkeypatch.setattr(server, "_DATASETS", {ds.repo_id: ds})
    monkeypatch.setattr(server, "_arm", None)  # force a real Arm
    out = TestClient(server.app).get("/api/replay", params={"dataset": ds.repo_id, "episode": 0}).json()
    nbody = len(out["body_names"])
    assert out["joints"] == list(sim.JOINTS)
    assert np.asarray(out["real_q"]).shape == (T, 6)
    assert np.asarray(out["sim_q"]).shape == (T, 6)
    assert np.asarray(out["real_poses"]).shape == (T, nbody, 7)
    assert np.asarray(out["sim_poses"]).shape == (T, nbody, 7)
    assert set(out["params"]) >= {"kp", "kv", "damping", "armature", "frictionloss", "latency_frames"}
    assert np.isfinite(out["gap"]["overall_deg"])
    json.dumps(out)  # strictly JSON-serialisable (no NaN)


def test_replay_shape(client, fakes):
    r = client.get("/api/replay", params={"dataset": FAKE_DS, "episode": 2})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["dataset"] == FAKE_DS and out["episode"] == 2
    assert out["fps"] == 30.0
    assert out["joints"] == list(JOINTS)
    assert out["body_names"] == BODIES
    assert out["params"] == FakeParams().to_dict()
    assert out["lock"] is None
    T = 20
    assert np.asarray(out["real_q"]).shape == (T, 6)
    assert np.asarray(out["sim_q"]).shape == (T, 6)
    assert np.asarray(out["real_poses"]).shape == (T, len(BODIES), 7)
    assert np.asarray(out["sim_poses"]).shape == (T, len(BODIES), 7)
    gap = out["gap"]
    assert set(gap["per_joint_deg"]) == set(JOINTS)
    assert gap["overall_deg"] > 0
    assert "gripper" not in gap["scored_joints"]
    # degrees, rounded to 4 decimals
    ep = server._DATASETS[FAKE_DS].episodes[2]
    assert out["real_q"][0][0] == pytest.approx(ep.state[0, 0], abs=1e-3)
    assert all(len(str(v).split(".")[-1]) <= 4 for v in out["real_q"][0])


def test_replay_with_lock_and_baseline(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    base = client.get("/api/replay", params={"dataset": FAKE_DS, "episode": 0}).json()
    locked = client.get("/api/replay", params={"dataset": FAKE_DS, "episode": 0, "lock": "unit-a"}).json()
    assert locked["params"]["kp"] == 50.0
    assert locked["lock"] == "unit-a"
    assert locked["gap"]["overall_deg"] < base["gap"]["overall_deg"]
    again = client.get("/api/replay", params={"dataset": FAKE_DS, "episode": 0, "lock": "unit-a", "baseline": 1}).json()
    assert again["params"] == base["params"]
    assert again["lock"] is None
    assert client.get("/api/replay", params={"dataset": FAKE_DS, "episode": 0, "lock": "ghost"}).status_code == 404
    assert client.get("/api/replay", params={"dataset": FAKE_DS, "episode": 99}).status_code == 404


def test_replay_with_explicit_params(client, fakes):
    base = client.get("/api/replay", params={"dataset": FAKE_DS, "episode": 0}).json()
    params = {**FakeParams().to_dict(), "kp": 50.0}
    r = client.post("/api/replay", json={"dataset": FAKE_DS, "episode": 0, "params": params})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["params"]["kp"] == 50.0
    assert out["lock"] is None
    assert out["gap"]["overall_deg"] < base["gap"]["overall_deg"]
    bad = client.post("/api/replay", json={"dataset": FAKE_DS, "episode": 99, "params": params})
    assert bad.status_code == 404


def test_replay_with_per_joint_param_lists(client, fakes):
    # Per-joint locks carry a list per parameter; the request model must let
    # them through to SimParams.from_dict rather than rejecting them at 422.
    per_joint = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    params = {**FakeParams().to_dict(), "damping": per_joint}
    r = client.post("/api/replay", json={"dataset": FAKE_DS, "episode": 0, "params": params})
    assert r.status_code == 200, r.text
    assert r.json()["params"]["damping"] == per_joint


def test_calibrate_job_lifecycle(client, fakes):
    r = client.post("/api/calibrate", json={"dataset": FAKE_DS, "n_train": 4, "n_eval": 2,
                                            "max_evaluations": 3, "name": "unit-a"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # late connect: the whole buffer is replayed, then the stream ends on done
    with client.stream("GET", f"/api/calibrate/{job_id}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = sse_events(resp)
    types_ = [e["type"] for e in events]
    assert types_ == ["progress", "progress", "progress", "done"]
    assert events[0]["evaluation"] == 1 and events[0]["best_params"]["kp"] == 50.0
    done = events[-1]
    assert done["lock"]["name"] == "unit-a"
    assert done["lock"]["dataset"] == FAKE_DS
    assert done["lock"]["params"]["kp"] == 50.0
    assert done["result"]["eval_score"] < done["result"]["baseline_eval_score"]
    assert done["result"]["train_episodes"] == [0, 1, 2, 3]
    assert done["result"]["eval_episodes"] == [4, 5]

    # a second subscriber gets the same replayed buffer
    with client.stream("GET", f"/api/calibrate/{job_id}/events") as resp:
        assert [e["type"] for e in sse_events(resp)] == types_

    st = client.get(f"/api/calibrate/{job_id}").json()
    assert st["status"] == "done"
    assert st["events_count"] == 4
    assert st["result"]["params"]["kp"] == 50.0

    # lock was saved to LOCK_DIR with the lock module's layout, digest over the train split
    path = fakes / "locks" / f"unit-a{LOCK_SUFFIX}"
    assert path.is_file()
    saved = json.loads(path.read_text())
    assert saved["name"] == "unit-a"
    assert saved["data_sha256"] == "digest-of-4"
    assert client.get("/api/locks/unit-a").json()["params"]["kp"] == 50.0

    assert client.get("/api/calibrate/nope").status_code == 404
    assert client.get("/api/calibrate/nope/events").status_code == 404


def test_calibrate_default_name_is_slug(client, fakes):
    job_id = client.post("/api/calibrate", json={"dataset": FAKE_DS}).json()["job_id"]
    with client.stream("GET", f"/api/calibrate/{job_id}/events") as resp:
        events = sse_events(resp)
    assert events[-1]["type"] == "done"
    assert events[-1]["lock"]["name"] == "fake__so101_unit_a"
    assert (fakes / "locks" / f"fake__so101_unit_a{LOCK_SUFFIX}").is_file()


def test_calibrate_rejects_bad_input(client, fakes):
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "n_train": 0}).status_code == 422
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "name": "../evil"}).status_code == 400
    assert client.post("/api/calibrate", json={}).status_code == 422


def test_calibrate_error_event(client, fakes):
    job_id = client.post("/api/calibrate", json={"dataset": "nope/none"}).json()["job_id"]
    with client.stream("GET", f"/api/calibrate/{job_id}/events") as resp:
        events = sse_events(resp)
    assert len(events) == 1 and events[0]["type"] == "error"
    assert "nope/none" in events[0]["message"]
    st = client.get(f"/api/calibrate/{job_id}").json()
    assert st["status"] == "error" and "nope/none" in st["error"]


def test_calibrate_cancel(client, fakes, monkeypatch):
    import threading
    gate = threading.Event()

    def slow_calibrate(train, eval, should_stop=None, on_progress=None, start=None, **kw):
        gate.wait(timeout=5)
        n = 0
        while not should_stop() and n < 100:
            n += 1
        return FakeResult(params=FakeParams(kp=7), baseline=start or FakeParams(), n_evaluations=n)

    monkeypatch.setattr(sys.modules["antares.calibrate"], "calibrate", slow_calibrate)
    job_id = client.post("/api/calibrate", json={"dataset": FAKE_DS}).json()["job_id"]
    assert client.get(f"/api/calibrate/{job_id}").json()["status"] == "running"
    r = client.post(f"/api/calibrate/{job_id}/cancel")
    assert r.json()["cancelled"] is True
    gate.set()
    with client.stream("GET", f"/api/calibrate/{job_id}/events") as resp:
        events = sse_events(resp)
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["n_evaluations"] == 0  # should_stop was honoured
    assert client.get(f"/api/calibrate/{job_id}").json()["cancelled"] is True


def test_locks_list_get_delete(client, fakes):
    assert client.get("/api/locks").json() == []
    write_lock(fakes, "unit-a", kp=50.0)
    write_lock(fakes, "unit-b", kp=20.0, dataset=FAKE_DS2)
    listing = client.get("/api/locks").json()
    assert sorted(l["name"] for l in listing) == ["unit-a", "unit-b"]
    a = next(l for l in listing if l["name"] == "unit-a")
    assert a["dataset"] == FAKE_DS
    assert a["params"]["kp"] == 50.0
    assert a["scores"]["eval_score"] == 1.2
    assert a["scores"]["baseline_eval_score"] == 3.3

    full = client.get("/api/locks/unit-a").json()
    assert full["result"]["per_joint_after"]["elbow_flex"] == 1.0
    assert full["antares_version"] == "0.0.1"

    assert client.get("/api/locks/ghost").status_code == 404
    assert client.get("/api/locks/..").status_code in (400, 404)

    assert client.delete("/api/locks/unit-a").json() == {"deleted": "unit-a"}
    assert client.delete("/api/locks/unit-a").status_code == 404
    assert [l["name"] for l in client.get("/api/locks").json()] == ["unit-b"]


def test_cross_matrix(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    write_lock(fakes, "unit-b", kp=20.0, dataset=FAKE_DS2)
    r = client.get("/api/cross", params={"datasets": f"{FAKE_DS},{FAKE_DS2}",
                                         "locks": "unit-a,unit-b", "n_train": 4, "n_eval": 2})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["rows"] == [FAKE_DS, FAKE_DS2]
    assert m["cols"] == ["unit-a", "unit-b", "default"]
    assert np.asarray(m["scores"]).shape == (2, 3)
    assert np.asarray(m["per_joint"]).shape == (2, 3)
    assert set(m["per_joint"][0][0]) == set(JOINTS)
    for row in m["scores"]:
        # higher kp -> smaller fake error; default kp=10 is worst
        assert row[0] < row[1] < row[2]

    r = client.get("/api/cross", params={"datasets": FAKE_DS, "n_train": 4, "n_eval": 2})
    assert r.json()["cols"] == ["default"]
    # default split (20 + 10) does not fit a 6-episode dataset -> 400, not a crash
    r = client.get("/api/cross", params={"datasets": FAKE_DS})
    assert r.status_code == 400 and "eval split" in r.json()["detail"]
    assert client.get("/api/cross", params={"datasets": "", "locks": ""}).status_code == 400
    assert client.get("/api/cross", params={"datasets": FAKE_DS, "locks": "ghost"}).status_code == 404


@pytest.fixture
def dr_env(monkeypatch, tmp_path):
    """Real sim + real CMA-ES on a tiny synthetic dataset, with a lock on disk."""
    pytest.importorskip("antares.sim")
    from test_calibrate import make_episodes

    eps = make_episodes(6, T=40)
    ds = Dataset(repo_id="synthetic/arm", robot_type="so101_follower", fps=30.0,
                 joint_names=JOINTS, episodes=tuple(eps))
    monkeypatch.setattr(server, "_DATASETS", {ds.repo_id: ds})
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "_JOBS", {})
    lock = {
        "name": "arm", "dataset": ds.repo_id, "robot_type": ds.robot_type, "data_sha256": "x",
        "params": {"kp": 300.0, "kv": 3.0, "damping": 0.5, "armature": 0.02,
                   "frictionloss": 0.05, "latency_frames": 1},
        "result": {"eval_episodes": [4, 5]},
        "created_at": "2026-10-17T00:00:00Z", "antares_version": "0.0.1",
    }
    (tmp_path / "arm.lock.json").write_text(json.dumps(lock))
    return tmp_path


def test_dr_job_lifecycle(dr_env):
    client = TestClient(server.app)
    r = client.post("/api/dr", json={"dataset": "synthetic/arm", "n_fits": 2, "n_train": 2,
                                     "max_evaluations": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    job_id = body["job_id"]
    assert body["events_url"] == f"/api/jobs/{job_id}/events"

    with client.stream("GET", body["events_url"]) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = sse_events(resp)
    types_ = [e["type"] for e in events]
    assert types_[-1] == "done"
    assert types_.count("fit") == 2
    assert "progress" in types_
    prog = next(e for e in events if e["type"] == "progress")
    assert prog["fit"] == 0 and prog["n_fits"] == 2 and "best_params" in prog
    fit = next(e for e in events if e["type"] == "fit")
    assert set(fit) >= {"fit", "n_fits", "params", "train_score", "train_episodes"}
    # fit events arrive after their fit's progress and before the next fit starts
    assert types_.index("fit") < types_.index("done")
    assert types_[types_.index("fit") - 1] == "progress"

    ranges = events[-1]["ranges"]
    assert ranges["lock"] == "arm" and ranges["name"] == "arm"  # lock defaults to the dataset slug
    assert ranges["dataset"] == "synthetic/arm" and ranges["n_fits"] == 2 and ranges["n_train"] == 2
    assert ranges["nominal"]["kp"] == 300.0
    for subset in ranges["fit_episodes"]:
        assert len(subset) == 2 and not ({4, 5} & set(subset))  # held-out split untouched
    for name, rng_ in ranges["ranges"].items():
        assert rng_["lo"] <= rng_["nominal"] <= rng_["hi"], name
        assert len(rng_["samples"]) == 2
    assert ranges["ranges"]["latency_frames"]["scale"] == "int"

    assert (dr_env / "arm.dr.json").is_file()
    assert client.get("/api/dr/arm").json() == ranges
    listing = client.get("/api/dr").json()
    assert [l["name"] for l in listing] == ["arm"]
    assert listing[0]["ranges"]["kp"]["lo"] == ranges["ranges"]["kp"]["lo"]
    assert "samples" not in listing[0]["ranges"]["kp"]

    st = client.get(f"/api/jobs/{job_id}").json()
    assert st["status"] == "done" and st["kind"] == "dr" and st["result"] == ranges
    assert client.get(f"/api/calibrate/{job_id}").json()["kind"] == "dr"  # same registry
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/events").status_code == 404


def test_dr_named_ranges_and_cancel(dr_env):
    client = TestClient(server.app)
    r = client.post("/api/dr", json={"dataset": "synthetic/arm", "lock": "arm", "name": "arm-dr",
                                     "n_fits": 1, "n_train": 1, "max_evaluations": 1})
    job_id = r.json()["job_id"]
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        events = sse_events(resp)
    assert events[-1]["type"] == "done" and events[-1]["ranges"]["name"] == "arm-dr"
    assert events[-1]["ranges"]["lock"] == "arm"
    assert (dr_env / "arm-dr.dr.json").is_file()
    assert client.get("/api/dr/arm-dr").status_code == 200

    # cancelling a dr job goes through the shared job registry
    r = client.post("/api/dr", json={"dataset": "synthetic/arm", "n_fits": 50, "n_train": 1})
    job_id = r.json()["job_id"]
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["cancelled"] is True
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        events = sse_events(resp)
    assert events[-1]["type"] in ("done", "error")
    if events[-1]["type"] == "done":
        assert events[-1]["ranges"]["n_fits"] < 50


def test_dr_rejects_bad_input(client, fakes):
    assert client.get("/api/dr").json() == []
    assert client.get("/api/dr/ghost").status_code == 404
    assert client.get("/api/dr/..").status_code in (400, 404)
    assert client.post("/api/dr", json={"dataset": FAKE_DS, "lock": "ghost"}).status_code == 404
    write_lock(fakes, "unit-a")
    assert client.post("/api/dr", json={"dataset": FAKE_DS, "lock": "unit-a", "n_fits": 0}).status_code == 422
    assert client.post("/api/dr", json={"dataset": FAKE_DS, "lock": "unit-a", "name": "../evil"}).status_code == 400
    assert client.post("/api/dr", json={}).status_code == 422


def test_arm_lock_serialises_sim_access(client, fakes):
    import threading
    arm = server._arm
    seen = []
    orig_apply = arm.apply

    def apply(params):
        assert server._ARM_LOCK.locked()
        seen.append(params)
        orig_apply(params)

    arm.apply = apply
    ths = [threading.Thread(target=lambda: client.get("/api/replay", params={"dataset": FAKE_DS, "episode": 0}))
           for _ in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert len(seen) == 4


def test_lock_export_endpoints(monkeypatch, tmp_path):
    """Real lock module + real exporter (MuJoCo), no network."""
    import io
    import zipfile

    import mujoco

    from antares.calibrate import CalibrationResult
    from antares.lock import RobotLock
    from antares.sim import SimParams

    monkeypatch.setattr(server, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(server, "LOCK_SUFFIX", LOCK_SUFFIX)
    p = SimParams(kp=120.0, kv=1.2, damping=0.1, armature=0.02, frictionloss=0.01, latency_frames=3)
    res = CalibrationResult(
        params=p, baseline=SimParams(), train_score=1.0, eval_score=1.1,
        baseline_train_score=5.0, baseline_eval_score=5.5,
        per_joint_before={j: 5.0 for j in JOINTS}, per_joint_after={j: 1.0 for j in JOINTS},
        history=[(0, 5.0)], n_evaluations=1, elapsed_s=0.1, train_episodes=[1], eval_episodes=[0],
    )
    RobotLock(name="unit-a", dataset=FAKE_DS, robot_type="so101_follower", data_sha256="0" * 64,
              params=p, result=res, created_at="2026-10-17T00:00:00Z", antares_version="0.0.1").save(
        tmp_path / f"unit-a{LOCK_SUFFIX}")
    client = TestClient(server.app)

    r = client.get("/api/locks/unit-a/export")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/xml")
    assert r.headers["content-disposition"] == 'attachment; filename="unit-a.xml"'
    xml = r.content.decode()
    assert xml.startswith('<mujoco model="so101">') and "unit-a" in xml and FAKE_DS in xml
    assert 'kp="120.0" kv="1.2"' in xml
    (tmp_path / "dl.xml").write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "dl.xml"))  # absolute meshdir resolves
    assert m.actuator_gainprm[0, 0] == 120.0

    r = client.get("/api/locks/unit-a/export.zip")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == 'attachment; filename="unit-a.zip"'
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"unit-a/unit-a.xml", "unit-a/unit-a.antares.json", "unit-a/scene.xml"} <= names
    assert sum(n.endswith(".stl") for n in names) == 18
    assert json.loads(zf.read("unit-a/unit-a.antares.json"))["latency_frames"] == 3
    zf.extractall(tmp_path / "z")
    mujoco.MjModel.from_xml_path(str(tmp_path / "z" / "unit-a" / "scene.xml"))

    assert client.get("/api/locks/ghost/export").status_code == 404
    assert client.get("/api/locks/ghost/export.zip").status_code == 404
    assert client.get("/api/locks/..%2Fx/export").status_code in (400, 404)
    (tmp_path / f"broken{LOCK_SUFFIX}").write_text(json.dumps({"name": "broken"}))
    assert client.get("/api/locks/broken/export").status_code == 500


def test_drift_job_lifecycle(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    assert client.get("/api/drift").json() == []
    assert client.get("/api/drift/unit-a").status_code == 404

    r = client.post("/api/drift", json={"dataset": FAKE_DS, "lock": "unit-a", "window": 2, "step": 2,
                                        "max_evaluations": 3})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        events = sse_events(resp)
    assert [e["type"] for e in events] == ["window"] * 3 + ["done"]
    assert [e["index"] for e in events[:3]] == [0, 1, 2]
    assert all(e["n_windows"] == 3 for e in events[:3])
    assert events[0]["window"]["episodes"] == [0, 1]
    assert events[2]["window"]["episodes"] == [4, 5]
    # fake_calibrate: train 1.0 vs the lock's 3.0 on every window
    assert events[0]["window"]["train_score"] == 1.0
    assert events[0]["window"]["lock_score"] == 3.0
    assert events[0]["window"]["params"]["kp"] == 50.0

    scan = events[-1]["scan"]
    assert scan["lock"] == "unit-a" and scan["dataset"] == FAKE_DS
    assert scan["window"] == 2 and scan["step"] == 2 and scan["n_windows_planned"] == 3
    assert len(scan["windows"]) == 3 and scan["cancelled"] is False
    assert set(scan["trends"]) >= {"kp", "latency_frames", "lock_score", "train_score"}
    # constant fits -> nothing drifts
    assert not any(scan["drifting"].values())
    assert scan["verdict"].startswith("no parameter or score drifts")

    st = client.get(f"/api/jobs/{job_id}").json()
    assert st["kind"] == "drift" and st["status"] == "done"
    assert st["result"]["lock"] == "unit-a" and "lock" not in st

    # saved next to the lock, served back, and listed
    path = fakes / "locks" / "unit-a.drift.json"
    assert path.is_file()
    assert client.get("/api/drift/unit-a").json() == json.loads(path.read_text())
    listing = client.get("/api/drift").json()
    assert len(listing) == 1 and listing[0]["lock"] == "unit-a"
    assert listing[0]["n_windows"] == 3 and listing[0]["drifting"] == []
    # the lock listing is unaffected by the sibling drift file
    assert [l["name"] for l in client.get("/api/locks").json()] == ["unit-a"]


def test_drift_rejects_bad_input(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    assert client.post("/api/drift", json={"dataset": FAKE_DS}).status_code == 422
    assert client.post("/api/drift", json={"dataset": FAKE_DS, "lock": "unit-a", "window": 0}).status_code == 422
    assert client.post("/api/drift", json={"dataset": FAKE_DS, "lock": "ghost"}).status_code == 404
    assert client.post("/api/drift", json={"dataset": FAKE_DS, "lock": "../evil"}).status_code == 400
    assert client.get("/api/drift/..").status_code in (400, 404)
    # a window wider than the dataset is a 400 reported through the job
    job_id = client.post("/api/drift", json={"dataset": FAKE_DS, "lock": "unit-a", "window": 50}).json()["job_id"]
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        events = sse_events(resp)
    assert len(events) == 1 and events[0]["type"] == "error"
    assert "episodes" in events[0]["message"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "error"
    assert not (fakes / "locks" / "unit-a.drift.json").exists()


def test_generic_job_routes_alias_calibrate(client, fakes):
    job_id = client.post("/api/calibrate", json={"dataset": FAKE_DS, "n_train": 4, "n_eval": 2}).json()["job_id"]
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        events = sse_events(resp)
    assert events[-1]["type"] == "done"
    st = client.get(f"/api/jobs/{job_id}").json()
    assert st["kind"] == "calibrate" and st["status"] == "done"
    assert st == client.get(f"/api/calibrate/{job_id}").json()
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["cancelled"] is True
    assert client.get("/api/jobs/nope").status_code == 404
