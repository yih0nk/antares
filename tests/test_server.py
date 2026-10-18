"""API tests. Everything heavy (MuJoCo, CMA-ES, the Hub) is stubbed through
``sys.modules`` so these run without the network or the sibling modules."""

from __future__ import annotations

import json
import time
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
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
    # the servo model's name and its gain, written only with a servo model (as SimParams)
    servo: str = "position"
    servo_gain: float = 1.0

    def to_dict(self):
        d = self.__dict__.copy()
        if self.servo == "position":
            del d["servo"], d["servo_gain"]
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def default(cls):
        return cls()


class FakeCommandMap:
    def __init__(self, gain, bias):
        self.gain, self.bias = np.asarray(gain, float), np.asarray(bias, float)

    @classmethod
    def identity(cls, n=6):
        return cls(np.ones(n), np.zeros(n))

    @classmethod
    def from_dict(cls, d):
        return cls.identity() if d is None else cls(d["gain"], d["bias"])

    def to_dict(self):
        return {"gain": self.gain.tolist(), "bias": self.bias.tolist()}

    def apply(self, u):
        return np.asarray(u) * self.gain + self.bias


class FakeArm:
    body_names = BODIES
    command_map = None

    def __init__(self, params=None, unit_map=None, robot=None):
        self.params = params or FakeParams()
        self.applied: list = []

    def apply(self, params):
        self.params = params
        self.applied.append(params)

    def real_q(self, ep):
        return np.radians(ep.state)

    def replay(self, ep):
        # error that shrinks as kp grows, so calibrated params "help"
        action = ep.action if self.command_map is None else self.command_map.apply(ep.action)
        return np.radians(action) + 0.1 / self.params.kp

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
    command_map: dict | None = None
    dynamics_only: dict | None = None
    robot: str = "so101"

    @classmethod
    def create(cls, dataset, robot_type, train_episodes, result, name=None,
               command_map=None, dynamics_only=None, robot="so101"):
        return cls(name or dataset.replace("/", "__"), dataset, robot_type,
                   f"digest-of-{len(list(train_episodes))}", result.params, result,
                   "2026-10-17T12:00:00Z", "0.0.1", command_map, dynamics_only, robot)

    def to_dict(self):
        return {
            "name": self.name, "dataset": self.dataset, "robot_type": self.robot_type,
            "robot": self.robot,
            "data_sha256": self.data_sha256, "params": self.params.to_dict(),
            "result": self.result.to_dict(), "created_at": self.created_at,
            "antares_version": self.antares_version,
            **({"command_map": self.command_map} if self.command_map is not None else {}),
            **({"dynamics_only": self.dynamics_only} if self.dynamics_only is not None else {}),
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
                   FakeResult(params, FakeParams()), d["created_at"], d.get("antares_version", ""),
                   d.get("command_map"), d.get("dynamics_only"), d.get("robot") or "so101")


@dataclass
class FakeCompReport:
    fps: float
    q_des: np.ndarray
    ctrl_raw: np.ndarray
    ctrl_comp: np.ndarray
    q_sim_raw: np.ndarray
    q_sim_comp: np.ndarray
    rms: dict
    gap_raw: FakeGap
    gap_shift: FakeGap
    gap_ilc: FakeGap
    ilc_iterations: int
    ilc_history_deg: tuple
    feasibility: dict


FAKE_FEAS = {
    "velocity_factor": 1.5, "ok": True, "flagged_joints": [], "max_range_excess_deg": 0.1,
    "clipped_frames": 0, "raw_clipped_frames": 0, "per_joint": {},
}


def fake_compensate_episode(arm, ep, params, method="ilc", real_vmax_deg_s=None, **kw):
    assert method in ("shift", "ilc")
    assert real_vmax_deg_s is not None and len(real_vmax_deg_s) == 6
    arm.apply(params)
    q_des = np.radians(ep.state)
    q_sim_raw = arm.replay(ep)
    shift = 0.02 if method == "shift" else 0.01
    ctrl_comp = q_des + shift
    gap = fake_tracking_error(q_des, q_sim_raw)
    return FakeCompReport(
        fps=ep.fps, q_des=q_des, ctrl_raw=np.radians(ep.action), ctrl_comp=ctrl_comp,
        q_sim_raw=q_sim_raw, q_sim_comp=q_des + shift / 2, rms={"raw": 3.0, "shift": 2.0, "ilc": 1.0},
        gap_raw=gap, gap_shift=gap, gap_ilc=gap, ilc_iterations=4, ilc_history_deg=(2.0, 1.5, 1.0),
        feasibility=dict(FAKE_FEAS),
    )


@dataclass
class FakeSummary:
    episodes: tuple

    def to_dict(self):
        return {
            "episodes": list(self.episodes), "rms": {"raw": 3.0, "shift": 2.0, "ilc": 1.0},
            "per_joint_deg": {}, "ilc_iterations": [4] * len(self.episodes),
            "n_frames": 20 * len(self.episodes), "feasibility": dict(FAKE_FEAS),
        }


def fake_compensate_episodes(arm, episodes, params, method="ilc", **kw):
    arm.apply(params)
    return FakeSummary(tuple(e.index for e in episodes))


def fake_held_out(ds, idx):
    by = {e.index: e for e in ds.episodes}
    missing = [i for i in idx if i not in by]
    if missing:
        raise ValueError(f"episodes {missing} not in {ds.repo_id}")
    return [by[i] for i in idx]


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
    # Modules that import names from the real antares.calibrate / antares.lock
    # must be loaded before those are stubbed below, or their first import
    # (which depends on which test files were collected first) fails.
    import antares.diagnose  # noqa: F401
    import antares.drift  # noqa: F401
    import antares.specificity  # noqa: F401
    import antares.lerobot_calib  # noqa: F401
    import antares.sensor  # noqa: F401

    sim = _module("antares.sim", JOINTS=JOINTS, SimParams=FakeParams, Arm=FakeArm, CommandMap=FakeCommandMap,
                  MODEL_PATH=tmp_path / "so101.xml",
                  model_geometry=lambda model=None: {"bodies": [], "geoms": [], "joints": []})
    gap = _module("antares.gap", GapReport=FakeGap, tracking_error=fake_tracking_error,
                  aggregate=fake_aggregate)
    from antares.calibrate import scoring_names  # the real one: pure, reads the robot off its argument

    cal = _module("antares.calibrate", calibrate=fake_calibrate, split_episodes=fake_split, PENALTY=1e3,
                  default_simulate=lambda command_map=None, robot=None: (lambda params, ep: (ep.state, ep.action)),
                  scoring_names=scoring_names)
    lock = _module("antares.lock", RobotLock=FakeLock, dataset_digest=lambda eps: "deadbeef",
                   slug=lambda r: r.replace("/", "__"), LOCK_DIR=tmp_path / "locks",
                   LOCK_SUFFIX=LOCK_SUFFIX,
                   # no seed spread in the fakes: only latency is pinned, as for a real old lock
                   pinned_params=lambda lock: ["latency_frames"], seed_info=lambda lock: None)
    comp = _module("antares.compensate", compensate_episode=fake_compensate_episode,
                   compensate_episodes=fake_compensate_episodes, held_out_episodes=fake_held_out,
                   max_joint_velocity=lambda episodes, arm: np.full(6, 100.0))
    for m in (sim, gap, cal, lock, comp):
        monkeypatch.setitem(sys.modules, m.__name__, m)

    monkeypatch.setattr(server, "_ARMS", {"so101": FakeArm()})
    monkeypatch.setattr(server, "_ARM_LOCKS", {})
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(server, "LOCK_SUFFIX", LOCK_SUFFIX)
    monkeypatch.setattr(server, "MESH_DIR", None)
    monkeypatch.setattr(server, "_DATASETS", {FAKE_DS: make_dataset(FAKE_DS),
                                              FAKE_DS2: make_dataset(FAKE_DS2, offset=-2.0)})
    monkeypatch.setattr(server, "_JOBS", {})
    monkeypatch.setattr(server, "_BASELINES", {})

    def no_network(repo_id, cache_dir=None):
        raise RuntimeError("network disabled in tests")

    import antares.data
    monkeypatch.setattr(antares.data, "load_dataset", no_network)
    monkeypatch.setattr(server, "fetch_hub_info", no_network)
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    return tmp_path


@pytest.fixture
def client():
    return TestClient(server.app)


def write_lock(tmp_path, name, kp=50.0, dataset=FAKE_DS, command_map=None):
    d = tmp_path / "locks"
    d.mkdir(exist_ok=True)
    lock = FakeLock(name, dataset, "so101_follower", "abc", FakeParams(kp=kp),
                    FakeResult(FakeParams(kp=kp), FakeParams()), "2026-10-17T00:00:00+00:00", "0.0.1")
    body = lock.to_dict()
    if command_map is not None:
        body["command_map"] = command_map
    (d / f"{name}{LOCK_SUFFIX}").write_text(json.dumps(body))
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
    monkeypatch.setattr(server, "_ARMS", {})  # force a real Arm
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
    assert out["dataset"] == FAKE_DS and out["episode"] == 2 and out["robot"] == "so101"
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


def test_replay_with_servo_model_params(client, fakes):
    # A --servo calibration's best_params carry the string ``servo``; the web
    # UI posts them for the live ghost, so the request model must accept it
    # and hand it to SimParams.from_dict rather than answering 422.
    params = {**FakeParams().to_dict(), "servo": "sts3215", "servo_gain": 1.2}
    r = client.post("/api/replay", json={"dataset": FAKE_DS, "episode": 0, "params": params})
    assert r.status_code == 200, r.text
    assert r.json()["params"]["servo"] == "sts3215" and r.json()["params"]["servo_gain"] == 1.2
    # a value that is neither a number, a list nor a string is still a 422
    bad = client.post("/api/replay", json={"dataset": FAKE_DS, "episode": 0, "params": {**params, "kp": {"a": 1}}})
    assert bad.status_code == 422


def test_replay_with_per_joint_param_lists(client, fakes):
    # Per-joint locks carry a list per parameter; the request model must let
    # them through to SimParams.from_dict rather than rejecting them at 422.
    per_joint = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    params = {**FakeParams().to_dict(), "damping": per_joint}
    r = client.post("/api/replay", json={"dataset": FAKE_DS, "episode": 0, "params": params})
    assert r.status_code == 200, r.text
    assert r.json()["params"]["damping"] == per_joint


def test_compensate_endpoint(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    r = client.get("/api/compensate", params={"dataset": FAKE_DS, "episode": 1, "lock": "unit-a"})
    assert r.status_code == 200, r.text
    out = r.json()
    json.dumps(out)
    assert out["dataset"] == FAKE_DS and out["episode"] == 1 and out["fps"] == 30.0
    assert out["lock"] == "unit-a" and out["method"] == "ilc"
    assert out["params"]["kp"] == 50.0
    assert out["validation"] == "sim-only" and "hardware" in out["note"]
    assert out["joints"] == list(JOINTS) and out["body_names"] == BODIES
    T = 20
    for key in ("q_des", "ctrl_raw", "ctrl_comp", "q_sim_raw", "q_sim_comp"):
        assert np.asarray(out[key]).shape == (T, 6), key
    assert np.asarray(out["sim_comp_poses"]).shape == (T, len(BODIES), 7)
    assert out["rms"] == {"raw": 3.0, "shift": 2.0, "ilc": 1.0}
    assert set(out["gap"]) == {"raw", "shift", "ilc"}
    assert out["ilc_iterations"] == 4 and out["ilc_history_deg"] == [2.0, 1.5, 1.0]
    assert out["feasibility"]["ok"] is True
    # degrees: q_des is the real measured trajectory
    ep = server._DATASETS[FAKE_DS].episodes[1]
    assert out["q_des"][0][0] == pytest.approx(ep.state[0, 0], abs=1e-3)
    # ILC vs shift differ in which commands come back
    ilc_comp = np.asarray(out["ctrl_comp"])
    shift = client.get("/api/compensate", params={"dataset": FAKE_DS, "episode": 1, "lock": "unit-a",
                                                   "method": "shift"}).json()
    assert shift["method"] == "shift"
    assert np.all(np.asarray(shift["ctrl_comp"]) > ilc_comp)
    # no lock -> default params, lock null
    bare = client.get("/api/compensate", params={"dataset": FAKE_DS, "episode": 1}).json()
    assert bare["lock"] is None and bare["params"] == FakeParams().to_dict()

    assert client.get("/api/compensate", params={"dataset": FAKE_DS, "episode": 1, "method": "pid"}).status_code == 400
    assert client.get("/api/compensate", params={"dataset": FAKE_DS, "episode": 1, "lock": "ghost"}).status_code == 404
    assert client.get("/api/compensate", params={"dataset": FAKE_DS, "episode": 99}).status_code == 404
    assert client.get("/api/compensate", params={"dataset": FAKE_DS}).status_code == 422


def test_compensate_heldout_helper(fakes):
    from fastapi import HTTPException

    lock = write_lock(fakes, "unit-a", kp=50.0).to_dict()
    lock["result"]["eval_episodes"] = [4, 5]
    out = server.compensate_heldout(server._DATASETS[FAKE_DS], lock)
    json.dumps(out)
    assert out["episodes"] == [4, 5] and out["n_frames"] == 40
    assert out["rms"]["ilc"] < out["rms"]["raw"]
    assert out["lock"] == "unit-a" and out["params"]["kp"] == 50.0 and out["validation"] == "sim-only"
    assert out["ilc_iterations"] == [4, 4]
    assert server._ARMS["so101"].params.kp == 50.0

    lock["result"]["eval_episodes"] = []
    with pytest.raises(HTTPException) as e:
        server.compensate_heldout(server._DATASETS[FAKE_DS], lock)
    assert e.value.status_code == 400
    lock["result"]["eval_episodes"] = [42]
    with pytest.raises(HTTPException) as e:
        server.compensate_heldout(server._DATASETS[FAKE_DS], lock)
    assert e.value.status_code == 400 and "42" in e.value.detail
    with pytest.raises(HTTPException):
        server.compensate_heldout(server._DATASETS[FAKE_DS], lock, method="pid")


def test_compensate_real_sim_integration(monkeypatch):
    """Real MuJoCo Arm + real compensate on a synthetic episode (no network)."""
    sim = pytest.importorskip("antares.sim")
    pytest.importorskip("antares.compensate")
    T = 60
    t = np.linspace(0, 1, T)
    ramp = 0.5 * (1 - np.cos(np.pi * np.minimum(t / 0.4, 1.0)))
    state = np.array([0.0, -40.0, 30.0, 10.0, 0.0, 20.0]) + 25.0 * np.sin(2 * np.pi * t[:, None] + np.arange(6)) * ramp[:, None]
    state[:, 5] = np.clip(state[:, 5], 0, 100)
    action = np.vstack([state[3:], np.repeat(state[-1:], 3, axis=0)])
    ep = Episode(index=0, fps=30.0, joint_names=JOINTS, state=state, action=action)
    ds = Dataset(repo_id="synthetic/arm", robot_type="so101_follower", fps=30.0,
                 joint_names=JOINTS, episodes=(ep,))
    monkeypatch.setattr(server, "_DATASETS", {ds.repo_id: ds})
    monkeypatch.setattr(server, "_ARMS", {})  # force a real Arm
    params = sim.SimParams(kp=90.0, kv=1.5, damping=0.05, armature=0.02, frictionloss=0.025, latency_frames=4)
    out = server.compensate_episode_dict(ds, 0, params, method="ilc", lock=None)
    json.dumps(out)
    assert np.asarray(out["ctrl_comp"]).shape == (T, 6)
    assert np.asarray(out["sim_comp_poses"]).shape == (T, len(out["body_names"]), 7)
    # raw reproduces the replay gap exactly
    replay = server.replay_episode(ds, 0, params)
    assert out["rms"]["raw"] == pytest.approx(replay["gap"]["overall_deg"], abs=1e-3)
    assert out["rms"]["ilc"] <= out["rms"]["shift"] < out["rms"]["raw"]
    assert out["ilc_iterations"] >= 1
    assert set(out["feasibility"]["per_joint"]) == set(JOINTS)


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


def test_calibrate_refuses_to_overwrite_a_saved_lock(client, fakes):
    """Two tabs (or a CLI run) starting a fit under one name must not clobber it."""
    write_lock(fakes, "unit-a", kp=50.0)
    r = client.post("/api/calibrate", json={"dataset": FAKE_DS, "name": "unit-a"})
    assert r.status_code == 409 and "already exists" in r.json()["detail"]
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "name": "unit-fresh"}).status_code == 200


def test_calibrate_rejects_bad_input(client, fakes):
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "n_train": 0}).status_code == 422
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "name": "../evil"}).status_code == 400
    assert client.post("/api/calibrate", json={}).status_code == 422
    # a factor selection the stage cannot fit, or that fits nothing, is a usage error
    r = client.post("/api/calibrate", json={"dataset": FAKE_DS, "factors": []})
    assert r.status_code == 400 and "nothing to fit" in r.json()["detail"]
    r = client.post("/api/calibrate", json={"dataset": FAKE_DS, "factors": ["+sensor_gain"]})
    assert r.status_code == 400 and "sensor" in r.json()["detail"]
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "factors": ["+nope"]}).status_code == 400


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


def test_calibrate_per_joint_option(client, fakes, monkeypatch):
    calls = []

    def recording_calibrate(train, eval, start=None, **kw):
        calls.append(kw)
        return FakeResult(params=FakeParams(kp=9), baseline=start or FakeParams())

    monkeypatch.setattr(sys.modules["antares.calibrate"], "calibrate", recording_calibrate)

    def run(body):
        job_id = client.post("/api/calibrate", json=body).json()["job_id"]
        with client.stream("GET", f"/api/calibrate/{job_id}/events") as resp:
            events = sse_events(resp)
        assert events[-1]["type"] == "done", events[-1]
        return events

    # default: shared only, and the per-joint kwargs are not even passed
    run({"dataset": FAKE_DS, "name": "shared-a"})
    assert "per_joint" not in calls[-1]
    assert client.get("/api/locks").json()[0]["space"] == "shared"

    # opt in: both per-joint kwargs reach calibrate
    run({"dataset": FAKE_DS, "name": "pj-a", "per_joint": True, "per_joint_max_evaluations": 42})
    assert calls[-1]["per_joint"] is True and calls[-1]["per_joint_max_evaluations"] == 42
    run({"dataset": FAKE_DS, "per_joint": True})
    assert calls[-1]["per_joint_max_evaluations"] == 600  # default budget

    # best of N seeds: forwarded only when asked for, bounded at 10
    run({"dataset": FAKE_DS, "name": "seeds-a"})
    assert "n_seeds" not in calls[-1]
    run({"dataset": FAKE_DS, "name": "seeds-b", "n_seeds": 3})
    assert calls[-1]["n_seeds"] == 3
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "n_seeds": 0}).status_code == 422
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "n_seeds": 11}).status_code == 422

    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "per_joint_max_evaluations": 0}).status_code == 422


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


def test_cross_matrix_reads_locks_from_another_directory(fakes):
    other = fakes / "fleet"
    other.mkdir()
    lock = FakeLock("fleet-a", FAKE_DS, "so101_follower", "abc", FakeParams(kp=40.0),
                    FakeResult(FakeParams(kp=40.0), FakeParams()), "2026-10-17T00:00:00+00:00", "0.0.1")
    (other / f"fleet-a{LOCK_SUFFIX}").write_text(json.dumps(lock.to_dict()))
    m = server.cross_matrix([FAKE_DS], ["fleet-a"], n_train=4, n_eval=2, lock_dir=other)
    assert m["cols"] == ["fleet-a", "default"] and m["scores"][0][0] < m["scores"][0][1]
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        server.cross_matrix([FAKE_DS], ["fleet-a"], n_train=4, n_eval=2)  # not in LOCK_DIR


def test_put_dataset_registers_for_get_dataset(fakes):
    ds = make_dataset("fake/so101_unit_c")
    server.put_dataset(ds)
    assert server.get_dataset("fake/so101_unit_c") is ds
    assert server.dataset_summary("fake/so101_unit_c")["loaded"] is True


def test_fleet_endpoints(client, fakes, monkeypatch):
    fleet_dir = fakes / "fleet"
    monkeypatch.setattr(server, "FLEET_DIR", fleet_dir)
    r = client.get("/api/fleet")
    assert r.status_code == 404 and "fleet scan" in r.json()["detail"]
    assert client.get("/api/fleet/cross").status_code == 404

    fleet_dir.mkdir()
    report = {"robots": [{"dataset": FAKE_DS, "params": {"kp": 50.0}}], "summary": {"n_robots": 1}, "cross": None}
    (fleet_dir / "fleet.json").write_text(json.dumps(report))
    r = client.get("/api/fleet")
    assert r.status_code == 200 and r.json() == report
    assert client.get("/api/fleet/cross").status_code == 404  # report without a cross matrix

    report["cross"] = {"rows": [FAKE_DS], "cols": ["a", "default"], "scores": [[1.0, 2.0]], "summary": {"n": 1}}
    (fleet_dir / "fleet.json").write_text(json.dumps(report))
    assert client.get("/api/fleet/cross").json() == report["cross"]

    (fleet_dir / "fleet.json").write_text("{not json")
    assert client.get("/api/fleet").status_code == 500


def test_arm_lock_serialises_sim_access(client, fakes):
    import threading
    arm = server._ARMS["so101"]
    seen = []
    orig_apply = arm.apply

    def apply(params):
        assert server.arm_lock("so101").locked()
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


def test_check_passes_on_matching_robot(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)  # FakeResult: eval_score 1.2, per joint 1.0
    r = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["passed"] is True and out["verdict"] == "ok"
    assert out["lock"] == "unit-a" and out["dataset"] == FAKE_DS
    assert out["reference"] == pytest.approx(1.2)
    assert out["limit"] == pytest.approx(1.8)
    assert out["score"] < out["limit"]
    assert out["n_episodes"] == 6 and out["episodes"] == [0, 1, 2, 3, 4, 5]
    assert set(out["per_joint"]) == set(JOINTS)
    pj = out["per_joint"]["elbow_flex"]
    assert set(pj) == {"score", "reference", "limit", "passed"}
    assert pj["reference"] == 1.0 and pj["limit"] == pytest.approx(1.5) and pj["passed"] is True
    assert out["worst_joint"] in JOINTS and out["worst_joint"] != "gripper"
    assert out["ratio"] == 1.5 and out["threshold_deg"] is None
    json.dumps(out)


def test_check_detects_drift_and_threshold_overrides(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    # FAKE_DS2 is recorded 2 units off the command: a different (drifted) arm
    r = client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-a"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["passed"] is False and out["verdict"] == "drifted"
    assert out["score"] > out["limit"]
    assert all(not j["passed"] for j in out["per_joint"].values())
    loose = client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-a",
                                            "threshold_deg": 10.0}).json()
    assert loose["passed"] is True and loose["limit"] == 10.0
    tight = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a", "ratio": 0.1}).json()
    assert tight["passed"] is False and tight["limit"] == pytest.approx(0.12)


def test_check_explain_adds_a_diagnosis_only_on_drift(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    ok = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a", "explain": True}).json()
    assert ok["verdict"] == "ok" and "diagnosis" not in ok
    r = client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-a", "explain": True,
                                        "explain_evaluations": 20})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["verdict"] == "drifted"
    d = out["diagnosis"]
    assert d["lock"] == "unit-a" and d["dataset"] == FAKE_DS2
    assert set(d) >= {"stale_deg", "refit_deg", "causes", "changes", "explained", "summary"}
    assert d["refit_episodes"] and d["held_out_episodes"]
    json.dumps(out)
    assert client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-a", "explain": True,
                                           "explain_evaluations": 1}).status_code == 422


def test_check_explain_keeps_the_verdict_when_the_diagnosis_cannot_run(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    # one scored episode is enough for a verdict but not for a refit/held-out split
    r = client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-a", "explain": True,
                                        "last": 1, "min_episodes": 1})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["verdict"] == "drifted" and out["diagnosis"] is None
    assert "two episodes" in out["diagnosis_error"]


def test_diagnose_job_streams_and_finishes(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    assert client.post("/api/diagnose", json={"dataset": FAKE_DS2, "lock": "ghost"}).status_code == 404
    assert client.post("/api/diagnose", json={"dataset": FAKE_DS2, "lock": "unit-a",
                                              "max_evaluations": 2}).status_code == 422
    job_id = client.post("/api/diagnose", json={"dataset": FAKE_DS2, "lock": "unit-a",
                                                "max_evaluations": 24}).json()["job_id"]
    for _ in range(200):
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] != "running":
            break
        time.sleep(0.02)
    assert st["status"] == "done" and st["kind"] == "diagnose", st
    assert st["result"]["lock"] == "unit-a" and 6 <= st["result"]["n_evaluations"] <= 24
    events = [json.loads(line[6:]) for line in client.get(f"/api/jobs/{job_id}/events").text.splitlines()
              if line.startswith("data: ")]
    assert [e["type"] for e in events][-1] == "done"
    assert [e["evaluation"] for e in events if e["type"] == "fit"] == list(range(1, st["result"]["n_evaluations"] + 1))


def test_check_last_and_insufficient_data(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    out = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a", "last": 2}).json()
    assert out["verdict"] == "insufficient_data" and out["passed"] is False
    assert out["score"] is None and out["per_joint"] == {}
    assert out["n_episodes"] == 2 and out["episodes"] == [4, 5]
    out = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a",
                                          "last": 2, "min_episodes": 2}).json()
    assert out["verdict"] == "ok" and out["episodes"] == [4, 5]


def test_check_rejects_bad_input(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    assert client.post("/api/check", json={"dataset": FAKE_DS, "lock": "ghost"}).status_code == 404
    assert client.post("/api/check", json={"dataset": FAKE_DS, "lock": "../x"}).status_code == 400
    assert client.post("/api/check", json={"dataset": "nope/none", "lock": "unit-a"}).status_code == 502
    assert client.post("/api/check", json={"dataset": FAKE_DS}).status_code == 422
    assert client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a", "ratio": 0}).status_code == 422
    assert client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a", "last": 0}).status_code == 422
    assert client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a",
                                           "threshold_deg": -1}).status_code == 422


def test_check_notifies_a_webhook_server_side(client, fakes, monkeypatch):
    import antares.notify as notify

    write_lock(fakes, "unit-a", kp=50.0)
    sent = []
    monkeypatch.setattr(notify, "send", lambda url, payload, timeout=10.0: sent.append((url, payload)) or 200)
    slack_url = "https://hooks.slack.com/services/T0/B0/xyz"

    out = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a"}).json()
    assert out["notified"] is None and out["notify_error"] is None

    # ok is quiet by default
    out = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a", "notify_url": slack_url}).json()
    assert out["verdict"] == "ok" and out["notified"] is False and out["notify_error"] is None
    assert sent == []

    # drifted posts, as Slack blocks for a hooks.slack.com URL
    out = client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-a", "notify_url": slack_url}).json()
    assert out["verdict"] == "drifted" and out["notified"] is True and out["notify_error"] is None
    ((url, payload),) = sent
    assert url == slack_url
    assert payload["blocks"][0]["type"] == "header"
    assert payload["text"].startswith(f"DRIFTED unit-a on {FAKE_DS2}: ")

    # any other host gets the generic envelope; notify_when widens the policy
    out = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a",
                                          "notify_url": "https://example.com/hook",
                                          "notify_when": "always"}).json()
    assert out["notified"] is True
    payload = sent[1][1]
    assert payload["event"] == "antares.check" and payload["verdict"] == "ok"
    assert payload["result"]["lock"] == "unit-a" and payload["result"]["verdict"] == "ok"
    out = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a", "last": 2,
                                          "notify_url": "https://example.com/hook",
                                          "notify_when": "failure", "notify_format": "slack"}).json()
    assert out["verdict"] == "insufficient_data" and out["notified"] is True
    assert "blocks" in sent[2][1] and sent[2][1]["text"].startswith("NO DATA unit-a")

    # a dead webhook never changes the verdict or the status code
    def boom(url, payload, timeout=10.0):
        raise notify.NotifyError("webhook returned HTTP 500: invalid_payload")

    monkeypatch.setattr(notify, "send", boom)
    r = client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-a", "notify_url": "https://example.com/hook"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["verdict"] == "drifted" and out["passed"] is False
    assert out["notified"] is False and out["notify_error"] == "webhook returned HTTP 500: invalid_payload"

    assert client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a", "notify_url": slack_url,
                                           "notify_when": "sometimes"}).status_code == 422
    assert client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a", "notify_url": slack_url,
                                           "notify_format": "xml"}).status_code == 422


def test_check_holds_arm_lock(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    arm = server._ARMS["so101"]
    orig = arm.replay

    def replay(ep):
        assert server.arm_lock("so101").locked()
        return orig(ep)

    arm.replay = replay
    assert client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a"}).status_code == 200


def test_get_dataset_reads_local_directory(fakes, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "recording"
    names = [f"{j}.pos" for j in JOINTS]
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "robot_type": "so101_follower", "fps": 30,
        "features": {"action": {"names": names}, "observation.state": {"names": names}},
    }))
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [0, 0, 1, 1], "frame_index": [0, 1, 0, 1],
        "observation.state": [[0.0] * 6] * 4, "action": [[1.0] * 6] * 4,
    }), root / "data" / "chunk-000" / "file-000.parquet")

    ds = server.get_dataset(str(root))
    assert ds.repo_id == str(root)
    assert ds.joint_names == JOINTS and len(ds.episodes) == 2
    assert server.get_dataset(str(root)) is ds  # cached
    assert server.dataset_summary(str(root))["loaded"] is True


# --------------------------------------------------------------------------
# residual / contact events
# --------------------------------------------------------------------------

PICK_DS = "fake/so101_pick"


def make_pick_dataset(repo_id: str = PICK_DS, n_episodes: int = 4, T: int = 300) -> Dataset:
    """Pick-and-place-like episodes: the gripper opens at 100, closes at 150
    (the grasp) and opens again at 220; the real arm deviates from the
    commanded motion for 30 frames around the grasp."""
    rng = np.random.default_rng(7)
    eps = []
    for i in range(n_episodes):
        action = np.zeros((T, 6))
        action[:, :5] = np.cumsum(rng.normal(0, 0.2, size=(T, 5)), axis=0)
        g = np.full(T, 1.5)
        g[100:150] = 22.0
        g[220:240] = 22.0
        action[:, 5] = g
        state = action + rng.normal(0, 0.3, size=(T, 6))
        state[145:175, 2] += 6.0  # elbow held back while grasping
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=state, action=action))
    return Dataset(repo_id=repo_id, robot_type="so101_follower", fps=30.0, joint_names=JOINTS, episodes=tuple(eps))


@pytest.fixture
def pick(fakes):
    server._DATASETS[PICK_DS] = make_pick_dataset()
    return fakes


def test_residual_endpoint_shape_and_events(client, pick):
    r = client.get("/api/residual", params={"dataset": PICK_DS, "episode": 1})
    assert r.status_code == 200, r.text
    out = r.json()
    T = 300
    assert out["dataset"] == PICK_DS and out["episode"] == 1 and out["fps"] == 30.0
    assert out["lock"] is None and out["params"] == FakeParams().to_dict()
    assert out["joints"] == list(JOINTS) and out["scored_joints"] == list(JOINTS[:5])
    assert np.asarray(out["residual"]).shape == (T, 6)
    assert np.asarray(out["z"]).shape == (T, 6)
    assert np.asarray(out["gripper_cmd"]).shape == (T,)
    assert out["grasps"] == [150, 240] and out["releases"] == [100, 220]
    assert out["grasp_windows"] == [[135, 166], [225, 256]]
    thr = out["gripper_thresholds"]
    assert 1.5 < thr["closed_below"] < thr["open_above"] < 22.0
    # residual is real - sim in degrees: the planted elbow bump shows up
    res = np.asarray(out["residual"])
    assert res[160, 2] > 4.0
    # the baseline came from the fallback (all episodes), on free-motion frames
    bl = out["baseline"]
    assert bl["joints"] == list(JOINTS) and bl["n_episodes"] == 4 and bl["free_frames_only"]
    assert sorted(bl["episodes"]) == [0, 1, 2, 3]
    assert len(bl["scale_deg"]) == 6 and all(s > 0 for s in bl["scale_deg"])
    # exactly one event, on the elbow, around the grasp
    evs = out["events"]
    assert len(evs) == 1, evs
    ev = evs[0]
    assert ev["joints"] == ["elbow_flex"] and ev["kind"] == "contact"
    assert 140 <= ev["start"] <= 146 and 174 <= ev["end"] <= 180
    assert ev["peak_z"] > 4.0 and ev["peak_deg"] > 4.0
    v = out["validation"]
    assert v["n_grasps"] == 2 and v["n_grasps_hit"] == 1
    assert v["n_events"] == 1 and v["n_events_in_grasp"] == 1 and v["n_events_far"] == 0
    assert v["energy_in_grasp"] > 0.5 * v["energy_total"]
    assert out["z_threshold"] == 4.0 and out["min_frames"] == 5 and out["warmup_frames"] == server.WARMUP_FRAMES


def test_residual_endpoint_params_and_errors(client, pick):
    assert client.get("/api/residual", params={"dataset": PICK_DS, "episode": 99}).status_code == 404
    assert client.get("/api/residual", params={"dataset": PICK_DS, "episode": 0, "lock": "ghost"}).status_code == 404
    assert client.get("/api/residual", params={"dataset": PICK_DS, "episode": 0, "z": 0}).status_code == 422
    assert client.get("/api/residual", params={"dataset": PICK_DS, "episode": 0, "min_frames": 0}).status_code == 422
    # a huge threshold finds nothing
    out = client.get("/api/residual", params={"dataset": PICK_DS, "episode": 0, "z": 1000}).json()
    assert out["events"] == [] and out["z_threshold"] == 1000.0


def test_residual_with_lock_uses_lock_params_and_eval_split(client, pick):
    write_lock(pick, "unit-a", kp=50.0, dataset=PICK_DS)
    path = pick / "locks" / f"unit-a{LOCK_SUFFIX}"
    d = json.loads(path.read_text())
    d["result"]["eval_episodes"] = [2, 3]
    path.write_text(json.dumps(d))
    out = client.get("/api/residual", params={"dataset": PICK_DS, "episode": 0, "lock": "unit-a"}).json()
    assert out["lock"] == "unit-a" and out["params"]["kp"] == 50.0
    assert sorted(out["baseline"]["episodes"]) == [2, 3]
    # the baseline is cached per (dataset, lock)
    assert set(server._BASELINES) == {(PICK_DS, "unit-a")}
    client.get("/api/residual", params={"dataset": PICK_DS, "episode": 1, "lock": "unit-a"})
    assert len(server._BASELINES) == 1
    client.get("/api/residual", params={"dataset": PICK_DS, "episode": 1})
    assert set(server._BASELINES) == {(PICK_DS, "unit-a"), (PICK_DS, "")}


def test_events_summary_endpoint(client, pick):
    r = client.get("/api/events", params={"dataset": PICK_DS})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["dataset"] == PICK_DS and out["lock"] is None
    assert len(out["episodes"]) == 4
    row = out["episodes"][0]
    assert row["episode"] == 0 and row["n_frames"] == 300 and row["held_out"] is True
    assert row["n_grasps"] == 2 and len(row["events"]) == 1
    s = out["summary"]
    assert s["n_episodes"] == 4 and s["n_grasps"] == 8 and s["n_grasps_hit"] == 4
    assert s["n_events"] == 4 and s["precision"] == 1.0 and s["recall"] == 0.5 and s["far_fraction"] == 0.0
    assert s["energy_in_grasp_fraction"] > s["chance"] and s["lift"] > 1.0
    assert out["summary_held_out"]["n_episodes"] == 4
    assert client.get("/api/events", params={"dataset": PICK_DS, "lock": "ghost"}).status_code == 404


SHIFT_MAP = {"gain": [1.0] * 6, "bias": [-2.0] * 6}  # FAKE_DS2 state = action - 2


def test_replay_uses_lock_command_map(client, fakes):
    write_lock(fakes, "unit-b", kp=50.0, dataset=FAKE_DS2, command_map=SHIFT_MAP)
    write_lock(fakes, "unit-b-plain", kp=50.0, dataset=FAKE_DS2)
    q = {"dataset": FAKE_DS2, "episode": 0}
    mapped = client.get("/api/replay", params={**q, "lock": "unit-b"}).json()
    plain = client.get("/api/replay", params={**q, "lock": "unit-b-plain"}).json()
    assert mapped["command_map"] == SHIFT_MAP
    assert plain["command_map"] is None
    assert mapped["gap"]["overall_deg"] < 0.5 * plain["gap"]["overall_deg"]
    # a later plain replay resets the shared arm to identity
    again = client.get("/api/replay", params={**q, "lock": "unit-b-plain"}).json()
    assert again["gap"]["overall_deg"] == plain["gap"]["overall_deg"]


def test_post_replay_accepts_command_map(client, fakes):
    body = {"dataset": FAKE_DS2, "episode": 0, "params": FakeParams(kp=50.0).to_dict()}
    plain = client.post("/api/replay", json=body).json()
    mapped = client.post("/api/replay", json={**body, "command_map": SHIFT_MAP}).json()
    assert mapped["command_map"] == SHIFT_MAP
    assert mapped["gap"]["overall_deg"] < plain["gap"]["overall_deg"]
    # naming a lock keeps its command map: hand-tuned params start from the
    # lock's own physics, not the identity map
    write_lock(fakes, "unit-b", kp=50.0, dataset=FAKE_DS2, command_map=SHIFT_MAP)
    from_lock = client.post("/api/replay", json={**body, "lock": "unit-b"}).json()
    assert from_lock["command_map"] == SHIFT_MAP and from_lock["lock"] is None
    assert from_lock["gap"]["overall_deg"] == mapped["gap"]["overall_deg"]
    # an explicit command_map still wins over the lock's
    ident = {"gain": [1.0] * 6, "bias": [0.0] * 6}
    override = client.post("/api/replay", json={**body, "lock": "unit-b", "command_map": ident}).json()
    assert override["gap"]["overall_deg"] == plain["gap"]["overall_deg"]
    assert client.post("/api/replay", json={**body, "lock": "ghost"}).status_code == 404
    bad = client.post("/api/replay", json={**body, "command_map": {"gain": [1.0] * 6}})
    assert bad.status_code == 400
    # the map exactly as a sensor-fitted lock stores it, "mode" and all
    as_saved = client.post("/api/replay", json={**body, "command_map": {**SHIFT_MAP, "mode": "sim"}})
    assert as_saved.status_code == 200 and as_saved.json()["command_map"] == SHIFT_MAP
    # a wrong-length map is a 400, not a 500 from inside the sim
    short = client.post("/api/replay", json={**body, "command_map": {"gain": [1.0] * 3, "bias": [0.0] * 3}})
    assert short.status_code == 400 and "command_map" in short.json()["detail"]


def test_cross_matrix_uses_command_map(client, fakes):
    write_lock(fakes, "unit-b", kp=50.0, dataset=FAKE_DS2, command_map=SHIFT_MAP)
    write_lock(fakes, "unit-b-plain", kp=50.0, dataset=FAKE_DS2)
    m = server.cross_matrix([FAKE_DS2], ["unit-b", "unit-b-plain"], n_train=4, n_eval=2)
    mapped, plain, _ = m["scores"][0]
    assert mapped < plain


def test_calibrate_with_fit_sensor_saves_command_map(client, fakes, monkeypatch):
    calls = {}

    def fake_calibrate_with_sensor(train, eval, **kw):
        calls.update(kw)
        res = fake_calibrate(train, eval, **{k: v for k, v in kw.items() if k != "mode"})
        return types.SimpleNamespace(
            result=res,
            command_map=FakeCommandMap(np.ones(6), np.full(6, -2.0)),
            dynamics_only=FakeResult(FakeParams(kp=40.0), FakeParams(), eval_score=2.0),
        )

    monkeypatch.setitem(sys.modules, "antares.sensor",
                        _module("antares.sensor", calibrate_with_sensor=fake_calibrate_with_sensor))
    result, lock, path = server.run_calibration(
        server.get_dataset(FAKE_DS2), n_train=4, n_eval=2, name="unit-b", fit_sensor=True)
    assert calls["max_evaluations"] == 150 and calls["mode"] == "sim"
    assert lock["command_map"] == {**SHIFT_MAP, "mode": "sim"}
    assert lock["dynamics_only"]["eval_score"] == 2.0
    saved = json.loads(path.read_text())
    assert saved["command_map"]["bias"] == SHIFT_MAP["bias"]
    replay = client.get("/api/replay", params={"dataset": FAKE_DS2, "episode": 0, "lock": "unit-b"}).json()
    assert replay["command_map"] == SHIFT_MAP


def test_calibrate_request_forwards_fit_sensor(client, fakes, monkeypatch):
    seen = {}

    def spy(ds, **kw):
        seen.update(kw)
        raise HTTPException(status_code=400, detail="stop here")

    monkeypatch.setattr(server, "run_calibration", spy)
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "sensor_mode": "bogus"}).status_code == 422
    job = client.post("/api/calibrate", json={"dataset": FAKE_DS, "fit_sensor": True,
                                              "sensor_mode": "kinematic"}).json()["job_id"]
    for _ in range(100):
        if client.get(f"/api/calibrate/{job}").json()["status"] != "running":
            break
        time.sleep(0.01)
    assert seen["fit_sensor"] is True and seen["sensor_mode"] == "kinematic"


def test_calibrate_request_forwards_fit_gravity_and_fit_deadband(client, fakes, monkeypatch):
    seen = {}

    def spy(ds, **kw):
        seen.update(kw)
        raise HTTPException(status_code=400, detail="stop here")

    monkeypatch.setattr(server, "run_calibration", spy)
    for body, want in (
        ({"dataset": FAKE_DS}, (False, False)),
        ({"dataset": FAKE_DS, "fit_gravity": True}, (True, False)),
        ({"dataset": FAKE_DS, "fit_gravity": True, "fit_deadband": True, "per_joint": True}, (True, True)),
    ):
        seen.clear()
        job = client.post("/api/calibrate", json=body).json()["job_id"]
        for _ in range(100):
            if client.get(f"/api/calibrate/{job}").json()["status"] != "running":
                break
            time.sleep(0.01)
        assert (seen["fit_gravity"], seen["fit_deadband"]) == want
    assert seen["per_joint"] is True


def test_check_uses_lock_command_map(client, fakes):
    write_lock(fakes, "unit-b", kp=50.0, dataset=FAKE_DS2, command_map=SHIFT_MAP)
    write_lock(fakes, "unit-b-plain", kp=50.0, dataset=FAKE_DS2)
    mapped = client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-b"}).json()
    plain = client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-b-plain"}).json()
    assert mapped["score"] < 0.5 * plain["score"]
    assert mapped["passed"] is True and plain["passed"] is False


def test_residual_uses_lock_command_map(client, pick):
    elbow_up = {"gain": [1.0] * 6, "bias": [0.0, 0.0, 2.0, 0.0, 0.0, 0.0]}
    write_lock(pick, "unit-a", kp=50.0, dataset=PICK_DS)
    write_lock(pick, "unit-a-mapped", kp=50.0, dataset=PICK_DS, command_map=elbow_up)
    q = {"dataset": PICK_DS, "episode": 1}
    plain = np.asarray(client.get("/api/residual", params={**q, "lock": "unit-a"}).json()["residual"])
    mapped = np.asarray(client.get("/api/residual", params={**q, "lock": "unit-a-mapped"}).json()["residual"])
    np.testing.assert_allclose(mapped[:, 2], plain[:, 2] - 2.0, atol=1e-3)
    np.testing.assert_allclose(mapped[:, :2], plain[:, :2], atol=1e-9)


def test_lock_simulate_holds_the_lock_command_map():
    """Real MuJoCo: drift and DR refits replay a mapped lock with its map."""
    from antares.sim import SimParams

    assert server.lock_simulate({"params": {}}) is None
    sim = server.lock_simulate({"command_map": {"gain": [1.0] * 6, "bias": [0.0, 5.0, 0.0, 0.0, 0.0, 0.0]}})
    T = 30
    action = np.zeros((T, 6))
    action[:, 1] = 20.0
    ep = Episode(index=0, fps=30.0, joint_names=JOINTS, state=np.zeros((T, 6)), action=action)
    _, mapped = sim(SimParams(), ep)
    plain = server.lock_simulate({"command_map": {"gain": [1.0] * 6, "bias": [0.0] * 6}})(SimParams(), ep)[1]
    assert mapped[-1, 1] > plain[-1, 1]


# --------------------------------------------------------------------------
# check history across sessions (``LOCK_DIR/<lock>.history.jsonl``)
# --------------------------------------------------------------------------

TREND_KEYS = {"n", "first", "last", "slope_deg_per_day", "r2", "latest_verdict", "n_drifted"}
ENTRY_KEYS = {"at", "lock", "dataset", "verdict", "passed", "score", "reference", "limit",
              "n_episodes", "episodes", "worst_joint", "per_joint", "antares_version"}


def _history_lines(fakes, name="unit-a"):
    p = fakes / "locks" / f"{name}.history.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.is_file() else None


def test_check_records_history_unless_told_not_to(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    ok = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a"}).json()
    assert "history_error" not in ok
    (rec,) = _history_lines(fakes)
    assert set(rec) == ENTRY_KEYS
    assert rec["lock"] == "unit-a" and rec["dataset"] == FAKE_DS and rec["verdict"] == "ok"
    assert rec["score"] == pytest.approx(ok["score"]) and rec["limit"] == pytest.approx(1.8)
    assert rec["episodes"] == [0, 1, 2, 3, 4, 5] and rec["worst_joint"] == ok["worst_joint"]
    assert rec["per_joint"] == {j: pytest.approx(v["score"]) for j, v in ok["per_joint"].items()}
    assert rec["at"].endswith("Z")

    client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-a", "record": False})
    assert len(_history_lines(fakes)) == 1

    # insufficient data is recorded too, with a null score
    out = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a", "last": 2}).json()
    assert out["verdict"] == "insufficient_data"
    recs = _history_lines(fakes)
    assert [r["verdict"] for r in recs] == ["ok", "insufficient_data"]
    assert recs[1]["score"] is None and recs[1]["per_joint"] == {} and recs[1]["n_episodes"] == 2


def test_history_get_entries_oldest_first_with_trend(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a"})
    client.post("/api/check", json={"dataset": FAKE_DS2, "lock": "unit-a"})
    r = client.get("/api/history/unit-a")
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) == {"lock", "entries", "trend"} and d["lock"] == "unit-a"
    assert [e["verdict"] for e in d["entries"]] == ["ok", "drifted"]
    assert [e["dataset"] for e in d["entries"]] == [FAKE_DS, FAKE_DS2]
    assert d["entries"][0]["at"] <= d["entries"][1]["at"]
    t = d["trend"]
    assert set(t) == TREND_KEYS
    assert t["n"] == 2 and t["n_drifted"] == 1 and t["latest_verdict"] == "drifted"
    assert t["first"] == d["entries"][0]["at"] and t["last"] == d["entries"][1]["at"]
    json.dumps(d)


def test_history_get_missing_lock_and_empty_history(client, fakes):
    assert client.get("/api/history/ghost").status_code == 404
    assert client.get("/api/history/..").status_code in (400, 404)
    assert client.get("/api/history/a%2Fb").status_code in (400, 404)
    write_lock(fakes, "unit-a", kp=50.0)
    d = client.get("/api/history/unit-a").json()
    assert d == {
        "lock": "unit-a",
        "entries": [],
        "trend": {"n": 0, "first": None, "last": None, "slope_deg_per_day": None, "r2": None,
                  "latest_verdict": None, "n_drifted": 0},
    }


def test_history_list_covers_every_lock(client, fakes):
    assert client.get("/api/history").json() == {"locks": []}
    write_lock(fakes, "unit-a", kp=50.0)
    write_lock(fakes, "unit-b", kp=50.0)
    client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a"})
    d = client.get("/api/history").json()
    assert [x["lock"] for x in d["locks"]] == ["unit-a", "unit-b"]
    a, b = d["locks"]
    assert set(a) == {"lock", "trend", "latest"} and set(a["trend"]) == TREND_KEYS
    assert a["trend"]["n"] == 1 and a["latest"]["verdict"] == "ok" and set(a["latest"]) == ENTRY_KEYS
    assert b["trend"]["n"] == 0 and b["latest"] is None


def test_history_get_skips_a_corrupt_line(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    client.post("/api/check", json={"dataset": FAKE_DS, "lock": "unit-a"})
    p = fakes / "locks" / "unit-a.history.jsonl"
    p.write_text(p.read_text() + "garbage\n")
    with pytest.warns(UserWarning, match="corrupt history line"):
        d = client.get("/api/history/unit-a").json()
    assert d["trend"]["n"] == 1 and len(d["entries"]) == 1


# --------------------------------------------------------------------------
# fleet dashboard
# --------------------------------------------------------------------------

COMMITTED_LOCK = Path(__file__).resolve().parents[1] / "locks" / "svla_so101_pickplace.lock.json"


def _copy_lock(lock_dir: Path, name: str, **changes) -> dict:
    """The committed demo lock under a new name (and optional edits)."""
    body = json.loads(COMMITTED_LOCK.read_text())
    body["name"] = name
    body.update(changes)
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / f"{name}{LOCK_SUFFIX}").write_text(json.dumps(body))
    return body


def test_dashboard_aggregates_locks(client, monkeypatch, tmp_path):
    lock_dir = tmp_path / "locks"
    monkeypatch.setattr(server, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(server, "LOCK_SUFFIX", LOCK_SUFFIX)
    monkeypatch.setattr(server, "FLEET_DIR", lock_dir / "fleet")
    monkeypatch.setattr(server, "_DATASETS", {})
    monkeypatch.setitem(sys.modules, "antares.history", None)  # import -> ImportError

    # no lock dir at all: an empty dashboard, not an error
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    assert r.json()["robots"] == [] and r.json()["fleet"] is None
    assert r.json()["generated_at"].endswith("Z")

    a = _copy_lock(lock_dir, "arm-a", seeds={"pinned": ["kp", "latency_frames"], "spread": {}})
    _copy_lock(lock_dir, "arm-b", created_at="2026-10-18T00:00:00Z")
    expected_after = a["result"]["eval_score"]
    expected_before = a["result"]["baseline_eval_score"]
    per_joint = a["result"]["per_joint_after"]
    body = {k: v for k, v in per_joint.items() if k != "gripper"}
    worst = max(body, key=body.get)
    assert per_joint["gripper"] > per_joint[worst]  # the gripper would win if it counted

    # a drift scan for arm-a only (verdict comes from the saved scan)
    (lock_dir / "arm-a.drift.json").write_text(json.dumps({"lock": "arm-a", "verdict": "no drift"}))
    # a fleet report that lists arm-b
    (lock_dir / "fleet").mkdir()
    (lock_dir / "fleet" / "fleet.json").write_text(json.dumps({
        "created_at": "2026-10-17T21:44:12Z",
        "robots": [{"lock": "arm-b", "dataset": a["dataset"], "fps": 30.0}],
        "summary": {"n_robots": 1, "median_improvement": 0.5},
    }))

    d = client.get("/api/dashboard").json()
    assert [r["name"] for r in d["robots"]] == ["arm-b", "arm-a"]  # newest first, like /api/locks
    by = {r["name"]: r for r in d["robots"]}

    ra = by["arm-a"]
    assert ra["dataset"] == a["dataset"] and ra["robot_type"] == a["robot_type"]
    assert ra["baseline_eval_score"] == pytest.approx(expected_before)
    assert ra["eval_score"] == pytest.approx(expected_after)
    assert ra["improvement"] == pytest.approx(1 - expected_after / expected_before, abs=1e-6)
    assert ra["worst_joint"] == worst and ra["worst_joint_deg"] == pytest.approx(per_joint[worst])
    assert ra["latency_frames"] == a["params"]["latency_frames"]
    assert ra["drift"] == {"exists": True, "verdict": "no drift"}
    assert ra["in_fleet"] is False and ra["fps"] is None
    assert ra["pinned"] == ["kp", "latency_frames"]
    assert ra["history"] == {"latest": None, "trend": None}

    rb = by["arm-b"]
    assert rb["created_at"] == "2026-10-18T00:00:00Z"
    assert rb["drift"] == {"exists": False, "verdict": None}
    assert rb["in_fleet"] is True and rb["fps"] == 30.0
    assert rb["pinned"] is None

    assert d["fleet"]["n_robots"] == 1 and d["fleet"]["median_improvement"] == 0.5
    assert d["fleet"]["created_at"] == "2026-10-17T21:44:12Z" and d["fleet"]["locks"] == ["arm-b"]


def test_dashboard_reads_check_history_when_available(client, monkeypatch, tmp_path):
    lock_dir = tmp_path / "locks"
    monkeypatch.setattr(server, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(server, "LOCK_SUFFIX", LOCK_SUFFIX)
    monkeypatch.setattr(server, "FLEET_DIR", lock_dir / "fleet")
    monkeypatch.setattr(server, "_DATASETS", {})
    _copy_lock(lock_dir, "arm-a")
    _copy_lock(lock_dir, "arm-b")

    entry = {"at": "2026-10-17T10:00:00Z", "lock": "arm-a", "dataset": FAKE_DS, "verdict": "drifted",
             "passed": False, "score": 2.1, "reference": 1.0, "limit": 1.5, "n_episodes": 5,
             "episodes": [0, 1, 2, 3, 4], "worst_joint": "elbow_flex", "per_joint": {"elbow_flex": 2.1},
             "antares_version": "0.0.1"}
    trend = {"n": 3, "first": "2026-10-10T10:00:00Z", "last": entry["at"], "slope_deg_per_day": 0.1,
             "r2": 0.9, "latest_verdict": "drifted", "n_drifted": 1}
    seen = {}

    def history_index(lock_dir):
        seen["lock_dir"] = Path(lock_dir)
        return {"locks": [{"lock": "arm-a", "trend": trend, "latest": entry}]}

    monkeypatch.setitem(sys.modules, "antares.history", _module("antares.history", history_index=history_index))
    by = {r["name"]: r for r in client.get("/api/dashboard").json()["robots"]}
    assert seen["lock_dir"] == lock_dir
    assert by["arm-a"]["history"] == {"latest": entry, "trend": trend}
    assert by["arm-b"]["history"] == {"latest": None, "trend": None}

    # a history module that blows up must not take the dashboard down
    def broken(lock_dir):
        raise RuntimeError("corrupt history")

    monkeypatch.setitem(sys.modules, "antares.history", _module("antares.history", history_index=broken))
    r = client.get("/api/dashboard")
    assert r.status_code == 200 and all(x["history"]["latest"] is None for x in r.json()["robots"])


# ---------------------------------------------------------------------------
# the follower's LeRobot calibration (real sim modules, no fakes)
# ---------------------------------------------------------------------------


def _real_dataset(repo_id="real/arm", n=3, T=12):
    eps = []
    for i in range(n):
        a = np.zeros((T, 6))
        a[:, 0] = np.linspace(0, 20, T)
        a[:, 5] = 50.0
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=a.copy(), action=a))
    return Dataset(repo_id=repo_id, robot_type="so101_follower", fps=30.0, joint_names=JOINTS, episodes=tuple(eps))


def _real_calibration(path, shift_joint=None, shift=0):
    from antares import lerobot_calib as lc
    from antares.sim import load_model

    model = load_model()
    out = {}
    for i, j in enumerate(JOINTS):
        span = int(round((model.jnt_range[i, 1] - model.jnt_range[i, 0]) / lc.RAD_PER_TICK))
        lo = 1000 if j == "gripper" else lc.HALF_TURN - span // 2
        if j == shift_joint:
            lo += shift
        out[j] = lc.MotorCalibration(i + 1, 0, 0, lo, lo + span)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lc.calibration_to_dict(out)))
    return out, model


def test_calibration_check_route(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_DATASETS", {"real/arm": _real_dataset()})
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "cal"))
    _real_calibration(tmp_path / "cal" / "robots" / "so101_follower" / "arm.json", "elbow_flex", 80)
    client = TestClient(server.app)
    assert client.get("/api/datasets/calibration-check", params={"dataset": "real/arm"}).status_code == 400
    r = client.get("/api/datasets/calibration-check", params={"dataset": "real/arm", "robot_id": "nope"})
    assert r.status_code == 404 and "no LeRobot calibration named 'nope'" in r.json()["detail"]
    r = client.get("/api/datasets/calibration-check", params={"dataset": "real/arm", "robot_id": "arm"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["dataset"] == "real/arm" and out["robot"] == "so101" and out["robot_id"] == "arm"
    assert out["source"].endswith("so101_follower/arm.json") and len(out["sha256"]) == 64
    assert out["flagged"] == ["elbow_flex"] and "use --calibration" in out["message"]
    rows = {j["joint"]: j for j in out["joints"]}
    assert rows["elbow_flex"]["max_deg"] == pytest.approx(80 * 360 / 4096, abs=0.15)
    assert set(rows["elbow_flex"]["at"]) == {"-100", "0", "100"} and rows["elbow_flex"]["range_min"] > 0
    by_path = client.get("/api/datasets/calibration-check",
                         params={"dataset": "real/arm", "calibration_path": str(tmp_path / "cal/robots/so101_follower/arm.json")})
    assert by_path.status_code == 200 and by_path.json()["flagged"] == ["elbow_flex"]
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert client.get("/api/datasets/calibration-check",
                      params={"dataset": "real/arm", "calibration_path": str(bad)}).status_code == 400
    assert client.get("/api/datasets/calibration-check",
                      params={"dataset": "real/arm", "calibration_path": str(tmp_path / "missing.json")}).status_code == 404


def test_replay_and_check_apply_the_lock_calibration_map(monkeypatch, tmp_path):
    """A lock with a calibration block replays on that unit map, not the model's."""
    from antares import lerobot_calib as lc
    from antares.sim import SimParams, UnitMap

    ds = _real_dataset()
    monkeypatch.setattr(server, "_DATASETS", {"real/arm": ds})
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path / "locks")
    calib, model = _real_calibration(tmp_path / "arm.json", "shoulder_pan", 200)  # 17.6 deg
    rec = lc.calibration_record(calib, model, path=tmp_path / "arm.json", robot_id="arm")
    lock = {"name": "cal-lock", "dataset": "real/arm", "robot_type": "so101_follower", "robot": "so101",
            "data_sha256": "x", "params": SimParams().to_dict(), "created_at": "now", "antares_version": "0",
            "result": {"params": SimParams().to_dict(), "baseline": SimParams().to_dict(), "train_score": 1.0,
                       "eval_score": 1.0, "baseline_train_score": 2.0, "baseline_eval_score": 2.0,
                       "n_evaluations": 1, "elapsed_s": 0.1, "per_joint_after": {j: 1.0 for j in JOINTS}},
            "calibration": rec}
    (tmp_path / "locks").mkdir()
    (tmp_path / "locks" / "cal-lock.lock.json").write_text(json.dumps(lock))
    um = server.unit_map_from_lock(lock)
    np.testing.assert_allclose(um.offset, UnitMap.from_dict(rec["unit_map"]).offset)
    assert server.unit_map_from_lock({"name": "plain"}) is None

    client = TestClient(server.app)
    with_lock = client.get("/api/replay", params={"dataset": "real/arm", "episode": 0, "lock": "cal-lock"}).json()
    plain = client.get("/api/replay", params={"dataset": "real/arm", "episode": 0}).json()
    assert with_lock["unit_map"] == rec["unit_map"] and plain["unit_map"] is None
    # the real arm's pan angle is read 17.6 deg further along under its own calibration
    shift = np.array(with_lock["real_q"])[:, 0] - np.array(plain["real_q"])[:, 0]
    np.testing.assert_allclose(shift, 200 * 360 / 4096, atol=0.05)
    # and the shared arm is back on the model's map afterwards
    arm = server.get_arm("so101")
    np.testing.assert_allclose(arm.unit_map.offset, UnitMap.from_model(arm.model).offset)

    res = server.run_check(ds, "cal-lock", threshold_deg=5.0, min_episodes=1)
    assert res.verdict in ("ok", "drifted") and res.n_episodes == 3
    np.testing.assert_allclose(arm.unit_map.offset, UnitMap.from_model(arm.model).offset)
    sim = server.lock_simulate(lock)
    real_q, _ = sim(SimParams(), ds.episodes[0])
    assert real_q[0, 0] == pytest.approx(um.to_rad(ds.episodes[0].state[0])[0])


def test_calibrate_request_forwards_calibration(client, fakes, monkeypatch):
    seen = {}

    def spy(ds, **kw):
        seen.update(kw)
        raise HTTPException(status_code=400, detail="stop here")

    monkeypatch.setattr(server, "run_calibration", spy)
    job = client.post("/api/calibrate", json={"dataset": FAKE_DS, "calibration_path": "/x/arm.json",
                                              "robot_type": "so100_follower"}).json()["job_id"]
    for _ in range(100):
        if client.get(f"/api/calibrate/{job}").json()["status"] != "running":
            break
        time.sleep(0.01)
    assert seen["calibration_path"] == "/x/arm.json" and seen["robot_id"] is None
    assert seen["robot_type"] == "so100_follower"


def test_run_calibration_records_the_calibration_in_the_lock(monkeypatch, tmp_path):
    """The real pipeline, two evaluations: the lock carries the calibration
    block and the fit ran on the calibration-derived unit map."""
    from antares import lerobot_calib as lc
    from antares.sim import UnitMap

    ds = _real_dataset(n=4, T=8)
    monkeypatch.setattr(server, "_DATASETS", {"real/arm": ds})
    monkeypatch.setattr(server, "LOCK_DIR", tmp_path / "locks")
    calib, model = _real_calibration(tmp_path / "arm.json", "wrist_flex", 50)
    result, lock, path = server.run_calibration(
        ds, n_train=3, n_eval=1, max_evaluations=2, name="cal-fit", calibration_path=str(tmp_path / "arm.json"))
    cal = lock["calibration"]
    assert cal["source"] == str(tmp_path / "arm.json") and cal["sha256"] == lc.file_sha256(tmp_path / "arm.json")
    assert cal["robot_type"] == "so101_follower" and cal["motors"]["wrist_flex"] == calib["wrist_flex"].to_dict()
    assert cal["unit_map_disagreement_deg"]["wrist_flex"] == pytest.approx(50 * 360 / 4096, abs=0.05)
    expected = lc.unit_map_from_calibration(calib, model)
    np.testing.assert_allclose(UnitMap.from_dict(cal["unit_map"]).offset, expected.offset)
    saved = json.loads(path.read_text())
    assert saved["calibration"]["robot_id"] is None and saved["calibration"]["anchor"] == "homing"
    # A robot id with no calibration file on this machine is not an error for
    # calibrate (the id also names the robot in the registry): the fit runs on
    # the model's unit map and the lock carries no calibration block.
    _, lock2, _ = server.run_calibration(ds, n_train=3, n_eval=1, max_evaluations=2, name="no-cal", robot_id="ghost")
    assert "calibration" not in lock2
    with pytest.raises(HTTPException) as e:
        server.resolve_calibration(None, "ghost")
    assert e.value.status_code == 404


# --------------------------------------------------------------------------
# robot registry: /api/registry, @<robot> on every lock route, dashboard fields
# --------------------------------------------------------------------------


def _register(client, robot, lock, **body):
    r = client.post(f"/api/registry/{robot}/register", json={"lock": lock, **body})
    assert r.status_code == 200, r.text
    return r.json()


def test_registry_routes_register_activate_rollback_remove(client, fakes):
    assert client.get("/api/registry").json() == {"version": 1, "path": str(fakes / "locks" / "registry.json"), "robots": []}
    assert client.get("/api/registry/arm_a").status_code == 404
    write_lock(fakes, "v1", kp=50.0)
    write_lock(fakes, "v2", kp=60.0)
    write_lock(fakes, "v3", kp=70.0)

    out = _register(client, "arm_a", "v1", note="first")
    reg = out["registration"]
    assert reg["activated"] and reg["version"] == 1 and reg["n_versions"] == 1 and reg["summary"].endswith("(first lock of this robot)")
    assert out["robot_id"] == "arm_a" and out["active"] == "v1" and out["robot_type"] == "so101_follower"
    assert out["lineage"][0]["note"] == "first" and out["lineage"][0]["is_active"] and out["lineage"][0]["exists"]
    assert out["lineage"][0]["pinned"] == ["latency_frames"]  # the stubbed lock module's answer
    assert out["lineage"][0]["params"]["kp"] == 50.0

    # the fake locks all score 1.2: the newcomer is within 10 % and takes over
    out = _register(client, "arm_a", "v2")
    assert out["registration"]["activated"] and out["active"] == "v2" and out["version"] == 2
    # activate=false registers without activating; a 404 for a lock that does not exist
    out = _register(client, "arm_a", "v3", activate=False)
    assert not out["registration"]["activated"] and out["active"] == "v2" and out["n_versions"] == 3
    assert client.post("/api/registry/arm_a/register", json={"lock": "ghost"}).status_code == 404
    assert client.post("/api/registry/-bad/register", json={"lock": "v1"}).status_code == 400  # invalid robot id
    assert client.post("/api/registry/arm_a/register", json={"lock": "v1", "max_regression": -1}).status_code == 422

    listing = client.get("/api/registry").json()
    (row,) = listing["robots"]
    assert row["robot_id"] == "arm_a" and row["active"] == "v2" and row["version"] == 2 and row["n_versions"] == 3
    assert [l["name"] for l in row["locks"]] == ["v1", "v2", "v3"]
    detail = client.get("/api/registry/arm_a").json()
    assert [l["version"] for l in detail["lineage"]] == [1, 2, 3]
    assert len(detail["param_drift"]["comparable"]["latency_frames"]["values"]) == 3
    assert "kp" in detail["param_drift"]["not_comparable"]

    r = client.post("/api/registry/arm_a/rollback")
    assert r.status_code == 200 and r.json()["active"] == "v1" and r.json()["previous_active"] == "v2"
    assert client.post("/api/registry/arm_a/rollback").status_code == 409  # oldest already
    r = client.post("/api/registry/arm_a/activate", json={"lock": "v3"})
    assert r.status_code == 200 and r.json()["active"] == "v3" and r.json()["previous_active"] == "v1"
    assert client.post("/api/registry/arm_a/activate", json={"lock": "nope"}).status_code == 400
    assert client.post("/api/registry/ghost/activate", json={"lock": "v1"}).status_code == 404
    assert client.post("/api/registry/ghost/rollback").status_code == 404

    r = client.delete("/api/registry/arm_a")
    assert r.status_code == 200 and r.json() == {"removed": "arm_a", "locks": ["v1", "v2", "v3"]}
    assert client.delete("/api/registry/arm_a").status_code == 404
    assert (fakes / "locks" / "v1.lock.json").is_file()
    assert client.get("/api/registry").json()["robots"] == []


def test_lock_routes_accept_a_robot_reference(client, fakes):
    write_lock(fakes, "unit-a", kp=50.0)
    write_lock(fakes, "unit-b", kp=60.0)
    _register(client, "arm_a", "unit-a")

    assert client.get("/api/locks/@arm_a").json()["name"] == "unit-a"
    assert client.get("/api/locks/@ghost").status_code == 404
    assert "unknown robot" in client.get("/api/locks/@ghost").json()["detail"]

    out = client.post("/api/check", json={"dataset": FAKE_DS, "lock": "@arm_a"}).json()
    assert out["lock"] == "unit-a" and out["verdict"] == "ok"
    (rec,) = _history_lines(fakes, "unit-a")
    assert rec["lock"] == "unit-a" and not (fakes / "locks" / "@arm_a.history.jsonl").exists()
    h = client.get("/api/history/@arm_a").json()
    assert h["lock"] == "unit-a" and len(h["entries"]) == 1
    assert client.get("/api/history/@ghost").status_code == 404
    assert client.post("/api/check", json={"dataset": FAKE_DS, "lock": "@ghost"}).status_code == 404

    r = client.get(f"/api/replay?dataset={FAKE_DS}&episode=0&lock=%40arm_a")
    assert r.status_code == 200 and r.json()["lock"] == "@arm_a"
    assert client.get(f"/api/compensate?dataset={FAKE_DS}&episode=0&lock=%40arm_a").status_code == 200
    assert client.post("/api/drift", json={"dataset": FAKE_DS, "lock": "@ghost"}).status_code == 404
    assert client.post("/api/diagnose", json={"dataset": FAKE_DS, "lock": "@ghost"}).status_code == 404

    # the reference follows the active lock
    _register(client, "arm_a", "unit-b", activate=True)
    assert client.get("/api/locks/@arm_a").json()["name"] == "unit-b"
    # a plain name never touches the registry, and a lock can not be deleted through a reference
    assert client.get("/api/locks/unit-a").json()["name"] == "unit-a"
    assert client.delete("/api/locks/@arm_a").status_code == 400
    assert (fakes / "locks" / "unit-b.lock.json").is_file()


def test_calibrate_with_robot_id_registers_the_lock(client, fakes):
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "robot_id": "bad/id"}).status_code == 400
    r = client.post("/api/calibrate", json={"dataset": FAKE_DS, "name": "unit-a", "robot_id": "arm_a"})
    assert r.status_code == 200
    job = r.json()["job_id"]
    with client.stream("GET", f"/api/calibrate/{job}/events") as resp:
        events = sse_events(resp)
    done = events[-1]
    assert done["type"] == "done" and done["lock"]["name"] == "unit-a"
    reg = done["registration"]
    assert reg["robot_id"] == "arm_a" and reg["lock"] == "unit-a" and reg["activated"] and reg["version"] == 1
    st = client.get(f"/api/jobs/{job}").json()
    assert st["status"] == "done" and st["registration"]["active"] == "unit-a"
    detail = client.get("/api/registry/arm_a").json()
    assert detail["active"] == "unit-a" and detail["lineage"][0]["dataset"] == FAKE_DS
    assert client.get("/api/locks/@arm_a").json()["name"] == "unit-a"

    # a second calibration of the same robot: same score, so it takes over by the rule
    r = client.post("/api/calibrate", json={"dataset": FAKE_DS, "name": "unit-b", "robot_id": "arm_a"})
    with client.stream("GET", f"/api/calibrate/{r.json()['job_id']}/events") as resp:
        done = sse_events(resp)[-1]
    assert done["registration"]["version"] == 2 and done["registration"]["activated"]
    # ...unless told not to
    r = client.post("/api/calibrate", json={"dataset": FAKE_DS, "name": "unit-c", "robot_id": "arm_a", "activate": False})
    with client.stream("GET", f"/api/calibrate/{r.json()['job_id']}/events") as resp:
        done = sse_events(resp)[-1]
    assert not done["registration"]["activated"] and client.get("/api/locks/@arm_a").json()["name"] == "unit-b"
    # without robot_id the done event has no registration key, as before
    r = client.post("/api/calibrate", json={"dataset": FAKE_DS, "name": "unit-d"})
    with client.stream("GET", f"/api/calibrate/{r.json()['job_id']}/events") as resp:
        done = sse_events(resp)[-1]
    assert "registration" not in done


def test_dashboard_rows_carry_registry_fields(client, monkeypatch, tmp_path):
    lock_dir = tmp_path / "locks"
    monkeypatch.setattr(server, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(server, "LOCK_SUFFIX", LOCK_SUFFIX)
    monkeypatch.setattr(server, "FLEET_DIR", lock_dir / "fleet")
    monkeypatch.setattr(server, "_DATASETS", {})
    monkeypatch.setitem(sys.modules, "antares.history", None)
    _copy_lock(lock_dir, "arm-a-v1", created_at="2026-10-01T00:00:00Z")
    _copy_lock(lock_dir, "arm-a-v2", created_at="2026-10-08T00:00:00Z")
    _copy_lock(lock_dir, "loose", created_at="2026-10-09T00:00:00Z")

    # no registry file: every row unregistered, as before
    by = {r["name"]: r for r in client.get("/api/dashboard").json()["robots"]}
    assert all(r["robot_id"] is None and r["is_active"] is None and r["version"] is None and r["n_versions"] is None
               for r in by.values())

    assert client.post("/api/registry/arm_a/register", json={"lock": "arm-a-v1"}).status_code == 200
    assert client.post("/api/registry/arm_a/register", json={"lock": "arm-a-v2"}).status_code == 200
    by = {r["name"]: r for r in client.get("/api/dashboard").json()["robots"]}
    assert by["arm-a-v1"]["robot_id"] == "arm_a" and by["arm-a-v1"]["is_active"] is False
    assert by["arm-a-v1"]["version"] == 1 and by["arm-a-v1"]["n_versions"] == 2
    assert by["arm-a-v2"]["robot_id"] == "arm_a" and by["arm-a-v2"]["is_active"] is True and by["arm-a-v2"]["version"] == 2
    assert by["loose"]["robot_id"] is None and by["loose"]["version"] is None

    # a corrupt registry must not take the dashboard down
    (lock_dir / "registry.json").write_text("{nope")
    r = client.get("/api/dashboard")
    assert r.status_code == 200 and all(x["robot_id"] is None for x in r.json()["robots"])
    assert client.get("/api/registry").status_code == 500


def test_calibrate_request_forwards_servo(client, fakes, monkeypatch):
    seen = {}

    def spy(ds, **kw):
        seen.update(kw)
        raise HTTPException(status_code=400, detail="stop here")

    monkeypatch.setattr(server, "run_calibration", spy)
    for body, want in (({"dataset": FAKE_DS}, "position"), ({"dataset": FAKE_DS, "servo": "sts3215"}, "sts3215")):
        seen.clear()
        job = client.post("/api/calibrate", json=body).json()["job_id"]
        for _ in range(100):
            if client.get(f"/api/calibrate/{job}").json()["status"] != "running":
                break
            time.sleep(0.01)
        assert seen["servo"] == want
    assert client.post("/api/calibrate", json={"dataset": FAKE_DS, "servo": "dynamixel"}).status_code == 422


def test_start_params_for_each_servo():
    from antares.sim import SimParams

    assert server.start_params() == SimParams()
    assert server.start_params("sts3215") == SimParams.sts3215()
    with pytest.raises(HTTPException):
        server.start_params("dynamixel")
