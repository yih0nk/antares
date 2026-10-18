"""Unit tests for the robot registry (antares/registry.py): file-based, no MuJoCo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from antares.registry import (
    DEFAULT_MAX_REGRESSION,
    REGISTRY_FILENAME,
    LockEntry,
    Registry,
    RegistryError,
    RobotEntry,
    UnknownRobot,
    decide_activation,
    is_robot_ref,
    param_drift,
    registry_path,
    robot_id_of,
    robot_ref,
)

PARAMS = {"kp": 100.0, "kv": 1.0, "damping": 0.1, "armature": 0.01, "frictionloss": 0.1, "latency_frames": 2}


def write_lock(lock_dir: Path, name: str, eval_score=1.0, baseline=4.0, created_at="2026-10-01T00:00:00Z",
               params=None, seeds=None, dataset="org/arm_a_daily", robot_type="so101_follower") -> dict:
    lock_dir.mkdir(parents=True, exist_ok=True)
    result = {"eval_score": eval_score, "baseline_eval_score": baseline, "params": params or PARAMS, "space": "shared"}
    if seeds is not None:
        result["seeds"] = seeds
    d = {"name": name, "dataset": dataset, "robot_type": robot_type, "created_at": created_at,
         "params": params or PARAMS, "result": result}
    (lock_dir / f"{name}.lock.json").write_text(json.dumps(d))
    return d


@pytest.fixture
def reg(tmp_path):
    return Registry.load(registry_path(tmp_path / "locks"))


def test_robot_refs():
    assert is_robot_ref("@arm_a") and not is_robot_ref("arm_a") and not is_robot_ref("x.lock.json")
    assert robot_ref("arm_a") == "@arm_a" and robot_id_of("@arm_a") == "arm_a" == robot_id_of("arm_a")
    for bad in ("@", "@../x", "@a/b", "", "@-x"):
        with pytest.raises(RegistryError):
            robot_id_of(bad)


def test_missing_file_is_an_empty_registry(tmp_path):
    r = Registry.load(tmp_path / "nowhere" / REGISTRY_FILENAME)
    assert r.robots() == [] and r.entries == {} and "arm_a" not in r
    assert r.resolve("plain_lock") == "plain_lock"  # names pass through untouched
    assert r.resolve("locks/x.lock.json") == "locks/x.lock.json"
    with pytest.raises(UnknownRobot, match="unknown robot 'arm_a'"):
        r.resolve("@arm_a")
    with pytest.raises(UnknownRobot):
        r.get("ghost")


def test_register_first_lock_is_active_and_save_round_trips(reg):
    lock_dir = reg.lock_dir
    write_lock(lock_dir, "arm_a-v1")
    res = reg.register("arm_a", "arm_a-v1", note="first fit")
    assert res.activated and res.version == 1 and res.n_versions == 1 and res.new
    assert res.active == "arm_a-v1" and res.previous_active is None
    assert "first lock" in res.reason and "now active" in res.summary()
    assert reg.active_lock("arm_a") == "arm_a-v1" == reg.resolve("@arm_a")
    assert reg.get("arm_a").robot_type == "so101_follower"
    assert reg.robot_for("arm_a-v1") == "arm_a" and reg.robot_for("nope") is None

    path = reg.save()
    assert path == lock_dir / REGISTRY_FILENAME
    raw = json.loads(path.read_text())
    assert raw["version"] == 1
    e = raw["robots"]["arm_a"]
    assert e["active"] == "arm_a-v1" and e["robot_type"] == "so101_follower"
    (lk,) = e["locks"]
    assert lk["name"] == "arm_a-v1" and lk["eval_score"] == 1.0 and lk["baseline_eval_score"] == 4.0
    assert lk["dataset"] == "org/arm_a_daily" and lk["note"] == "first fit"
    assert lk["created_at"] == "2026-10-01T00:00:00Z" and lk["registered_at"].endswith("Z")
    assert e["created_at"].endswith("Z") and e["updated_at"].endswith("Z")
    # no temp file left behind
    assert sorted(p.name for p in lock_dir.iterdir()) == ["arm_a-v1.lock.json", REGISTRY_FILENAME]

    back = Registry.load(path)
    assert back.to_dict() == reg.to_dict()
    assert back.robots() == ["arm_a"] and back.active_lock("@arm_a") == "arm_a-v1"


def test_register_accepts_dicts_robotlocks_and_paths(reg, tmp_path):
    d = write_lock(reg.lock_dir, "by-dict", eval_score=0.9)
    assert reg.register("r", d).lock == "by-dict"

    class FakeLock:
        def to_dict(self):
            return {**d, "name": "by-object"}

    assert reg.register("r", FakeLock()).lock == "by-object"
    other = tmp_path / "elsewhere" / "by-path.lock.json"
    other.parent.mkdir()
    other.write_text(json.dumps({**d, "name": "by-path"}))
    assert reg.register("r", str(other)).lock == "by-path"
    assert reg.get("r").names == ["by-dict", "by-object", "by-path"]

    with pytest.raises(FileNotFoundError, match="no lock named 'ghost'"):
        reg.register("r", "ghost")
    with pytest.raises(RegistryError, match="robot reference"):
        reg.register("r", "@r")
    with pytest.raises(RegistryError, match="invalid lock name"):
        reg.register("r", "../etc")
    with pytest.raises(RegistryError, match="invalid robot id"):
        reg.register("bad/id", d)


def test_activation_rule(reg):
    lock_dir = reg.lock_dir
    write_lock(lock_dir, "v1", eval_score=1.0)
    write_lock(lock_dir, "v2", eval_score=1.08, created_at="2026-10-08T00:00:00Z")  # within 10 %
    write_lock(lock_dir, "v3", eval_score=1.3, created_at="2026-10-15T00:00:00Z")  # 20 % worse
    write_lock(lock_dir, "v4", eval_score=None, created_at="2026-10-16T00:00:00Z")
    write_lock(lock_dir, "v5", eval_score=0.8, created_at="2026-10-17T00:00:00Z")

    reg.register("arm_a", "v1")
    r2 = reg.register("arm_a", "v2")
    assert r2.activated and r2.version == 2 and "within 10%" in r2.reason
    assert reg.active_lock("arm_a") == "v2"

    r3 = reg.register("arm_a", "v3")
    assert not r3.activated and r3.version == 3 and r3.active == "v2" and r3.previous_active == "v2"
    assert "20% worse" in r3.reason and "limit 10%" in r3.reason
    assert r3.summary().startswith("registered lock v3 as v3 of 3 for robot arm_a; not activated:")
    assert r3.summary().endswith("v2 stays active")
    assert reg.active_lock("arm_a") == "v2"

    # a looser limit, or --activate, lets it through
    reg2 = Registry.from_dict(reg.to_dict(), reg.path)
    assert reg2.register("arm_a", "v3", max_regression=0.5).activated
    reg3 = Registry.from_dict(reg.to_dict(), reg.path)
    forced = reg3.register("arm_a", "v3", activate=True)
    assert forced.activated and forced.reason == "activation forced" and not forced.new
    assert reg3.active_lock("arm_a") == "v3"

    # no score on the newcomer: never auto-activated
    r4 = reg.register("arm_a", "v4")
    assert not r4.activated and "no held-out score" in r4.reason
    # activate=False registers only
    r5 = reg.register("arm_a", "v5", activate=False)
    assert not r5.activated and r5.reason == "registered without activation" and reg.active_lock("arm_a") == "v2"
    assert reg.get("arm_a").names == ["v1", "v2", "v3", "v4", "v5"]

    with pytest.raises(RegistryError, match="max_regression"):
        reg.register("arm_a", "v5", max_regression=-1)


def test_decide_activation_edge_cases():
    a = LockEntry("a", eval_score=None)
    b = LockEntry("b", eval_score=1.0)
    assert decide_activation(a, b, None, DEFAULT_MAX_REGRESSION) == (
        True, "held-out 1.000 deg; the active lock a has no held-out score")
    ok, reason = decide_activation(b, b, None, 0.1)
    assert ok and reason == "already the active lock"
    ok, reason = decide_activation(b, LockEntry("c", eval_score=1.1), None, 0.1)
    assert ok and "within 10%" in reason
    ok, reason = decide_activation(b, LockEntry("c", eval_score=1.1001), None, 0.1)
    assert not ok


def test_reregistering_refreshes_the_entry_in_place(reg):
    write_lock(reg.lock_dir, "v1", eval_score=1.0)
    write_lock(reg.lock_dir, "v2", eval_score=1.0, created_at="2026-10-08T00:00:00Z")
    reg.register("arm_a", "v1", note="keep me")
    reg.register("arm_a", "v2")
    write_lock(reg.lock_dir, "v1", eval_score=0.7, created_at="2026-10-09T00:00:00Z")  # refitted under the same name
    r = reg.register("arm_a", "v1")
    assert not r.new and r.version == 1 and r.n_versions == 2 and r.activated
    e = reg.get("arm_a").entry("v1")
    assert e.eval_score == 0.7 and e.created_at == "2026-10-09T00:00:00Z" and e.note == "keep me"
    again = reg.register("arm_a", "v1")
    assert not again.activated and again.summary().endswith("still active")


def test_activate_rollback_remove(reg):
    for i, score in enumerate((1.0, 0.9, 0.8), start=1):
        write_lock(reg.lock_dir, f"v{i}", eval_score=score)
        reg.register("arm_a", f"v{i}")
    assert reg.active_lock("arm_a") == "v3"

    assert reg.rollback("arm_a") == ("v3", "v2")
    assert reg.active_lock("arm_a") == "v2"
    assert reg.rollback("@arm_a") == ("v2", "v1")
    with pytest.raises(RegistryError, match="nothing to roll back to"):
        reg.rollback("arm_a")

    assert reg.activate("arm_a", "v3") == "v1"
    assert reg.active_lock("arm_a") == "v3"
    with pytest.raises(RegistryError, match="not in this robot's lineage"):
        reg.activate("arm_a", "v9")
    with pytest.raises(UnknownRobot):
        reg.activate("ghost", "v1")

    removed = reg.remove("arm_a")
    assert removed.names == ["v1", "v2", "v3"] and reg.robots() == []
    assert (reg.lock_dir / "v1.lock.json").is_file()  # lock files are never touched
    with pytest.raises(UnknownRobot):
        reg.remove("arm_a")
    reg.entries["empty"] = RobotEntry()
    with pytest.raises(UnknownRobot, match="has no active lock"):
        reg.active_lock("empty")
    with pytest.raises(RegistryError, match="no active lock to roll back"):
        reg.rollback("empty")


def test_lineage_joins_lock_files_and_pinned(reg):
    seeds = {"pinned": ["kp", "latency_frames"], "spread": {"kp": {"min": 90.0, "max": 110.0, "ratio": 1.22}}}
    write_lock(reg.lock_dir, "v1", eval_score=1.0, seeds=seeds)
    write_lock(reg.lock_dir, "v2", eval_score=0.9, created_at="2026-10-08T00:00:00Z",
               params={**PARAMS, "kp": 120.0, "latency_frames": 3})
    reg.register("arm_a", "v1", note="seeded")
    reg.register("arm_a", "v2")
    reg.register("arm_a", {"name": "v3-missing", "dataset": "x", "result": {"eval_score": 1.5}})  # too bad to activate
    (reg.lock_dir / "v2.lock.json").write_text(json.dumps({**json.loads((reg.lock_dir / "v2.lock.json").read_text()),
                                                           "result": {"eval_score": 0.85, "baseline_eval_score": 4.0}}))

    rows = reg.lineage("@arm_a")
    assert [r["version"] for r in rows] == [1, 2, 3]
    v1, v2, v3 = rows
    assert v1["name"] == "v1" and v1["exists"] and v1["note"] == "seeded" and not v1["is_active"]
    assert v1["pinned"] == ["kp", "latency_frames"] and v1["spread"]["kp"]["ratio"] == 1.22
    assert v1["params"]["kp"] == 100.0 and v1["space"] == "shared"
    assert v2["is_active"] and v2["pinned"] == ["latency_frames"] and v2["spread"] is None
    assert v2["eval_score"] == 0.85  # the file wins over what was registered
    assert v2["params"]["latency_frames"] == 3
    assert not v3["exists"] and v3["params"] is None and v3["pinned"] == [] and v3["eval_score"] == 1.5

    s = reg.summary("arm_a")
    assert s["robot_id"] == "arm_a" and s["active"] == "v2" and s["version"] == 2 and s["n_versions"] == 3
    assert s["eval_score"] == 0.9 and s["last_calibrated"] is None  # v3 has no created_at
    assert [x["name"] for x in s["locks"]] == ["v1", "v2", "v3-missing"]
    assert reg.index_by_lock()["v2"] == {"robot_id": "arm_a", "is_active": True, "version": 2, "n_versions": 3}
    assert reg.index_by_lock()["v3-missing"]["is_active"] is False

    drift = param_drift(rows)
    assert drift["versions"] == [1, 2]
    assert drift["comparable"] == {"latency_frames": {"values": [2, 3], "change": "+1 frame"}}
    assert drift["not_comparable"] == {"kp": [2], "kv": [1, 2], "damping": [1, 2], "armature": [1, 2], "frictionloss": [1, 2]}


def test_param_drift_formats_ratios_and_per_joint():
    rows = [
        {"version": 1, "exists": True, "params": {"kp": 100.0, "latency_frames": [2, 2, 2, 2, 2, 2]}, "pinned": ["kp", "latency_frames"]},
        {"version": 2, "exists": True, "params": {"kp": 150.0, "latency_frames": [3, 3, 3, 3, 3, 3]}, "pinned": ["kp", "latency_frames"]},
        {"version": 3, "exists": False, "params": None, "pinned": []},
    ]
    d = param_drift(rows)
    assert d["comparable"]["kp"]["change"] == "x1.50" and d["comparable"]["latency_frames"]["change"] == "+1 frame"
    assert param_drift(rows[:1])["comparable"]["kp"]["change"] == "-"
    same = param_drift([rows[0], {**rows[0], "version": 2}])
    assert same["comparable"]["kp"]["change"] == "unchanged"


def test_corrupt_or_foreign_files_are_errors(tmp_path):
    p = tmp_path / REGISTRY_FILENAME
    p.write_text("{not json")
    with pytest.raises(RegistryError, match="not a valid registry"):
        Registry.load(p)
    p.write_text(json.dumps({"version": 1, "robots": []}))
    with pytest.raises(RegistryError, match="'robots' map"):
        Registry.load(p)
    p.write_text(json.dumps({"version": 99, "robots": {}}))
    with pytest.raises(RegistryError, match="newer"):
        Registry.load(p)
    p.write_text(json.dumps({"version": 1, "robots": {"arm_a": {"locks": [{"nope": 1}]}}}))
    with pytest.raises(RegistryError, match="needs a name"):
        Registry.load(p)
    # a sparse but valid file loads with defaults
    p.write_text(json.dumps({"robots": {"arm_a": {"active": "x", "locks": [{"name": "x"}]}}}))
    r = Registry.load(p)
    assert r.active_lock("arm_a") == "x" and r.get("arm_a").robot_type is None
    assert r.lineage("arm_a")[0]["exists"] is False


def test_save_is_atomic_against_a_failing_rename(tmp_path, monkeypatch):
    reg = Registry.load(tmp_path / "locks" / REGISTRY_FILENAME)
    write_lock(reg.lock_dir, "v1")
    reg.register("arm_a", "v1")
    reg.save()
    before = reg.path.read_text()

    def boom(src, dst):
        Path(src).unlink()
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", boom)
    reg.register("arm_a", {"name": "v2", "result": {"eval_score": 0.5}})
    with pytest.raises(OSError):
        reg.save()
    assert reg.path.read_text() == before  # the old file is intact; nothing half-written
    assert [p.name for p in reg.lock_dir.iterdir() if p.name.endswith(".tmp")] == []
