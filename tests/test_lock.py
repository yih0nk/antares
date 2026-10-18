import json
import math
from pathlib import Path

import numpy as np
import pytest

from antares import __version__
from antares.calibrate import CalibrationResult, SimParams
from antares.data import Episode
from antares.lock import (
    DEFAULT_PINNED,
    LOCK_DIR,
    RobotLock,
    dataset_digest,
    list_locks,
    param_spread,
    pinned_params,
    seed_info,
    slug,
)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def episodes(n=3, T=20, seed=0, fps=30.0):
    rng = np.random.default_rng(seed)
    return [
        Episode(index=i, fps=fps, joint_names=JOINTS,
                state=rng.normal(size=(T, 6)), action=rng.normal(size=(T, 6)))
        for i in range(n)
    ]


def fake_result(**over) -> CalibrationResult:
    kw = dict(
        params=SimParams(kp=1500.0, kv=3.0, damping=0.4, armature=0.02, frictionloss=0.08, latency_frames=1),
        baseline=SimParams(),
        train_score=1.5,
        eval_score=1.7,
        baseline_train_score=4.0,
        baseline_eval_score=4.2,
        per_joint_before={j: 4.0 + i for i, j in enumerate(JOINTS)},
        per_joint_after={j: 1.0 + 0.1 * i for i, j in enumerate(JOINTS)},
        history=[(0, 4.0), (1, 3.0), (2, 1.5)],
        n_evaluations=2,
        elapsed_s=0.5,
        train_episodes=[0, 1],
        eval_episodes=[2],
    )
    kw.update(over)
    return CalibrationResult(**kw)


def test_slug():
    assert slug("lerobot/svla_so101_pickplace") == "svla_so101_pickplace"
    assert slug("Some User/My Robot v2!") == "my_robot_v2"
    assert slug("plain") == "plain"
    assert slug("") == "robot"


def test_dataset_digest_stable_and_order_independent():
    eps = episodes()
    d1 = dataset_digest(eps)
    assert len(d1) == 64 and int(d1, 16)
    assert dataset_digest(episodes()) == d1  # same content -> same digest
    assert dataset_digest(list(reversed(eps))) == d1  # order of episodes irrelevant
    assert dataset_digest(eps[:2]) != d1  # different episode set
    bumped = episodes()
    bumped[1].state[3, 2] += 1e-6
    assert dataset_digest(bumped) != d1  # any value change
    reindexed = [Episode(index=e.index + 10, fps=e.fps, joint_names=e.joint_names,
                         state=e.state, action=e.action) for e in eps]
    assert dataset_digest(reindexed) != d1  # indices are part of it
    assert dataset_digest([]) == dataset_digest([])


def test_lock_save_load_round_trip(tmp_path):
    eps = episodes()
    res = fake_result()
    lock = RobotLock.create("lerobot/svla_so101_pickplace", "so101_follower", eps[:2], res)
    assert lock.name == "svla_so101_pickplace"
    assert lock.params == res.params
    assert lock.data_sha256 == dataset_digest(eps[:2])
    assert lock.antares_version == __version__
    assert lock.created_at.endswith("Z") and "T" in lock.created_at

    path = lock.save(tmp_path / "locks" / "x.lock.json")
    assert path.exists()
    raw = json.loads(path.read_text())
    assert raw["params"]["latency_frames"] == 1
    assert raw["result"]["history"] == [[0, 4.0], [1, 3.0], [2, 1.5]]

    back = RobotLock.load(path)
    assert back == lock
    assert back.result.params == res.params
    assert back.result.history == res.history
    assert back.result.per_joint_after == res.per_joint_after
    assert back.to_dict() == lock.to_dict()


def test_lock_default_path_uses_lock_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lock = RobotLock.create("org/arm", "so101", episodes(1), fake_result(), name="arm_a")
    path = lock.save()
    assert path == LOCK_DIR / "arm_a.lock.json"
    assert (tmp_path / "locks" / "arm_a.lock.json").exists()


def test_from_dict_accepts_result_dict_and_nan(tmp_path):
    res = fake_result(eval_score=float("nan"))
    lock = RobotLock.create("org/arm", "so101", episodes(1), res)
    d = lock.to_dict()
    assert d["result"]["eval_score"] is None  # NaN is not JSON
    back = RobotLock.from_dict(json.loads(json.dumps(d)))
    assert math.isnan(back.result.eval_score)
    assert back.params == res.params


def test_list_locks(tmp_path):
    assert list_locks(tmp_path / "nope") == []
    for name in ("b_arm", "a_arm"):
        RobotLock.create("org/x", "so101", episodes(1), fake_result(), name=name).save(
            tmp_path / f"{name}.lock.json"
        )
    (tmp_path / "junk.lock.json").write_text("{not json")
    (tmp_path / "other.json").write_text("{}")
    with pytest.warns(UserWarning, match="junk"):
        locks = list_locks(tmp_path)
    assert [lk.name for lk in locks] == ["a_arm", "b_arm"]


# ---------------------------------------------------------------------------
# the locks shipped in the repo (written before per-joint parameters existed)
# ---------------------------------------------------------------------------

REPO_LOCKS = Path(__file__).resolve().parent.parent / "locks"


def test_shipped_locks_load_as_scalar_params():
    paths = sorted(REPO_LOCKS.glob("svla_so10[01]_pickplace.lock.json"))
    assert len(paths) == 2, "expected the SO-100 and SO-101 demo locks"
    for path in paths:
        raw = json.loads(path.read_text())
        lock = RobotLock.load(path)
        assert not lock.params.has_per_joint  # written as scalars, read as scalars
        assert lock.params.forcerange == 2.94  # absent in the file: MJCF default
        assert lock.params.to_dict() == {**raw["params"], "forcerange": 2.94}
        assert lock.params == lock.result.params
        # the lock survives a save/load round trip untouched
        back = RobotLock.from_dict(json.loads(json.dumps(lock.to_dict())))
        assert back == lock
        # written before seeds were recorded: no seed block, only latency is pinned
        assert "seeds" not in raw["result"] and lock.result.seeds is None
        assert all("seed" not in s for s in raw["result"].get("stages", []))
        assert pinned_params(lock) == pinned_params(raw) == ["latency_frames"]
        assert param_spread(lock) is None and param_spread(raw) is None


def _cached_dataset(repo_id: str):
    """The real dataset from the local Hugging Face cache, or skip."""
    import os

    from antares.data import load_dataset

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return load_dataset(repo_id)
    except Exception as exc:  # noqa: BLE001 - any download/cache problem means skip
        pytest.skip(f"{repo_id} not in the local HF cache: {exc}")


@pytest.mark.slow
def test_so101_lock_replays_unchanged():
    """The SO-101 lock must drive the sim exactly as it did when it was fitted.

    The lock stores the held-out score computed at fit time; replaying the
    same eval split with the loaded parameters has to reproduce it, and the
    scalar and broadcast (per-joint tuple) forms must give the same sim
    trajectory on episode 0.
    """
    from antares.calibrate import split_episodes
    from antares.gap import aggregate, tracking_error
    from antares.sim import Arm

    lock = RobotLock.load(REPO_LOCKS / "svla_so101_pickplace.lock.json")
    ds = _cached_dataset(lock.dataset)
    assert ds.robot_type == lock.robot_type
    train, eval_ = split_episodes(ds.episodes, 20, 10, seed=0)
    assert [e.index for e in train] == lock.result.train_episodes
    assert [e.index for e in eval_] == lock.result.eval_episodes
    assert dataset_digest(train) == lock.data_sha256

    arm = Arm(lock.params)
    report = aggregate([tracking_error(arm.real_q(e), arm.replay(e)) for e in eval_])
    # The shipped locks were fitted before Arm.apply recomputed the qpos0-derived
    # constants (mj_setConst); that moved replays by ~1e-5 rad, so the stored
    # scores reproduce to ~1e-5 deg rather than bit-for-bit.
    assert report.overall_deg == pytest.approx(lock.result.eval_score, abs=1e-4)
    for joint, deg in lock.result.per_joint_after.items():
        assert report.per_joint_deg[joint] == pytest.approx(deg, abs=1e-4)

    ep0 = ds.episodes[0]
    scalar_q = arm.replay(ep0)
    arm.apply(lock.params.broadcast())
    np.testing.assert_array_equal(arm.replay(ep0), scalar_q)


def test_lock_command_map_round_trip(tmp_path):
    from antares.sim import CommandMap

    cmap = CommandMap(gain=[1.0, 0.95, 1.05, 1.0, 1.0, 1.0], bias=[0.0, -2.0, 1.5, 0.0, 0.0, 0.0])
    only = fake_result(eval_score=2.5).to_dict()
    lock = RobotLock.create("lerobot/x", "so101", episodes(), fake_result(),
                            command_map=cmap.to_dict(), dynamics_only=only)
    back = RobotLock.load(lock.save(tmp_path / "x.lock.json"))
    assert back.command_map == cmap.to_dict()
    assert back.dynamics_only["eval_score"] == 2.5
    np.testing.assert_allclose(CommandMap.from_dict(back.command_map).bias, cmap.bias)


def test_lock_without_command_map_omits_it():
    d = RobotLock.create("lerobot/x", "so101", episodes(), fake_result()).to_dict()
    assert "command_map" not in d and "dynamics_only" not in d
    assert RobotLock.from_dict(d).command_map is None


def test_committed_locks_still_load():
    from pathlib import Path

    from antares.sim import CommandMap

    locks = list_locks(Path(__file__).resolve().parent.parent / "locks")
    assert locks
    for lk in locks:
        # locks from before sensor fitting have none; sensor locks parse
        cmap = CommandMap.from_dict(lk.command_map)
        assert cmap.gain.shape == cmap.bias.shape == (6,)
        # single-seed locks: no spread recorded, so only latency counts as pinned
        assert lk.result.seeds is None and seed_info(lk) is None
        assert pinned_params(lk) == list(DEFAULT_PINNED) == ["latency_frames"]
        assert param_spread(lk) is None
        # and re-serializing adds nothing a loader before seeds would trip on
        d = lk.to_dict()
        assert "seeds" not in d["result"] and all("seed" not in s for s in d["result"]["stages"])
        assert RobotLock.from_dict(json.loads(json.dumps(d))) == lk


def test_committed_fleet_locks_still_load():
    from pathlib import Path

    fleet = list_locks(Path(__file__).resolve().parent.parent / "locks" / "fleet")
    assert len(fleet) >= 10
    for lk in fleet:
        assert lk.result.seeds is None and pinned_params(lk) == ["latency_frames"]
        assert RobotLock.from_dict(json.loads(json.dumps(lk.to_dict()))) == lk


# ---------------------------------------------------------------------------
# best-of-N seeds in a lock
# ---------------------------------------------------------------------------


def seeds_block(pinned=("kp", "latency_frames")) -> dict:
    p0 = SimParams(kp=100.0, kv=1.0, damping=0.1, armature=0.01, frictionloss=0.1, latency_frames=2)
    p1 = p0.replace(kp=150.0, kv=5.0, latency_frames=3)
    return {
        "n": 2, "seeds": [0, 100], "chosen": 1,
        "eval_scores": [1.9, 1.7], "train_scores": [1.6, 1.5],
        "params": [p0.to_dict(), p1.to_dict()],
        "spread": {
            "kp": {"min": 100.0, "max": 150.0, "ratio": 1.5},
            "kv": {"min": 1.0, "max": 5.0, "ratio": 5.0},
            "damping": {"min": 0.1, "max": 0.1, "ratio": 1.0},
            "armature": {"min": 0.01, "max": 0.01, "ratio": 1.0},
            "frictionloss": {"min": 0.1, "max": 0.1, "ratio": 1.0},
            "latency_frames": {"min": 2.0, "max": 3.0, "range": 1.0},
        },
        "pinned": list(pinned),
    }


def test_lock_with_seeds_round_trips_and_reports_pinned(tmp_path):
    res = fake_result(seeds=seeds_block())
    lock = RobotLock.create("lerobot/x", "so101", episodes(), res, name="seeded")
    path = lock.save(tmp_path / "seeded.lock.json")
    raw = json.loads(path.read_text())
    assert raw["result"]["seeds"]["chosen"] == 1
    assert raw["result"]["seeds"]["spread"]["kp"] == {"min": 100.0, "max": 150.0, "ratio": 1.5}
    assert raw["result"]["seeds"]["spread"]["latency_frames"] == {"min": 2.0, "max": 3.0, "range": 1.0}
    assert raw["result"]["seeds"]["pinned"] == ["kp", "latency_frames"]

    back = RobotLock.load(path)
    assert back == lock and back.result.seeds == seeds_block()
    for view in (back, raw, back.result):
        assert seed_info(view) == seeds_block()
        assert pinned_params(view) == ["kp", "latency_frames"]
        assert param_spread(view)["kv"]["ratio"] == 5.0

    # an explicit empty pinned list is respected: nothing is physical
    none_pinned = RobotLock.create("lerobot/x", "so101", episodes(), fake_result(seeds=seeds_block(())))
    assert pinned_params(none_pinned) == [] and pinned_params(none_pinned.to_dict()) == []
    # NaN scores survive as null
    nan_block = {**seeds_block(), "eval_scores": [float("nan"), 1.7]}
    d = RobotLock.create("lerobot/x", "so101", episodes(), fake_result(seeds=nan_block)).to_dict()
    assert d["result"]["seeds"]["eval_scores"][0] is None
    assert math.isnan(RobotLock.from_dict(json.loads(json.dumps(d))).result.seeds["eval_scores"][0])


def test_pinned_params_tolerates_odd_inputs():
    assert pinned_params({}) == ["latency_frames"]
    assert pinned_params({"result": {}}) == ["latency_frames"]
    assert pinned_params({"result": {"seeds": None}}) == ["latency_frames"]
    assert pinned_params({"seeds": {"pinned": ["kv"]}}) == ["kv"]  # a bare result dict
    assert param_spread({"result": {"seeds": {"pinned": []}}}) is None
    assert pinned_params(object()) == ["latency_frames"]
    assert DEFAULT_PINNED == ("latency_frames",)
