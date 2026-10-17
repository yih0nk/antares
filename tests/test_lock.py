import json
import math

import numpy as np
import pytest

from antares import __version__
from antares.calibrate import CalibrationResult, SimParams
from antares.data import Episode
from antares.lock import LOCK_DIR, RobotLock, dataset_digest, list_locks, slug

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
