"""antares.validate: pure checks on meta/info.json and loaded data. No network, no MuJoCo."""

from __future__ import annotations

import numpy as np
import pytest

from antares import validate as V
from antares.data import Dataset, Episode
from antares.validate import Verdict, robot_family, validate_dataset, validate_dataset_info

SO101_NAMES = [f"{j}.pos" for j in V.JOINTS]
SO100_NAMES = [f"main_{j}" for j in V.JOINTS]
BI_NAMES = [f"left_{j}.pos" for j in V.JOINTS] + [f"right_{j}.pos" for j in V.JOINTS]


def info(**over):
    base = {
        "codebase_version": "v3.0",
        "robot_type": "so101_follower",
        "fps": 30,
        "total_episodes": 50,
        "total_frames": 12000,
        "features": {
            "action": {"names": list(SO101_NAMES)},
            "observation.state": {"names": list(SO101_NAMES)},
        },
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# robot_family
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rt, family",
    [
        ("so100", "so100"),
        ("so100_follower", "so100"),
        ("SO-ARM100", "so100"),
        ("so100-blue", "so100"),
        ("so101", "so101"),
        ("so101_follower", "so101"),
        ("so-101", "so101"),
        ("so_arm101", "so101"),
        ("so_follower", "so101"),
        ("so-follower", "so101"),
        ("koch", None),
        ("", None),
        (None, None),
        ("bi_so100_follower", None),
        ("bi_so_follower", None),
        ("piper_follower", None),
        ("xlerobot", None),
    ],
)
def test_robot_family(rt, family):
    assert robot_family(rt) == family


# --------------------------------------------------------------------------
# validate_dataset_info
# --------------------------------------------------------------------------


def test_accepts_so101_and_so100_naming():
    v = validate_dataset_info(info(), "a/b")
    assert v.ok and v.reasons == [] and v.warnings == []
    assert v.robot_family == "so101" and v.robot_type == "so101_follower"
    assert (v.fps, v.n_episodes, v.n_frames, v.codebase_version) == (30.0, 50, 12000, "v3.0")
    assert v.joint_names == list(V.JOINTS)

    old = info(
        codebase_version="v2.1",
        robot_type="so100",
        features={"action": {"names": SO100_NAMES}, "observation.state": {"names": SO100_NAMES}},
    )
    v = validate_dataset_info(old, "a/b")
    assert v.ok and v.robot_family == "so100"
    nested = info(features={"action": {"names": {"motors": SO100_NAMES}}, "observation.state": {"names": {"motors": SO100_NAMES}}})
    assert validate_dataset_info(nested).ok
    for rt in ("so-arm101", "SO_ARM100", "so100_follower", "so-101", "so_follower"):
        assert validate_dataset_info(info(robot_type=rt)).ok, rt


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"robot_type": "koch"}, "robot_type 'koch' is not an SO-100/101"),
        ({"robot_type": ""}, "robot_type missing"),
        ({"robot_type": "bi_so100_follower"}, "bimanual"),
        ({"codebase_version": "v1.6"}, "codebase_version v1.6"),
        ({"fps": 60}, "fps 60 not in [30, 50]"),
        ({"fps": 0}, "fps missing"),
        ({"total_episodes": 4}, "4 episodes < 5"),
        ({"features": {"action": {"names": SO101_NAMES[:5]}, "observation.state": {"names": SO101_NAMES}}}, "action has 5 joints, not 6"),
        ({"features": {"action": {"names": BI_NAMES}, "observation.state": {"names": BI_NAMES}}}, "12 joints, not 6 (a bimanual"),
        ({"features": {"action": {"names": SO101_NAMES}, "observation.state": {"names": list(reversed(SO101_NAMES))}}}, "not the SO-101 joints"),
        ({"features": {"action": {"names": [f"joint_{i}" for i in range(6)]}, "observation.state": {"names": SO101_NAMES}}}, "not the SO-101 joints"),
        ({"features": {"action": {"shape": [6]}, "observation.state": {"names": SO101_NAMES}}}, "action has no joint names"),
        ({"features": {}}, "no joint names"),
    ],
)
def test_rejects_with_reason(over, needle):
    v = validate_dataset_info(info(**over), "a/b")
    assert not v.ok
    assert any(needle in r for r in v.reasons), v.reasons
    assert "rejected" in v.summary()


def test_collects_every_reason_not_just_the_first():
    v = validate_dataset_info(info(robot_type="koch", fps=60, total_episodes=1))
    assert len(v.reasons) == 3
    assert v.robot_family is None


def test_warnings_do_not_reject():
    v = validate_dataset_info(info(total_episodes=8), "a/b")
    assert v.ok and any("only 8 episodes" in w for w in v.warnings)
    v = validate_dataset_info(info(), "u/so101_bench_sim_v2")
    assert v.ok and any("simulation" in w for w in v.warnings)
    v = validate_dataset_info(info(robot_type="so101_mujoco"))
    assert v.ok and any("simulation" in w for w in v.warnings)
    v = validate_dataset_info(info(), "u/so101_simple_task")  # "simple" is not "sim"
    assert v.ok and v.warnings == []
    v = validate_dataset_info(info(), "u/community_dataset_v3")
    assert v.ok and any("merge" in w for w in v.warnings)
    v = validate_dataset_info({**info(), "codebase_version": None})
    assert v.ok and any("codebase_version missing" in w for w in v.warnings)
    assert "warning" in v.summary()


def test_episode_minimum_is_configurable():
    assert not validate_dataset_info(info(total_episodes=14), min_episodes=15).ok
    v = validate_dataset_info(info(total_episodes=14), min_episodes=5, recommended_episodes=15)
    assert v.ok and v.warnings


def test_verdict_round_trip():
    v = validate_dataset_info(info(total_episodes=8), "a/b")
    d = v.to_dict()
    assert set(d) >= {"ok", "robot_type", "robot_family", "reasons", "warnings", "fps", "n_episodes"}
    assert Verdict.from_dict({**d, "extra": 1}) == v


# --------------------------------------------------------------------------
# validate_dataset (loaded data)
# --------------------------------------------------------------------------


def make_dataset(n=20, T=30, scale=1.0, grip_hi=100.0, robot_type="so100", fps=30.0, joints=V.JOINTS, nan_in=(), noise=0.5):
    rng = np.random.default_rng(0)
    eps = []
    for i in range(n):
        body = rng.uniform(-90, 90, size=(T, 5)) * scale
        grip = rng.uniform(0, grip_hi, size=(T, 1))
        action = np.concatenate([body, grip], axis=1)
        state = action + rng.normal(0, noise, size=action.shape)
        if i in nan_in:
            state[3, 2] = np.nan
        eps.append(Episode(index=i, fps=fps, joint_names=tuple(joints), state=state, action=action))
    return Dataset(repo_id="x/y", robot_type=robot_type, fps=fps, joint_names=tuple(joints), episodes=tuple(eps))


def test_dataset_ok():
    v = validate_dataset(make_dataset())
    assert v.ok and v.warnings == []
    assert v.robot_family == "so100" and v.n_episodes == 20 and v.n_frames == 600


def test_dataset_unit_warnings():
    deg = validate_dataset(make_dataset(scale=1.8))  # body up to +-162
    assert deg.ok and any("look like degrees" in w for w in deg.warnings)
    rad = validate_dataset(make_dataset(scale=0.03, grip_hi=1.7, noise=0.01))  # everything within +-3
    assert rad.ok and any("look like radians" in w for w in rad.warnings)
    grip = validate_dataset(make_dataset(grip_hi=180.0))
    assert grip.ok and any("gripper spans" in w for w in grip.warnings)


def test_dataset_nans_and_minimums():
    v = validate_dataset(make_dataset(n=20, nan_in=(1, 2)))
    assert v.ok and any("2 of 20 episodes contain NaN" in w for w in v.warnings)
    v = validate_dataset(make_dataset(n=6, nan_in=(0, 1)))
    assert not v.ok and any("4 usable episodes < 5" in r for r in v.reasons)
    v = validate_dataset(make_dataset(n=8))
    assert v.ok and any("only 8 usable episodes" in w for w in v.warnings)


def test_dataset_rejects_wrong_joints_and_fps():
    v = validate_dataset(make_dataset(joints=tuple(reversed(V.JOINTS))))
    assert not v.ok and any("not the SO-101 joints" in r for r in v.reasons)
    v = validate_dataset(make_dataset(fps=60.0))
    assert not v.ok and any("fps 60" in r for r in v.reasons)


def test_merge_verdicts_folds_warnings():
    a = validate_dataset_info(info(total_episodes=8), "u/sim_so101")
    b = validate_dataset(make_dataset(n=8))
    m = V.merge_verdicts(a, b)
    assert m.ok and m.codebase_version == "v3.0"
    assert any("simulation" in w for w in m.warnings) and any("usable episodes" in w for w in m.warnings)
    bad = V.merge_verdicts(validate_dataset_info(info(fps=60)), b)
    assert not bad.ok
