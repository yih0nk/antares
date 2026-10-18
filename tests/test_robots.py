"""antares.robots: the model registry, and both vendored MJCFs through Arm."""

from __future__ import annotations

import numpy as np
import pytest

from antares import robots
from antares.data import Episode
from antares.robots import REGISTRY, RobotModel, get_robot, model_for_dataset
from antares.sim import JOINTS, Arm, SimParams
from antares.validate import JOINTS as CANONICAL


def test_registry_entries_are_complete():
    assert set(REGISTRY) >= {"so101", "so100"}
    assert robots.DEFAULT_ROBOT == "so101"
    for name, r in REGISTRY.items():
        assert r.name == name
        assert r.model_path.is_file(), r.model_path
        assert r.scene_path.is_file(), r.scene_path
        assert r.mesh_dir.is_dir() and any(r.mesh_dir.glob("*.stl")), r.mesh_dir
        assert (r.model_path.parent / "LICENSE").is_file()
        assert len(r.joint_names) == 6 and r.canonical_joints == CANONICAL
        d = r.to_dict()
        assert d["joints"] == list(CANONICAL) and d["mjcf_joint_names"] == list(r.joint_names)


def test_get_robot_default_and_unknown():
    assert get_robot() is REGISTRY["so101"]
    assert get_robot(None) is REGISTRY["so101"]
    assert get_robot("SO100") is REGISTRY["so100"]
    with pytest.raises(ValueError, match="unknown robot"):
        get_robot("koch")


def test_robot_model_needs_six_joints(tmp_path):
    with pytest.raises(ValueError):
        RobotModel("x", tmp_path, tmp_path, tmp_path, ("a", "b"), "two joints")


@pytest.mark.parametrize(
    "robot_type, model",
    [("so101_follower", "so101"), ("so_follower", "so101"), ("so100", "so101"), ("koch", "so101"), (None, "so101")],
)
def test_model_for_dataset_follows_the_family_table(robot_type, model):
    assert model_for_dataset(robot_type) == model
    assert robots.model_for_family("so100") == robots.MODEL_FOR_FAMILY["so100"]


@pytest.mark.parametrize("name", ["so101", "so100"])
def test_both_models_load_and_map_joints_in_canonical_order(name):
    arm = Arm(robot=name)
    assert arm.robot_name == name
    assert arm.model.nq == 6 and arm.model.nu == 6
    assert arm.model_path == REGISTRY[name].model_path
    # actuator i drives joint i, and the joints follow the canonical order
    assert list(arm.model.actuator_trnid[:, 0]) == list(range(6))
    um = arm.unit_map
    assert um.scale.shape == (6,) and np.all(um.scale > 0)
    # body joints: -100 and 100 land on the joint range; gripper: 0 and 100
    body = np.full(6, 100.0)
    body[JOINTS.index("gripper")] = 100.0
    np.testing.assert_allclose(um.to_rad(body), um.hi, atol=1e-9)
    lo = np.full(6, -100.0)
    lo[JOINTS.index("gripper")] = 0.0
    np.testing.assert_allclose(um.to_rad(lo), um.lo, atol=1e-9)


def test_default_arm_is_so101_and_wrong_robot_for_a_model_is_caught():
    assert Arm().robot_name == "so101"
    with pytest.raises(RuntimeError, match="MJCF joints"):
        Arm(robot="so100", model_path=REGISTRY["so101"].model_path)


@pytest.mark.slow
def test_so100_model_replays_an_episode():
    arm = Arm(robot="so100", params=SimParams.default())
    T = 40
    action = np.zeros((T, 6))
    action[:, 0] = np.linspace(0, 30, T)
    action[:, 5] = 50.0
    ep = Episode(index=0, fps=30.0, joint_names=JOINTS, state=action.copy(), action=action)
    sim_q = arm.replay(ep)
    assert sim_q.shape == (T, 6)
    np.testing.assert_allclose(sim_q[0], arm.unit_map.to_rad(action[0]))
    assert np.all(np.isfinite(sim_q))
    # the pan joint followed its ramp
    assert sim_q[-1, 0] > sim_q[0, 0] + 0.2
