import numpy as np
import pytest

from antares.data import Episode
from antares.sim import JOINTS, UnitMap, load_model
from antares.task import Tip, dataset_thresholds, find_grasps

FPS = 30.0
OPEN, SHUT = 80.0, 2.0
POSE_A = [0.0, -60.0, 60.0, 30.0, 0.0]
POSE_B = [20.0, -10.0, 20.0, 40.0, 0.0]  # at the object
POSE_C = [-25.0, -15.0, 25.0, 35.0, 10.0]  # where it is put down


def pick_place_episode(index=0, held_jaw=25.0):
    """A -> B (open), close at B, carry to C, open at C, back to A."""
    segments = [
        (POSE_A, OPEN, OPEN, 20),
        (POSE_B, OPEN, OPEN, 20),
        (POSE_B, SHUT, held_jaw, 20),  # grasp at frame 40
        (POSE_C, SHUT, held_jaw, 20),
        (POSE_C, OPEN, OPEN, 20),  # release at frame 80
        (POSE_A, OPEN, OPEN, 20),
    ]
    state, action = [], []
    for pose, cmd, jaw, n in segments:
        for _ in range(n):
            action.append([*pose, cmd])
            state.append([*pose, jaw])
    return Episode(index, FPS, JOINTS, np.asarray(state), np.asarray(action))


@pytest.fixture(scope="module")
def tip():
    return Tip()


@pytest.fixture(scope="module")
def unit_map():
    return UnitMap.from_model(load_model())


def test_tip_matches_site_and_handles_batches(tip, unit_map):
    q = unit_map.to_rad(np.array([*POSE_B, 0.0]))
    one = tip(q)
    assert one.shape == (3,)
    batch = tip(np.stack([q, q]))
    assert batch.shape == (2, 3)
    np.testing.assert_allclose(batch[0], one)
    # Moving shoulder_pan swings the tip around the pan axis (vertical, at
    # x = 0.0388 in the base frame): height and reach from it stay the same.
    q2 = q.copy()
    q2[0] += 0.5
    other = tip(q2)
    assert other[2] == pytest.approx(one[2], abs=1e-9)
    axis = np.array([0.0388353, 0.0])
    assert np.hypot(*(other[:2] - axis)) == pytest.approx(np.hypot(*(one[:2] - axis)), abs=1e-6)


def test_find_grasps_locates_pick_and_place(tip, unit_map):
    ep = pick_place_episode()
    thr = dataset_thresholds([ep])
    grasps = find_grasps(ep, thr, tip, unit_map)
    assert len(grasps) == 1
    g = grasps[0]
    assert g.frame == 40
    assert g.release_frame == 80
    np.testing.assert_allclose(g.xyz, tip(unit_map.to_rad(np.array([*POSE_B, 0.0]))), atol=1e-9)
    np.testing.assert_allclose(g.release_xyz, tip(unit_map.to_rad(np.array([*POSE_C, 0.0]))), atol=1e-9)


def test_grasp_without_release(tip, unit_map):
    ep = pick_place_episode()
    cut = Episode(ep.index, FPS, JOINTS, ep.state[:70], ep.action[:70])
    (g,) = find_grasps(cut, dataset_thresholds([ep]), tip, unit_map)
    assert g.release_frame is None and g.release_xyz is None
    assert g.to_dict()["release_xyz"] is None
