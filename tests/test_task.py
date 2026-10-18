import numpy as np
import pytest

from antares.data import Episode
from antares.sim import JOINTS, UnitMap, load_model
from antares.task import Tip, dataset_thresholds, empty_level, find_grasps

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
    grasps = find_grasps(ep, thr, tip, unit_map, empty_level([ep], thr))
    assert len(grasps) == 1
    g = grasps[0]
    assert g.frame == 40
    assert g.release_frame == 80
    np.testing.assert_allclose(g.xyz, tip(unit_map.to_rad(np.array([*POSE_B, 0.0]))), atol=1e-9)
    np.testing.assert_allclose(g.release_xyz, tip(unit_map.to_rad(np.array([*POSE_C, 0.0]))), atol=1e-9)


def test_grasp_without_release(tip, unit_map):
    ep = pick_place_episode()
    cut = Episode(ep.index, FPS, JOINTS, ep.state[:70], ep.action[:70])
    thr = dataset_thresholds([ep])
    (g,) = find_grasps(cut, thr, tip, unit_map, empty_level([ep], thr))
    assert g.release_frame is None and g.release_xyz is None
    assert g.to_dict()["release_xyz"] is None


def test_held_and_missed_grasps(tip, unit_map):
    held = pick_place_episode(0, held_jaw=25.0)
    missed = pick_place_episode(1, held_jaw=SHUT + 0.5)
    thr = dataset_thresholds([held, missed])
    empty = empty_level([held, missed], thr)
    assert empty == pytest.approx(SHUT)
    (g,) = find_grasps(held, thr, tip, unit_map, empty)
    assert g.held and g.jaw_gap == pytest.approx(25.0 - SHUT)
    assert g.jaw_deg == pytest.approx(np.degrees(unit_map.to_rad(np.array([0, 0, 0, 0, 0, 25.0]))[5]))
    (m,) = find_grasps(missed, thr, tip, unit_map, empty)
    assert not m.held and m.jaw_gap == pytest.approx(0.5)


def test_distribution_clouds_summary_and_roundtrip(tmp_path, tip, unit_map):
    from antares.task import TaskDistribution

    eps = [pick_place_episode(0), pick_place_episode(1, held_jaw=SHUT + 0.5), pick_place_episode(2)]
    td = TaskDistribution.from_episodes(eps, name="toy", dataset="local/toy")
    assert len(td.grasps) == 3
    assert td.picks.shape == (2, 3) and td.places.shape == (2, 3)
    assert td.missed_episodes == [1]
    s = td.summary()
    assert (s["held"], s["missed"]) == (2, 1)
    np.testing.assert_allclose(s["pick"]["mean"], tip(unit_map.to_rad(np.array([*POSE_B, 0.0]))), atol=1e-9)
    assert td.start_q.shape == (3, 6)

    path = td.save(tmp_path / "toy.task.json")
    back = TaskDistribution.load(path)
    assert back.to_dict() == td.to_dict()
    assert td.default_path(tmp_path).name == "toy.task.json"


def test_sample_draws_near_logged_points():
    from antares.task import TaskDistribution

    td = TaskDistribution.from_episodes([pick_place_episode(0)], name="toy")
    draws = td.sample(200, np.random.default_rng(0), jitter_m=0.01)
    picks = np.array([d["pick"] for d in draws])
    assert np.all(np.abs(picks - td.picks[0]) < 0.06)
    assert picks.std(axis=0) == pytest.approx([0.01] * 3, rel=0.2)
    assert all(len(d["start_q"]) == 6 and d["place"] is not None for d in draws)


def test_sample_needs_a_held_grasp():
    from antares.task import TaskDistribution

    td = TaskDistribution.from_episodes([pick_place_episode(0, held_jaw=SHUT)], name="toy")
    with pytest.raises(ValueError):
        td.sample(1)


def test_thresholds_catch_a_shallow_close(tip, unit_map):
    deep = pick_place_episode(0)
    # Same motion, but the leader is only closed to 30 on the cube.
    shallow = pick_place_episode(1)
    act = shallow.action.copy()
    act[act[:, 5] == SHUT, 5] = 30.0
    shallow = Episode(1, FPS, JOINTS, shallow.state, act)
    thr = dataset_thresholds([deep, shallow])
    assert thr.closed_below > 30.0
    assert len(find_grasps(shallow, thr, tip, unit_map, empty_level([deep, shallow], thr))) == 1


def test_held_read_before_the_operator_lets_go(tip, unit_map):
    """No release is detected (the leader only opens a little), so the closed
    stretch runs to the end; the jaw must still be read while it holds."""
    ep = pick_place_episode(0)
    st, act = ep.state.copy(), ep.action.copy()
    act[80:, 5] = SHUT + 5.0  # a shallow open the thresholds do not see
    st[80:100, 5] = SHUT + 5.0  # cube dropped, jaw at the command
    st[100:, 5] = SHUT  # and shut again
    ep = Episode(0, FPS, JOINTS, st, act)
    thr = dataset_thresholds([ep])
    (g,) = find_grasps(ep, thr, tip, unit_map, empty_level([ep], thr))
    assert g.held and g.jaw_gap == pytest.approx(25.0 - SHUT)
    assert g.release_frame is None  # the jaw dropped to the command, it never opened past the cube


def test_release_found_from_the_jaw_opening(tip, unit_map):
    ep = pick_place_episode(0)
    st, act = ep.state.copy(), ep.action.copy()
    act[80:, 5] = SHUT + 5.0  # the leader barely opens
    st[80:, 5] = 40.0  # but the follower's jaw opens past the cube
    ep = Episode(0, FPS, JOINTS, st, act)
    thr = dataset_thresholds([ep])
    (g,) = find_grasps(ep, thr, tip, unit_map, empty_level([ep], thr))
    assert g.held and g.release_frame == 80
    np.testing.assert_allclose(g.release_xyz, tip(unit_map.to_rad(np.array([*POSE_C, 0.0]))), atol=1e-9)


def _segments_episode(index, segments):
    state, action = [], []
    for pose, cmd, jaw, n in segments:
        for _ in range(n):
            action.append([*pose, cmd])
            state.append([*pose, jaw])
    return Episode(index, FPS, JOINTS, np.asarray(state), np.asarray(action))


def test_kinds_miss_pick_regrab_idle(tip, unit_map):
    from antares.task import TaskDistribution

    ep = _segments_episode(0, [
        (POSE_A, OPEN, OPEN, 20),
        (POSE_B, SHUT, SHUT, 20),  # miss: closes on nothing
        (POSE_B, OPEN, OPEN, 20),
        (POSE_B, SHUT, 25.0, 20),  # pick
        (POSE_B, OPEN, OPEN, 20),  # let go in place: a re-grab, not carried
        (POSE_B, SHUT, 25.0, 20),  # pick again
        (POSE_C, SHUT, 25.0, 20),
        (POSE_C, OPEN, OPEN, 20),  # placed
        (POSE_A, SHUT, SHUT, 20),  # idle: shut after letting go
    ])
    td = TaskDistribution.from_episodes([ep], name="toy")
    assert [g.kind for g in td.grasps] == ["miss", "pick", "pick", "idle"]
    assert [g.carried for g in td.grasps] == [False, False, True, False]
    assert td.picks.shape == (2, 3) and td.places.shape == (1, 3)
    s = td.summary()
    assert (s["held"], s["carried"], s["missed"], s["idle"]) == (2, 1, 1, 1)
    assert s["missed_episodes"] == [0] and s["episodes_without_pick"] == []
    assert all(d["place"] is not None for d in td.sample(20, np.random.default_rng(0)))
