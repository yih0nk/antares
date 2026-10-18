"""Object detection and tracking on a synthetic scene with a known object."""

from __future__ import annotations

import json

import numpy as np
import pytest

from antares.data import Dataset, Episode
from antares.objects import (
    ColourPrior,
    ObjectFit,
    describe,
    fit_objects,
    footprint_on_plane,
    learn_prior,
    load_object_fit,
    load_track,
    pixel_rays,
    ray_plane,
    save_object_fit,
    track_episode,
    triangulate,
)
from antares.project import ArmHulls, Camera
from antares.scene import fingertips
from antares.silhouette import BackgroundModel

pytest.importorskip("cv2")

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
W, H = 320, 240
TABLE_Z = -0.02
HALF = 0.015
PLANE_Z = TABLE_Z + HALF
SIZE = (0.04, 0.02)  # the synthetic object's footprint (long, short)
RED = (200, 60, 60)
GREY = 150
#: Straight down over the workspace, x to the top of the image.
CAM = Camera.look_at([0.22, 0.0, 0.45], [0.22, 0.0, 0.0], 50.0, W, H, up=(1.0, 0.0, 0.0))
#: A second camera off to the side for the triangulation check.
CAM2 = Camera.look_at([0.5, -0.35, 0.3], [0.22, 0.0, 0.0], 50.0, W, H)


def render(cam: Camera, objects: list[np.ndarray]) -> np.ndarray:
    """Grey table with a red box (SIZE x 2*HALF) centred at each of ``objects``."""
    import cv2

    img = np.full((cam.height, cam.width, 3), GREY, np.uint8)
    for c in objects:
        hx, hy = SIZE[0] / 2, SIZE[1] / 2
        corners = np.array([[c[0] + sx, c[1] + sy, c[2] + sz] for sx in (-hx, hx) for sy in (-hy, hy) for sz in (-HALF, HALF)])
        uv, depth = cam.project(corners)
        hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
        cv2.fillConvexPoly(img, np.round(hull * 16).astype(np.int32).reshape(-1, 1, 2), RED, lineType=cv2.LINE_8, shift=4)
    return img


def background_model(cam: Camera = CAM) -> BackgroundModel:
    """Fitted on frames with the object in different places, so the median is the bare table."""
    rng = np.random.default_rng(0)
    frames = [render(cam, [np.array([0.15 + 0.12 * rng.random(), -0.1 + 0.2 * rng.random(), PLANE_Z])]) for _ in range(7)]
    return BackgroundModel.fit(np.stack(frames))


@pytest.fixture(scope="module")
def arm():
    from antares.sim import Arm

    return Arm()


@pytest.fixture(scope="module")
def hulls(arm):
    """No arm silhouette: the synthetic frames hold no arm, and where the real
    arm's hulls would land in this camera is not what these tests are about."""
    return ArmHulls(geoms=(), body_of_point=np.zeros(0, dtype=int), slices=(), points_body=np.zeros((0, 3)))


def rest_state(T: int) -> np.ndarray:
    """The arm folded back away from the camera's view of the table."""
    st = np.zeros((T, 6))
    st[:, 1] = -95.0  # shoulder_lift back
    st[:, 2] = 95.0  # elbow folded
    st[:, 5] = 90.0  # gripper open
    return st


def pick_episode(index: int, start: np.ndarray, grasp: int = 40, release: int = 70, T: int = 100) -> tuple[Episode, np.ndarray]:
    """An episode whose arm sits still while the gripper closes at ``grasp``
    and opens at ``release`` holding something (jaw stops short of empty),
    then shuts on nothing at the end. Returns it with the end position of the
    object: right under the fingertip at the release, on the plane."""
    st = rest_state(T)
    act = st.copy()
    act[grasp:release, 5] = 5.0
    st[grasp + 3 : release, 5] = 20.0  # the jaw stops on the object
    act[T - 10 :, 5] = 5.0
    st[T - 8 :, 5] = 2.0  # closed on nothing: the empty level
    return Episode(index=index, fps=30.0, joint_names=JOINTS, state=st, action=act), start


def frames_for(arm, ep: Episode, start: np.ndarray, end: np.ndarray, cam: Camera = CAM):
    """Frame source: the object at ``start`` before the grasp, unseen while held, at ``end`` after the release."""
    g, r = int(np.flatnonzero(np.diff((ep.action[:, 5] < 50).astype(int)) == 1)[0] + 1), None
    rel = np.flatnonzero(np.diff((ep.action[:, 5] < 50).astype(int)) == -1)
    r = int(rel[0] + 1) if rel.size else len(ep)
    for t in range(len(ep)):
        yield t, render(cam, [start] if t < g else ([] if t <= r else [end]))


def test_rays_and_plane_round_trip():
    p = np.array([[0.25, 0.03, PLANE_Z], [0.18, -0.08, PLANE_Z]])
    uv, depth = CAM.project(p)
    assert np.all(depth > 0)
    back = ray_plane(CAM, uv, PLANE_Z)
    np.testing.assert_allclose(back, p, atol=1e-9)
    d = pixel_rays(CAM, uv)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0)
    # a ray looking away from the plane gives NaN
    up = Camera.look_at([0.2, 0.0, 0.5], [0.2, 0.0, 1.0], 50.0, W, H, up=(1.0, 0.0, 0.0))
    assert np.isnan(ray_plane(up, np.array([[W / 2, H / 2]]), PLANE_Z)).all()


def test_triangulate_two_cameras():
    p = np.array([0.24, -0.02, PLANE_Z])
    (ua,), _ = CAM.project(p[None])
    (ub,), _ = CAM2.project(p[None])
    q, gap = triangulate(CAM, ua, CAM2, ub)
    np.testing.assert_allclose(q, p, atol=1e-6)
    assert gap < 1e-6


def test_footprint_from_a_box_on_the_plane():
    c = np.array([0.24, 0.0, PLANE_Z])
    img = render(CAM, [c])
    mask = (img[:, :, 0] > 180) & (img[:, :, 1] < 100)
    long_, short, _ = footprint_on_plane(mask, CAM, PLANE_Z)
    # top-down the box's top face is its footprint, plus a little of the sides
    assert SIZE[0] * 0.9 < long_ < SIZE[0] * 1.3 and SIZE[1] * 0.9 < short < SIZE[1] * 1.4


def test_learn_prior_finds_the_red_object(arm, hulls):
    bgm = background_model()
    samples = []
    for x in (0.18, 0.22, 0.26):
        frame = render(CAM, [np.array([x, 0.02, PLANE_Z])])
        samples.append((frame, np.zeros((H, W), np.uint8)))
    prior = learn_prior(samples, bgm)
    assert prior.n_samples == 3 and prior.area_px > 50
    assert abs(prior.rgb[0] - RED[0]) < 20 and prior.rgb[1] < 100
    lab = np.asarray(prior.lab)
    assert lab[1] > 20  # red: positive a*
    assert "red" in describe(prior, SIZE, (GREY, GREY, GREY))
    assert "4.0 x 2.0 cm" in describe(prior, SIZE, None)
    with pytest.raises(ValueError):
        learn_prior([(render(CAM, []), np.zeros((H, W), np.uint8))], bgm)
    d = prior.to_dict()
    assert ColourPrior.from_dict(json.loads(json.dumps(d))).lab == prior.lab


def test_track_episode_vision_then_held_then_vision(arm, hulls):
    bgm = background_model()
    start = np.array([0.26, 0.05, PLANE_Z])
    ep, _ = pick_episode(0, start)
    tips = fingertips(arm, arm.fk(arm.unit_map.to_rad(ep.state)))
    end = np.array([tips[70, 0] + 0.08, tips[70, 1], PLANE_Z])  # slid a little way from the jaws
    prior = learn_prior([(render(CAM, [start]), np.zeros((H, W), np.uint8))], bgm)
    tr = track_episode(ep, frames_for(arm, ep, start, end), "top", CAM, bgm, arm, hulls, prior, [(40, 70)], PLANE_Z)
    names = tr.source_names()
    assert names[:40] == ["vision"] * 40
    assert set(names[40:71]) == {"held"}
    assert "vision" in names[71:]
    np.testing.assert_allclose(tr.xyz[:40], np.tile(start, (40, 1)), atol=0.004)
    np.testing.assert_allclose(tr.xyz[50], tips[50], atol=1e-9)
    assert tr.start_pose is not None and np.linalg.norm(tr.start_pose - start) < 0.004
    assert tr.end_source == "vision" and np.linalg.norm(tr.end_pose - end) < 0.006
    assert tr.footprint is not None and 0.03 < tr.footprint[0] < 0.055 and 0.015 < tr.footprint[1] < 0.03
    assert tr.vision_vs_fk == pytest.approx(np.linalg.norm(start - tips[40]), abs=0.004)
    assert np.isfinite(tr.end_vs_fk)
    assert tr.n_visible == int(tr.visible.sum()) > 40
    d = tr.to_dict()
    assert len(d["xyz"]) == len(ep) and d["bbox"][0] is not None and d["bbox"][50] is None
    assert d["start_pose"] is not None and d["source"][50] == "held"


def test_track_without_a_pick_keeps_vision_and_says_so(arm, hulls):
    bgm = background_model()
    start = np.array([0.2, -0.04, PLANE_Z])
    ep, _ = pick_episode(1, start)
    prior = learn_prior([(render(CAM, [start]), np.zeros((H, W), np.uint8))], bgm)
    tr = track_episode(ep, ((t, render(CAM, [start])) for t in range(len(ep))), "top", CAM, bgm, arm, hulls, prior, [], PLANE_Z)
    assert tr.picks == [] and "no held grasp" in tr.note
    assert tr.end_source == "none" and tr.end_pose is None and np.isnan(tr.vision_vs_fk)
    assert tr.visible.all()


def test_fit_objects_over_a_dataset_and_save(arm, hulls, tmp_path):
    bgm = background_model()
    starts = {0: np.array([0.26, 0.05, PLANE_Z]), 1: np.array([0.2, -0.06, PLANE_Z]), 2: np.array([0.23, 0.0, PLANE_Z])}
    eps = [pick_episode(i, s)[0] for i, s in starts.items()]
    ds = Dataset(repo_id="fake/objects", robot_type="so101", fps=30.0, joint_names=JOINTS, episodes=tuple(eps))
    tips = fingertips(arm, arm.fk(arm.unit_map.to_rad(eps[0].state)))
    end = np.array([tips[70, 0] + 0.08, tips[70, 1], PLANE_Z])
    events = []
    fit, tracks = fit_objects(
        ds, arm, "unit", "top", CAM, bgm, TABLE_Z,
        object_half_height=HALF, table_rgb=(GREY, GREY, GREY), hulls=hulls,
        frame_source=lambda ep: frames_for(arm, ep, starts[ep.index], end),
        on_progress=events.append,
    )
    assert [e["stage"] for e in events] == ["prior", "track", "track", "track"]
    assert fit.n_tracked == 3 and fit.start["n"] == 3 and fit.end["n"] == 3
    np.testing.assert_allclose(fit.start["points"], [starts[i] for i in range(3)], atol=0.005)
    assert fit.footprint is not None and 0.03 < fit.footprint[0] < 0.055
    assert fit.vision_vs_fk_m["n"] == 3 and len(fit.vision_vs_fk_offset_m) == 3
    assert "red" in fit.description and "table" in fit.description
    assert fit.mass_kg is None  # no payload estimate attached
    entry = fit.lock_entry()
    assert entry["footprint_m"] is not None and entry["start_mean"] is not None and entry["n_tracked"] == 3

    # save: JSON + tracks npz + the lock's objects field
    from antares.lock import RobotLock
    from tests.test_lock import fake_result

    RobotLock.create("fake/objects", "so101", [], fake_result(), name="unit").save(tmp_path / "unit.lock.json")
    path = save_object_fit(fit, "unit", tmp_path, tracks)
    assert path == tmp_path / "unit.objects.json" and (tmp_path / "objects" / "unit_top_tracks.npz").is_file()
    back = load_object_fit("unit", tmp_path)
    assert isinstance(back, ObjectFit) and back.description == fit.description and len(back.tracks) == 3
    lock = json.loads((tmp_path / "unit.lock.json").read_text())
    assert lock["objects"]["description"] == fit.description and lock["objects"]["camera"] == "top"
    tr = load_track(back, 1, tmp_path)
    assert tr is not None and tr.source_names() == tracks[1].source_names()
    np.testing.assert_allclose(tr.xyz, tracks[1].xyz, atol=1e-5)
    assert tr.start_pose is not None and tr.picks == [(40, 70)]
    assert load_track(back, 99, tmp_path) is None
    assert load_object_fit("ghost", tmp_path) is None
    # the vision start poses feed the task distribution
    s = back.sample_start(5, np.random.default_rng(1))
    assert s.shape == (5, 3)
