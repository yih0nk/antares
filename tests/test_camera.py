"""Camera fitting: video index, projector, segmentation, the fit, and the lock/export plumbing.

Everything here runs offline: the only "video" is frames rendered from the
sim silhouette itself, so the expected answer is known exactly.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from antares.data import Dataset, Episode

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from antares import video  # noqa: E402
from antares.camera import (  # noqa: E402
    CameraFit,
    SilhouetteObjective,
    Space,
    fit_camera,
    load_camera_fits,
    pick_distinct,
    posed_points,
    sample_cameras,
    save_camera_fit,
    spread_frames,
)
from antares.project import (  # noqa: E402
    Camera,
    arm_hulls,
    iou,
    mat_to_quat,
    mat_to_rotvec,
    project_polygons,
    quat_to_mat,
    rasterize,
    rotvec_to_mat,
    silhouette,
)
from antares.silhouette import BackgroundModel, mask_polygons  # noqa: E402

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


# --------------------------------------------------------------------------
# video: metadata and episode -> file mapping (no decoding)
# --------------------------------------------------------------------------


def test_camera_names_and_wrist_detection():
    assert video.camera_name("observation.images.up") == "up"
    assert video.camera_name("up") == "up"
    assert video.camera_key("side") == "observation.images.side"
    assert video.is_fixed_camera("observation.images.top")
    assert not video.is_fixed_camera("observation.images.wrist")
    assert not video.is_fixed_camera("gripper_cam")


def test_cameras_from_info_reads_video_features():
    info = {
        "fps": 30,
        "features": {
            "observation.state": {"dtype": "float32"},
            "observation.images.up": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {"video.width": 640, "video.height": 480, "video.fps": 30},
            },
            "observation.images.wrist": {"dtype": "video", "shape": [240, 320, 3]},
        },
    }
    cams = {c.name: c for c in video.cameras_from_info(info)}
    assert set(cams) == {"up", "wrist"}
    assert (cams["up"].width, cams["up"].height, cams["up"].fixed) == (640, 480, True)
    assert (cams["wrist"].width, cams["wrist"].height, cams["wrist"].fixed) == (320, 240, False)


def test_segments_from_episodes_concatenates_in_order():
    segs = video.segments_from_episodes([(1, 60), (0, 30)], fps=30.0)
    assert segs[0].from_ts == 0.0 and segs[0].to_ts == pytest.approx(1.0)
    assert segs[1].from_ts == pytest.approx(1.0) and segs[1].length == 60


def test_frames_uses_cache_and_decodes_missing(tmp_path, monkeypatch):
    """Cached frames are read from disk; the rest go through one decode call."""
    monkeypatch.setattr(video, "FRAME_CACHE", tmp_path / "frames")
    seg = video.Segment(episode=2, chunk=0, file=0, from_ts=10.0, to_ts=12.0, length=60)
    cam = video.CameraInfo(name="up", key="observation.images.up", width=8, height=6, fps=30.0, fixed=True)
    calls = []

    def fake_decode(path, times, fps):
        calls.append(list(times))
        return np.stack([np.full((6, 8, 3), int(round((t - 10.0) * 30)), dtype=np.uint8) for t in times])

    monkeypatch.setattr(video, "episode_segments", lambda repo, camera, ds=None: {2: seg})
    monkeypatch.setattr(video, "find_camera", lambda repo, camera: cam)
    monkeypatch.setattr(video, "video_file", lambda repo, camera, chunk=0, file=0: tmp_path / "v.mp4")
    monkeypatch.setattr(video, "decode_at", fake_decode)

    out = video.frames("fake/ds", "up", 2, [5, 1, 5])
    assert out.shape == (3, 6, 8, 3)
    assert [int(o[0, 0, 0]) for o in out] == [5, 1, 5]
    assert len(calls) == 1 and calls[0] == pytest.approx([10 + 1 / 30, 10 + 5 / 30])

    out2 = video.frames("fake/ds", "up", 2, [1, 7])
    assert len(calls) == 2 and calls[1] == pytest.approx([10 + 7 / 30])  # only frame 7 decoded
    assert [int(o[0, 0, 0]) for o in out2] == [1, 7]

    with pytest.raises(video.VideoUnavailable):
        video.frames("fake/ds", "up", 2, [60])


# --------------------------------------------------------------------------
# project: rotations, pinhole, hulls, silhouette
# --------------------------------------------------------------------------


def test_rotation_round_trips():
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = rng.normal(size=3)
        r *= rng.uniform(0, 3.0) / np.linalg.norm(r)
        R = rotvec_to_mat(r)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.allclose(rotvec_to_mat(mat_to_rotvec(R)), R, atol=1e-9)
        assert np.allclose(quat_to_mat(mat_to_quat(R)), R, atol=1e-9)


def test_look_at_camera_puts_target_at_image_centre_and_up_is_up():
    cam = Camera.look_at([1.0, 0.5, 0.8], [0.0, 0.0, 0.1], fovy=60.0, width=640, height=480)
    uv, depth = cam.project(np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.3]]))
    assert np.allclose(uv[0], [320.0, 240.0])
    assert depth[0] == pytest.approx(math.sqrt(1 + 0.25 + 0.49))
    assert uv[1][1] < 240.0  # a higher point is higher in the image (v down)
    assert uv[1][0] == pytest.approx(320.0, abs=1e-6)  # and on the vertical centre line
    # Points behind the camera get non-positive depth.
    _, d = cam.project(np.array([[2.0, 1.0, 1.5]]))
    assert d[0] < 0
    # The quaternion is MuJoCo wxyz for the same rotation.
    assert np.allclose(quat_to_mat(cam.quat), cam.R)
    assert Camera.from_dict(cam.to_dict()).fovy == 60.0


def test_focal_from_fovy():
    cam = Camera(pos=[0, 0, 0], rotvec=[0, 0, 0], fovy=90.0, width=200, height=100)
    assert cam.focal == pytest.approx(50.0)
    assert cam.fovx == pytest.approx(math.degrees(2 * math.atan(2.0)))


def test_arm_hulls_cover_visual_geoms():
    from antares.sim import Arm

    arm = Arm()
    hulls = arm_hulls(arm.model)
    names = {g.body_name for g in hulls.geoms}
    assert {"base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"} <= names
    assert "camera_mount" not in names
    assert all(len(g.verts) >= 4 for g in hulls.geoms)
    assert hulls.n_points == sum(len(g.verts) for g in hulls.geoms)
    # Body-frame hull vertices move rigidly with the body under FK.
    pts0 = hulls.world_points(arm.fk(np.zeros(6)))
    pts1 = hulls.world_points(arm.fk(np.array([1.0, 0, 0, 0, 0, 0])))
    s, e = hulls.slices[[g.body_name for g in hulls.geoms].index("base")]
    assert np.allclose(pts0[s:e], pts1[s:e])  # the base does not move with the pan
    s, e = hulls.slices[[g.body_name for g in hulls.geoms].index("upper_arm")]
    assert not np.allclose(pts0[s:e], pts1[s:e])


def test_silhouette_is_filled_polygons_in_view():
    from antares.sim import Arm

    arm = Arm()
    hulls = arm_hulls(arm.model)
    cam = Camera.look_at([0.6, 0.5, 0.4], [0.0, 0.0, 0.12], fovy=50.0, width=160, height=120)
    pts = hulls.world_points(arm.fk(np.zeros(6)))
    mask, polys = silhouette(hulls, pts, cam)
    assert mask.shape == (120, 160) and mask.dtype == np.uint8
    assert 0.02 < mask.mean() < 0.5
    assert len(polys) == len(hulls.geoms)
    assert all(len(p) >= 3 for p in polys)
    # Rasterizing the same polygons at half scale gives the same coverage.
    half = rasterize(polys, 80, 60, scale=0.5)
    assert half.mean() == pytest.approx(mask.mean(), abs=0.03)
    # A camera looking away sees nothing.
    away = Camera.look_at([0.6, 0.5, 0.4], [1.2, 1.0, 0.8], fovy=50.0, width=160, height=120)
    m2, p2 = silhouette(hulls, pts, away)
    assert m2.sum() == 0 and all(len(p) == 0 for p in p2)


def test_iou():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2] = True
    b[1:3] = True
    assert iou(a, b) == pytest.approx(4 / 12)
    assert iou(a, a) == 1.0
    assert iou(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0


# --------------------------------------------------------------------------
# silhouette: background subtraction on synthetic frames
# --------------------------------------------------------------------------


def _synthetic_frames(rng, n=16, H=60, W=80, dark_arm=True):
    """A static 'base' block plus a moving 'arm' bar, over a flat background
    with a static same-coloured distractor the arm never touches."""
    bg_col = (225, 225, 230) if dark_arm else (40, 40, 45)
    arm_col = (25, 25, 25) if dark_arm else (240, 140, 30)
    frames, masks = [], []
    for i in range(n):
        f = np.tile(np.array(bg_col, np.uint8), (H, W, 1)).copy()
        m = np.zeros((H, W), np.uint8)
        f[48:58, 36:46] = arm_col  # base (static)
        m[48:58, 36:46] = 1
        f[4:12, 4:12] = arm_col  # distractor (static, never touched)
        f[40:48, 38:44] = arm_col  # shoulder (static, connects bar and base)
        m[40:48, 38:44] = 1
        # The arm: a bar pivoting about the shoulder top, sweeping a half circle.
        a = math.pi * (0.1 + 0.8 * i / n)
        tip = (int(41 + 26 * math.cos(a)), int(40 - 22 * math.sin(a)))
        cv2.line(f, (41, 40), tip, arm_col, 5)
        cv2.line(m, (41, 40), tip, 1, 5)
        noise = rng.integers(-6, 7, (H, W, 3))
        frames.append(np.clip(f.astype(int) + noise, 0, 255).astype(np.uint8))
        masks.append(m)
    return np.stack(frames), np.stack(masks)


@pytest.mark.parametrize("dark_arm", [True, False])
def test_background_model_recovers_static_base_and_drops_distractor(dark_arm):
    rng = np.random.default_rng(1)
    frames, truth = _synthetic_frames(rng, dark_arm=dark_arm)
    bgm = BackgroundModel.fit(frames)
    assert bgm.appearance.sign == (-1 if dark_arm else 1)
    assert bgm.static_arm[50, 40] == 1  # the base is in the static-arm region
    assert bgm.static_arm[8, 8] == 0  # the distractor is not
    ious = [iou(bgm.mask(f), t) for f, t in zip(frames, truth)]
    assert min(ious) > 0.7, ious
    m = bgm.mask(frames[3])
    assert m[8, 8] == 0 and m[50, 40] == 1
    polys = mask_polygons(m)
    assert polys and all(len(p) >= 3 for p in polys)


# --------------------------------------------------------------------------
# camera: frame selection, search space, the fit on synthetic video
# --------------------------------------------------------------------------


def _dataset(rng, n_eps=6, T=50):
    eps = []
    for i in range(n_eps):
        base = rng.uniform(-60, 60, 6)
        state = base + 30 * np.sin(np.linspace(0, 3, T)[:, None] + rng.uniform(0, 6, 6))
        state[:, 5] = np.clip(state[:, 5] + 50, 0, 100)
        state[:, :5] = np.clip(state[:, :5], -100, 100)
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=state, action=state))
    return Dataset(repo_id="fake/so101_cam", robot_type="so101", fps=30.0, joint_names=JOINTS, episodes=tuple(eps))


def test_spread_frames_spans_episodes_and_avoids_the_ends():
    ds = _dataset(np.random.default_rng(0))
    picks = spread_frames(ds.episodes, 3, 12)
    assert len(picks) == 12
    assert {e for e, _ in picks} == {0, 2, 4}
    assert all(4 <= f <= 45 for _, f in picks)
    other = spread_frames(ds.episodes, 3, 6, phase=0.5)
    assert {e for e, _ in other} == {1, 3, 5}


def test_space_round_trip():
    centre = np.array([0.05, 0.0, 0.12])
    ref = Camera.look_at([0.5, 0.4, 0.5], [0, 0, 0.1], 55.0, 64, 48)
    space = Space(ref=ref, centre=centre)
    x0 = space.encode(ref)
    assert np.allclose(x0[[4, 6]], 0.5)  # the reference sits mid-box in azimuth and roll
    assert np.allclose(space.decode(x0).pos, ref.pos, atol=1e-9)
    assert np.allclose(space.decode(x0).R, ref.R, atol=1e-9)
    # A camera elsewhere, rolled and tilted off the reference, survives the round trip.
    R = ref.R @ rotvec_to_mat([0.2, -0.1, 0.3])
    cam = Camera(pos=[0.3, -0.2, 0.7], rotvec=mat_to_rotvec(R), fovy=70.0, width=64, height=48)
    back = space.decode(space.encode(cam))
    assert np.allclose(back.pos, cam.pos, atol=1e-6)
    assert np.allclose(back.R, cam.R, atol=1e-6)
    assert back.fovy == pytest.approx(70.0)
    # A top-down camera (the degenerate look-at case) round-trips too.
    top = Camera.look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 50.0, 64, 48, up=(1.0, 0.0, 0.0))
    back = Space(ref=top, centre=centre).decode(Space(ref=top, centre=centre).encode(top))
    assert np.allclose(back.pos, top.pos, atol=1e-6) and np.allclose(back.R, top.R, atol=1e-6)


def test_sample_cameras_look_at_the_target():
    rng = np.random.default_rng(0)
    cams = sample_cameras(rng, 50, 64, 48, target=(0.1, 0.0, 0.1))
    for c in cams:
        uv, depth = c.project(np.array([[0.1, 0.0, 0.1]]))
        assert depth[0] > 0.2
        assert 0 <= uv[0][0] <= 64 and 0 <= uv[0][1] <= 48  # the arm is in the frame
    chosen = pick_distinct(cams, [float(c.pos[2]) for c in cams], 5, min_dist=0.2)
    assert len(chosen) == 5
    assert all(np.linalg.norm(cams[i].pos - cams[j].pos) >= 0.2 for i in chosen for j in chosen if i != j)


def test_camera_fit_round_trip_and_summary():
    fit = CameraFit(
        camera="up", key="observation.images.up", dataset="d", pos=(1, 2, 3), rotvec=(0.1, 0.2, 0.3),
        quat=(1, 0, 0, 0), fovy=50.0, width=640, height=480, fit_iou=0.7, heldout_iou=0.65,
        fit_frames=((0, 1), (2, 3)), heldout_frames=((4, 5),), fit_ious=(0.7, 0.7), heldout_ious=(0.65,),
        n_starts=8, best_start=2, start_ious=(0.3,) * 8, n_evaluations=100, elapsed_s=1.0,
    )
    d = json.loads(json.dumps(fit.to_dict()))
    assert CameraFit.from_dict(d) == fit
    assert "fit_ious" not in fit.summary() and fit.summary()["fit_iou"] == 0.7
    assert np.allclose(fit.to_camera().pos, [1, 2, 3])


def test_save_camera_fit_writes_sidecar_and_updates_lock(tmp_path):
    from antares.lock import RobotLock
    from tests.test_lock import fake_result

    lock = RobotLock.create("fake/ds", "so101", [], fake_result(), name="unit")
    lock.save(tmp_path / "unit.lock.json")
    assert RobotLock.load(tmp_path / "unit.lock.json").cameras == {}
    fit = CameraFit(
        camera="up", key="observation.images.up", dataset="fake/ds", pos=(0.5, 0.1, 0.5), rotvec=(0, 0, 0),
        quat=(1, 0, 0, 0), fovy=50.0, width=640, height=480, fit_iou=0.7, heldout_iou=0.65, fit_frames=(),
        heldout_frames=(), fit_ious=(), heldout_ious=(), n_starts=1, best_start=0, start_ious=(0.5,),
        n_evaluations=10, elapsed_s=1.0,
    )
    p = save_camera_fit(fit, "unit", tmp_path)
    assert p == tmp_path / "unit.camera.json"
    assert load_camera_fits("unit", tmp_path)["up"] == fit
    entry = RobotLock.load(tmp_path / "unit.lock.json").cameras["up"]
    assert entry["quat"] == [1, 0, 0, 0] and entry["fit_iou"] == 0.7 and "fit_ious" not in entry
    assert Camera.from_dict(entry).fovy == 50.0
    # A second camera merges rather than replaces.
    fit2 = CameraFit.from_dict({**fit.to_dict(), "camera": "side", "key": "observation.images.side"})
    save_camera_fit(fit2, "unit", tmp_path)
    assert set(load_camera_fits("unit", tmp_path)) == {"up", "side"}
    assert set(RobotLock.load(tmp_path / "unit.lock.json").cameras) == {"up", "side"}
    assert load_camera_fits("nope", tmp_path) == {}


def test_export_writes_fitted_cameras(tmp_path):
    import mujoco

    from antares.export import ExportError, export_mjcf, insert_cameras
    from antares.lock import RobotLock
    from tests.test_lock import fake_result

    cam = Camera.look_at([0.5, 0.1, 0.5], [0, 0, 0.1], 50.0, 640, 480)
    entry = {**cam.to_dict(), "key": "observation.images.up", "fit_iou": 0.7, "heldout_iou": 0.65}
    lock = RobotLock.create("fake/ds", "so101", [], fake_result(), name="unit").replace(cameras={"up": entry})
    out = export_mjcf(lock, tmp_path / "unit.xml", bundle_assets=False)
    xml = out.read_text()
    assert '<camera name="up"' in xml and "IoU 0.700 fit / 0.650 held-out" in xml
    model = mujoco.MjModel.from_xml_path(str(out))
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "up")
    assert cid >= 0
    assert np.allclose(model.cam_pos[cid], cam.pos, atol=1e-9)
    assert np.allclose(model.cam_quat[cid], cam.quat, atol=1e-6) or np.allclose(model.cam_quat[cid], -cam.quat, atol=1e-6)
    assert model.cam_fovy[cid] == pytest.approx(50.0)
    with pytest.raises(ExportError):
        insert_cameras(xml, {"up": entry})  # already there
    with pytest.raises(ExportError):
        insert_cameras(xml, {"bad name": entry})


def test_fit_camera_recovers_a_known_camera(monkeypatch):
    """Frames rendered from the sim with a known camera; the fit must find it."""
    from antares.sim import Arm

    rng = np.random.default_rng(3)
    ds = _dataset(rng)
    arm = Arm()
    hulls = arm_hulls(arm.model)
    W, H = 128, 96
    truth = Camera.look_at([0.55, 0.35, 0.45], [0.0, 0.05, 0.12], 55.0, W, H)
    cache: dict[tuple[int, int], np.ndarray] = {}

    def render(ep, f):
        if (ep, f) not in cache:
            pts = posed_points(arm, hulls, ds, [(ep, f)])[0]
            mask, _ = silhouette(hulls, pts, truth)
            rgb = np.full((H, W, 3), 230, np.uint8)
            rgb[mask.astype(bool)] = 20
            cache[(ep, f)] = rgb
        return cache[(ep, f)]

    cam_info = video.CameraInfo(name="up", key="observation.images.up", width=W, height=H, fps=30.0, fixed=True)
    monkeypatch.setattr(video, "find_camera", lambda repo, camera: cam_info)
    monkeypatch.setattr(video, "frames", lambda repo, camera, ep, idx, ds=None, cache=True: np.stack([render(ep, i) for i in idx]))

    events = []
    fit = fit_camera(
        ds, "up", n_frames=12, n_episodes=3, n_heldout=4, n_starts=3, n_samples=800,
        start_evaluations=150, refine_evaluations=250, seed=0, arm=arm, on_progress=events.append,
    )
    assert fit.width == W and fit.height == H
    assert fit.fit_iou > 0.75, fit.fit_iou
    assert fit.heldout_iou > 0.7, fit.heldout_iou
    assert len(fit.fit_frames) == 12 and len(fit.heldout_frames) == 4
    assert not ({e for e, _ in fit.fit_frames} & {e for e, _ in fit.heldout_frames})
    assert [e.stage for e in events][0] == "sample" and events[-1].stage == "heldout"
    # Distance and field of view trade off against each other in a silhouette,
    # so the position is only loosely pinned; the viewing direction is not.
    assert np.linalg.norm(np.array(fit.pos) - truth.pos) < 0.25
    forward = -fit.to_camera().R[:, 2]
    assert np.degrees(np.arccos(np.clip(forward @ -truth.R[:, 2], -1, 1))) < 15
    assert fit.n_starts == 3 and fit.n_evaluations > 800

    # The objective agrees with the saved IoU, and the truth scores at least as well.
    masks_fit = np.stack([render(e, f)[..., 0] < 128 for e, f in fit.fit_frames])
    obj = SilhouetteObjective(hulls, posed_points(arm, hulls, ds, fit.fit_frames), masks_fit)
    assert obj(fit.to_camera()) == pytest.approx(fit.fit_iou, abs=0.08)
    assert obj(truth) > 0.9

    # Cancellation stops early with a usable (if rough) fit.
    calls = {"n": 0}

    def stop_soon():
        calls["n"] += 1
        return calls["n"] > 3

    partial = fit_camera(ds, "up", n_frames=8, n_episodes=2, n_heldout=2, n_starts=2, n_samples=200,
                         start_evaluations=40, refine_evaluations=40, arm=arm, should_stop=stop_soon)
    assert partial.cancelled and 0.0 <= partial.fit_iou <= 1.0


def test_fit_camera_rejects_wrist_cameras(monkeypatch):
    ds = _dataset(np.random.default_rng(0))
    info = video.CameraInfo(name="wrist", key="observation.images.wrist", width=8, height=6, fps=30.0, fixed=False)
    monkeypatch.setattr(video, "find_camera", lambda repo, camera: info)
    with pytest.raises(ValueError, match="wrist"):
        fit_camera(ds, "wrist")
