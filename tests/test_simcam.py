"""Policy camera frames rendered from the sim (antares.simcam).

MuJoCo's offscreen renderer needs an OpenGL context; without one (a
headless CI box with no EGL/OSMesa) the rendering tests skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from antares.data import Episode
from antares.project import Camera
from antares.sim import JOINTS, UnitMap, load_model

ROOT = Path(__file__).resolve().parents[1]
W, H = 160, 120
GREY = 200


def _cams():
    from antares.simcam import SimCameras

    cam = Camera.look_at((0.45, 0.25, 0.35), (0.15, 0.0, 0.05), 50.0, W, H)
    try:
        sc = SimCameras({"up": cam}, {"up": np.full((H, W, 3), GREY, np.uint8)}, object_size=(0.04, 0.02, 0.02),
                        object_rgb=(220, 40, 40), object_emission=1.0)
        sc.render(np.zeros(sc.n_arm_q))
    except Exception as e:  # noqa: BLE001 - no GL context
        pytest.skip(f"no OpenGL context for mujoco.Renderer: {e}")
    return sc


@pytest.fixture(scope="module")
def cams():
    sc = _cams()
    yield sc
    sc.close()


def test_render_changes_only_the_arm_and_object_pixels(cams):
    q = np.zeros(cams.n_arm_q)
    img = cams.render(q)["up"]
    rgb, arm, obj = cams.layers(q)["up"]
    assert img.shape == (H, W, 3) and arm.any() and not obj.any()  # no object position: hidden
    assert np.all(img[~arm] == GREY) and np.array_equal(img[arm], rgb[arm])
    with_obj = cams.render(q, (0.25, -0.05, 0.0))["up"]
    _, arm2, obj2 = cams.layers(q, (0.25, -0.05, 0.0))["up"]
    assert obj2.sum() > 10 and not (arm2 & obj2).any()
    assert np.mean(with_obj[obj2][:, 0]) > np.mean(with_obj[obj2][:, 1]) + 50  # red
    alone = cams.layers(q, (0.25, -0.05, 0.0), object_only=True)["up"]
    assert not alone[1].any() and alone[2].sum() >= obj2.sum()


def test_render_over_a_given_background(cams):
    bg = np.zeros((H, W, 3), np.uint8)
    q = np.zeros(cams.n_arm_q)
    img = cams.render(q, backgrounds={"up": bg})["up"]
    arm = cams.layers(q)["up"][1]
    assert np.all(img[~arm] == 0)
    with pytest.raises(ValueError, match="background"):
        cams.render(q, backgrounds={"up": np.zeros((10, 10, 3), np.uint8)})
    with pytest.raises(ValueError, match="joints"):
        cams.render(np.zeros(3))


def test_policy_inputs_are_lerobot_images(cams):
    pytest.importorskip("torch")
    from antares.policy import StepContext

    f = cams.policy_inputs(object_offset=(0.0, 0.0, 0.01))
    x = f(StepContext(0, 30.0, q_rad=np.zeros(cams.n_arm_q), object_xyz=np.array([0.25, -0.05, 0.0])))
    t = x["observation.images.up"]
    assert tuple(t.shape) == (3, H, W) and float(t.min()) >= 0.0 and float(t.max()) <= 1.0
    with pytest.raises(ValueError, match="q_rad"):
        f(StepContext(0, 30.0))


def _episode(index: int, offset_deg: float) -> Episode:
    um = UnitMap.from_model(load_model())
    q = np.zeros((5, 6))
    q[:, 0] = np.radians(offset_deg + 10.0 * np.arange(5))
    units = um.to_units(q)
    return Episode(index=index, fps=30.0, joint_names=JOINTS, state=units, action=units)


def test_video_arm_takes_the_nearest_recorded_pose_and_cuts_its_object(cams):
    from antares.simcam import VideoArm

    um = UnitMap.from_model(load_model())
    eps = [_episode(3, 0.0), _episode(4, 5.0)]
    frames = {}

    def frame(ep, cam, t):
        frames[(ep, cam, t)] = True
        return np.full((H, W, 3), 10 * ep + t, np.uint8)

    full = lambda rgb: np.ones(rgb.shape[:2], np.uint8)  # noqa: E731 - "the whole frame is arm"
    track = {4: np.tile([0.25, -0.05, 0.0], (5, 1))}
    va = VideoArm(eps, um, frame, {"up": full}, objects=track, object_margin_px=2)
    q = np.zeros(cams.n_arm_q)
    q[0] = np.radians(26.0)  # nearest: episode 4 frame 2 (25 deg)
    ep, t, d = va.nearest(q)
    assert (ep, t) == (4, 2) and d == pytest.approx(1.0 / np.sqrt(6), rel=1e-6)
    img = cams.render(q, None, arm=va)["up"]
    obj = cams.layers(np.zeros(cams.n_arm_q), track[4][2], object_only=True)["up"][2]
    assert va.last[:2] == (4, 2) and va.distances == [pytest.approx(d)]
    cv2 = pytest.importorskip("cv2")
    cut = cv2.dilate(obj.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # margin 2 px
    assert np.all(img[cut] == GREY)  # that frame's own object was cut out of the arm mask
    assert np.all(img[~cut] == 42)  # everywhere else: episode 4's frame 2
    with pytest.raises(ValueError, match="empty"):
        VideoArm([], um, frame, {"up": full})


def test_from_lock_uses_the_committed_fits():
    pytest.importorskip("cv2")
    from antares.simcam import SimCameras

    lock = json.loads((ROOT / "locks" / "svla_so101_pickplace.lock.json").read_text())
    try:
        sc = SimCameras.from_lock(lock, ROOT / "locks", cameras=["up", "side"])
    except Exception as e:  # noqa: BLE001
        if "Renderer" in repr(e) or "GL" in repr(e):
            pytest.skip(str(e))
        raise
    assert set(sc.cameras) == {"up", "side"} and sc.backdrops["up"].shape == (480, 640, 3)
    long, short, height = sc.object_size
    assert 0.03 < long < 0.05 and 0.01 < short < 0.03 and height == short
    with pytest.raises(ValueError, match="camera fits"):
        SimCameras.from_lock(lock, ROOT / "locks", cameras=["wrist"])
