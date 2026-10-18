"""The static scene from the logs: table, workspace, backdrops, sim view, export."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np
import pytest

from antares.data import Dataset, Episode
from antares.project import Camera, arm_hulls, rasterize
from antares.scene import (
    OBJECT_HALF_HEIGHT,
    Backdrop,
    SceneFit,
    TablePlane,
    Workspace,
    backdrop_quad,
    check_backdrop,
    fingertips,
    fit_plane,
    fit_scene,
    fit_table,
    fit_workspace,
    hull_xy,
    load_scene_fit,
    polygon_mask,
    project_quad,
    save_scene_fit,
    sim_view,
    table_quad,
    table_region,
    tip_frame,
)
from antares.sim import JOINTS, Arm

pytest.importorskip("cv2")


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def arm():
    return Arm()


def pick_place_dataset(n_episodes: int = 3, T: int = 60) -> Dataset:
    """Episodes whose gripper closes once low and opens once higher, with a
    little joint motion so the fingertips move around."""
    eps = []
    rng = np.random.default_rng(0)
    for i in range(n_episodes):
        state = np.zeros((T, 6))
        state[:, 0] = np.linspace(-30, 30, T) + 5 * i  # pan sweeps
        state[:, 1] = -20 + 10 * np.sin(np.linspace(0, math.pi, T))
        state[:, 2] = 30 + 10 * np.cos(np.linspace(0, math.pi, T))
        state[:, 3] = 20 + rng.normal(0, 1, T)
        action = state.copy()
        action[:, 5] = 90.0
        action[T // 3 : 2 * T // 3, 5] = 5.0  # closed in the middle third
        state[:, 5] = action[:, 5]
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=state, action=action))
    return Dataset(repo_id="fake/ds", robot_type="so101", fps=30.0, joint_names=JOINTS, episodes=tuple(eps))


# -- fingertips -----------------------------------------------------------------


def test_tip_frame_is_the_gripperframe_site_on_the_gripper(arm):
    body, offset = tip_frame(arm.model)
    assert arm.body_names[body] == "gripper"
    assert offset.shape == (3,) and np.linalg.norm(offset) > 0.05  # ~10 cm down the jaws


def test_fingertips_follow_the_body_poses(arm):
    poses = arm.fk(np.zeros((4, 6)))
    tips = fingertips(arm, poses)
    assert tips.shape == (4, 3)
    assert np.allclose(tips[0], fingertips(arm, poses[0]))
    body, offset = tip_frame(arm.model)
    assert np.allclose(np.linalg.norm(tips[0] - poses[0, body, :3]), np.linalg.norm(offset))


# -- table -----------------------------------------------------------------------


def test_fit_table_uses_the_low_grasp_cluster_minus_the_object():
    rng = np.random.default_rng(1)
    low = np.column_stack([rng.uniform(0.1, 0.3, 40), rng.uniform(-0.2, 0.2, 40), 0.010 + rng.normal(0, 0.001, 40)])
    high = np.column_stack([rng.uniform(0.1, 0.3, 10), rng.uniform(-0.2, 0.2, 10), 0.08 + rng.normal(0, 0.002, 10)])
    t = fit_table(np.vstack([low, high]), min_fingertip_z=-0.004, episode_min_z=[-0.004, -0.002, -0.003])
    assert t.n_grasps == 50 and t.n_on_table == 40
    assert t.z_grasp == pytest.approx(0.010, abs=0.001)
    assert t.z == pytest.approx(0.010 - OBJECT_HALF_HEIGHT, abs=0.001)
    assert t.horizontal and t.normal == (0.0, 0.0, 1.0) and t.tilt_deg < 1.0
    assert t.fingertip_z["n"] == 50 and t.fingertip_z["p90"] > 0.05 and t.min_fingertip_z == -0.004
    assert t.episode_min_z["spread_mm"] == pytest.approx(2.0)
    assert t.height_at(1.0, 2.0) == t.z


def test_fit_table_drops_the_horizontal_assumption_for_a_clearly_tilted_spread_of_grasps():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0.0, 0.4, 60), rng.uniform(-0.2, 0.2, 60)
    z = 0.01 + 0.2 * x  # ~11 deg tilt about y
    t = fit_table(np.column_stack([x, y, z]), min_fingertip_z=0.0)
    assert not t.horizontal and t.tilt_deg == pytest.approx(math.degrees(math.atan(0.2)), abs=0.5)
    assert t.normal[2] > 0.9 and t.normal[0] < 0
    assert t.height_at(0.4, 0.0) - t.height_at(0.0, 0.0) == pytest.approx(0.08, abs=1e-6)


def test_fit_table_keeps_horizontal_when_the_grasps_lie_on_a_line():
    x = np.linspace(0.0, 0.4, 30)
    pts = np.column_stack([x, np.zeros_like(x), 0.01 + 0.2 * x])  # tilted but only along x
    t = fit_table(pts, min_fingertip_z=0.0)
    assert t.horizontal


def test_fit_table_needs_a_grasp():
    with pytest.raises(ValueError):
        fit_table(np.zeros((0, 3)), 0.0)


def test_fit_plane_normal_points_up():
    n, tilt = fit_plane(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]]))
    assert np.allclose(n, [0, 0, 1]) and tilt == 0.0
    n2, _ = fit_plane(np.array([[0, 0, 0], [1, 0, 0]]))
    assert np.allclose(n2, [0, 0, 1])


# -- workspace -------------------------------------------------------------------


def test_workspace_box_and_hull():
    pts = np.array([[0, 0, 0], [1, 0, 0.1], [1, 1, 0.2], [0, 1, 0.3], [0.5, 0.5, 0.05]])
    ws = fit_workspace(pts)
    assert ws.min == (0, 0, 0) and ws.max == (1, 1, 0.3) and ws.size == (1.0, 1.0, 0.3)
    assert len(ws.hull_xy) == 4 and (0.5, 0.5) not in ws.hull_xy
    assert Workspace.from_dict(ws.to_dict()) == ws
    assert hull_xy(np.array([[0, 0], [1, 1]])).shape == (2, 2)
    assert hull_xy(np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]])).shape == (2, 2)  # collinear


# -- projection ------------------------------------------------------------------


def test_project_quad_clips_corners_behind_the_camera():
    cam = Camera.look_at([0.0, 0.0, 0.5], [1.0, 0.0, 0.0], 60.0, 320, 240)  # looking along +x, slightly down
    quad = np.array([[-1, -1, 0], [3, -1, 0], [3, 1, 0], [-1, 1, 0.0]])  # runs under the camera
    poly = project_quad(quad, cam)
    assert len(poly) >= 4  # clipped against the near plane, not dropped
    assert np.isfinite(poly).all()
    mask = polygon_mask(poly, 320, 240)
    assert mask[200:, :].mean() > 0.9  # the ground fills the bottom of the frame
    assert project_quad(np.array([[-3, -1, 0], [-2, -1, 0], [-2, 1, 0], [-3, 1, 0.0]]), cam).shape == (0, 2)


def test_table_quad_sits_on_the_plane():
    t = TablePlane(-0.02, (0, 0, 1), 0, True, 1, 1, -0.005, 0.015, {}, {}, {}, -0.02)
    ws = Workspace((0.0, -0.2, 0.0), (0.3, 0.2, 0.1), (), 1)
    q = table_quad(t, ws, 0.1)
    assert q.shape == (4, 3) and np.allclose(q[:, 2], -0.02)
    assert q[:, 0].min() == pytest.approx(-0.1) and q[:, 1].max() == pytest.approx(0.3)


# -- backdrops -------------------------------------------------------------------


def _cell_image(W: int = 320, H: int = 240) -> np.ndarray:
    """A bright table filling the bottom two thirds, a grey wall above, a dark box on the table."""
    img = np.full((H, W, 3), 150, np.uint8)
    img[H // 3 :] = 235
    img[H // 2 : H // 2 + 30, 40:90] = 40
    return img


def test_table_region_is_the_dominant_colour_blob():
    mask, colour = table_region(_cell_image())
    assert mask.shape == (240, 320) and mask.dtype == np.uint8
    assert mask[200:, 200:].mean() > 0.95 and mask[:60].mean() < 0.05
    assert mask[135:145, 50:80].mean() < 0.2  # the box is a hole
    assert all(c > 220 for c in colour)


def test_check_backdrop_scores_the_projected_plane_against_the_region():
    cam = Camera.look_at([0.45, 0.0, 0.5], [0.15, 0.0, 0.0], 50.0, 320, 240)
    t = TablePlane(-0.02, (0, 0, 1), 0, True, 1, 1, -0.005, 0.015, {}, {}, {}, -0.02)
    ws = Workspace((0.0, -0.2, -0.02), (0.3, 0.2, 0.1), (), 1)
    bd = check_backdrop("up", cam, _cell_image(), t, ws, "scene/x_up_backdrop.jpg")
    assert bd.width == 320 and bd.height == 240 and len(bd.plane_polygon) >= 4
    assert 0.0 < bd.plane_frac < 1.0 and bd.plane_in_region > 0.5
    assert 0.0 <= bd.iou <= 1.0 and bd.region_in_plane <= 1.0
    hw, hh = backdrop_quad(cam, 3.0)
    assert bd.half_height == pytest.approx(hh) and hw / hh == pytest.approx(320 / 240)
    assert hh == pytest.approx(3.0 * math.tan(math.radians(25.0)))
    assert Backdrop.from_dict(json.loads(json.dumps(bd.to_dict()))) == bd


# -- the fit -------------------------------------------------------------------


def test_fit_scene_on_a_synthetic_pick_place(arm):
    ds = pick_place_dataset()
    cam = Camera.look_at([0.45, 0.0, 0.5], [0.15, 0.0, 0.0], 50.0, 320, 240)
    fit = fit_scene(ds, arm, "unit", {"up": cam}, {"up": _cell_image()})
    assert fit.lock == "unit" and fit.dataset == "fake/ds" and fit.robot == "so101"
    assert len(fit.grasp_points) == 3 and len(fit.release_points) == 3
    assert fit.table.n_grasps == 3 and fit.table.horizontal
    assert fit.table.z == pytest.approx(fit.table.z_grasp - OBJECT_HALF_HEIGHT)
    assert fit.table.min_fingertip_z <= fit.table.z_grasp
    assert fit.table.jaw_low_z["n"] == 3 and fit.table.jaw_low_z["median"] < fit.table.z_grasp
    assert fit.workspace.n_frames == fit.n_frames == 180 and fit.n_episodes == 3
    assert set(fit.backdrops) == {"up"} and fit.backdrops["up"].path == "scene/unit_up_backdrop.jpg"
    back = SceneFit.from_dict(json.loads(json.dumps(fit.to_dict())))
    assert back == fit
    entry = fit.lock_entry()
    assert entry["table_z"] == fit.table.z and entry["n_grasps"] == 3 and entry["backdrops"] == {"up": "scene/unit_up_backdrop.jpg"}


def test_fit_scene_without_cameras_has_no_backdrops(arm):
    fit = fit_scene(pick_place_dataset(2), arm, "unit")
    assert fit.backdrops == {} and fit.table.n_grasps == 2


def test_save_scene_fit_writes_json_backdrops_and_lock_field(arm, tmp_path):
    from antares.lock import RobotLock
    from tests.test_lock import fake_result

    lock = RobotLock.create("fake/ds", "so101", [], fake_result(), name="unit")
    lock.save(tmp_path / "unit.lock.json")
    cam = Camera.look_at([0.45, 0.0, 0.5], [0.15, 0.0, 0.0], 50.0, 320, 240)
    fit = fit_scene(pick_place_dataset(), arm, "unit", {"up": cam}, {"up": _cell_image()})
    path = save_scene_fit(fit, "unit", tmp_path, {"up": _cell_image()})
    assert path == tmp_path / "unit.scene.json"
    assert (tmp_path / "scene" / "unit_up_backdrop.jpg").is_file()
    assert load_scene_fit("unit", tmp_path) == fit
    assert load_scene_fit("nope", tmp_path) is None
    saved = RobotLock.load(tmp_path / "unit.lock.json")
    assert saved.scene is not None and saved.scene["table_z"] == fit.table.z
    assert saved.cameras == {} and RobotLock.from_dict(saved.to_dict()) == saved
    # old locks without the field load with None and do not write it
    assert RobotLock.from_dict({k: v for k, v in saved.to_dict().items() if k != "scene"}).scene is None
    assert "scene" not in lock.to_dict()


# -- sim view --------------------------------------------------------------------


def test_sim_view_paints_the_arm_over_the_backdrop(arm):
    hulls = arm_hulls(arm.model)
    cam = Camera.look_at([0.45, 0.0, 0.5], [0.0, 0.0, 0.1], 50.0, 320, 240)
    poses = arm.fk(np.zeros(6))
    bg = _cell_image()
    poly = project_quad(np.array([[-0.1, -0.3, -0.02], [0.4, -0.3, -0.02], [0.4, 0.3, -0.02], [-0.1, 0.3, -0.02]]), cam)
    img, drawn = sim_view(bg, cam, hulls, poses, None, dark_arm=True)
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
    assert len(drawn) > 10
    sil = rasterize(drawn, 320, 240).astype(bool)
    assert sil.any()
    # the arm's pixels are dark; away from the arm (and its 1 px anti-aliased
    # outline) the image is the untouched backdrop
    import cv2

    near = cv2.dilate(sil.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert img[sil].mean() < 130 and np.array_equal(img[~near], bg[~near])
    # the table polygon is a faint tint plus an outline, never opaque
    tinted, _ = sim_view(bg, cam, hulls, poses, poly, dark_arm=True)
    inside = polygon_mask(poly, 320, 240).astype(bool) & ~near
    assert 0 < np.abs(tinted[inside].astype(int) - bg[inside].astype(int)).mean() < 40
    # the backdrop is resized to the camera when sizes differ
    img2, _ = sim_view(np.full((60, 80, 3), 200, np.uint8), cam, hulls, poses, None, dark_arm=False)
    assert img2.shape == (240, 320, 3) and img2[sil].mean() > 120
    # a camera looking away sees only the backdrop
    away = Camera.look_at([0.45, 0.0, 0.5], [5.0, 0.0, 5.0], 50.0, 320, 240)
    img3, drawn3 = sim_view(bg, away, hulls, poses, None)
    assert drawn3 == [] and np.array_equal(img3, bg)
