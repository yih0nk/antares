"""The vision object in the export and the task distribution, on a hand-built fit."""

from __future__ import annotations

import json

import numpy as np
import pytest

from antares.objects import ColourPrior, ObjectFit, ObjectTrack, save_object_fit

pytest.importorskip("cv2")

T = 60


PRIOR = ColourPrior(lab=(54.0, 39.0, 21.0), spread=(14.0, 7.0, 7.0), area_px=680.0, rgb=(200, 102, 97), n_samples=3)


def _track(episode: int, start=(0.24, 0.05, -0.007), end=(0.03, -0.15, -0.007)) -> ObjectTrack:
    xyz = np.tile(np.asarray(start, dtype=np.float64), (T, 1))
    source = np.zeros(T, dtype=np.int8)
    source[20:41] = 1  # held
    source[41:] = 0
    xyz[41:] = end
    xyz[20:41] = [0.22, 0.04, 0.05]
    visible = source == 0
    centroid = np.full((T, 2), np.nan)
    centroid[visible] = [320.0, 400.0]
    bbox = np.full((T, 4), np.nan)
    bbox[visible] = [308, 380, 24, 40]
    return ObjectTrack(
        episode=episode, camera="up", frames=np.arange(T), xyz=xyz, visible=visible, source=source,
        centroid=centroid, bbox=bbox, area=np.where(visible, 680, 0), picks=[(20, 40)],
        start_pose=np.asarray(start), end_pose=np.asarray(end), end_source="vision", footprint=(0.038, 0.018),
        fk_grasp=np.array([0.23, 0.045, -0.008]), fk_release=np.array([0.03, -0.15, 0.07]),
        vision_vs_fk=0.012, end_vs_fk=0.077, n_visible=int(visible.sum()),
    )


def _fit(payload: dict | None = None) -> tuple[ObjectFit, list[ObjectTrack]]:
    tracks = [_track(i, start=(0.24 - 0.02 * i, 0.05, -0.007)) for i in range(3)]
    pts = [t.start_pose for t in tracks]
    fit = ObjectFit(
        lock="unit", dataset="fake/objects", camera="up", robot="so101", table_z=-0.022, object_half_height=0.015,
        prior=PRIOR, description="a red object, about 3.8 x 1.8 cm on a light table",
        tracks=[t.summary() for t in tracks],
        start={"n": 3, "mean": np.mean(pts, axis=0).tolist(), "std": np.std(pts, axis=0).tolist(), "min": np.min(pts, axis=0).tolist(), "max": np.max(pts, axis=0).tolist(), "points": np.asarray(pts).tolist()},
        end={"n": 3, "mean": [0.03, -0.15, -0.007], "std": [0.0, 0.0, 0.0], "min": [0.03, -0.15, -0.007], "max": [0.03, -0.15, -0.007], "points": [[0.03, -0.15, -0.007]] * 3},
        footprint_m={"long": {"n": 3, "median": 0.038, "q1": 0.037, "q3": 0.039, "iqr": 0.002, "max": 0.04}, "short": {"n": 3, "median": 0.018, "q1": 0.017, "q3": 0.019, "iqr": 0.002, "max": 0.02}},
        vision_vs_fk_m={"n": 3, "median": 0.012, "q1": 0.01, "q3": 0.015, "iqr": 0.005, "max": 0.02},
        end_vs_fk_m={"n": 3, "median": 0.077, "q1": 0.07, "q3": 0.08, "iqr": 0.01, "max": 0.08},
        vision_vs_fk_offset_m=[0.01, 0.005, 0.0], vision_vs_fk_centred_m={"n": 3, "median": 0.004, "q1": 0.003, "q3": 0.005, "iqr": 0.002, "max": 0.006},
        payload=payload, n_episodes=3, n_tracked=3, elapsed_s=1.0, created_at="2026-10-17T00:00:00Z",
    )
    return fit, tracks


PAYLOAD_OK = {
    "kp": {"shoulder_lift": 12.0, "elbow_flex": 12.0}, "episodes": [], "mass_kg": 0.04, "mass_iqr_kg": 0.01,
    "mass": {"n": 3, "median": 0.04, "q1": 0.035, "q3": 0.045, "iqr": 0.01, "std": 0.005},
    "mass_by_joint": {"shoulder_lift": {"n": 3, "median": 0.041, "q1": 0.03, "q3": 0.05, "iqr": 0.02, "std": 0.01}, "elbow_flex": {"n": 3, "median": 0.039, "q1": 0.03, "q3": 0.05, "iqr": 0.02, "std": 0.01}},
    "payload_deg": {"shoulder_lift": {"n": 3, "median": 0.5, "q1": 0.4, "q3": 0.6, "iqr": 0.2, "std": 0.1}, "elbow_flex": {"n": 3, "median": 0.4, "q1": 0.3, "q3": 0.5, "iqr": 0.2, "std": 0.1}},
    "motion_deg": {"shoulder_lift": {"n": 3, "median": 0.1, "q1": 0.0, "q3": 0.2, "iqr": 0.2, "std": 0.1}, "elbow_flex": {"n": 3, "median": 0.1, "q1": 0.0, "q3": 0.2, "iqr": 0.2, "std": 0.1}},
    "mass_per_step_kg": {"shoulder_lift": 0.013, "elbow_flex": 0.012}, "mass_noise_floor_kg": {"shoulder_lift": 0.01, "elbow_flex": 0.01},
    "lever_m": {"shoulder_lift": 0.13, "elbow_flex": 0.14}, "n_ok": 3, "n_episodes": 3, "verdict": "consistent: median 40 g, IQR 10 g over 3 picks", "assumptions": ["quasi-static"],
}


def test_export_scene_uses_the_vision_object(tmp_path):
    import mujoco

    from antares.export import OBJECTS_FILENAME, export_scene
    from antares.task import OBJECT_BODY, PLACE_SITE, TaskDistribution
    from tests.test_export import make_lock
    from tests.test_task import pick_place_episode

    fit, tracks = _fit(PAYLOAD_OK)
    lock = make_lock()
    scene = export_scene(lock, tmp_path / "out", objects_fit=fit)
    m = mujoco.MjModel.from_xml_path(str(scene))
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, PLACE_SITE)
    assert body >= 0 and site >= 0 and m.nq == 6 + 7
    geom = int(np.flatnonzero(m.geom_bodyid == body)[0])
    np.testing.assert_allclose(m.geom_size[geom], [0.019, 0.009, 0.015], atol=1e-6)
    assert m.body_mass[body] == pytest.approx(0.04, abs=1e-6)
    np.testing.assert_allclose(m.body_pos[body][:2], fit.start["mean"][:2], atol=1e-4)
    assert m.body_pos[body][2] == pytest.approx(fit.table_z + 0.015, abs=1e-4)
    np.testing.assert_allclose(m.site_pos[site], fit.end["mean"], atol=1e-4)
    side = json.loads((tmp_path / "out" / OBJECTS_FILENAME).read_text())
    assert side["mass_kg"] == 0.04 and side["start"]["n"] == 3 and side["footprint_m"] == [0.038, 0.018]
    assert "Object from the video" in scene.read_text()

    # below-the-noise payload: the assumed mass is used and said so
    fit2, _ = _fit({**PAYLOAD_OK, "verdict": "below the noise: ...", "mass_kg": 5.8})
    assert fit2.mass_kg is None and fit2.measured_mass_kg == 5.8
    scene2 = export_scene(lock, tmp_path / "out2", objects_fit=fit2)
    m2 = mujoco.MjModel.from_xml_path(str(scene2))
    assert m2.body_mass[mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)] == pytest.approx(0.03, abs=1e-6)
    assert "assumed (payload estimate below the noise" in scene2.read_text()

    # objects fit wins over the task cube; the task is saved with vision picks
    td = TaskDistribution.from_episodes([pick_place_episode(0), pick_place_episode(1)], name="toy")
    scene3 = export_scene(lock, tmp_path / "out3", task=td, objects_fit=fit)
    m3 = mujoco.MjModel.from_xml_path(str(scene3))
    g3 = int(np.flatnonzero(m3.geom_bodyid == mujoco.mj_name2id(m3, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY))[0])
    np.testing.assert_allclose(m3.geom_size[g3], [0.019, 0.009, 0.015], atol=1e-6)
    saved = json.loads((tmp_path / "out3" / "task.json").read_text())
    assert saved["source"] == "vision"
    assert saved["grasps"][0]["xyz"] == [round(v, 4) for v in tracks[0].start_pose] or saved["grasps"][0]["xyz"] == list(tracks[0].start_pose)

    # the saved fit is picked up from lock_dir
    save_object_fit(fit, lock.name, tmp_path / "locks", tracks)
    scene4 = export_scene(lock, tmp_path / "out4", lock_dir=tmp_path / "locks")
    assert (tmp_path / "out4" / OBJECTS_FILENAME).is_file() and "Object from the video" in scene4.read_text()


def test_task_with_vision_picks():
    from antares.task import TaskDistribution
    from tests.test_task import pick_place_episode

    td = TaskDistribution.from_episodes([pick_place_episode(0), pick_place_episode(1)], name="toy")
    assert td.source == "fk"
    v = td.with_vision_picks({0: (0.3, 0.1, 0.0)})
    assert v.source == "vision" and v.picks[0].tolist() == [0.3, 0.1, 0.0]
    assert v.picks[1].tolist() == td.picks[1].tolist()  # no vision pose: FK kept
    assert TaskDistribution.from_dict(v.to_dict()).source == "vision"
    assert TaskDistribution.from_dict({**td.to_dict(), "source": None} | {"source": "fk"}).source == "fk"
