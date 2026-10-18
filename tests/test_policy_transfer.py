import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import policy_transfer as pt  # noqa: E402


def _ref(T: int = 120, fps: float = 30.0) -> pt.Ref:
    t = np.arange(T) / fps
    q = np.stack([0.3 * np.sin(2 * np.pi * (0.2 + 0.05 * j) * t + j) for j in range(6)], axis=1)
    return pt.Ref("x", 0, fps, q)


@pytest.mark.parametrize("lat", [1, 3, 5])
def test_closed_loop_plant_matches_open_loop_rollout(lat):
    """The frame-by-frame plant applies latency exactly like Arm.rollout."""
    ref = _ref()
    spec = pt.ArmSpec("a", pt.SimParams(kp=90.0, kv=1.5, latency_frames=lat))
    plant = pt.Plant()
    plant.set(spec)
    q_loop = plant.track(pt.raw_controller, ref)
    q_open = plant.arm.rollout(ref.q, ref.q[0], ref.fps)
    np.testing.assert_allclose(q_loop, q_open, atol=1e-12)


def test_payload_is_applied_and_removed():
    plant = pt.Plant()
    base = pt.ArmSpec("a", pt.SimParams(latency_frames=2))
    m0 = float(plant.arm.model.body_mass[plant._gripper])
    plant.set(pt.ArmSpec("a+payload", base.params, 0.15))
    assert plant.arm.model.body_mass[plant._gripper] == pytest.approx(m0 + 0.15)
    plant.set(base)
    assert plant.arm.model.body_mass[plant._gripper] == pytest.approx(m0)


def test_initial_linear_controller_sends_the_reference():
    ctrl = pt.Linear(pt.Linear.init())
    window = np.random.default_rng(0).normal(size=(pt.H + 1, 6))
    q = np.zeros(6)
    np.testing.assert_allclose(ctrl(0, q, q, window, np.zeros((pt.K, 6))), window[0])


def test_end_to_end_on_synthetic_refs(tmp_path):
    robots = json.loads(pt.FLEET.read_text())["robots"][:6]
    arrays, meta = {}, []
    for i, r in enumerate(robots):
        q = _ref(80).q
        units = pt.Arm().unit_map.to_units(q)
        arrays[f"{r['lock']}/0/state"] = units.astype(np.float32)
        arrays[f"{r['lock']}/0/action"] = units.astype(np.float32)
        meta.append({"lock": r["lock"], "dataset": r["dataset"], "fps": 30.0, "joint_names": [], "episode_indices": [0]})
    arrays["meta"] = np.array([json.dumps(m) for m in meta])
    refs = tmp_path / "refs.npz"
    np.savez_compressed(refs, **arrays)

    out = tmp_path / "out"
    rc = pt.main(
        ["--refs", str(refs), "--out", str(out), "--conditions", "lock", "fleet", "--n-test", "2",
         "--generations", "2", "--batch", "2", "--train-seeds", "1", "--workers", "1"]
    )
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())["summary"]
    assert {"raw", "shift_own_latency", "ilc_own_model", "learned_lock", "learned_fleet"} <= set(s["nominal"])
    assert s["nominal"]["ilc_own_model"]["mean_deg"] < s["nominal"]["raw"]["mean_deg"]
