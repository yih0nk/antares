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


def test_latency_zero_lock_is_evaluated_on_the_plant_it_runs_on():
    """A latency-0 lock behaves like latency 1 in the closed loop; the ILC
    ceiling and the own-latency shift are built for that same plant."""
    ref = _ref()
    zero = pt.ArmSpec("z", pt.SimParams(kp=90.0, kv=1.5, latency_frames=0))
    one = pt.ArmSpec("o", pt.SimParams(kp=90.0, kv=1.5, latency_frames=1))
    plant = pt.Plant()
    plant.set(zero)
    q0 = plant.track(pt.raw_controller, ref)
    plant.set(one)
    q1 = plant.track(pt.raw_controller, ref)
    np.testing.assert_allclose(q0, q1, atol=1e-12)
    assert list(pt.effective_latency(zero.params)) == [1] * 6
    ilc_row = pt._eval_one(("ilc_own_model", zero, ref, {"type": "ilc"}))
    raw_row = pt._eval_one(("raw", zero, ref, {"type": "raw"}))
    assert ilc_row["rms_deg"] < 0.2 * raw_row["rms_deg"]  # a real ceiling, not 60x worse than in rollout
    assert ilc_row["unstable"] is False


def test_deadband_lock_runs_the_same_physics_as_rollout():
    ref = _ref()
    spec = pt.ArmSpec("d", pt.SimParams(kp=90.0, kv=1.5, latency_frames=2, deadband=0.01))
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
    cached = json.loads((out / "train_lock_s0.json").read_text())
    assert cached["settings"]["generations"] == 2 and cached["settings"]["batch_size"] == 2

    # a cached training is reused only with the same settings: a different
    # generation count retrains instead of relabelling the old controller
    rc = pt.main(
        ["--refs", str(refs), "--out", str(out), "--conditions", "lock", "--n-test", "2",
         "--generations", "3", "--batch", "2", "--train-seeds", "1", "--workers", "1"]
    )
    assert rc == 0
    assert json.loads((out / "train_lock_s0.json").read_text())["settings"]["generations"] == 3


def test_refs_convert_with_each_arms_own_units(tmp_path):
    """The fleet's SO-100 demo lock is on so101_deg: its logged degrees become
    radians as degrees, every other arm through the SO-101 default map."""
    from antares.sim import Arm

    state = np.tile(np.linspace(-30.0, 30.0, 60)[:, None], (1, 6))
    meta = [{"lock": lock, "fps": 30.0, "episode_indices": [0]}
            for lock in ("lerobot__svla_so100_pickplace", "lerobot__svla_so101_pickplace")]
    arrays = {f"{m['lock']}/0/state": state for m in meta}
    arrays["meta"] = np.array([json.dumps(m) for m in meta])
    np.savez_compressed(tmp_path / "refs.npz", **arrays)
    refs = pt.load_refs(tmp_path / "refs.npz", max_len=1000)
    deg = refs["lerobot__svla_so100_pickplace"][0].q
    norm = refs["lerobot__svla_so101_pickplace"][0].q
    np.testing.assert_allclose(deg, Arm(robot="so101_deg").unit_map.to_rad(state))
    np.testing.assert_allclose(norm, Arm().unit_map.to_rad(state))
    assert not np.allclose(deg[:, :5], norm[:, :5])
