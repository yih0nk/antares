"""Fleet transfer experiment: does a sim built from real logs train a
controller that works better on arms it has never seen?

The controller is the low-level layer under any LeRobot policy: given the
next few frames of a joint trajectory (what a policy's action chunk says
the arm should do), choose the command to send this frame so the arm
actually follows it. It is trained in one kind of simulator and tested on
held-out fleet arms (their locks stand in for the real robots).

Everything is sim-to-sim. The test arms are built from the same model the
training arms are, so the numbers are optimistic about the real world; the
``payload`` variant adds a mass the locks do not model to show what happens
outside the model class.

Usage (from the repo root)::

    python experiments/policy_transfer.py --refs refs.npz --out results/

``refs.npz`` holds measured joint trajectories per fleet dataset (dataset
units), written by ``dump_refs.py`` on a machine that can reach the Hub.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cma
import mujoco
import numpy as np

from antares.closedloop import FrameStepper
from antares.compensate import ilc
from antares.dr import DRRanges, sample, uniform_ranges
from antares.sim import GRIPPER_INDEX, JOINTS, Arm, SimParams

ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / "locks" / "fleet" / "fleet.json"
DR_PATH = ROOT / "locks" / "svla_so101_pickplace.dr.json"
SOURCE_LOCK = "lerobot__svla_so101_pickplace"  # the single-arm sims (C, D)

SCORED = [i for i in range(len(JOINTS)) if i != GRIPPER_INDEX]
H = 8  # reference frames ahead the controller sees (q_ref[t..t+H])
K = 5  # past commands it sees (covers the fleet's latency range 1..5)
N_FEAT = (H + 1) + 1 + K  # tracking errors, velocity, past commands
PAYLOAD_KG = 0.15  # unmodelled mass at the gripper for the payload test


# --------------------------------------------------------------------------
# Arms and references
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    name: str
    params: SimParams
    payload_kg: float = 0.0


def fleet_arms() -> list[ArmSpec]:
    robots = json.loads(FLEET.read_text())["robots"]
    return [ArmSpec(r["lock"], SimParams.from_dict(r["params"])) for r in robots]


def split_arms(arms: list[ArmSpec], n_test: int, seed: int) -> tuple[list[ArmSpec], list[ArmSpec]]:
    """Hold out ``n_test`` arms. The source arm of the single-arm sims stays in training.

    The fleet scan keeps one dataset per Hub account, so no account (and
    likely no physical arm) lands on both sides.
    """
    rng = np.random.default_rng(seed)
    pool = [a for a in arms if a.name != SOURCE_LOCK]
    test_idx = set(rng.choice(len(pool), size=n_test, replace=False).tolist())
    test = [a for i, a in enumerate(pool) if i in test_idx]
    train = [a for a in arms if a not in test]
    return train, test


@dataclass(frozen=True)
class Ref:
    arm: str  # lock name of the dataset the motion came from
    episode: int
    fps: float
    q: np.ndarray  # (T, 6) measured joint positions, radians


def load_refs(path: Path, max_len: int) -> dict[str, list[Ref]]:
    """Measured trajectories per arm, converted with the default unit map (fleet locks carry none)."""
    z = np.load(path, allow_pickle=False)
    raw = z["meta"]
    if raw.ndim == 0:  # one JSON string holding the whole list, or a single entry
        loaded = json.loads(str(raw))
        meta = loaded if isinstance(loaded, list) else [loaded]
    else:
        meta = [json.loads(str(m)) for m in raw]
    unit_map = Arm().unit_map
    out: dict[str, list[Ref]] = {}
    for m in meta:
        refs = []
        for ep in m["episode_indices"]:
            state = z[f"{m['lock']}/{ep}/state"].astype(np.float64)
            if len(state) < 2 * (H + K) + 10:
                continue
            refs.append(Ref(m["lock"], int(ep), float(m["fps"]), unit_map.to_rad(state)[:max_len]))
        if refs:
            out[m["lock"]] = refs
    return out


# --------------------------------------------------------------------------
# Closed-loop rollout
# --------------------------------------------------------------------------


class Plant:
    """One arm stepped frame by frame, with the lock's command latency."""

    def __init__(self) -> None:
        self.arm = Arm()
        m = self.arm.model
        self._gripper = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._mass0 = float(m.body_mass[self._gripper])
        self._inertia0 = m.body_inertia[self._gripper].copy()
        self._payload = 0.0

    def set(self, spec: ArmSpec) -> None:
        m = self.arm.model
        if spec.payload_kg != self._payload:
            m.body_mass[self._gripper] = self._mass0 + spec.payload_kg
            m.body_inertia[self._gripper] = self._inertia0 * (1.0 + spec.payload_kg / self._mass0)
            self._payload = spec.payload_kg
        self.arm.apply(spec.params)  # also recomputes the mass-derived constants

    def track(self, controller, ref: Ref) -> np.ndarray:
        """Run ``controller`` against ``ref`` from its initial state; return sim q (T, 6).

        Stepping (substeps, clipping, deadband) is :class:`FrameStepper`, the
        same physics as :meth:`Arm.rollout`; only the latency buffer lives here.
        """
        arm = self.arm
        lo, hi = arm.ctrl_range
        lat = effective_latency(arm.params)
        q_ref = ref.q
        T = len(q_ref)
        # The window the controller reads runs past the end: hold the last frame.
        q_pad = np.concatenate([q_ref, np.repeat(q_ref[-1:], H, axis=0)])
        sent = np.repeat(q_ref[:1], T + K, axis=0)  # sent[K + t] = command sent at t
        q = np.empty((T, 6))
        cols = np.arange(6)
        with FrameStepper(arm, q_ref[0], ref.fps) as stepper:
            q[0] = stepper.q
            prev = q_ref[0]
            for t in range(T):
                cur = q[t]
                u = controller(t, cur, cur - prev, q_pad[t : t + H + 1], sent[t : t + K][::-1])
                sent[K + t] = np.clip(u, lo, hi)
                if t == T - 1:
                    break
                # Same convention as Arm.rollout: the step into frame t+1 applies the
                # command sent at t+1-lat. A command computed from the state at t
                # can act at the earliest on that step, so latency 0 behaves like 1.
                q[t + 1] = stepper.step(sent[K + np.maximum(t + 1 - lat, 0), cols])
                prev = cur
        return q


def effective_latency(params: SimParams) -> np.ndarray:
    """Per-joint command latency the closed loop can realize: a command computed
    from frame t acts on the step into t+1 at the earliest, so a lock's latency
    0 behaves like 1. The ILC ceiling and the own-latency shift use the same."""
    return np.maximum(params.per_joint("latency_frames").astype(int), 1)


def rms_deg(q_sim: np.ndarray, q_ref: np.ndarray) -> float:
    err = np.degrees(q_sim[:, SCORED] - q_ref[:, SCORED])
    return float(np.sqrt(np.mean(err**2)))


# --------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------


def lock_features(params: SimParams) -> np.ndarray:
    """What the conditioned controller is told about the arm: latency and response time.

    kp and kv are not identified separately by the fits, so the controller
    gets the step response's 63% rise time (in frames) instead, which is.
    """
    return np.array([float(params.per_joint("latency_frames").mean()) / 5.0, response_frames(params) / 10.0])


_RESPONSE_CACHE: dict[str, float] = {}


def response_frames(params: SimParams, fps: float = 30.0) -> float:
    """Frames for the elbow to cover 63% of a 10 degree step, latency excluded."""
    key = json.dumps(params.to_dict(), sort_keys=True)
    if key in _RESPONSE_CACHE:
        return _RESPONSE_CACHE[key]
    arm = Arm(params=params)
    j = JOINTS.index("elbow_flex")
    q0 = np.zeros(6)
    ctrl = np.zeros((60, 6))
    ctrl[:, j] = math.radians(10.0)
    q = arm.rollout(ctrl, q0, fps, latency_frames=0)
    frac = (q[:, j] - q[0, j]) / math.radians(10.0)
    hit = np.flatnonzero(frac >= 0.63)
    out = float(hit[0]) if hit.size else 60.0
    _RESPONSE_CACHE[key] = out
    return out


class Linear:
    """``u = q + sum_f w_f * feature_f + b_joint``, the same weights on every joint.

    Features, per joint, relative to the current position q: the tracking
    errors ``q_ref[t+k] - q`` for k = 0..H, the last frame's velocity and
    the last K commands sent. With ``n_cond`` > 0 the weights are an affine
    function of the arm's lock features (one policy for the whole fleet).
    """

    def __init__(self, theta: np.ndarray, n_cond: int = 0, cond: np.ndarray | None = None):
        n_w = N_FEAT * (1 + n_cond)
        w = theta[:n_w].reshape(1 + n_cond, N_FEAT)
        z = np.concatenate([[1.0], cond if cond is not None else np.zeros(n_cond)])
        self.w = z @ w
        self.b = theta[n_w : n_w + 6]

    @staticmethod
    def size(n_cond: int = 0) -> int:
        return N_FEAT * (1 + n_cond) + 6

    @staticmethod
    def init(n_cond: int = 0) -> np.ndarray:
        """Start at "send the reference": weight 1 on the frame-0 error."""
        theta = np.zeros(Linear.size(n_cond))
        theta[0] = 1.0
        return theta

    def __call__(self, t, q, v, window, past):
        feats = np.concatenate([window - q, v[None], past - q])  # (N_FEAT, 6)
        return q + self.w @ feats + self.b


def raw_controller(t, q, v, window, past):
    return window[0]


def shift_controller(lat: int):
    lat = min(int(lat), H)
    return lambda t, q, v, window, past: window[lat]


# --------------------------------------------------------------------------
# Training sims
# --------------------------------------------------------------------------


def make_sampler(kind: str, train: list[ArmSpec]):
    """Return ``draw(rng) -> ArmSpec`` for one training condition.

    ``lock`` and ``lock_dr`` share one source arm, the nominal of the
    committed ``antares dr`` ranges (the SO-101 lock those ranges were
    derived around), so the only difference between the two conditions is
    the randomization.
    """
    if kind == "default":
        return lambda rng: ArmSpec("default", SimParams())
    if kind == "default_dr30":
        ranges = uniform_ranges(SimParams(), 0.3, name="default+-30%")
        return lambda rng: ArmSpec("default_dr30", sample(ranges, rng, 1)[0])
    if kind in ("lock", "lock_dr"):
        if not DR_PATH.is_file():
            raise FileNotFoundError(f"{DR_PATH} is missing: run `antares dr` on {SOURCE_LOCK} first")
        ranges = DRRanges.load(DR_PATH)
        if kind == "lock":
            source = ArmSpec(SOURCE_LOCK, ranges.nominal)
            return lambda rng: source
        return lambda rng: ArmSpec("lock_dr", sample(ranges, rng, 1)[0])
    if kind in ("fleet", "fleet_cond"):
        return lambda rng: train[int(rng.integers(len(train)))]
    raise ValueError(kind)


CONDITIONS = ("default", "default_dr30", "lock", "lock_dr", "fleet", "fleet_cond")


# --------------------------------------------------------------------------
# Training: CMA-ES on closed-loop tracking error
# --------------------------------------------------------------------------

_PLANT: Plant | None = None


def _plant() -> Plant:
    global _PLANT
    if _PLANT is None:
        _PLANT = Plant()
    return _PLANT


def _score(args) -> float:
    theta, batch, n_cond = args
    plant = _plant()
    errs = []
    for spec, ref, cond in batch:
        plant.set(spec)
        q = plant.track(Linear(theta, n_cond, cond), ref)
        e = rms_deg(q, ref.q)
        errs.append(e if math.isfinite(e) else 1e3)
    return float(np.mean(errs))


def crop(ref: Ref, rng: np.random.Generator, length: int) -> Ref:
    if len(ref.q) <= length:
        return ref
    s = int(rng.integers(len(ref.q) - length))
    return Ref(ref.arm, ref.episode, ref.fps, ref.q[s : s + length])


def train(
    kind: str,
    train_arms: list[ArmSpec],
    train_refs: list[Ref],
    pool: Pool,
    generations: int,
    batch_size: int,
    seed: int,
    crop_len: int = 150,
    log=print,
) -> dict:
    n_cond = 2 if kind == "fleet_cond" else 0
    draw = make_sampler(kind, train_arms)
    rng = np.random.default_rng(seed)
    es = cma.CMAEvolutionStrategy(Linear.init(n_cond), 0.1, {"seed": seed + 1, "verbose": -9})
    history = []
    t0 = time.time()
    for g in range(generations):
        # One batch per generation shared by the whole population (common random numbers).
        batch = []
        for _ in range(batch_size):
            spec = draw(rng)
            ref = crop(train_refs[int(rng.integers(len(train_refs)))], rng, crop_len)
            batch.append((spec, ref, lock_features(spec.params) if n_cond else None))
        xs = es.ask()
        # The distribution mean the generation was drawn from is scored in the
        # same parallel pass as the population (no serial rollout while the
        # workers sit idle).
        mean_theta = np.asarray(es.mean)
        fs_all = pool.map(_score, [(np.asarray(x), batch, n_cond) for x in xs] + [(mean_theta, batch, n_cond)])
        fs, mean_f = fs_all[:-1], fs_all[-1]
        es.tell(xs, fs)
        history.append({"gen": g, "best": float(min(fs)), "mean_theta": mean_f})
        if g % 10 == 0 or g == generations - 1:
            log(f"[{kind}] gen {g:3d}  best {min(fs):.3f}  mean {mean_f:.3f}  ({time.time() - t0:.0f}s)")
    theta = np.asarray(es.mean)
    return {
        "kind": kind,
        "theta": theta.tolist(),
        "n_cond": n_cond,
        "history": history,
        "elapsed_s": time.time() - t0,
        "settings": train_settings(train_arms, train_refs, generations, batch_size, seed, crop_len),
    }


def train_settings(train_arms, train_refs, generations, batch_size, seed, crop_len) -> dict:
    """What a cached training result must match to be reused."""
    return {
        "generations": int(generations),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "crop_len": int(crop_len),
        "train_arms": sorted(a.name for a in train_arms),
        "n_train_refs": len(train_refs),
        "max_len": max(len(r.q) for r in train_refs) if train_refs else 0,
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _eval_one(args) -> dict:
    name, spec, ref, how = args
    plant = _plant()
    plant.set(spec)
    if how["type"] == "raw":
        ctrl = raw_controller
    elif how["type"] == "shift":
        ctrl = shift_controller(how["lat"] if how["lat"] is not None else int(effective_latency(spec.params)[0]))
    elif how["type"] == "linear":
        theta = np.asarray(how["theta"])
        cond = lock_features(spec.params) if how["n_cond"] else None
        ctrl = Linear(theta, how["n_cond"], cond)
    elif how["type"] == "ilc":
        # Open-loop upper bound: commands optimized in the test arm's lock (which
        # does not know about the payload), then sent to the arm as it really is.
        # The lock is given the latency the closed loop realizes (a latency-0
        # lock behaves like 1 here), so the ceiling is measured on the same plant.
        lat = effective_latency(spec.params)
        shared = int(lat[0]) if len(set(lat.tolist())) == 1 else tuple(int(v) for v in lat)
        plant.set(ArmSpec(spec.name, spec.params.replace(latency_frames=shared)))
        cmds = ilc(plant.arm, ref.q, ref.q[0], ref.fps, latency=int(lat.max())).ctrl
        plant.set(spec)
        ctrl = lambda t, q, v, window, past: cmds[t]  # noqa: E731
    else:
        raise ValueError(how)
    q = plant.track(ctrl, ref)
    rms = rms_deg(q, ref.q)
    unstable = not math.isfinite(rms)
    if unstable:  # a blown-up rollout scores the same penalty as in training, never NaN
        rms = 1e3
        per_joint = np.full(6, 1e3)
    else:
        per_joint = np.sqrt(np.mean(np.degrees(q - ref.q) ** 2, axis=0))
    return {
        "controller": name,
        "arm": spec.name,
        "payload_kg": spec.payload_kg,
        "ref": f"{ref.arm}/{ref.episode}",
        "rms_deg": rms,
        "unstable": unstable,
        "per_joint_deg": {JOINTS[i]: float(per_joint[i]) for i in SCORED},
    }


def evaluate(controllers: dict[str, dict], arms: list[ArmSpec], refs: dict[str, list[Ref]], pool: Pool, n_refs: int):
    jobs = []
    for spec in arms:
        own = refs.get(spec.name.removesuffix("+payload"), [])[:n_refs]
        for ref in own:
            for name, how in controllers.items():
                jobs.append((name, spec, ref, how))
    return pool.map(_eval_one, jobs, chunksize=4)


def summarize(rows: list[dict]) -> dict:
    """Per controller (training seeds pooled), nominal and payload test arms."""
    out: dict[str, dict] = {}
    for variant in ("nominal", "payload"):
        sel = [r for r in rows if (r["payload_kg"] > 0) == (variant == "payload")]
        by: dict[str, list] = {}
        for r in sel:
            by.setdefault(re.sub(r"_s\d+$", "", r["controller"]), []).append(r)
        out[variant] = {}
        for c, rs in by.items():
            per_arm: dict[str, list] = {}
            for r in rs:
                per_arm.setdefault(r["arm"], []).append(r["rms_deg"])
            arm_means = [float(np.mean(v)) for v in per_arm.values()]
            per_seed: dict[str, list] = {}
            for r in rs:
                per_seed.setdefault(r["controller"], []).append(r["rms_deg"])
            seed_means = [float(np.mean(v)) for v in per_seed.values()]
            out[variant][c] = {
                "mean_deg": float(np.mean([r["rms_deg"] for r in rs])),
                "median_deg": float(np.median([r["rms_deg"] for r in rs])),
                "worst_arm_deg": float(max(arm_means)),
                "elbow_deg": float(np.mean([r["per_joint_deg"]["elbow_flex"] for r in rs])),
                "seed_range_deg": [min(seed_means), max(seed_means)],
                "n": len(rs),
            }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/policy_transfer"))
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--generations", type=int, default=120)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--eval-refs", type=int, default=10, help="reference episodes per test arm")
    p.add_argument("--max-len", type=int, default=600, help="frames per reference episode")
    p.add_argument("--seed", type=int, default=0, help="arm split and first training seed")
    p.add_argument("--train-seeds", type=int, default=3, help="independent trainings per condition")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    refs = load_refs(args.refs, args.max_len)
    arms = [a for a in fleet_arms() if a.name in refs]
    train_arms, test_arms = split_arms(arms, args.n_test, args.seed)
    # Ten episodes per arm, so the two 50-episode datasets do not dominate.
    train_refs = [r for a in train_arms for r in refs[a.name][:10]]
    print(f"{len(train_arms)} training arms ({len(train_refs)} reference episodes), {len(test_arms)} test arms")
    print("test arms:", ", ".join(a.name for a in test_arms))

    trained = {}
    with Pool(args.workers) as pool:
        for kind in args.conditions:
            for s in range(args.train_seeds):
                key = f"{kind}_s{s}"
                path = args.out / f"train_{key}.json"
                wanted = train_settings(train_arms, train_refs, args.generations, args.batch, args.seed + 1000 * s, 150)
                if path.exists():
                    cached = json.loads(path.read_text())
                    if cached.get("settings") == wanted:
                        trained[key] = cached
                        print(f"[{key}] loaded {path}")
                        continue
                    print(f"[{key}] {path} was trained with other settings; retraining")
                res = train(kind, train_arms, train_refs, pool, args.generations, args.batch, args.seed + 1000 * s)
                path.write_text(json.dumps(res))
                trained[key] = res

        mean_lat = int(round(np.mean([a.params.per_joint("latency_frames")[0] for a in train_arms])))
        controllers = {
            "raw": {"type": "raw"},
            f"shift_fleet_mean_{mean_lat}": {"type": "shift", "lat": mean_lat},
            "shift_own_latency": {"type": "shift", "lat": None},
            "ilc_own_model": {"type": "ilc"},
        }
        for kind, res in trained.items():
            controllers[f"learned_{kind}"] = {"type": "linear", "theta": res["theta"], "n_cond": res["n_cond"]}

        payload = [ArmSpec(a.name + "+payload", a.params, PAYLOAD_KG) for a in test_arms]
        t0 = time.time()
        rows = evaluate(controllers, test_arms + payload, refs, pool, args.eval_refs)
        print(f"evaluated {len(rows)} rollouts in {time.time() - t0:.0f}s")

    summary = summarize(rows)
    (args.out / "rows.json").write_text(json.dumps(rows))
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "train_arms": [a.name for a in train_arms],
                "test_arms": [a.name for a in test_arms],
                "settings": vars(args) | {"refs": str(args.refs), "out": str(args.out)},
                "train_final": {k: v["history"][-1] for k, v in trained.items()},
            },
            indent=2,
        )
    )
    for variant, table in summary.items():
        print(f"\n{variant}: mean / median / worst arm / elbow (deg)")
        for c, s in sorted(table.items(), key=lambda kv: kv[1]["mean_deg"]):
            print(
                f"  {c:28s} {s['mean_deg']:6.3f} {s['median_deg']:6.3f}"
                f" {s['worst_arm_deg']:6.3f} {s['elbow_deg']:6.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
