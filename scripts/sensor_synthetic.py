"""Synthetic check of the sensor-calibration fit on the real MuJoCo arm.

A hidden "real" arm uses the committed SO-101 lock's dynamics plus a known
command-map mismatch (a few dataset units of offset, a few percent of
scale). Its replays, with measurement noise, become the logged state. Then
the dynamics-only fit and the joint fit run on the same train/eval split,
and the held-out errors and the recovered map are printed.

    .venv/bin/python scripts/sensor_synthetic.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from antares.calibrate import split_episodes
from antares.data import Episode
from antares.lock import RobotLock
from antares.sensor import calibrate_with_sensor
from antares.sim import JOINTS, Arm, CommandMap

TRUE_MAP = CommandMap(
    gain=[1.03, 0.97, 1.04, 0.98, 1.02, 1.0],
    bias=[1.5, -3.0, 2.5, -2.0, 1.0, 0.0],
)


def make_episodes(arm: Arm, true_map: CommandMap, n: int, T: int, fps: float, noise: float, seed: int) -> list[Episode]:
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    eps = []
    for i in range(n):
        action = np.zeros((T, 6))
        for j in range(6):
            for _ in range(3):
                f = rng.uniform(0.1, 0.7)
                action[:, j] += rng.uniform(-20, 20) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        action[:, 5] = 30 + np.abs(action[:, 5])
        # teleop pauses: hold the command still now and then, as operators do
        for start in range(40, T - 30, 90):
            action[start : start + 30] = action[start]
        q0 = arm.unit_map.to_rad(true_map.apply(action[0]))
        seed_ep = Episode(i, fps, JOINTS, arm.unit_map.to_units(np.tile(q0, (T, 1))), action)
        sim_q = arm.replay(seed_ep)
        state = arm.unit_map.to_units(sim_q) + rng.normal(0, noise, size=(T, 6))
        eps.append(Episode(i, fps, JOINTS, state, action))
    return eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=14)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--max-evaluations", type=int, default=150)
    ap.add_argument("--noise", type=float, default=0.2, help="state noise, dataset units")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mismatch", action="store_true", help="control: the hidden arm has an identity map")
    ap.add_argument("--mode", choices=("sim", "kinematic"), default="sim")
    args = ap.parse_args()
    true_map = CommandMap.identity() if args.no_mismatch else TRUE_MAP

    truth = RobotLock.load("locks/svla_so101_pickplace.lock.json").params
    hidden = Arm(params=truth, command_map=true_map)
    eps = make_episodes(hidden, true_map, args.episodes, args.frames, 30.0, args.noise, args.seed)
    train, eval_ = split_episodes(eps, args.episodes - 4, 4, seed=args.seed)

    t0 = time.perf_counter()
    out = calibrate_with_sensor(train, eval_, max_evaluations=args.max_evaluations, seed=args.seed,
                                mode=args.mode)
    only, final = out.dynamics_only, out.result
    print(f"mode {args.mode}, {'no mismatch' if args.no_mismatch else 'hidden mismatch'}: "
          f"{len(train)} train / {len(eval_)} eval episodes, {time.perf_counter() - t0:.0f}s")
    print(f"held-out RMSE: default {final.baseline_eval_score:.3f}  dynamics only {only.eval_score:.3f}  "
          f"with sensor {final.eval_score:.3f} deg")
    print("true params  ", truth.to_dict())
    print("dyn-only fit ", only.params.to_dict())
    print("sensor fit   ", final.params.to_dict())
    print(f"{'joint':<14}{'true bias':>10}{'fit bias':>10}{'true gain':>11}{'fit gain':>10}")
    for j, name in enumerate(JOINTS):
        print(f"{name:<14}{true_map.bias[j]:>10.2f}{out.command_map.bias[j]:>10.2f}"
              f"{true_map.gain[j]:>11.3f}{out.command_map.gain[j]:>10.3f}")


if __name__ == "__main__":
    main()
