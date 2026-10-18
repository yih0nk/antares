"""Does a data-derived DR range beat a hand-picked one?

Scores the lock's held-out episodes under (a) the lock's own parameters,
(b) parameter sets sampled from its data-derived ranges
(``locks/<lock>.dr.json``), (c) +-30% around the lock, and (c') +-30% with
latency held at the lock's value, and prints the README table.

    .venv/bin/python scripts/dr_compare.py svla_so101_pickplace
"""

from __future__ import annotations

import argparse

import numpy as np

from antares.data import load_dataset
from antares.dr import DR_SUFFIX, DRRanges, ParamRange, sample, uniform_ranges
from antares.gap import tracking_error
from antares.lock import LOCK_DIR, LOCK_SUFFIX, RobotLock
from antares.sim import Arm, SimParams


def score(arm: Arm, params: SimParams, episodes) -> float:
    arm.apply(params)
    return tracking_error([arm.real_q(e) for e in episodes], [arm.replay(e) for e in episodes]).overall_deg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lock")
    ap.add_argument("--n", type=int, default=64, help="parameter sets sampled per range")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lock = RobotLock.load(LOCK_DIR / f"{args.lock}{LOCK_SUFFIX}")
    derived = DRRanges.load(LOCK_DIR / f"{args.lock}{DR_SUFFIX}")
    ds = load_dataset(lock.dataset)
    by_index = {e.index: e for e in ds.episodes}
    episodes = [by_index[i] for i in lock.result.eval_episodes]
    arm = Arm()

    pm30 = uniform_ranges(lock.params, 0.3)
    fixed = dict(pm30.ranges)
    lat = int(lock.params.latency_frames)
    fixed["latency_frames"] = ParamRange(nominal=lat, lo=lat, hi=lat, scale="int")
    pm30_fixed = DRRanges(name="pm30_fixed_latency", nominal=pm30.nominal, ranges=fixed)

    print(f"{lock.name}: {len(episodes)} held-out episodes, {args.n} samples per range")
    print(f"MJCF defaults: {score(arm, SimParams.default(), episodes):.2f} deg")
    print("| parameters | mean | median | min .. max |")
    print("|---|---|---|---|")
    print(f"| (a) lock nominal | {score(arm, lock.params, episodes):.2f} | | |")
    for label, ranges in (
        ("(b) data-derived ranges", derived),
        ("(c) hand-picked +-30%", pm30),
        ("(c') +-30%, latency fixed at nominal", pm30_fixed),
    ):
        rng = np.random.default_rng(args.seed)
        s = np.array([score(arm, p, episodes) for p in sample(ranges, rng, args.n)])
        print(f"| {label} | {s.mean():.2f} | {np.median(s):.2f} | {s.min():.2f} .. {s.max():.2f} |")


if __name__ == "__main__":
    main()
