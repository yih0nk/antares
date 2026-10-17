"""Domain randomization from data, in a training loop.

Each step samples simulator parameters from the ranges Antares derived for
one robot (``antares dr``), applies them to the arm, and replays a logged
episode so you can see how far the randomized sim lands from the real one.
Swap the replay for your env step and keep the first two lines of the loop.

    .venv/bin/python examples/randomize_env.py
"""

import numpy as np

from antares.data import load_dataset
from antares.dr import DRRanges, apply, sample
from antares.gap import tracking_error
from antares.sim import Arm

ranges = DRRanges.load("locks/svla_so101_pickplace.dr.json")
ds = load_dataset(ranges.dataset)
arm = Arm()
rng = np.random.default_rng(0)

for step, params in enumerate(sample(ranges, rng, 10)):
    apply(arm, params)  # a different plausible sim every step
    ep = ds.episodes[rng.integers(len(ds.episodes))]
    gap = tracking_error(arm.real_q(ep), arm.replay(ep))
    print(f"step {step}: kp={params.kp:6.1f} kv={params.kv:.2f} latency={params.latency_frames}  "
          f"episode {ep.index:2d}  gap {gap.overall_deg:.2f} deg")
