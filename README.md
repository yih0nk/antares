# Antares

**A simulator calibrated to *this* robot, from its own logs.**

Every physical robot is a slightly different deploy target: servo gains
drift, joints wear, calibrations differ. A policy that works in a generic
sim meets a specific arm in the real world. Antares takes a few minutes of
one robot's recorded episodes, fits the simulator to that unit, and shows
you how much of the sim-to-real gap closed.

## How it works

1. **Load real logs.** Commanded actions and measured joint positions
   from a [LeRobot](https://github.com/huggingface/lerobot) dataset (no
   video needed).
2. **Replay in sim.** Start MuJoCo from the real initial state, feed the
   same commands, and measure where the sim drifts from reality.
3. **Calibrate.** CMA-ES fits actuator gain, damping, armature, friction,
   and latency to that robot's episodes, scored on held-out ones.
4. **See it.** The web UI overlays the real arm and the sim ghost in 3D
   and streams the fit live, so you watch the ghost lock on.

First embodiment: the SO-100/101 arm, using public community datasets.

## Run it

```sh
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/antares serve            # API on :8000

cd web && npm install && npm run dev   # UI on :5173
```

Or build the UI once (`cd web && npm run build`) and `antares serve` alone
serves it on :8000.

In the viewer, the white arm is the logged robot and the blue ghost is the
simulator replaying the same commands. Pick an episode, scrub the timeline,
switch the ghost between the default sim and a saved lock, or press
**Calibrate** to fit a new lock and watch the ghost lock on as the error
drops.

Inspect a dataset from the CLI:

```sh
.venv/bin/antares inspect lerobot/svla_so101_pickplace
```

### Export

A calibration is saved as a lock (`locks/<name>.lock.json`). Export one as
a standalone MJCF to use it in any MuJoCo / RL pipeline without Antares:

```sh
.venv/bin/antares locks                               # list saved locks
.venv/bin/antares export svla_so101_pickplace         # -> export/svla_so101_pickplace/
.venv/bin/antares export svla_so101_pickplace -o out/ --no-bundle   # meshdir -> package assets
python -m mujoco.viewer --mjcf export/svla_so101_pickplace/scene.xml
```

The output directory holds:

| File | What |
|---|---|
| `<name>.xml` | the SO-101 MJCF with the fitted `kp`/`kv`, `damping`/`armature`/`frictionloss` baked into the `sts3215` class, plus a header comment with the lock's provenance and held-out scores |
| `<name>.antares.json` | sidecar: all params including `latency_frames` (MJCF cannot express command latency; delay each `ctrl` by this many dataset frames in your control loop), the dataset-unit -> radian map, scores |
| `scene.xml` | the arm plus floor and light, for the viewer |
| `assets/*.stl` | the meshes (omitted with `--no-bundle`, which points `meshdir` at the package's copy) |

The same files come from the API: `GET /api/locks/<name>/export` is the XML
as a download and `GET /api/locks/<name>/export.zip` is a zip of the whole
directory. From Python, `antares.export.export_mjcf(lock, path)` /
`export_scene(lock, dir)`; `load_exported_arm(path)` builds an `Arm` from
an export, and its replay matches `Arm(params=lock.params)` to 1e-9.

## Drift over a session

Episodes are recorded in order, so a servo that heats up or a joint that
wears during the session should show up as parameters that stop fitting.
`antares drift` slides a window over the episodes, refits the sim on each
window warm-started from a saved lock (small CMA-ES step, 60 evaluations,
a few seconds per window) and records, per window, the fitted parameters,
their training error and the error of the lock's static parameters on the
same episodes:

```sh
.venv/bin/antares drift lerobot/svla_so101_pickplace --lock svla_so101_pickplace \
    --window 10 --step 5
```

Each parameter and both scores get a least-squares slope against episode
index (log10 space for the log-scaled parameters), an r², and a 95%
bootstrap interval over windows. A quantity is flagged as drifting only if
the interval excludes zero *and* the fitted line moves more than 10% from
the first window to the last. The scan is saved next to the lock as
`locks/<name>.drift.json`; the API exposes it as `POST /api/drift` (a job
streamed over `/api/jobs/{id}/events`, one event per window) and
`GET /api/drift/{name}`.

What the two demo datasets show (window 10, step 5, nine windows each):

* **SO-100**: the lock's error grows from 0.71° to about 0.83° over the
  session (+16%, r² 0.65, significant). The refits do not recover it
  (they beat the lock by under 3% on any window) and no parameter moves
  consistently, so this is a harder tail of the session, not wear the
  six parameters can describe. Most of the growth is in `elbow_flex`
  (1.11° to 1.32°) and `shoulder_lift`.
* **SO-101**: the lock's error is flat (1.04° to 1.14°, +5%). The scan
  flags `armature` (+269%), but with r² 0.36, a 4% mean refit gain and two
  windows where the refit found nothing better than the lock itself, that
  is scatter in a weakly constrained parameter rather than drift. The
  verdict says so.

Overlapping windows are not independent, so the bootstrap interval is
optimistic; treat a flag as "worth a look", not as a measurement of wear.

## Domain randomization from data

Policies are usually trained with simulator parameters randomized inside
hand-picked ranges ("friction +-30%"). A lock lets you measure the ranges
instead: refit the sim on several random subsets of the robot's episodes,
each warm-started from the lock with a small CMA-ES step, and read how far
the fitted parameters scatter. That scatter is what the data leaves
undetermined, and so what a policy for this arm should be robust to.

```sh
.venv/bin/antares dr lerobot/svla_so101_pickplace        # ~30 s, 8 refits
```

For the SO-101 lock (8 refits x 12 episodes x 60 evaluations; `lo`/`hi`
are the 5th/95th percentiles of the fitted values in log space, widened to
contain the lock value; latency is the integer min/max):

| param | nominal | lo | hi | spread |
|---|---|---|---|---|
| kp | 93.7 | 50.1 | 108.9 | -46% .. +16% |
| kv | 1.57 | 0.62 | 5.07 | -61% .. +223% |
| damping | 0.053 | 0.017 | 0.271 | -68% .. +414% |
| armature | 0.0197 | 0.0018 | 0.031 | -91% .. +56% |
| frictionloss | 0.025 | 0.0028 | 0.108 | -89% .. +328% |
| latency_frames | 4 | 3 | 4 | -1 .. +0 frames |

The ranges are saved as `locks/<name>.dr.json` with their provenance
(dataset, lock, episode subsets, per-fit scores, seed). Use them in a
training loop with `antares.dr.sample` and `apply`; see
`examples/randomize_env.py`.

Does the derived range beat a hand-picked one? Tracking RMSE on the lock's
10 held-out episodes, averaging over 64 parameter sets sampled from each
range (MJCF defaults score 6.01 deg):

| parameters | mean | median | min .. max |
|---|---|---|---|
| (a) lock nominal | 1.05 | | |
| (b) data-derived ranges | 1.39 | 1.20 | 0.97 .. 3.29 |
| (c) hand-picked +-30% | 1.53 | 1.69 | 1.04 .. 2.10 |
| (c') +-30%, latency fixed at nominal | 1.07 | 1.07 | 1.04 .. 1.13 |

The hand-picked range is wrong in both directions at once. On latency,
+-30% around 4 frames allows 5, which the data never supports and which
alone costs about half a degree on average (c vs c'). On the continuous parameters
it is far too narrow: within +-30% the sim is nearly indistinguishable from
nominal (c'), so a policy trained there sees no diversity. The
data-derived ranges span 2-5x on `kv`, `damping` and `frictionloss`, keep
latency where the data puts it, and still average 1.39 deg, 4.3x better
than the uncalibrated sim. Some samples beat the lock itself (min 0.97),
because the lock was fit on different episodes.

Caveat worth knowing: most refits push `kp` to 50, the lower edge of the
search space, so `kp.lo` is set by the bound, not the data. The replay
cannot separate a low gain from a low velocity gain over these episodes.

API: `POST /api/dr {dataset, lock?, n_fits?, n_train?, max_evaluations?,
seed?, name?}` starts a job and returns `{job_id, status_url, events_url}`;
the job streams `progress`, `fit` and `done` events at
`/api/jobs/{id}/events` like calibration does. `GET /api/dr` lists saved
ranges, `GET /api/dr/{name}` returns one.

## Per-joint calibration

The shared fit gives all six servos the same gain, damping, armature,
friction and latency. That leaves the elbow as the worst joint on both demo
arms (SO-101 held-out: 2.03° vs under 1° for the others), which makes
sense: each STS3215 carries a different load, from the shoulder lifting the
whole arm to the wrist lifting almost nothing.

Every `SimParams` field can therefore be a scalar *or* a tuple of six
per-joint values, and `forcerange` (the actuator force limit, MJCF default
2.94 N·m) joins the parameters. Scalar locks written before this change
load and replay unchanged.

`antares calibrate --per-joint` runs a two-stage fit:

1. the shared fit as before (6 dims, `--max-evaluations`, default 150), then
2. a per-joint refinement over 42 dims (7 parameters × 6 joints), started
   from the shared optimum with a small CMA-ES step
   (`--per-joint-sigma0`, default 0.05) for `--per-joint-evaluations`
   more evaluations (default 600, about a minute at 20 training episodes).

```sh
.venv/bin/antares calibrate lerobot/svla_so101_pickplace --per-joint \
    --max-evaluations 200 --per-joint-evaluations 1200 --per-joint-sigma0 0.1 \
    --name svla_so101_pickplace_perjoint
```

The HTTP API takes the same options (`per_joint`, `per_joint_max_evaluations`
in `POST /api/calibrate`); `per_joint` defaults to false so the live demo
stays fast.
Progress events carry a `stage` field (`shared` / `per_joint`) and the
result records a per-stage summary.

Held-out tracking RMSE in degrees, `split_episodes(…, 20, 10, seed=0)`,
gripper excluded from the overall score. The per-joint locks are
`locks/*_perjoint.lock.json`; their shared stage reproduces the shipped
shared locks exactly.

| | SO-101 default | SO-101 shared | SO-101 per-joint | SO-100 default | SO-100 shared | SO-100 per-joint |
|---|---|---|---|---|---|---|
| **overall** | 6.007 | 1.054 | **0.812** | 2.744 | 0.791 | **0.530** |
| shoulder_pan | 8.049 | 0.349 | 0.363 | 3.224 | 0.433 | 0.320 |
| shoulder_lift | 6.832 | 0.968 | 1.034 | 3.540 | 0.829 | 0.709 |
| elbow_flex | 5.355 | 2.031 | **1.316** | 2.516 | 1.322 | **0.601** |
| wrist_flex | 3.797 | 0.476 | 0.455 | 1.955 | 0.432 | 0.408 |
| wrist_roll | 5.085 | 0.387 | 0.390 | 2.136 | 0.567 | 0.520 |
| evaluations | – | 200 | 200 + 1200 | – | 150 | 150 + 1200 |
| per-joint sigma0 | – | – | 0.1 | – | – | 0.05 |

The per-joint stage is sensitive to its step size and budget. Held-out
score after refining each shared optimum, CMA-ES's default population
of 15, seed 0:

| sigma0 | SO-101, 600 evals | SO-101, 1200 | SO-100, 600 | SO-100, 1200 |
|---|---|---|---|---|
| 0.03 | 0.990 | – | – | – |
| 0.05 | 0.991 | 0.974 | 0.552 | **0.530** |
| 0.1 | 1.034 | **0.812** | 0.643 | 0.555 |
| 0.2 | 1.054 (never beat the start) | – | – | – |

A small step improves quickly but can settle in a nearby basin; on SO-101
sigma0 0.1 sits flat for ~500 evaluations, then finds a much better one.
The default (0.05, 600 evaluations) is the safe choice within a minute;
for a lock worth keeping, try both step sizes with a 1200 budget.

The per-joint locks also transfer across the two arms better than the
shared ones (`antares cross`, held-out splits): the SO-100 per-joint lock
scores 1.014° on SO-101 against 1.138° for the SO-100 shared lock, and the
SO-101 per-joint lock 0.769° on SO-100 against 0.846°.

Caveat: the gripper is not scored (object contacts are not logged), so its
per-joint parameters are unconstrained and should not be read as physical.

## Gate deploys on your robot's drift

A lock is a snapshot of how one arm behaved on the day it was calibrated.
Servos wear, belts loosen, someone recalibrates: the arm stops matching its
sim, and a policy that passed in sim meets a different robot. `antares
check` is the test suite for that. Record a few fresh episodes, replay them
through the locked sim, and fail if the tracking error has grown past
`max(--threshold, --ratio * <the lock's held-out score>)`:

```sh
# fresh logs from the Hub, or a LeRobot dataset directory you just recorded
.venv/bin/antares check lerobot/svla_so101_pickplace --lock svla_so101_pickplace
.venv/bin/antares check ./recordings/2026-10-17 --lock my_arm --last 10 --threshold 2
```

```
joint          rmse (deg)  reference  limit  status
-------------  ----------  ---------  -----  -----------------
shoulder_pan   0.405       0.349      0.524  PASS  <- worst
shoulder_lift  0.854       0.968      1.452  PASS
elbow_flex     2.095       2.031      3.046  PASS
...
overall        1.067       1.054      1.582  PASS
OK: lerobot/svla_so101_pickplace vs lock svla_so101_pickplace: 1.067 deg RMSE on 10 episodes (reference 1.054, limit 1.582)
```

Exit codes: `0` pass, `1` drifted, `2` usage error or too few episodes
(`--min-episodes`, default 5). `--json` prints the full `CheckResult`. The
same check is `POST /api/check` with
`{"dataset", "lock", "threshold_deg"?, "ratio"?, "last"?}`.

In CI, the repo is a composite GitHub Action. Point it at your dataset and
lock and the job fails when the robot drifts:

```yaml
- uses: actions/checkout@v4
- uses: your-org/antares@main
  with:
    dataset: your-org/so101_arm_a_daily   # or a directory in the workspace
    lock: so101_arm_a                     # locks/so101_arm_a.lock.json
    threshold: "2"                        # optional, degrees
    last: "10"                            # optional, newest N episodes
```

`.github/workflows/robot-check.yml` is a working example (manual and
weekly) against the public SO-101 dataset and the lock under `locks/`. The
tests in `tests/test_check.py` prove the gate bites: real episodes pass,
the same episodes with 3 frames of extra lag or excursions scaled to 0.9
(a weak servo) fail.

## Layout

| Path | What |
|---|---|
| `antares/data.py` | LeRobot dataset loader (tabular only) |
| `antares/sim.py` | MuJoCo model, unit mapping, replay |
| `antares/calibrate.py` | CMA-ES parameter fit |
| `antares/gap.py` | Tracking-error metrics |
| `antares/lock.py` | Calibrated sim saved per robot as a lock file |
| `antares/export.py` | Lock -> standalone MJCF, sidecar, scene, zip |

| `antares/drift.py` | Sliding-window refits and trend statistics over a session |

| `antares/lock.py` | Calibrated sim saved per robot (`locks/*.lock.json`) |
| `antares/dr.py` | Domain-randomization ranges derived from refits (`locks/*.dr.json`) |

| `antares/check.py` | Pre-deploy drift gate (`antares check`, `/api/check`) |
| `action.yml` | Composite GitHub Action running the gate in CI |
| `antares/server.py` | FastAPI backend for the UI |
| `examples/` | `randomize_env.py`: DR ranges in a training loop |
| `web/` | React + three.js viewer |

## Status

Weekend hackathon project. Data loading, replay, calibration, robot locks
and the web viewer work end to end on the SO-101 and SO-100 demo datasets.
