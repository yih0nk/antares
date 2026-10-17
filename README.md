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

## Is a lock specific to its arm?

A lock is only worth having per robot if it beats another arm's lock on
this arm, by more than refitting the same arm moves the score. `antares
specificity` runs that test on the newest shared (one value for all
joints) lock of each dataset, scoring the held-out episodes of the split
that lock was fitted with, and writes `results/robot_specificity.{md,json}`:

```sh
.venv/bin/antares specificity                # the two demo arms
.venv/bin/antares specificity --synthetic    # same study on simulated arms with known parameters
.venv/bin/antares identify lerobot/svla_so101_pickplace --lock svla_so101_pickplace
```

Per arm it reports the cross matrix with a verdict and margin, a same-arm
control (refit on 20 episodes the lock never saw), a swap test (refit
with the other arm's `kp`/`kv` pinned), two extra CMA-ES seeds on the same
episodes, a linearized identifiability table (how far each parameter can
move before the training error rises 0.1°, alone and with the others
refit), the latency profile, and the error along the path to the other
lock. `antares cross` also prints the verdict and takes `--json`.

What the two demo arms show with the current shared locks (held-out
error, full tables in [`results/robot_specificity.md`](results/robot_specificity.md)):

| | SO-101 data | SO-100 data |
|---|---|---|
| own lock | **0.74°** | 0.79° |
| other seeds, same episodes | 0.91°, 1.06° | **0.57°**, 0.66° |
| same arm, refit on other episodes | 1.05° | 0.80° |
| own fit, other arm's kp/kv pinned | 1.09° | 0.65° |
| other arm's lock | 1.27° (1.73×) | 0.96° (1.22×) |

* **The arms are distinguishable.** Each arm's own lock beats the other
  arm's, and the best fit found for each arm beats the other arm's lock by
  about 1.7× on both (0.74° vs 1.27°, 0.57° vs 0.96°).
* **A single fit is not reliable.** The same episodes with another CMA-ES
  seed score up to 1.4× worse on SO-101, and two seeds beat SO-100's own
  lock (0.57° vs 0.79°). Which basin a 150-200 evaluation run lands in
  matters as much as which arm it is fitting, so the cross margins above
  carry that noise. Keeping the best of several seeds would make a lock,
  and every check built on it, much more repeatable.
* **The stiffness gap is not a difference between the arms.** The locks
  sit at kp 10.5 (SO-101) and 1427 (SO-100), but SO-100's two better
  seeds land at kp 13-14, and pinning SO-100's fit to SO-101's `kp`/`kv`
  scores 0.65°, better than its own lock. Across seeds `kp` and `kv`
  spread 25-125× at similar error.
* **The linearized table is local only.** It finds SO-100's lock tightly
  pinned (every parameter within ×1.4-1.6, condition number 3), yet better
  fits sit 100× away. On SO-101 it shows `damping` unconstrained and
  traded off against `kv` (×8800 with the others refit, ×1.9 alone).
* **Latency is the best-pinned parameter**: one frame either side of the
  lock costs 0.3-0.9° with everything else held fixed.

The synthetic study ([`results/synthetic/robot_specificity.md`](results/synthetic/robot_specificity.md))
gives each simulated arm one real lock's parameters as its truth (kp 10.5
vs 1427). There the own lock wins by 2.2× and 5.4×, but the fit still
misses the truth (SO-100-like arm: kp 182 vs 1427 at 0.40° against
0.35°), and the seeds on one arm range from 0.55° to 1.91°. So the
optimizer spread above is a property of the fit, not of the real data.
The fleet scan's cross matrix over many arms is the larger version of
this test.

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

## Contact detection from the residual

Once the sim reproduces the arm's free-space motion (the SO-101 lock
tracks held-out episodes to 1.05 deg RMSE), whatever is left in
`real - sim` is something the sim does not model. In a pick-and-place
log that should be the arm touching, grasping or pushing an object, so a
sustained deviation is a contact event read off the joint encoders alone,
with no force sensor. `antares/events.py` implements this, pure numpy:

1. **Residual.** `residuals(real_q, sim_q)` is the signed per-joint
   difference in degrees.
2. **Baseline.** `residual_baseline(episodes, arm)` replays the lock's
   held-out episodes and takes, per joint, the median and `1.4826 * MAD`
   of the residual on *free-motion* frames (nothing in the gripper, at
   least 15 frames from any gripper open/close), plus its lag-1
   autocorrelation. The gripper command is binarized with hysteresis
   around thresholds fitted by 2-means on the dataset's own command
   values, so no fixed 0..100 convention is assumed.
3. **Events.** `detect_events(residual, baseline, z=4, min_frames=5)`
   z-scores the scored joints, enters an event at `|z| >= z`, leaves at
   `z/2`, bridges gaps of up to 3 frames and drops runs shorter than
   `min_frames`. Each event carries the joints that crossed, the peak z,
   the signed peak residual, and a kind: `contact` for a long one-sided
   deviation, `anomaly` for a brief or sign-flipping one.
4. **Proxy label.** The gripper command closing (open -> closed) marks a
   grasp. `validate_episode` counts grasps whose +-15 frame window holds
   an event (recall), events inside a grasp window (precision), events
   far from any gripper transition, and the share of residual energy that
   falls inside grasp windows against the share of frames there.

```sh
.venv/bin/antares events lerobot/svla_so101_pickplace --episode 8 --lock svla_so101_pickplace
.venv/bin/antares events lerobot/svla_so101_pickplace --all --lock svla_so101_pickplace
```

`GET /api/residual?dataset=&episode=&lock=` returns `residual` and `z`
as `(T, 6)` arrays in degrees, `events`, `gripper_cmd`, `grasps`,
`releases`, `grasp_windows` and the `baseline`, so the UI can draw the
residual on the timeline and flash the ghost when it separates from the
real arm. `GET /api/events?dataset=&lock=` runs every episode and pools
the validation.

What the two demo datasets give, at `z=4`, `min_frames=5`, all 50
episodes (held-out episodes in parentheses):

| | SO-101 lock | SO-100 lock |
|---|---|---|
| grasps / events | 68 / 92 | 89 / 213 |
| recall (grasps with an event in +-15 frames) | 100% (100%) | 57% (59%) |
| precision (events in a grasp window) | 61% (55%) | 33% (34%) |
| events far from any gripper transition | 21% | 44% |
| residual energy inside grasp windows | 23% (22%) | 11% (10%) |
| frames inside grasp windows (chance) | 18% (16%) | 14% (14%) |
| lift | 1.29x (1.37x) | 0.79x (0.74x) |

Read these with care. On the SO-101 the residual does concentrate around
grasps, but weakly (1.3x chance), and the perfect recall comes from
events that are long: the sim deviates by 3-5 deg on the elbow for the
whole stretch where the arm is out over the table, from before the grasp
until after the release, so one event covers both. On the SO-100 the
residual is *less* concentrated in grasp windows than chance. In both
datasets the dominant unmodelled effect is pose-dependent (the real elbow
sags or lags differently when the arm is extended) rather than contact
with a light object, and the free-motion baseline, estimated mostly on
frames where the arm rests, makes every reach look like an excursion. The
detector is a faithful "the sim and the robot disagree here" sensor; the
claim that the disagreement *is* contact needs a baseline conditioned on
pose and speed, or heavier objects, before it holds up. The gripper
command is itself only a proxy: it also closes after a release, so some
of the "false" positives and misses are label noise.

## Lag compensation (sim-validated)

The SO-101 lock says its servo answers with about 4 frames of latency
(130 ms at 30 fps) and a soft position gain, so the commanded trajectory
leads the real arm by several degrees. Given the calibrated model,
`antares/compensate.py` computes commands that make the arm actually
follow a desired trajectory:

- **shift**: send `q_des[t + L]` at frame `t` (pure-latency baseline).
- **ilc**: iterative learning control in the calibrated sim. Start from
  the shifted commands, roll out, add `gain × (q_des − q_sim)` aligned by
  the latency (with a 3-frame moving average so the update does not
  amplify frequencies the servo cannot follow), repeat until the tracking
  error stops improving. Commands are clipped to the actuator range.

**Every number below is sim-only.** The desired trajectory is the real
measured trajectory of a held-out episode (what the operator had the arm
doing), and tracking is judged against the calibrated MuJoCo model, never
against hardware. The raw row therefore equals the lock's gap score by
construction; the other rows say how much of that gap the compensator
removes *in the sim*. Whether the physical servo agrees is exactly the
question the lock cannot answer on its own.

RMS tracking error in degrees (gripper excluded), 10 held-out episodes
per dataset, with each dataset's own lock:

| dataset | latency | raw commands | shift | ILC | ILC iterations |
|---|---|---|---|---|---|
| `lerobot/svla_so101_pickplace` | 4 frames | 1.054 | 0.133 | 0.048 | 4 per episode |
| `lerobot/svla_so100_pickplace` | 3 frames | 0.791 | 0.477 | 0.119 | 3–6 per episode |

Feasibility (also sim-only): the compensated commands stay within the
actuator range, exceed the original commands' range by at most 0.6°, and
their peak velocity is 0.99× (SO-101) and 1.15× (SO-100) the fastest
joint motion seen in the real data; the tool flags any joint above 1.5×.
Frames sitting on the control limit are reported for the raw and the
compensated commands side by side, because a desired trajectory that
rides a joint limit (the SO-100 gripper and wrist roll do) legitimately
pins its commands there.

```sh
.venv/bin/antares compensate lerobot/svla_so101_pickplace --episode 0 --lock svla_so101_pickplace
.venv/bin/antares compensate lerobot/svla_so101_pickplace --lock svla_so101_pickplace --heldout
.venv/bin/antares compensate lerobot/svla_so101_pickplace --episode 0 --lock svla_so101_pickplace \
    --method ilc --out commands.npy    # (T,6) in dataset units
```

The API mirrors it: `GET /api/compensate?dataset=&episode=&lock=&method=`
returns `fps`, `joints`, `q_des`, `ctrl_raw`, `ctrl_comp`, `q_sim_raw`,
`q_sim_comp` (all `(T,6)` in degrees), `rms: {raw, shift, ilc}`,
`ilc_iterations`, `feasibility`, and the compensated ghost's body poses so
the viewer can overlay it.

Caveats: the first `latency_frames` frames of any trajectory cannot be
corrected (nothing sent at `t ≥ 0` reaches the arm before `t = L`), the
compensator trusts the lock, so unmodelled effects (backlash, load,
temperature) carry straight through, and ILC is open-loop per trajectory:
it needs the desired path ahead of time and does not react to disturbances.

### Fleet scan

Every community SO-100/101 dataset on the Hub was recorded on somebody's
own arm, so each one is a different physical robot. `antares fleet scan`
searches the Hub, keeps the datasets the simulator can replay (SO-100/101
robot type, the six canonical joints, 30 or 50 fps, at least 15 episodes, a
tabular part under 200 MB and 200 parquet files, not a simulation or a merge
of several people's recordings, one dataset per account), calibrates each
one from the default parameters and writes one lock per robot plus a report:

```sh
.venv/bin/antares fleet scan --limit 60 --budget-s 2400   # locks/fleet/*.lock.json, fleet.json
.venv/bin/antares fleet list                              # the table below
curl localhost:8000/api/fleet          # fleet.json
curl localhost:8000/api/fleet/cross    # the partial cross matrix
```

The committed scan (`locks/fleet/fleet.json`, 2026-10-17) inspected 166 Hub
hits, accepted 60 and calibrated **40 robots** in 60 minutes, with no
failures (20 candidates were left when the budget ran out). The 106
rejections were mostly a second dataset from the same account (30), v1
datasets without `meta/info.json` (24), v2 datasets stored as one parquet
file per episode (15), simulation recordings (8), too few episodes (7) and
bimanual or non-SO arms (14).

* **Every robot improved.** Held-out tracking RMSE went from a median of
  3.45 deg to 1.18 deg; the median robot closed **59 %** of its gap (IQR
  52–70 %, range 22–84 %).
* **The fitted parameters are all over the place**, which is the point of
  calibrating per robot. Latency spans 1–5 frames (33–167 ms at 30 fps),
  median 3 frames (100 ms), IQR 2–4. Actuator gain `kp` spans 50–2954
  (1.8 decades) with a median of 345 and IQR 93–1032; the MJCF default is
  998. Damping, armature and friction each span more than two decades.
* **Per-robot calibration beats borrowing a neighbour's.** On a 6-robot
  cross matrix (the two demo arms plus the next four), a robot scored on
  its own lock averages 1.14 deg, on another robot's lock 1.54 deg (1.35x;
  per-robot median 1.24x), and on the defaults 4.60 deg (4.0x). Any
  calibrated lock beats the defaults on all 30 (robot, other lock) pairs,
  and 5 of 6 robots do best on their own lock.
* **Outliers worth a look.** `theconstruct-ai/so101_pickplace_auto_v1` and
  `legalaspro/so101-pick-and-place-cube-lerobot-30hz` start at 0.08 and
  0.12 deg: their commands equal the measured state, so they are not
  teleoperation logs and the sim matches them trivially.
  `kmg0620/so101_bomb_pick_place_v1` starts at 21.3 deg and only gets to
  14.9; no lock in the fleet explains it, which points at a recording in
  different units or a badly calibrated arm rather than at the servos.
  `OkzRIKU/so101-turn-up-drone` closes only 22 % of its gap, with the gain
  pinned at the bottom of the search range and the maximum latency;
  `codywang/so101_pickup_test` is left at 4.4 deg, almost all of it in
  `wrist_roll` (9.3 deg), which smells like a joint-offset problem rather
  than servo dynamics. `SGPatil/so101_table_cleanup` fits an armature 130x
  below the fleet median. Score outliers are kept out of the cross matrix so
  one unexplained arm does not dominate it.

## Layout

| Path | What |
|---|---|
| `antares/data.py` | LeRobot dataset loader (tabular only) |
| `antares/sim.py` | MuJoCo model, unit mapping, replay and rollout |
| `antares/calibrate.py` | CMA-ES parameter fit |
| `antares/gap.py` | Tracking-error metrics |
| `antares/lock.py` | Calibrated sim saved per robot as a lock file |
| `antares/export.py` | Lock -> standalone MJCF, sidecar, scene, zip |

| `antares/drift.py` | Sliding-window refits and trend statistics over a session |

| `antares/lock.py` | Calibrated sim saved per robot (`locks/*.lock.json`) |
| `antares/dr.py` | Domain-randomization ranges derived from refits (`locks/*.dr.json`) |

| `antares/check.py` | Pre-deploy drift gate (`antares check`, `/api/check`) |
| `action.yml` | Composite GitHub Action running the gate in CI |

| `antares/events.py` | Contact events from the real-sim residual |

| `antares/compensate.py` | Lag compensation (shift, ILC), sim-validated |
| `antares/specificity.py` | Cross-arm verdict, identifiability, swap and same-arm checks |
| `antares/synthetic.py` | Simulated arm datasets with known parameters |
| `results/` | Robot-specificity reports (`antares specificity`) |
| `antares/server.py` | FastAPI backend for the UI |
| `examples/` | `randomize_env.py`: DR ranges in a training loop |

| `antares/fleet.py` | Hub discovery, fleet scan, fleet statistics |
| `locks/` | Committed locks for the demo arms; `locks/fleet/` for the fleet scan |
| `web/` | React + three.js viewer |

## Status

Weekend hackathon project. Data loading, replay, calibration, robot locks
and the web viewer work end to end on the SO-101 and SO-100 demo datasets.
