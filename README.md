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

Inspect a dataset from the CLI:

```sh
.venv/bin/antares inspect lerobot/svla_so101_pickplace
```

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

## Layout

| Path | What |
|---|---|
| `antares/data.py` | LeRobot dataset loader (tabular only) |
| `antares/sim.py` | MuJoCo model, unit mapping, replay |
| `antares/calibrate.py` | CMA-ES parameter fit |
| `antares/gap.py` | Tracking-error metrics |
| `antares/events.py` | Contact events from the real-sim residual |
| `antares/server.py` | FastAPI backend for the UI |
| `web/` | React + three.js viewer |

## Status

Weekend hackathon project. Data loading and the UI shell work; replay
and calibration are in progress.
