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
4. **Fit the sensor calibration (optional).** With `--fit-sensor`,
   Antares also fits a per-joint offset and scale between the logged
   commands and the measured joints, and saves it in the lock.
5. **See it.** The web UI overlays the real arm and the sim ghost in 3D
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

#### Other simulators

Most SO-101 training runs on Isaac Lab, ManiSkill or mjlab rather than raw
MuJoCo. `--format` renders the same fitted parameters as that simulator's
own actuator configuration, written into the export directory next to the
MJCF (`antares/export_targets.py`):

```sh
.venv/bin/antares export svla_so101_pickplace --format all         # isaaclab + maniskill + mjlab + json
.venv/bin/antares export svla_so101_pickplace --format isaaclab --fps 30
```

| `--format` | Files | What |
|---|---|---|
| `mjcf` (default) | the table above | unchanged |
| `isaaclab` | `<name>_isaaclab.py`, `<name>_isaaclab.json` | `SO101_ACTUATOR_CFG` (an `ImplicitActuatorCfg`), `SO101_ARTICULATION_CFG` around a USD path placeholder, `CONTROL_HZ` / `COMMAND_DELAY_STEPS`, and `EventTerm`s for `randomize_actuator_gains` / `randomize_joint_parameters` when `locks/<name>.dr.json` exists |
| `maniskill` | `<name>_maniskill.py`, `<name>_maniskill.json` | `PDJointPosControllerConfig` for the six joints, absolute and delta variants, `force_limit` from `forcerange` |
| `mjlab` | `<name>_mjlab.py` | an `EntityCfg` loading the exported MJCF through `mujoco.MjSpec`, with `XmlActuatorCfg` wrapping its `<position>` actuators and the latency as `delay_min_lag` / `delay_max_lag` |
| `json` | `<name>.params.json` | the sidecar plus every target's numbers, latency in seconds, pinned parameters and DR ranges |
| `all` | all of the above | |

Every file starts with the lock's provenance (dataset, `created_at`,
held-out error before -> after, pinned parameters, seed spread) and the
unit assumptions. Per-joint locks emit one value per joint (`{joint: value}`
dicts for Isaac Lab, lists for ManiSkill); scalar locks one value shared by
the six joints. A lock with a fitted command map (`--fit-sensor`) says so
in every file and carries the map in the JSON. The generated Python
imports the simulator inside a `try`, so the numbers stay importable
without it. API: `GET /api/locks/<name>/export?format=isaaclab` returns
the main file of a target (`mjcf`, the default, is still the XML; an unknown
format is 422) and `GET /api/locks/<name>/export.zip?targets=all` adds every
target to the zip.

Isaac Lab (verified against 2.3.2): point `SO101_USD_PATH` at your SO-101
USD (LeIsaac's `so101_follower.usd` uses these six joint names), then

```python
from svla_so101_pickplace_isaaclab import SO101_ARTICULATION_CFG, SO101_RANDOMIZE_GAINS, COMMAND_DELAY_STEPS

robot = SO101_ARTICULATION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")   # in your SceneCfg
class EventsCfg:
    gains = SO101_RANDOMIZE_GAINS                                           # data-derived DR, startup mode
# in the action term / wrapper: a deque of COMMAND_DELAY_STEPS targets, pop the oldest each step
```

ManiSkill (3.0.1): the file defines `pd_joint_pos` / `pd_joint_delta_pos`
with the lock's gains in place of the stock agent's `1e3 / 1e2 / 100`:

```python
from svla_so101_pickplace_maniskill import controller_configs

class SO101(BaseAgent):
    @property
    def _controller_configs(self):
        return controller_configs       # joint names must match your URDF
```

mjlab (1.6.0): the MJCF is the artefact; the snippet loads it and sets the
delay:

```python
from svla_so101_pickplace_mjlab import SO101_ENTITY_CFG   # reads svla_so101_pickplace.xml next to it

scene = SceneCfg(entities={"robot": SO101_ENTITY_CFG}, ...)
```

Unit mapping (MuJoCo hinge joints in radians; Isaac Lab stores drive gains
per radian and converts to the USD per-degree convention itself, SAPIEN
wraps PhysX articulations in radians):

| fitted (MuJoCo) | Isaac Lab | ManiSkill | caveat |
|---|---|---|---|
| `kp` N·m/rad | `stiffness` | `stiffness` | one-to-one |
| `kv` + `damping` N·m·s/rad | `damping` | `damping` | PhysX drives have one damping slot; both terms are −b·q̇ at zero target velocity, so the sum is the exact equivalent there (the JSON keeps both) |
| `armature` kg·m² | `armature` (on the actuator cfg) | `JOINT_ARMATURE` constant | not settable from the ManiSkill controller config; unverified on SAPIEN 3 |
| `frictionloss` N·m | `friction` | `friction=0.0` + `JOINT_FRICTIONLOSS_NM` | Isaac Lab's `friction` is an effort on Isaac Sim 5.0+ but a unitless coefficient on 4.5; SAPIEN's is a PhysX coefficient with no torque conversion, so the N·m value is emitted beside it |
| `forcerange` N·m | `effort_limit_sim` | `force_limit` | one-to-one |
| `latency_frames` | `COMMAND_DELAY_STEPS` at `CONTROL_HZ` | same | no simulator expresses it; mjlab takes it as `delay_min_lag` / `delay_max_lag` in physics steps of the MJCF's 0.005 s timestep |

Latency is converted with the dataset's frame rate: `meta/info.json` from
the local Hugging Face cache when the dataset has been loaded, else 30 fps
(both demo datasets record at 30); `--fps` overrides it and each file
records which it used. "Delay each command by N control steps at F Hz" is
the whole instruction; a per-joint lock gives six N.

## Bring your own dataset

The two demo datasets are pinned in the picker, but the point is *your*
robot. Three ways in:

* **Any Hub id.** In the UI, **+ add dataset** → type `owner/name` → **Check**
  fetches only `meta/info.json` and shows the verdict (reasons in red,
  warnings in amber) before anything is downloaded; **Load** then pulls the
  parquet files behind the usual overlay. From the API:
  `GET /api/datasets/validate?dataset=<id>` (the verdict, cheap) and
  `POST /api/datasets/load?dataset=<id>` (400 with the reasons if the
  dataset cannot be replayed; warnings ride along in the summary when it can).
* **A zip upload.** Drop a `.zip` of a LeRobot dataset directory
  (`meta/info.json` + `data/**/*.parquet`; videos may be present and are
  ignored) on the same popover, or `POST /api/datasets/upload` (multipart
  field `file`; a bare `.parquet` is accepted with an `info` field holding
  `meta/info.json`; optional `name`). The archive is extracted with
  traversal checks under `uploads/<name>-<hash>/` (`ANTARES_UPLOAD_DIR`),
  capped at 200 MB (413), validated, loaded and registered as
  `upload/<name>`; `DELETE /api/datasets?dataset=upload/<name>` removes it.
  Uploads survive a server restart.
* **A directory on disk.** Every CLI command takes a LeRobot dataset
  directory wherever it takes a Hub id, and so does the API
  (`dataset=/path/to/recording`):

  ```sh
  .venv/bin/antares inspect ./recordings/2026-10-17      # verdict + summary, no sim
  .venv/bin/antares calibrate ./recordings/2026-10-17 --name my_arm --seeds 3
  .venv/bin/antares replay upload/bench_arm --episode 0 --lock my_arm
  ```

`GET /api/datasets` lists everything loaded in the process with
`source` (`demo` | `hub` | `upload` | `local`), `warnings`, `robot_family`
(what the dataset says it was recorded on) and `robot` (the registry model
that simulates it). The UI shows those as a chip next to the picker.

### What a dataset needs

`antares/validate.py` is the one place that decides (the fleet scan, the
API and `antares inspect` all call it):

| check | outcome |
|---|---|
| `robot_type` recognisable: `so100*`, `so101*`, `so_follower` (any spelling), or a registered robot's `robot_types` (ALOHA, Trossen AI, Piper, Koch); a dataset in `antares datasets` or given `--robot` uses that model | reject otherwise; bimanual SO rigs (`bi_so100_follower`, 12 joints) are rejected with their own reason |
| the robot's columns in its order (SO: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` after stripping `main_` / `.pos`; others: their profile's `joints` or `aliases`), same names and order in `action` and `observation.state` | reject |
| `fps` 30 or 50 on the SO arms; any rate from 5 to 500 Hz on others unless the profile fixes one | reject |
| episodes: at least 5 for uploads and Hub ids (the fleet scan wants 15) | reject below, warn below 15 |
| `codebase_version` v2.x / v3.x | reject other versions, warn when missing |
| name looks like a simulator (`sim`, `mujoco`, `isaac`, …) or a merge (`merged`, `community`, …) | warning (the fleet scan rejects) |
| loaded data: NaN/inf episodes, episodes under 2 frames | warning, skipped |
| loaded data: body joints outside about [-120, 120] or gripper outside [-5, 105], or everything within ±3.3 | warning: "values look like degrees / radians; replay will be wrong" |

### Supported robots

`antares/robots.py` is a registry of MuJoCo models: `so101` (MuJoCo
Menagerie `robotstudio_so101`, the default) and `so100` (Menagerie
`trs_so_arm100`, vendored under `antares/assets/so100/` with its licence).
`Arm(robot="so100")`, `/api/model?robot=so100`,
`/api/model/mesh/<file>?robot=so100`, `antares replay --robot so100` and
`antares calibrate --robot so100` all work; the server keeps one `Arm` per
model, each behind its own lock, `/api/replay` reports the `robot` it used,
and every lock records its `robot` (older locks read as `so101`).

Which model a dataset gets is `MODEL_FOR_FAMILY` in `robots.py`. Both
families currently map to the SO-101 model, decided on the committed
`svla_so100_pickplace` lock's 10 held-out episodes (RMS degrees, gripper
excluded):

| model | default params | fresh 150-eval calibration (seed 0) |
|---|---|---|
| so101 (Menagerie file, implicitfast @ 5 ms) | 2.744 | **0.787** (the committed lock) |
| so100 (Menagerie file verbatim, Euler @ 2 ms) | 2.720 | 0.697 |
| so100 with so101's `<option>` block (implicitfast @ 5 ms) | 2.737 | 0.777 (lands on the same parameters as the so101 lock) |

The SO-100 geometry does not change the answer: stepped the same way the
two files are within 0.01° and find the same optimum, and the committed
so100 lock scores 0.791 on the so100 model. The verbatim so100 file's 11%
gain comes from its finer timestep, which costs 17 physics substeps per
frame instead of 7 (2.4x slower replay; the live 150-evaluation
calibration would take 90 s instead of 45 s) and would invalidate every
committed SO-100 number in this README. So SO-100 datasets stay on the
SO-101 model and the SO-100 meshes are there for anyone who wants them.
The exporter (`antares export`) writes SO-100 locks on the SO-101 MJCF.

To add an SO-style robot inside the package: vendor its MJCF, meshes and LICENSE under
`antares/assets/<name>/`; add a `RobotModel` to `REGISTRY` with the MJCF's
joint names in canonical order (the SO-100 file calls them
`Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw`) and a
`unit_map_factory` if LeRobot's normalized units do not map onto the joint
ranges the way `UnitMap.from_model` assumes; teach
`validate.robot_family` its `robot_type` spellings; point the family at the
model in `MODEL_FOR_FAMILY`. `tests/test_robots.py` loads every registry
entry through `Arm`. Other arms are described in [Other robots](#other-robots).

## Other robots

`antares robots` lists every arm antares can simulate:

| name | robot | columns | model |
|---|---|---|---|
| `so101`, `so100` | SO-101 / SO-100 | 6 (5 + gripper), LeRobot ±100 / 0..100 | vendored |
| `aloha` | ALOHA (two ViperX 300s) | 14 (2 × 6 + 2 grippers), radians, grippers 0..1 | Menagerie `aloha` |
| `trossen_ai` | Trossen AI stationary (two WidowX AI) | 14, radians, gripper carriage in metres | Menagerie `trossen_wxai` |
| `piper` | AgileX Piper | 7 (6 + gripper), radians, jaw opening in metres | Menagerie `agilex_piper` |
| `koch` | Koch v1.1 (low-cost robot arm) | 6, LeRobot ±100 / 0..100 | Menagerie `low_cost_robot_arm` |
| `so101_deg`, `koch_deg` | the same arms recorded in degrees (LeRobot ≥ 0.5 `use_degrees`, or the 2024 Koch datasets) | 6 | as above |

Only the SO models ship in the package. The others are downloaded from
MuJoCo Menagerie at a pinned commit the first time they are used (or with
`antares fetch-model all`) into `~/.cache/antares/models`
(`$ANTARES_MODEL_CACHE`), so a lock fitted today replays on the same
geometry later.

A robot profile (`RobotModel` in `antares/robots.py`) says what antares
used to assume for the SO arms:

- `joints`: the dataset columns, in order, and `joint_names`: the MJCF joint
  each one drives (each must have exactly one joint actuator). Joints the
  MJCF couples to a mapped one with an `<equality joint>` (a second finger)
  follow it. `aliases` accepts other column spellings (`motor_0`, `..._joint_6`).
- `grippers`: the columns left out of the score (grasp contact isn't
  simulated). The first one marks grasps for contact detection.
- `units` per column: `range_m100_100`, `range_0_100`, `range_0_1`,
  `degrees` (zero at the middle of the range, LeRobot's `DEGREES` mode),
  `degrees_raw`, `radians` or `native` (metres for a prismatic gripper),
  with an optional `unit_scale` and separate `action_units` when commands
  and readings differ.
- `default_params`: `"sts3215"` (SO: the shared servo defaults) or
  `"model"`: start from each joint's own values in the MJCF. On those robots
  one fitted kp, damping or similar is a multiplier of each joint's own
  value, so a wrist and a shoulder that differ 50x keep their proportions.
- `bounds` overrides per factor, `fps`, `robot_types` patterns, `bimanual`.

Add your own without editing the package: write `[[robot]]` tables in a TOML
file (`name`, `joints`, `mjcf_joints`, and `model = "path/to/robot.xml"` or
`menagerie = {directory, xml}`, plus any field above) and point
`$ANTARES_ROBOTS` at it.

`antares datasets` lists public LeRobot datasets of these arms to test on,
with the model each replays on (the catalog in `antares/catalog.py`; notes
on why each was picked are in the PR). `antares calibrate <id>` picks the
model from the catalog; `--robot` overrides it for any dataset.

Everything downstream runs on the lock's robot: `check`, `--explain`,
`drift`, `dr`, `events`, `compensate`, `specificity`, the viewer (OBJ and
STL meshes) and `export`. Robots without a shared servo class are exported
through `mujoco.MjSpec`: the lock's values are written on each mapped joint
and actuator, and the exported model replays the lock to within 1e-6 rad
(`tests/test_other_robots.py`).

## What the fit accounts for

`antares factors` prints every factor the fit can account for, from one
registry (`antares/factors.py`) that the fit, `--explain`, drift, DR,
export and the lock all read:

```
name            what                 stages     range                  pinned by data
kp              Servo stiffness      shared     10..5000 N*m/rad       no
kv              Servo velocity gain  shared     0.1..50 N*m*s/rad      no
damping         Joint damping        shared     0.01..10 N*m*s/rad     no
armature        Rotor inertia        shared     0.0001..0.5 kg*m^2     partly
frictionloss    Joint friction       shared     0.001..2 N*m           partly
latency_frames  Command latency      shared     0..5 frames            yes
forcerange      Torque limit         per_joint  0.5..30 N*m            untested
sensor_gain     Joint reading scale  sensor     0.75..1.33 x           yes
sensor_bias     Joint zero offset    sensor     -25..25 dataset units  yes
```

"Pinned by data" is what the robot-specificity study and the fleet showed:
latency and the sensor map are measurements, stiffness and damping trade
off against each other at equal error (refits of one arm span 25x), so read
them as a fit. `-v` prints what each one explains, the data it needs and
the evidence. The same command lists effects antares knows about but
doesn't model yet (contact, payload, backlash, temperature, cable drag,
encoder resolution) and what each would need.

Choose what a fit accounts for:

```bash
antares calibrate <id> --factors -kp                    # all defaults but kp
antares calibrate <id> --factors latency_frames,damping # only these two
```

Add a factor from a TOML file (`examples/factors.toml`): `target` names a
MuJoCo `MjModel` field (`opt.gravity`, `body_mass`, `jnt_stiffness`,
`actuator_gainprm:0`), `mode` is `scale` (starts at 1) or `set`, and
`per_joint = true` gives one value per joint. Anything a field can't
express takes `apply = "module:function"`, a function of `(arm, values)`.

```bash
antares calibrate <id> --factors-file examples/factors.toml --factors +gravity_scale
```

`$ANTARES_FACTORS` loads the same files for the server and every command.
An added factor starts as "untested"; run `antares identify` on the lock
before reading its value as a measurement. Every lock records which factors
each stage fitted (`"factors"` in the lock JSON), and the viewer's
calibrate panel has the same checklist.

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

* **SO-100**: the lock's error grows by 17% over the session (r² 0.66,
  significant) and the windowed refits' own training error grows with it
  (+20%, r² 0.61), so refitting does not recover it and no parameter
  moves consistently. This is a harder tail of the session, not wear the
  six parameters can describe.
* **SO-101**: the lock's error falls slightly (-5%, significant but under
  the 10% bar) and no parameter drifts. `armature` moves the most (+201%
  first to last window, r² 0.31), but its interval includes zero, so that
  is scatter in a weakly constrained parameter. The verdict is "no drift".


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
| kp | 10.54 | 10.02 | 11.14 | -5% .. +6% |
| kv | 0.136 | 0.110 | 0.558 | -19% .. +310% |
| damping | 0.0123 | 0.0104 | 0.0459 | -15% .. +274% |
| armature | 0.0028 | 0.00034 | 0.0065 | -88% .. +128% |
| frictionloss | 0.094 | 0.047 | 0.102 | -50% .. +8% |
| latency_frames | 4 | 3 | 4 | -1 .. +0 frames |

The ranges are saved as `locks/<name>.dr.json` with their provenance
(dataset, lock, episode subsets, per-fit scores, seed). Use them in a
training loop with `antares.dr.sample` and `apply`; see
`examples/randomize_env.py`.

Does the derived range beat a hand-picked one? Tracking RMSE on the lock's
10 held-out episodes, averaging over 64 parameter sets sampled from each
range (MJCF defaults score 6.01 deg; `scripts/dr_compare.py svla_so101_pickplace`):

| parameters | mean | median | min .. max |
|---|---|---|---|
| (a) lock nominal | 0.74 | | |
| (b) data-derived ranges | 1.03 | 0.89 | 0.59 .. 1.87 |
| (c) hand-picked +-30% | 1.32 | 1.51 | 0.73 .. 2.17 |
| (c') +-30%, latency fixed at nominal | 0.77 | 0.76 | 0.72 .. 0.91 |

The hand-picked range is wrong in both directions at once. On latency,
+-30% around 4 frames allows 5, which the data never supports and which
alone costs about half a degree on average (c vs c'). On the continuous parameters
it is far too narrow: within +-30% the sim is nearly indistinguishable from
nominal (c'), so a policy trained there sees no diversity. The
data-derived ranges span 4-5x on `kv` and `damping` and almost 20x on
`armature`, keep latency where the data puts it, and still average
1.03 deg, 5.8x better than the uncalibrated sim. Some samples beat the
lock itself (min 0.59), because the lock was fit on different episodes.


Caveat worth knowing: the lock and its refits sit at `kp` 10-11, right on
the lower edge of the search space (10), so the narrow `kp` range and
`kp.lo` are set by the bound, not the data. The replay cannot separate a
low gain from a low velocity gain over these episodes.


API: `POST /api/dr {dataset, lock?, n_fits?, n_train?, max_evaluations?,
seed?, name?}` starts a job and returns `{job_id, status_url, events_url}`;
the job streams `progress`, `fit` and `done` events at
`/api/jobs/{id}/events` like calibration does. `GET /api/dr` lists saved
ranges, `GET /api/dr/{name}` returns one.

## Does calibration help a policy transfer?

The replay numbers above are a proxy. What a user wants to know is whether
a policy trained in the calibrated sim works better on the real arm than
one trained in the default sim, and nothing with a physical arm can run in
CI, so `antares transfer` tests it sim-to-sim: a MuJoCo arm with *hidden*
true parameters stands in for the real robot (arm `so101` has the SO-101
lock's parameters as its truth, `so100` the SO-100 lock's, as the synthetic
specificity study did), and from each hidden arm Antares synthesizes 50
teleop-style episodes, fits a lock to 20 of them with `calibrate(n_seeds=3,
max_evaluations=150)` and derives DR ranges with `antares.dr` (8 refits x
12 episodes x 60 evaluations), never seeing the truth. The task is
joint-space reaching at 30 fps: from a random start pose reach a random
target pose, both uniform inside +-70 units on the five body joints, within
60 frames; success means every body joint within 2 deg of the target over
the last 5 frames (2 deg is two to three times the calibrated sim's own
held-out residual and a third of the default sim's gap, and 167 ms of
holding rules out passing through the window while still moving). The
policy observes joint positions, velocities and the target and emits a
position command, `u = t + kp (t - q) - kd v + 0.1 MLP(q, t)` with
per-joint gains and a 6-unit tanh hidden layer (111 parameters), trained
with separable CMA-ES (150 generations of 18, 24 fixed training tasks per
seed) to minimize mean plus final tracking error minus a success bonus, so
it has to learn how far past the target to command and how much to lead a
soft, laggy servo. Per hidden arm five sims are trained in: the default
`SimParams()`, the fitted lock, the fitted lock with a parameter set drawn
from its data-derived DR ranges per task and generation, the true
parameters (oracle), and the default sim with hand-picked +-30% DR; every
policy is scored on 64 held-out start/target pairs in the sim it was
trained in and on the hidden arm, over three training seeds (0, 1, 2).
Mean +- sd over seeds; the untrained policy `u = target` scores 20% / 0.98
deg on `so101` and 100% / 0.01 deg on `so100`.

| variant | hidden arm | success in train sim | success on hidden arm | final error on hidden arm (deg) |
|---|---|---|---|---|
| default sim | so101 | 100% +- 0 | 20% +- 0 | 0.98 +- 0.00 |
| fitted lock | so101 | 89% +- 3 | **86% +- 2** | **0.58 +- 0.05** |
| fitted lock + data-derived DR | so101 | 86% +- 2 | 82% +- 2 | 0.61 +- 0.08 |
| oracle (true parameters) | so101 | 83% +- 7 | 83% +- 7 | 0.67 +- 0.18 |
| default sim + hand-picked +-30% DR | so101 | 100% +- 0 | 29% +- 2 | 1.66 +- 0.65 |
| default sim | so100 | 100% +- 0 | 100% +- 0 | 0.01 +- 0.00 |
| fitted lock | so100 | 89% +- 2 | 100% +- 0 | 0.32 +- 0.27 |
| fitted lock + data-derived DR | so100 | 34% +- 12 | 74% +- 25 | 0.77 +- 0.25 |
| oracle (true parameters) | so100 | 100% +- 0 | 100% +- 0 | 0.01 +- 0.00 |
| default sim + hand-picked +-30% DR | so100 | 100% +- 0 | 100% +- 0 | 0.14 +- 0.05 |

What it shows:

* **On `so101` (soft servo, kp 10.5, 4 frames of latency) calibration is
  the difference between failing and working.** The default-sim policy is
  perfect in its own sim and reaches 20% on the hidden arm, exactly the
  untrained policy: the default sim has no gravity sag or lag to learn
  from, so it learns nothing. The lock-trained policy reaches 86% with
  final error 0.98 -> 0.58 deg, lock + data-derived DR 82%, and the
  hand-picked +-30% DR around the default sim 29%, because a +-30% band
  around the wrong stiffness never contains the right one. The oracle
  lands at 83% +- 7, below the lock, so it is not a practical upper bound
  here: the CMA-ES training variance (sd 7 points over three seeds) is as
  large as the gap between the fitted and the true parameters, and the
  fitted lock (held-out replay 0.52 deg against 0.34 deg for the truth) is
  already close enough to the arm that the policy cannot tell them apart.
* **On `so100` (stiff servo, kp 1427) the task does not discriminate.**
  The untrained policy already succeeds on 100% of the tasks at 0.01 deg,
  so the default sim is good enough and calibration has nothing to add.
  The fitted lock still reaches 100% but with 0.32 deg of final error, and
  lock + DR drops to 74% +- 25 (one seed 45%, two 89%): the fit landed at
  kp 120 / kv 0.23 against a truth of 1427 / 41.6, the gain versus
  velocity-gain degeneracy the "Repeatable locks" section describes, and
  the data-derived ranges (kp 120..176) inherit that basin rather than the
  arm's real uncertainty, so the policy is trained for an arm softer than
  the one it meets. The lock itself replays the hidden arm at 0.43 deg
  against 0.39 deg for the truth; a reaching policy feels a parameter
  error the replay metric barely registers.
* **Conclusion.** Calibration helps when the default sim is wrong in a way
  the task feels (sag and lag on a soft servo), is harmless when it is not,
  and DR ranges are only as good as the basin the fit found, which is why
  best-of-N seeds and the pinned-parameter list matter downstream of the
  fit too.

What it does not show. It is sim-to-sim: the hidden arm is the same MuJoCo
model class as the sim, so backlash, temperature, cable drag, servo deadband
and everything else the six parameters cannot express are absent, and a
fitted lock can only be as wrong as the fit, never as wrong as the model.
Observations are noise-free, the task is reaching without contact, and the
policy is a 111-parameter controller, not a learned visuomotor policy. A
run on the physical arm is the next step; the committed result is the
sim-to-sim evidence that the mechanism exists.

Full tables, per-seed 95% bootstrap intervals and the fitted parameters are
in [`results/policy_transfer.md`](results/policy_transfer.md); the JSON
next to it holds every per-task outcome and every trained policy. The run
took 19.4 min on a 10-core laptop (fits about 200 s per arm, 272-363 s per
policy, 30 policies in parallel); `--quick` runs the same pipeline with tiny
budgets in about a second and is what the tests exercise.

```sh
.venv/bin/antares transfer run --seeds 3 --out results/policy_transfer
.venv/bin/antares transfer report results/policy_transfer.json
```

## A policy in the loop

The transfer experiment above uses a hand-built reaching controller on a
synthetic arm. This section puts a policy *learned from the real logs* into
both simulators and measures how differently they judge it. If the stock
sim and the calibrated sim disagree about what the same policy does, the
stock sim would have misjudged it; the lock's held-out replay (0.74° vs
6.0° on SO-101) says which of the two to believe. Everything runs on a
laptop CPU in well under a minute per dataset; nothing is driven on
hardware.

**The policy** (`antares/policy.py`, `pip install -e ".[policy]"` for
torch). A state-based chunked policy, ACT without the images: the last
K=2 measured joint states in, the next H=10 commanded actions out, a 3x256
MLP (151k parameters). It gets two more inputs that stand in for what the
cameras would tell an image policy, because a pick-and-place policy that
does not know where the object is cannot be run closed-loop at all: the
episode's **grasp point** (the fingertip at the operator's first pick, read
from the gripper command and FK exactly as `antares task` does) and
nothing else (a clock was tried and made the policy worse held-out and in
closed loop: it learns *when* the operator closed rather than *where*).
Trained with MSE on the lock's 20 training episodes with 1 unit of Gaussian
noise on the observed states (so the sim's own states in closed loop do not
throw it), early-stopped on the lock's 10 held-out episodes, which it
therefore never fits either; 3-5 s on a CPU. Two trivial baselines on the
same windows say whether it learned anything: *hold state* (command the
last measured state for the whole chunk) and *repeat command* (repeat the
previous logged command, which a deployed policy does not even have).
Held-out episodes, RMS over the body joints in degrees, chunk / first
action of the chunk, and the raw MSE in dataset units²:

| dataset | predictor | held-out chunk (deg) | held-out first action (deg) | held-out MSE (units²) | train chunk (deg) |
|---|---|---|---|---|---|
| SO-101 | **policy** | **5.47** | 3.61 | **24.4** | 4.26 |
| SO-101 | hold state | 11.81 | 5.72 | 105.3 | 12.09 |
| SO-101 | repeat command | 8.41 | **1.46** | 53.7 | 8.63 |
| SO-100 | **policy** | **3.70** | 2.39 | **10.9** | 2.82 |
| SO-100 | hold state | 5.80 | 2.62 | 28.5 | 5.80 |
| SO-100 | repeat command | 4.30 | **0.76** | 14.9 | 4.31 |

The policy beats both baselines over the chunk on both arms (seeds 0-2:
5.47-5.63° on SO-101, 3.61-3.70° on SO-100). On the first action alone
"repeat the last command" wins, as it must: commands are smooth at 30 fps
and the policy does not see them. The phase is not moot, but the policy is
a modest imitator (3.6° / 2.4° from the operator's commands), and that is
the thing being judged below, not a strong one.

**Shadow (open loop)** (`antares/shadow.py`). For each held-out episode the
policy is fed the *real* observations frame by frame, its first action per
frame is collected (receding horizon), and those commands are rolled
through `Arm.rollout` from the real initial state in the default sim, the
shared lock and the per-joint lock. Three distances come out: the policy's
commands from the logged ones (imitation), each sim's trajectory from the
real one, and the two sims' trajectories from *each other* under the same
commands, which is how much the stock sim misjudges this policy. At the
frames the operator closed the gripper, FK gives where each sim thought the
fingertip was against where it really was.

| dataset | sim | policy: sim vs real (deg) | vs stock sim (deg) | fingertip at grasp vs real (cm) | vs stock sim (cm) | logged commands: sim vs real (deg) |
|---|---|---|---|---|---|---|
| SO-101 | stock | 6.40 | – | 2.25 | – | 6.01 |
| SO-101 | shared lock | **3.52** | **5.78** | **1.18** | 1.68 | 0.74 |
| SO-101 | per-joint lock | 3.53 | 5.71 | 1.24 | 1.62 | 0.66 |
| SO-100 | stock | 2.96 | – | 0.83 | – | 2.74 |
| SO-100 | shared lock | **2.37** | **2.61** | 0.86 | 0.61 | 0.79 |
| SO-100 | per-joint lock | 2.33 | 2.76 | 0.99 | 0.93 | 0.54 |

Policy vs logged commands: 3.60° (SO-101), 2.38° (SO-100). Seeds 0-2 move
the SO-101 divergence between 5.78 and 5.90°. Per joint on SO-101 the stock
and the calibrated sim disagree by 7.7° on the shoulder pan, 6.5° on the
shoulder lift, 5.1° on the elbow and the wrist roll, 3.8° on the wrist
flex; the worst frames are the fast reaches (episode 19, frames 36-37:
20.5° RMS between the two sims; episode 16, frame 86: 18.5°).

**Closed loop** (`antares/closedloop.py`). The policy drives each sim from
the real initial state: it observes the sim's joint state (through the
lock's unit map, in dataset units), emits a chunk, the sim executes the
first action with the lock's latency, repeat for the episode's length, with
the same grasp point in every sim. The stock sim's zero latency is stepped
as one frame (a command cannot act on the step that produced the state it
was computed from); with the logged commands fed back regardless of the
observation the loop reproduces `Arm.replay` to 1e-9, which pins its
timing to the calibration's. Metrics, no objects: the trajectory against
the real episode's (whole episode, and the *approach* up to the real grasp
frame), the fingertip distance to the real grasp point at the frame the
policy closes the gripper, and the share of the 10 held-out episodes where
that distance is within 2 cm, a proxy success rate. `+shift` is the cheap
lag compensation a chunk allows: send `chunk[L]` instead of `chunk[0]`
(`compensate.shift_compensate` inside the chunk); it only exists where
there is a latency to shift.

| dataset | sim | trajectory RMS (deg) | approach RMS (deg) | fingertip at close vs real grasp (cm) | closest approach (cm) | within 2 cm | closed the gripper | close offset (frames) |
|---|---|---|---|---|---|---|---|---|
| SO-101 | stock | 43.8 | 38.8 | 9.6 | 4.1 | 2/10 | 9/10 | -84 |
| SO-101 | shared lock | 42.1 | 33.2 | 10.0 | 3.8 | 3/10 | 8/10 | 0 |
| SO-101 | per-joint lock | 43.1 | 32.8 | 7.3 | 2.7 | 4/10 | 8/10 | -13 |
| SO-101 | shared lock + shift | 39.9 | 35.3 | **2.2** | **1.1** | **5/10** | 10/10 | -74 |
| SO-101 | per-joint lock + shift | 41.8 | 34.7 | 4.5 | 1.7 | 6/10 | 10/10 | -59 |
| SO-100 | stock | 59.5 | 53.9 | 21.0 | 9.7 | 0/10 | 4/10 | -139 |
| SO-100 | shared lock | 40.8 | 41.3 | 23.4 | 10.5 | 0/10 | 2/10 | -46 |
| SO-100 | per-joint lock | 38.0 | 39.6 | 18.1 | 11.8 | 0/10 | 2/10 | +51 |
| SO-100 | shared lock + shift | 49.0 | 47.9 | 14.3 | 10.1 | 0/10 | 3/10 | -107 |
| SO-100 | per-joint lock + shift | 48.5 | 45.0 | 20.5 | 11.5 | 0/10 | 2/10 | -93 |

Across training seeds 0-2 the SO-101 successes are 2/4/2 (stock), 3/5/3
(shared lock), 4/5/2 (per-joint), 5/7/4 (shared + shift), 6/4/2 (per-joint
+ shift). The committed numbers are seed 0.

**The honest reading.**

* *Open loop, the stock sim misjudges the policy by as much as it misjudges
  the arm.* On SO-101 the two sims' pictures of what the policy does are
  5.8° RMS apart, as large as the stock sim's own replay error (6.0°) and
  larger than the policy's imitation error (3.6°): which simulator you
  evaluate in moves the trajectory more than the policy's own mistakes
  do. At the grasp instants the stock sim puts the fingertip 2.25 cm from
  where the real jaws were, the calibrated sim 1.2 cm; the two sims are
  1.7 cm apart there, about half the width of the cube being picked. On
  the stiff SO-100 the same numbers are 2.6° and 0.6 cm: the stock sim is
  less wrong about that arm, so it is less wrong about the policy.
* *Closed loop, the stock sim gets the timing wrong, and the ranking is
  the same direction but inside the noise.* In the stock sim the policy
  closes the gripper 84 frames (2.8 s) early on average: the stiff,
  lag-free servo arrives at the goal long before the real arm would, and
  a state-conditioned policy closes when it gets there. In the calibrated
  sim the close lands within a frame of the real one. The proxy success
  itself is low everywhere (2/10 stock, 3/10 calibrated, 4/10 per-joint);
  the calibrated sims rank at or above the stock one in every seed, but
  the margin (one or two episodes) is within seed-to-seed variation, so
  these runs do not show that calibration changes the success verdict of
  this policy. The chunk shift does: sending `chunk[L]` lifts the
  calibrated sim to 5/10 (4-7 across seeds) with the fingertip 1.1 cm from
  the grasp point at its closest, and it is a compensation the stock sim
  cannot even express, having no latency to shift.
* *Whole-episode trajectory RMS is not a sim verdict here.* It is 40°+ in
  every sim because the policy only knows the pick point: after the grasp
  it has no place target and wanders, so most of that number is the place
  phase. The approach RMS (up to the real grasp) is the comparable one:
  38.8° stock vs 33.2° calibrated on SO-101.
* *On SO-100 the sims agree that this policy does not work closed-loop*
  (0/10 everywhere, the gripper closes in 2-4 of 10 episodes), even though
  it is the better imitator. Calibration changes the trajectory (59.5° to
  40.8°) but not the verdict; the task's bimodal grasps (table and
  container) are more than a state-only policy with one goal point learns
  from 20 episodes.
* *What this is not.* A state-only policy handed the grasp point, judged
  without objects or contact by a 2 cm proxy, and every difference is
  sim-vs-sim. What makes the calibrated sim's verdict the one to trust is
  the lock's held-out replay against the real arm, not anything in this
  section.

```sh
.venv/bin/pip install -e ".[policy]"
.venv/bin/antares policy train lerobot/svla_so101_pickplace --lock svla_so101_pickplace
.venv/bin/antares policy evaluate lerobot/svla_so101_pickplace --lock svla_so101_pickplace --mode both
```

The policy is saved as `locks/<lock>.policy.pt` with a `.policy.json`
sidecar (normalization, K, H, features, split, metrics), the evaluations as
`locks/<lock>.policy.shadow.json` and `.policy.closedloop.json`. API:
`POST /api/policy/train {dataset, lock, epochs?, seed?, name?}` and
`POST /api/policy/evaluate {dataset, lock, policy?, mode: "shadow" |
"closedloop", robots?: ["default", "lock", "perjoint", <lock name>...]}`
are jobs streamed over `/api/jobs/<id>/events` (training progress per
epoch with train/eval MSE; evaluation progress per episode with its row);
`GET /api/policy?lock=` lists saved policies with their metrics,
`GET /api/policy/results?lock=` the saved tables, and
`GET /api/policy/viewer?dataset=&lock=&episode=&mode=` recomputes one
held-out episode and returns `(T, nbody, 7)` poses per sim, like
`/api/replay`. In the UI the Policy section trains and evaluates with
progress, shows the stock-vs-calibrated table, and adds two ghosts to the
viewer: the policy in the stock sim (grey) and in the calibrated sim
(green) for the current held-out episode.

**With a GPU.** `antares.policy.load_lerobot_policy("<hub id or dir>")`
wraps any LeRobot `PreTrainedPolicy` (ACT, diffusion, SmolVLA) behind the
same `Policy` interface (`lerobot` is not an Antares dependency; the
wrapper is tested against a stub), dataset units in and out like the
LeRobot runtime, so the shadow and closed-loop evaluators take it as is.
What changes is the observation: an image policy needs its camera frames.
In shadow mode the observations are real, so the dataset's own video
frames (`antares.video.frames`) can go into `extra_inputs(ctx)`. In
closed loop the frame has to be rendered from the sim, which is what the
fitted camera and scene are for (see [Cameras from the logs](#cameras-from-the-logs)
and [Scene from the logs](#scene-from-the-logs)): the exported `scene.xml`
carries the table and the backdrops at the fitted camera pose, so
MuJoCo's offscreen renderer through camera `up` or `side` produces the
policy's input, or `antares.scene.sim_view` composites it without OpenGL.
The grasp-point goal is not needed then; the images carry it.

## Per-joint calibration

The shared fit gives all six servos the same gain, damping, armature,
friction and latency. That leaves the elbow as the worst joint on both demo
arms (SO-101 held-out: 1.12° vs 0.81° or less for the others), which makes
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
shared locks exactly (all four refit with the `kp` floor at 10).

| | SO-101 default | SO-101 shared | SO-101 per-joint | SO-100 default | SO-100 shared | SO-100 per-joint |
|---|---|---|---|---|---|---|
| **overall** | 6.007 | 0.736 | **0.656** | 2.744 | 0.787 | **0.538** |
| shoulder_pan | 8.049 | 0.522 | 0.446 | 3.224 | 0.356 | 0.336 |
| shoulder_lift | 6.832 | 0.813 | 0.736 | 3.540 | 0.840 | 0.662 |
| elbow_flex | 5.355 | 1.121 | **1.061** | 2.516 | 1.329 | **0.703** |
| wrist_flex | 3.797 | 0.548 | 0.371 | 1.955 | 0.436 | 0.361 |
| wrist_roll | 5.085 | 0.463 | 0.383 | 2.136 | 0.557 | 0.522 |
| evaluations | – | 200 | 200 + 1200 | – | 150 | 150 + 1200 |

With the lower `kp` floor the shared fit already reaches most of what the
per-joint stage used to add on SO-101 (0.736 shared now against 0.812
per-joint before); the per-joint stage still pays off most on the SO-100
elbow.

The per-joint stage is sensitive to its step size and budget. Held-out
score after refining each shared optimum, CMA-ES's default population
of 15, seed 0 (measured on the earlier shared locks, fit with the `kp`
floor at 50, so the absolute numbers are higher than the table above):

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
scores 1.069° on SO-101 against 1.269° for the SO-100 shared lock, and the
SO-101 per-joint lock 0.812° on SO-100 against 0.964°.


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

### Repeatable locks: best of N seeds

One CMA-ES run is deterministic for its seed but not representative. On
the same SO-101 episodes, seeds 0/1/2 land at kp 10.5 / 27 / 260 with
held-out errors of 0.74° / 0.91° / 1.06°; on SO-100 they land at kp
1427 / 14 / 13 at 0.79° / 0.57° / 0.66°
([`results/robot_specificity.md`](results/robot_specificity.md)). The
tracking error is repeatable to within about 1.4×; the parameter values
are not, because several parameter sets reproduce the arm equally well.
A parameter value read off a single lock is therefore a fit, not a
measurement.

`--seeds N` runs the shared stage from N seeds (`seed`, `seed+100`, ...),
each from the same start with the full budget, keeps the candidate with
the best held-out score, and records every candidate in the lock under
`result.seeds`. From the candidates it computes each parameter's spread
and marks a parameter **pinned** when the seeds agree on it: a log-scaled
parameter (`kp`, `kv`, `damping`, `armature`, `frictionloss`) when its
max/min across seeds is at most 2, `latency_frames` when it moves by at
most one frame. The per-joint stage, if asked for, runs once from the
winner.

```sh
.venv/bin/antares calibrate lerobot/svla_so101_pickplace --seeds 3
```

Three seeds cost three times the shared budget (about a minute on the
demo datasets) and are what a lock worth keeping should use; the default
stays at 1 so the live demo stays fast. `antares.lock.pinned_params(lock)`
returns the pinned list, and for a lock without seed spread the
conservative default `["latency_frames"]`: one frame either way costs
0.3-0.9° with everything else held fixed, so latency is the one parameter
a single fit does pin down.

Drift, diagnose and the fleet statistics now read only pinned parameters
as physical. A drift scan still reports every trend, but its verdict only
announces a parameter as drifting when the lock pins it (scores always
count), and says when an unpinned parameter trends. `--explain` only
names a cause the lock pins; fleet statistics take `pinned_only`, and
`antares fleet list` prints which parameters are pinned across the
fleet's locks. The committed locks were fitted with one seed, so for them
that is latency only.

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
shoulder_pan   0.592       0.522      0.783  PASS  <- worst
shoulder_lift  0.841       0.813      1.220  PASS
elbow_flex     1.122       1.121      1.682  PASS
...
overall        0.752       0.736      1.103  PASS
OK: lerobot/svla_so101_pickplace vs lock svla_so101_pickplace: 0.752 deg RMSE on 50 episodes (reference 0.736, limit 1.103); worst joint shoulder_pan 0.592 deg (limit 0.783)

```

Exit codes: `0` pass, `1` drifted, `2` usage error or too few episodes
(`--min-episodes`, default 5). `--json` prints the full `CheckResult`. The
same check is `POST /api/check` with
`{"dataset", "lock", "threshold_deg"?, "ratio"?, "last"?, "record"?}`.

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

### Record then check, one command

A technician at the arm should not have to assemble a `lerobot-record`
command and then a `check` command. `antares record-check` does both:

```sh
.venv/bin/antares record-check --lock so101_arm_a \
    --robot-port /dev/tty.usbmodem585A0076841 --robot-id arm_a \
    --teleop-port /dev/tty.usbmodem58760431551
```

```
recording 5 episode(s) -> recordings/arm_a/20261017-093005
  lerobot-record --robot.type=so101_follower --robot.port=/dev/tty.usbmodem585A0076841 --robot.id=arm_a ... --dataset.push_to_hub=false
... LeRobot's own prompts and keyboard controls (n: next, r: re-record, q: quit) ...
OK: recordings/arm_a/20261017-093005 vs lock so101_arm_a: 1.067 deg RMSE on 5 episodes (reference 1.054, limit 1.582)
PASS arm_a: 1.067 deg (limit 1.582)
```

It runs `lerobot-record` with the technician's terminal attached (follower
`--robot-type so101_follower`, leader `--teleop-type so101_leader`,
`--teleop-id <robot-id>_leader` by default; `--robot-id` must be the id the
arm was calibrated under in LeRobot), writes the episodes to `--root`
(default `recordings/<robot-id>/<YYYYMMDD-HHMMSS>`, so runs never overwrite
each other) and never pushes to the Hub. It then runs exactly what `antares
check <root> --lock <name>` runs, so `--threshold`, `--ratio`,
`--min-episodes` and `--json` mean the same thing, and ends with one banner
line, `PASS <robot-id>: ...` or `FAIL <robot-id>: ...`. Exit codes are the
check's (`0` pass, `1` drifted, `2` usage or too few episodes) plus `3`
when recording itself failed. `--episodes` defaults to 5 to satisfy the
check's `--min-episodes 5` and is raised with a warning when set lower.

Other knobs: `--episode-time S` / `--reset-time S` (LeRobot's
`episode_time_s` / `reset_time_s`, 60 s each by default), `--task "..."`,
`--extra=--robot.cameras=...` to pass any other `lerobot-record` flag
through verbatim, `--dry-run` to print the two commands and exit 0, and
`--skip-record` to check a directory a previous run already recorded.

Prerequisite: LeRobot with its recording scripts and Feetech motor support,
`pip install 'lerobot[core_scripts,feetech]'`. LeRobot needs Python >= 3.12
and Antares runs on 3.11, so they usually sit in separate environments;
point `--recorder /path/to/env/bin/lerobot-record` at LeRobot's when it is
not on `PATH`. The recorded directory is an ordinary LeRobot dataset
(`meta/info.json`, `data/*.parquet`): keep it as the record of that day's
check, run `antares check` or `--explain` on it again later, or upload it
with `hf upload <user>/<name> recordings/arm_a/20261017-093005 --repo-type dataset`.

### Check history

One check says whether the arm matches its sim today. A shop running arms
for weeks wants that answer per robot over time, so every `antares check`
(and `POST /api/check`) appends one JSON line to
`locks/<lock>.history.jsonl` next to the lock: when it ran, the dataset,
the verdict, the score, the limit, the worst joint and the per-joint
errors. Insufficient-data checks are recorded too, with a null score; a
check that could not run is itself a signal. `--no-record` skips it and
`--history-dir` moves the file. The check's last line is the trend so far:

```
history: 7 checks since 2026-09-30, +0.021 deg/day (r² 0.41), last drifted 2026-10-10
```

`antares history <lock>` tabulates the entries (`--last N` for the newest
N) and `antares history <lock> --json` prints what the API serves:
`GET /api/history/{lock}` is `{"lock", "entries": [...oldest first],
"trend": {"n", "first", "last", "slope_deg_per_day", "r2",
"latest_verdict", "n_drifted"}}` (404 when the lock does not exist, empty
entries and an `n` of 0 when it has no checks yet), and `GET /api/history`
lists every lock with its trend and latest entry. The slope is a
least-squares fit of the score in degrees against days since the first
check, so it measures how fast the tracking error is growing *if the
checks are comparable*; a handful of checks on different tasks or episode
counts will scatter (low r²) and the slope then says nothing about wear.
Compare like with like: the same task, the same `--last`, the same lock.

In the Action the history is a workspace file (`record-history: "false"`
turns it off), so commit it from the job or keep it as an artifact:

```yaml
- uses: actions/upload-artifact@v4
  with: { name: robot-check-history, path: locks/*.history.jsonl }
```

### One identity per robot: the registry

A lock is one arm on one day. Recalibrate and you have a new lock under a
new name, and every `check`, workflow and dashboard row that named the old
one has to be re-pointed. `locks/registry.json` gives each physical robot
one id instead (LeRobot's robot id, the `--robot-id` you record with), a
versioned lineage of the locks fitted to it, and one **active** lock:

```json
{"version": 1,
 "robots": {"arm_a": {"robot_type": "so101_follower", "active": "arm_a-20261017",
                      "locks": [{"name": "arm_a-20261001", "created_at": "...", "dataset": "...",
                                 "eval_score": 1.05, "baseline_eval_score": 4.1, "note": null,
                                 "registered_at": "..."}, {"name": "arm_a-20261017", "...": "..."}],
                      "created_at": "...", "updated_at": "..."}}}
```

The registry only references locks by name; the lock files are untouched
and an unregistered lock keeps working everywhere exactly as before.

```sh
.venv/bin/antares calibrate ./recordings/arm_a/2026-10-17 --seeds 3 --name arm_a-20261017 --robot-id arm_a
#   saved lock arm_a-20261017 -> locks/arm_a-20261017.lock.json
#   registered lock arm_a-20261017 as v2 of 2 for robot arm_a; now active (held-out 1.050 deg vs 1.120 for arm_a-20261001 (within 10%))
.venv/bin/antares robots                                  # one row per robot
.venv/bin/antares robots show arm_a                       # the lineage and the parameter drift
.venv/bin/antares robots register arm_a --lock some_lock [--activate] [--note "belt replaced"]
.venv/bin/antares robots activate arm_a arm_a-20261001    # pick any version
.venv/bin/antares robots rollback arm_a                   # back to the version before the active one
.venv/bin/antares robots remove arm_a                     # forget the robot; its lock files stay
```

```
robot  active lock     version  held-out (deg)  locks  last calibrated  last check
-----  --------------  -------  --------------  -----  ---------------  ----------------
arm_a  arm_a-20261017  v2 of 2  1.050           2      2026-10-17       ok 2026-10-17
```

**The robot reference.** Wherever a `--lock` or a `lock` is accepted,
`@arm_a` means "the active lock of robot `arm_a`": `antares check <dataset>
--lock @arm_a`, `antares history @arm_a`, `antares export @arm_a`, `drift`,
`dr`, `compensate`, `events`, `replay`, `synth`, `camera-fit`, and the API
(`GET /api/locks/@arm_a`, `POST /api/check {"lock": "@arm_a"}`,
`GET /api/history/@arm_a`, ...). The CLI and the server resolve it in one
place (`antares.registry.Registry.resolve`), and everything downstream, the
check history included, uses the lock's own name, so a recalibration or a
rollback changes what `@arm_a` points at and nothing else. `@` is not a
legal character in a lock name, so the two can never be confused; a lock
cannot be deleted through a reference. `antares record-check --robot-id
arm_a` no longer needs `--lock` once the robot is registered.

**Activation rule.** A robot's first lock becomes active. A later one is
activated when its held-out `eval_score` is not worse than the active lock's
by more than 10 % (`--max-regression`, a fraction); a lock without a score
is never activated automatically; `--activate` forces it. Otherwise the lock
is registered but left inactive and the command says why:

```
registered lock arm_a-20261024 as v3 of 3 for robot arm_a; not activated: held-out 1.480 deg is 41% worse than arm_a-20261017's 1.050 (limit 10%); force with --activate or `antares robots activate`; arm_a-20261017 stays active
```

`antares robots rollback arm_a` activates the version before the active
one, for when a new calibration passed the rule but the arm checks worse
on it. `antares history --robot arm_a` concatenates the check history of
every lock in the lineage in time order, each entry tagged with its lock
(`*` marks the active one), so the robot's record survives recalibration.

**The lineage view and the pinned caveat.** `antares robots show arm_a`
lists every version (created, dataset, held-out before → after, which
parameters are pinned, last check, note) and then a parameter-drift table
across versions. That table only includes parameters that are *pinned in
every version* (see [best of N seeds](#repeatable-locks-best-of-n-seeds)):
a single-seed fit pins only `latency_frames`, and `kp`, `kv`, `damping`,
`armature` and `frictionloss` trade off against each other at the same
tracking error, so a difference between two fits of them is not a change
in the arm. They are listed as "not comparable across fits" rather than
tabulated. Calibrate with `--seeds 3` and the pinned set grows to whatever
the seeds agreed on.

```
parameter drift across versions (pinned in every version only):
param           v1  v2  change
--------------  --  --  --------
latency_frames  2   3   +1 frame
not comparable across fits (unpinned, so each value is a fit, not a measurement): kp (v1, v2), kv (v1, v2), ...
```

The same data is `GET /api/registry` (every robot), `GET
/api/registry/{robot_id}` (`lineage` with `pinned`, `spread`, `params` and
`is_active` per version, plus `param_drift`), `POST
/api/registry/{robot_id}/register {"lock", "activate"?, "note"?,
"max_regression"?}`, `POST .../activate {"lock"}`, `POST .../rollback` and
`DELETE /api/registry/{robot_id}`. `POST /api/calibrate` takes `robot_id`
(and `activate`) and the job's `done` event carries the `registration`.
Every `GET /api/dashboard` row has `robot_id`, `is_active`, `version` and
`n_versions` (all null when unregistered), and the Fleet view groups rows by
robot: one header row per robot with the active lock's numbers and a
"v2 of 2" badge, expandable to the lineage, with unregistered locks under
"unassigned".

In the Action, name the robot instead of a lock and the job follows
whatever is active, so a recalibration never needs a workflow change:

```yaml
- uses: your-org/antares@main
  with:
    dataset: your-org/so101_arm_a_daily
    robot: arm_a            # locks/registry.json in the workspace; same as lock: "@arm_a"
```

### What changed? `--explain`

A failed check says the arm no longer matches its sim. `--explain` also
says *why*: it refits the lock on the fresh episodes as one change per
parameter on top of it (a factor on `kp`, `kv`, `damping`, `armature`,
`frictionloss`, a shift of whole frames on `latency_frames`), then tries
each change *alone* on the lock. A change is named as the cause when it
alone recovers at least 20% of what the full refit recovers. The refit uses
every other episode and both are scored on the rest, so the gap is not
overfitting.

Each change has to earn its size: the refit minimizes the error plus 0.2
deg per unit of |ln factor| (a frame of lag is nearly free), then drops any
change that lowers the error by less than 0.05 deg. That prune spends
evaluations held back from the search, so the whole diagnosis stays
within `--explain-evaluations` (default 300, at least 20) and stops when
a job is cancelled. Without the penalty, on the real arm a large `kv`
swing stood in for injected lag, and unmodified recordings showed a 0.14
deg "gain" from parameters pulling against each other. On the real SO-101
held-out episodes it now names injected lag of 1, 2 and 3 frames exactly
(every factor left at +0%), reports scaled-down motion as not explained,
and reports "no change found" on the unmodified recordings (the lock
within its usual error of its reference, and no refit improving on it
held out by more than 0.05 deg).

```sh
# no arm handy? synthesize one with known wear, then check it
.venv/bin/antares synth /tmp/worn --lock svla_so101_pickplace --wear latency_frames=1 --episodes 10
.venv/bin/antares check /tmp/worn --lock svla_so101_pickplace --explain
```

```
DRIFTED: /tmp/worn vs lock svla_so101_pickplace: 2.123 deg RMSE on 10 episodes (reference 0.736, limit 1.103); ...

what changed (refit on episodes 0,2,4,6,8, scored on 1,3,5,7,9; lock alone 2.124 deg):
param           change    alone (deg)  recovers
--------------  --------  -----------  --------  --------
kp              +0%       2.124        +0.000
...
latency_frames  +1 frame  0.232        +1.892    <- cause
all together              0.232        +1.892
lock svla_so101_pickplace on /tmp/worn: 2.124 deg, 0.232 deg after refitting the changes (100% of the excess recovered); likely cause: latency_frames +1 frame (alone recovers 1.89 deg)
```

With `--wear kp=0.6` instead it names `kp -41%`. When the refit recovers
less than half of the excess error, the change is not one the sim's
dynamics can express (a sensor offset, contacts, a harder task), and the
diagnosis says that instead of guessing a parameter. The parameters trade
off against each other, so read several named causes as "these moved
together", not as independent measurements.

`python examples/drift_demo.py` runs a ten-week wear story on the MuJoCo
arm (gain fading 5% a week, friction building 20% a week, one more frame of
lag from week 6). The check fails from week 6, and the named causes match
the wear applied: the extra frame of lag every time, the friction with it,
and by week 10 the weaker gain too (the friction has grown past the x4 the
search allows, so the fit leans on the gain for the rest):

```
week  error (deg)  limit  verdict  named cause                                           actual wear
0     0.231        1.103  ok       -                                                     none
2     0.489        1.103  ok       -                                                     kp -10%, frictionloss +44%
4     1.067        1.103  ok       -                                                     kp -19%, frictionloss +107%
6     3.643        1.103  DRIFTED  latency_frames +1 frame, frictionloss +157%           kp -26%, frictionloss +199%, latency_frames +1 frame
8     4.754        1.103  DRIFTED  latency_frames +1 frame, frictionloss +300%           kp -34%, frictionloss +330%, latency_frames +1 frame
10    6.312        1.103  DRIFTED  latency_frames +1 frame, frictionloss +295%, kp -50%  kp -40%, frictionloss +519%, latency_frames +1 frame
```

Causes are only named when the drift is explained,
so the summary, the table and the Action's output agree; a lock without
a held-out score of its own cannot be judged and says so. `POST
/api/check` takes `"explain": true` and adds a `diagnosis` object to a
drifted result (or `diagnosis_error` when the refit could not run, the
verdict standing either way); `POST /api/diagnose` runs the same refit
as a cancellable job with progress events, on its own MuJoCo arm so the
other endpoints keep answering. The GitHub Action takes `explain: "true"`
and sets a `cause` output.

### Alerts

A gate that fails only in a CI log is easy to miss. `--notify` posts the
verdict to a webhook, so the team hears about a drifted arm where it talks:

```sh
.venv/bin/antares check ./recordings/2026-10-17 --lock my_arm --notify "$ANTARES_WEBHOOK_URL"
.venv/bin/antares check ./recordings/2026-10-17 --lock my_arm --notify-when failure  # URL read from $ANTARES_WEBHOOK_URL
.venv/bin/antares notify-test --url "$ANTARES_WEBHOOK_URL"                            # posts a synthetic OK to confirm the hook
```

```
DRIFTED my_arm on ./recordings/2026-10-17: 2.10 deg RMSE (limit 1.58)
worst joint elbow_flex: 3.21 deg (limit 3.05)
failed joints (2): elbow_flex 3.21/3.05, shoulder_lift 1.60/1.45
10 episodes, 3000 frames, 2026-10-17T06:04:12+00:00
```

`--notify-when` is `drifted` (default), `failure` (drifted, too few
episodes, or a check that could not run at all) or `always`;
`--notify-dry-run` prints the payload instead of posting it. A webhook
that is down only prints `warning: alert not sent: ...` on stderr; the
exit code is the check's, never the alert's.

For Slack, create an app with Incoming Webhooks switched on, add a webhook
to the channel you want and copy its `https://hooks.slack.com/services/...`
URL. Antares recognises that host and posts Block Kit (a header, the
summary, one field per joint); `--notify-format slack|json` overrides the
detection. Any other URL gets a generic JSON POST:

```json
{
  "event": "antares.check",
  "verdict": "drifted",
  "text": "DRIFTED my_arm on ./recordings/2026-10-17: 2.10 deg RMSE (limit 1.58)\nworst joint ...",
  "result": {"passed": false, "verdict": "drifted", "score": 2.1, "limit": 1.58, "per_joint": {"...": {}}, "...": "..."}
}
```

`result` is the full `CheckResult`, as `--json` prints it. A check that
could not run (missing lock, unreadable dataset) posts `"verdict": "error"`
with `"result": null` and an `"error"` string, for `failure` and `always`
only. `POST /api/check` takes the same options as `notify_url`,
`notify_when` and `notify_format`, posts server-side after the check, and
adds `notified` (`true`, `false`, or `null` when no URL was given) and
`notify_error` to the response. In the Action, pass the URL from a
repository secret so it never appears in the workflow file or the logs:

```yaml
- uses: your-org/antares@main
  with:
    dataset: your-org/so101_arm_a_daily
    lock: so101_arm_a
    webhook: ${{ secrets.ANTARES_WEBHOOK_URL }}   # Slack incoming webhook or any JSON endpoint
    notify-when: failure                          # optional: drifted (default), failure, always
```

## Contact detection from the residual

Once the sim reproduces the arm's free-space motion (the SO-101 lock
tracks held-out episodes to 0.74 deg RMSE), whatever is left in
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
| grasps / events | 68 / 112 | 89 / 196 |
| recall (grasps with an event in +-15 frames) | 100% (100%) | 54% (71%) |
| precision (events in a grasp window) | 53% (61%) | 33% (41%) |
| events far from any gripper transition | 33% | 44% |
| residual energy inside grasp windows | 20% (19%) | 11% (10%) |
| frames inside grasp windows (chance) | 18% (16%) | 14% (14%) |
| lift | 1.12x (1.17x) | 0.78x (0.75x) |

Read these with care. On the SO-101 the residual does concentrate around
grasps, but weakly (1.1x chance), and the perfect recall comes from
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
| `lerobot/svla_so101_pickplace` | 4 frames | 0.736 | 0.923 | 0.108 | 7–10 per episode |
| `lerobot/svla_so100_pickplace` | 3 frames | 0.787 | 0.369 | 0.120 | 3–5 per episode |

On the SO-101 the pure shift makes tracking worse than the raw commands.
Its lock's `kp` of about 10 makes the sim servo respond as a slow lag, not
a pure 4-frame delay, so shifting the commands alone does not help; ILC
does, with more iterations than on the stiffer SO-100.


Feasibility (also sim-only): the compensated commands stay within the
actuator range, exceed the original commands' range by at most 0.8°, and
their peak velocity is 1.04× (SO-101) and 1.11× (SO-100) the fastest
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
one from the default parameters and writes one lock per robot plus a report.
A dataset the default parameters already track to under 0.5 deg is
rejected after its fit and gets no lock: no physical arm in the fleet comes
within 1.2 deg of the MJCF defaults, so such a log was most likely recorded
in a simulator under a name that does not say so:

```sh
.venv/bin/antares fleet scan --limit 60 --budget-s 3300   # locks/fleet/*.lock.json, fleet.json
.venv/bin/antares fleet list                              # the table below
curl localhost:8000/api/fleet          # fleet.json
curl localhost:8000/api/fleet/cross    # the partial cross matrix
```

The committed scan (`locks/fleet/fleet.json`, 2026-10-18, `kp` floor 10)
inspected 166 Hub hits, accepted 60 and calibrated all of them in 51
minutes, with no failures; **58 robots** are kept. The 108 rejections were
mostly a second dataset from the same account (30), v1 datasets without
`meta/info.json` (24), bimanual or non-SO arms (16), v2 datasets stored as
one parquet file per episode (15), simulation recordings (10, two of them
caught by the default-sim rule above), too few episodes (7) and merges (3).

* **Every robot improved.** Held-out tracking RMSE went from a median of
  3.45 deg to 1.09 deg; the median robot closed **63 %** of its gap (IQR
  57–70 %, range 6–87 %).
* **The fitted parameters are all over the place**, which is the point of
  calibrating per robot. Latency spans 0–5 frames (0–167 ms at 30 fps),
  median 3 frames (100 ms), IQR 3–4. Actuator gain `kp` spans 10–4582
  (2.7 decades) with a median of 81 and IQR 17–571; the MJCF default is
  998. Ten robots fit `kp` under 12, at or near the floor of 10, so the low
  end is still set partly by the bound. Damping, armature, friction and
  `kv` each span two to three decades. The fleet's own fit of
  `lerobot/svla_so101_pickplace` (150 evaluations) lands at `kp` 15 and
  0.771 deg, next to the demo lock's `kp` 10.5 and 0.736 deg at 200.
* **Per-robot calibration beats borrowing a neighbour's.** On a 6-robot
  cross matrix (the two demo arms plus the next four), a robot scored on
  its own lock averages 1.00 deg, on another robot's lock 1.54 deg (1.53x;
  per-robot median 1.53x), and on the defaults 4.60 deg (4.6x). Any
  calibrated lock beats the defaults on all 30 (robot, other lock) pairs,
  and 5 of 6 robots do best on their own lock.
* **Outliers worth a look.** `kmg0620/so101_bomb_pick_place_v1` starts at
  21.3 deg and only gets to 14.8; no lock in the fleet explains it, which
  points at a recording in different units or a badly calibrated arm
  rather than at the servos. `aivanni/so101-algebra` is similar (12.6 to
  8.0 deg, with `kp` at 4582 and no latency). `codywang/so101_pickup_test`
  closes only 6 % of its gap and is left at 6.3 deg, mostly in
  `wrist_roll` (13.7 deg), which smells like a joint-offset problem rather
  than servo dynamics. `OkzRIKU/so101-turn-up-drone` closes 25 %.
  `dongdong123123/so101_stack_cubes` fits an armature 58x below the fleet
  median, at the bottom of the search range. Score outliers are kept out
  of the cross matrix so one unexplained arm does not dominate it.

## Gravity load and servo deadband

The contact section above blames the elbow's leftover error on pose. `antares
sag` checks that: it replays a lock's held-out episodes and regresses each
joint's residual on the torque that holds the real pose against gravity (with
velocity as a covariate), prints the residual by torque bin and by motion
(still, moving up, moving down), and converts the slope into an implied
gravity load.

```bash
antares sag lerobot/svla_so101_pickplace --lock svla_so101_pickplace
antares calibrate lerobot/svla_so100_pickplace --fit-gravity      # + gravity_scale
antares calibrate lerobot/svla_so100_pickplace --fit-deadband     # + servo deadband
```

On both demo arms the elbow error is a step, not a slope. On SO-101 it sits
near +0.9° in the two most loaded torque bins and near -0.5° in the other
three, which differ by only 0.03 N·m. The elbow's load barely changes in
pick-and-place logs (std 0.02-0.04 N·m), so these datasets cannot measure
elbow sag directly. `sag` reports no implied scale for a joint whose torque
varies by less than 0.05 N·m.

Two optional terms can be fitted, and both are left out of a lock when unused,
so old locks load and replay unchanged:

* `gravity_scale` scales the gravity vector (exported as `<option gravity>`).
  A soft servo's static droop is about `gravity_scale * g(q) / kp`, so this
  sets the droop apart from how fast `kp` makes the joint track.
* `deadband` (rad, shared or per joint) is a band of position error in which
  the servo applies no torque. It stands for the STS3215 firmware deadzone
  plus gear backlash. MJCF cannot express it, so it travels in the export
  sidecar.

`POST /api/calibrate` takes the same two flags (`fit_gravity`, `fit_deadband`)
and the UI's calibrate panel has a checkbox for each; a lock's parameter table
shows the two terms when the lock carries them.

Held-out RMS (deg) at an equal budget of 300 shared evaluations. Seed 0 and
seed 1 use different train/eval splits.

| dataset | seed | dynamics only | + gravity | + deadband | + both |
|---|---|---|---|---|---|
| SO-101 | 0 | 0.701 | 0.690 | 0.997 | 0.950 |
| SO-101 | 1 | 1.137 | 0.588 | | |
| SO-100 | 0 | 0.760 | 0.524 | 0.540 | 0.577 |
| SO-100 | 1 | 0.800 | 0.537 | | |

The gravity term wins all four pairs. It works by keeping the fit off the
gain ridge (see per-joint calibration): without it, three of four fits drove
`kp` to between 730 and 2,600, and with it every fit landed between 19 and 36.
The fitted scale is 1.3 to 1.9, which reads as a heavier arm than the MJCF.
That number is inferred to stand in for whatever else makes the real servo
droop, not measured as a mass. The deadband helps SO-100 about as much as
gravity does, but makes SO-101 worse, so it stays opt-in. A per-joint deadband
fit (600 evaluations) put the SO-101 elbow's band at 1.4°, but scored 0.674°
held-out against 0.593° without it.

#### Fleet dashboard

The UI's **Fleet** toggle (top bar, next to Viewer) is the one screen for a
company with many arms: stat tiles (robots, median gap closed, robots drifted
at their last check, robots never checked) over a sortable table with one row
per lock in the lock directory: dataset, calibration date, default → calibrated
held-out error, gap closed, fitted latency in frames and ms, the last `antares
check` (date, verdict, score against its limit), the trend of those checks in
deg/day with its r², the worst body joint after calibration, and the pinned
parameters when a seeded lock records them. Clicking a row expands it to a
sparkline of every check's score over time against the lock's reference and
limit, and the per-joint error of the last check next to the lock's own. All of
it comes from one route:

```sh
curl localhost:8000/api/dashboard   # {"robots": [...], "fleet": <fleet.json summary or null>, "generated_at": ...}
```

Each robot joins the lock with its `.drift.json` verdict, whether the fleet
scan produced it, and its latest check and trend from the check history (the
view fetches `/api/history/<lock>` on expand; a 404 means never checked).


## Sensor calibration

LeRobot normalizes the commands (the leader arm's readings) and the
measured state (the follower's) with two separate per-arm calibrations, so
the same number can mean slightly different angles on each side. No
dynamics parameter can model that, so a plain fit bends stiffness and
latency to hide it. `--fit-sensor` fits a per-joint
`command = gain * logged + bias` (a `CommandMap`, dataset units, gripper
left at identity) and stores it in the lock:

```sh
.venv/bin/antares calibrate lerobot/svla_so101_pickplace --fit-sensor --name so101_sensor
.venv/bin/antares calibrate lerobot/svla_so101_pickplace --fit-sensor --sensor-mode kinematic
```

Two ways to fit it (`--sensor-mode`, or `sensor_mode` in
`POST /api/calibrate`):

* **`sim`** (default): Levenberg-Marquardt on the replay residual with the
  dynamics held fixed, alternating with a CMA-ES refit of the dynamics
  under the new map, then a last map fit.
* **`kinematic`**: a least-squares line through frames where command and
  joint have both been still for 10 frames, with no sim, frozen before the
  physics fit so it cannot absorb dynamics error. Its catch: a still joint
  under gravity sags below its command by an amount the sim models through
  `kp`, and this fit puts that sag into the bias.

The table compares against a dynamics-only fit given the **same** budget,
counting every pass over the training episodes, the map fits' included:
it is refit round for round under the identity map, plus one more refit
with the simulations the map fits spent, so a gain is not just extra
evaluations. The lock's `n_evaluations` and stages count the map fits
the same way. It also works with `--per-joint` (later rounds refit in
the per-joint space directly, since the shared stage would only average
the per-joint values back together).

Synthetic check (`scripts/sensor_synthetic.py`): a hidden MuJoCo arm with
the SO-101 lock's dynamics, 0.2-unit state noise and teleop-style pauses,
10 train / 4 held-out episodes, held-out RMSE in degrees:

| hidden arm | default sim | dynamics only | `sim` mode | `kinematic` mode |
|---|---|---|---|---|
| calibration mismatch (bias up to 3 units, gain ±4%) | 7.40 | 2.34 | **0.51** | 1.56 |
| no mismatch (control) | 7.15 | 0.46 | **0.40** | 1.58 |

(Measured with the current SO-101 lock, whose refit `kp` of about 10
makes the arm sag a lot under gravity.) `sim` mode recovers the hidden
bias to within 0.06 units and the gain to within 0.004, and on the arm
with no mismatch keeps the map within 0.08 units of identity. `kinematic`
mode's sag caveat bites on this arm: it reads the gravity sag as a bias
(+3.0 units on the shoulder lift and +2.6 on the elbow of the arm with
no mismatch, and the hidden -3.0 shoulder-lift bias comes out as -0.1),
and the dynamics then refit around that, so it ends worse than fitting
the dynamics alone.
On the two demo datasets (20 train / 10 held-out episodes, seed 0, two
rounds of 150 evaluations, with the wider `kp` range), held-out RMSE in degrees.
"Dynamics only" is a fresh shared fit given the same budget under the
identity map (300 evaluations plus the map fits' simulations), which is
why it beats the committed shared locks (0.736 SO-101 at 200 evaluations,
0.787 SO-100 at 150). The `sim` columns are the committed
`locks/svla_so101_pickplace_sensor_sim.lock.json` and
`locks/svla_so100_pickplace_sensor_sim.lock.json` (`_sim` is the sensor
mode, not a simulated robot), which a test replays against their stored
scores:

| | SO-101 dynamics only | SO-101 `sim` | SO-101 `kinematic` | SO-100 dynamics only | SO-100 `sim` | SO-100 `kinematic` |
|---|---|---|---|---|---|---|
| **overall** | 0.708 | **0.624** | 0.633 | 0.745 | **0.466** | 0.480 |
| shoulder_pan | 0.407 | 0.373 | 0.368 | 0.326 | 0.317 | 0.311 |
| shoulder_lift | 0.801 | 0.651 | 0.665 | 0.854 | 0.462 | 0.504 |
| elbow_flex | 1.145 | 1.027 | 1.049 | 1.248 | 0.683 | 0.701 |
| wrist_flex | 0.485 | 0.425 | 0.420 | 0.416 | 0.312 | 0.308 |
| wrist_roll | 0.394 | 0.385 | 0.383 | 0.458 | 0.458 | 0.465 |

Both real arms carry a real mismatch, mostly on the elbow (`sim` bias
+1.6 units on SO-101 and +2.2 on SO-100, gain about 0.98) and, on the
SO-100, the shoulder lift (+1.1 units, gain 1.010). The fitted map cuts
the SO-100's held-out error by 37% and the SO-101's by 12%. `kinematic`
puts more into the gravity-loaded joints (an SO-101 elbow bias of +3.5),
as its sag caveat predicts, and is never better overall.

Locks fitted this way carry a `command_map`. Replay, the viewer, `cross`,
`check`, residuals and events, `compensate` (whose `--out` commands come
back in raw units), `export` (in the sidecar) and the drift and DR refits
(which hold the map fixed) all use it. Locks without one behave as before.

## The task and the sensors, from the logs

A lock matches the arm. A policy trained in sim also needs the rest of the
world to match: where objects are and how noisy the joint readings are.
Both can be read off the same LeRobot logs (cameras are covered in
[Cameras from the logs](#cameras-from-the-logs)).

### Where objects were picked and placed

```sh
.venv/bin/antares task lerobot/svla_so101_pickplace     # -> locks/<name>.task.json
```

Each time the gripper command closes, forward kinematics of the measured
joints at the MJCF's `gripperframe` site gives where the gripper tip was;
each time it lets go gives where the object ended up. The jaw says whether
anything was in it: with an object between them the jaws stop open by its
width, on a miss they close all the way (the dataset's empty-closed level
of the measured jaw; the leader's closed command has a level of its own,
since the two arms are calibrated separately, and is only ever compared
with that). The jaw is read once it has settled, 0.3 s after the command
closes; a close the command undoes sooner is a blip across the hysteresis
band, not a grasp. Each close is a *pick*, a *miss* (empty, before
anything was picked), or an *idle* close (empty, after letting go). Only
picks that moved the object more than 3 cm give a place position.

On the two demo datasets (gripper tip in the base frame):

| | picks | carried | missed | idle | pick x, y (cm) | place x, y, z (cm) | jaw on the cube |
|---|---|---|---|---|---|---|---|
| SO-101 | 51 in 50 episodes | 50 | 0 | 27 | 3..24, -14..13 | 2.8, -15.3, 7.0 (±1) | -2.3° |
| SO-100 | 50 in 49 episodes | 49 | 1 | 49 | 19..33, -8..14 | 8.4, -12.4, 8.0 (±1) | +5.8° |

Picks spread over a 20 x 27 cm area on SO-101 and land in one spot about
7 cm up (on top of something). That is the distribution a sim environment
for this task should draw from, and `TaskDistribution.sample(n)` does
(pick, place and start pose, with 1 cm of jitter).
`antares export <lock> --task <name>` adds a free 3 cm cube at the mean
pick and a marker at the mean place to `scene.xml`;
`antares.task.set_task(model, data, instance)` resets a scene to one draw.

The SO-101 pick heights come out 0.3-1.0 cm below the table plane, so the
lock's kinematics put the tip slightly low there; the exported cube is
lifted to rest on the floor.

### Joint-reading noise

```sh
.venv/bin/antares noise lerobot/svla_so101_pickplace    # -> locks/<name>.noise.json
```

While a joint's command has not moved for half a second, whatever its
reading still does is sensor noise. Pooled over those stretches:

| joint | SO-101 std | lag-1 corr | step | SO-100 std | lag-1 corr | step |
|---|---|---|---|---|---|---|
| shoulder_pan | 0.030° | 0.33 | 0.088° | 0.045° | 0.78 | 0.097° |
| shoulder_lift | 0.013° | 0.79 | 0.084° | 0.060° | 0.51 | 0.088° |
| elbow_flex | 0.035° | 0.80 | 0.087° | 0.017° | 0.86 | 0.085° |
| wrist_flex | 0.066° | 0.67 | 0.081° | 0.018° | 0.77 | 0.084° |
| wrist_roll | 0.030° | 0.59 | 0.078° | 0.046° | 0.39 | 0.141° |

The readings barely move: the noise is smaller than one encoder step
(about 0.09°), so a parked joint mostly reads the same value. The common
habit of adding ~0.5-1° of white Gaussian noise to joint observations is
10-30x too much for these arms. `NoiseModel.apply(q)` adds noise of the
measured size and colour (AR(1)) to simulated readings and rounds them to
the same steps, on the same lattice: the model also keeps one real reading
per joint (`origin_rad`) because the unit map's offset puts the dataset's
lattice off the plain multiples of the step in radians.


### Use your arm's LeRobot calibration

Everything above infers the meaning of the dataset's numbers from the
MJCF: `UnitMap.from_model` spreads -100..100 over each joint's modelled
range. But LeRobot already measured that for your arm. `lerobot-calibrate`
writes one file per follower with, for every motor, the homing offset it
set and the encoder range (`range_min`/`range_max`, in 4096-tick STS3215
steps) the joint was moved through, and that file defines the dataset's
normalized values exactly:

```
~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<robot.id>.json   # LeRobot 0.4.x
~/.cache/huggingface/lerobot/calibration/robots/so_follower/<robot.id>.json      # LeRobot >= 0.5
```

(`$HF_LEROBOT_CALIBRATION` moves the root; both directory names are tried.)
Hand it to Antares by path or by the `--robot.id` you gave LeRobot:

```sh
.venv/bin/antares inspect my/dataset --robot-id my_awesome_follower_arm      # diagnose first
.venv/bin/antares calibrate my/dataset --robot-id my_awesome_follower_arm    # or --calibration PATH
.venv/bin/antares calibrate my/dataset --calibration ~/.cache/huggingface/lerobot/calibration/robots/so_follower/arm.json --fit-sensor
curl "localhost:8000/api/datasets/calibration-check?dataset=my/dataset&robot_id=my_awesome_follower_arm"
```

`antares/lerobot_calib.py` turns the file into the arm's own unit map:
`rad(n) = mid + k * (present(n) - 2047)`, with `present(n)` LeRobot's inverse
normalization (`(n + 100) / 200 * (range_max - range_min) + range_min`),
`k = 2*pi/4096` per tick, the half turn 2047 the pose LeRobot's homing made the
calibration pose read, and `mid` the MJCF mid-range that pose is taken to be
(LeRobot asks for the arm in the middle of its range; `anchor="range"`
centres the recorded range instead). The gripper is anchored at its closed
stop (`range_min` = MJCF `lo`), one unit = span/100 ticks. `drive_mode` flips
the direction; the homing offset is already folded into the ranges LeRobot
recorded and is kept for the record. The module docstring has the LeRobot
sources and versions.

What it buys:

* **An exact unit map.** The replay is driven to, and reads back, the
  physical angles this arm's numbers mean, instead of the model's guess:
  gravity, forward kinematics, the viewer's ghost, camera fits and exports
  all see the right pose. The lock records it (`calibration`: source path,
  robot id, sha256, the per-joint `range_min`/`range_max`/`homing_offset`,
  the map and its disagreement with the model in degrees), and `replay`,
  `check`, `/api/replay` and the lock-based refits rebuild the map from the
  lock, so it stays self-contained.
* **A measured starting point for `--fit-sensor`.** Under the arm's own map
  the prior is identity: the follower un-normalizes every command and
  normalizes every reading with the same ranges, so the file leaves no
  command-vs-state disagreement to express, and the fit measures only what
  the file cannot (servo load error, dead band).
* **A diagnosis instead of a bent servo.** `inspect --calibration` prints,
  per joint, the calibration map minus the model map in degrees at
  -100/0/+100 and flags anything over 2 degrees with "the model's assumed
  range does not match this arm's calibration; use --calibration when
  calibrating". A wrist roll calibrated as LeRobot's full turn (0..4095) over
  a model range of 320 degrees, or a range the arm never reached one end of,
  shows up here as a scale or offset in degrees rather than as fitted
  dynamics. A caveat on what it fixes: the follower closes its loop in its
  own ticks, so command and state share the map and a map error leaves the
  dataset-unit tracking residual almost untouched; an offset-looking
  residual such as `codywang`'s wrist roll is diagnosed by this table, not
  removed by it.

Only the **follower's** file matters. A teleoperation dataset's `action` is
the leader's normalized reading, but the follower un-normalizes it in its
own calibration before writing `Goal_Position`, and `observation.state` is
the follower's reading in that same calibration; the leader's file never
enters what the arm did, so it is not needed for replay. LeRobot >= 0.5
records body joints in degrees by default (`use_degrees=True`); the module
handles that normalization (`norm_mode="degrees"`) but the CLI assumes the
-100..100 range mode Antares replays. Community Hub datasets almost never
ship their calibration file and `meta/info.json` does not embed it, so this
is for your own arm; without a file, nothing changes.

## Cameras from the logs

Every LeRobot dataset ships video from fixed cameras, and the joint angles
are known at every frame. `antares camera-fit` solves for the camera pose
that makes the sim arm overlap the real one, so the UI can draw the sim
arm's outline over the real video and an exported MJCF carries a
`<camera>` per fitted camera.

```sh
.venv/bin/pip install -e ".[video]"      # PyAV (AV1 via dav1d), OpenCV, SciPy
.venv/bin/antares camera-fit lerobot/svla_so101_pickplace --camera up --lock svla_so101_pickplace
.venv/bin/antares camera-list svla_so101_pickplace
```

How it works, with no OpenGL and no learned segmenter:

1. **Frames.** Only the one video file an episode lives in is downloaded;
   `meta/episodes` maps episode frames to timestamps in it. Decoded frames
   are cached under `.cache/antares/frames/`.
2. **Real silhouette.** The camera is static and the arm moves, so the
   per-pixel median over frames spread across episodes is the background.
   A Lab colour histogram of moving pixels against the background then
   recovers what subtraction cannot see (the static base, the rest pose
   that leaks into the median), restricted to regions the moving arm
   touches in most frames so same-coloured clutter is dropped. The contrast
   sign (black arm on a white table for the SO-101, orange on a dark table
   for the SO-100) is learned from the frames.
3. **Sim silhouette.** Each visual mesh is reduced to its convex hull once;
   `Arm.fk` poses the hulls, a pinhole camera (MuJoCo convention) projects
   them, and `cv2.fillConvexPoly` fills each one.
4. **Fit.** CMA-ES maximises the mean IoU of the two silhouettes over 40
   frames from 4 episodes, searched in look-at coordinates (target point,
   distance, azimuth, elevation, roll, fov). 3000 random cameras on a
   hemisphere over the arm are scored at 80x60 first, the 8 best distinct
   ones get a short CMA-ES run at 160x120, and the best is refined at full
   resolution. 12 frames from other episodes are scored as a held-out
   check. About a minute per camera.

The demo datasets, 40 fit frames / 12 held-out frames each:

| dataset | camera | fit IoU | held-out IoU | notes |
|---|---|---|---|---|
| svla_so101_pickplace | `up` | 0.82 | 0.83 | clean; residual is hull slack and the real gripper reaching lower than the sim's |
| svla_so101_pickplace | `side` | 0.58 | 0.72 | the arm leaves the top of the frame in many fit frames (per-frame IoU down to 0.08) |
| svla_so100_pickplace | `top` | 0.63 | 0.60 | black servo and gripper parts vanish against the black table, so the real mask is the orange and white parts only |

Fits are saved as `locks/<lock>.camera.json` (frames used, per-frame IoUs)
and a compact copy in the lock's `cameras` field, which `antares export`
writes into `<worldbody>` as `<camera name="up" pos=... quat=... fovy=.../>`.
The API: `POST /api/camera/fit` (a job, kind `camera`), `GET /api/camera?lock=`,
`GET /api/camera/cameras?dataset=`, and `GET /api/camera/overlay?dataset=&camera=&episode=&frame=&lock=`
returns the real frame (JPEG data URL), the sim outline polygons, the
segmented arm's polygon and their IoU. The UI's Camera section shows that
overlay at the playhead whenever playback is paused.

Caveats: silhouettes pin the viewing direction well but trade distance
against field of view (a farther camera with a narrower fov looks nearly
the same), so expect the position to be loose along the optical axis;
wrist cameras are skipped; the sim is the SO-101 model for both datasets.

### Video delay

```sh
.venv/bin/antares camera-delay lerobot/svla_so101_pickplace --camera up --episodes 8
```

A policy trained on images and joint states assumes both show the same
instant. `camera-delay` measures how many frames a camera's video lags the
joint log: the share of pixels that change between frames rises and falls
with joint speed, so cross-correlating the two over +-10 frames and
refining the peak gives the lag per episode (median reported). Any camera
works, wrist cameras included. On the demo datasets, 8 episodes each
(positive: video is late):

| dataset | camera | median lag | per-episode range | peak corr |
|---|---|---|---|---|
| svla_so101_pickplace | `up` | -0.37 frames (-12 ms) | -0.92..+0.15 | 0.75-0.91 |
| svla_so101_pickplace | `side` | -0.33 frames (-11 ms) | -0.72..-0.13 | 0.66-0.92 |
| svla_so100_pickplace | `top` | +2.88 frames (+96 ms) | +2.13..+2.92 | 0.54-0.65 |
| svla_so100_pickplace | `wrist` | +1.96 frames (+65 ms) | +1.46..+2.95 | 0.25-0.45 |

The SO-101 cameras match the joint log to within half a frame. The SO-100
top camera is consistently about 3 frames late, so a sim for that setup
should show the policy the arm as it was ~100 ms earlier. The wrist number
is weak (low correlation, since the whole view moves with the arm) and is
best read as "late, by a similar amount".

## Scene from the logs

With the cameras fitted, the rest of what they see is static, and the
logs say where it is. `antares scene-fit` recovers the table, the
workspace and the camera backdrops, so the exported `scene.xml` looks
like the real cell and the UI can show the sim's own view next to the
real frame.

```sh
.venv/bin/antares scene-fit lerobot/svla_so101_pickplace --lock svla_so101_pickplace
.venv/bin/antares export svla_so101_pickplace      # scene.xml now has the table and backdrops
python -m mujoco.viewer --mjcf export/svla_so101_pickplace/scene.xml   # look through camera "up"
```

How it works (`antares/scene.py`, FK and OpenCV only):

1. **Table height from grasps.** The gripper command closing is a grasp
   (`antares.events`), and at a grasp the fingertips are on an object that
   sits on the table. `Arm.fk` at every grasp instant gives the fingertip
   (the MJCF's `gripperframe` site, between the jaws) in the base frame.
   Grasps within 15 mm of the lowest decile of fingertip height are "on
   the table"; the rest are re-grasps in the air or at a container. The
   median of that cluster minus an **assumed object half-height of 15 mm**
   is the table. The assumption is stated in the result, not hidden: the
   raw fingertip-height percentiles, the lowest hull vertex of the jaws
   at each grasp, and the lowest fingertip over every frame of every
   episode (the arm never goes below the table) are all saved alongside.
2. **Plane normal.** The SO-101 base is bolted to the table, so the plane
   is **assumed horizontal**. A plane is still fitted through the on-table
   grasp points and its tilt reported; it replaces the assumption only
   when it exceeds 5 degrees with the points spread at least 10 cm in both
   x and y.
3. **Workspace.** The bounding box and xy convex hull of the fingertip
   positions over all episodes, plus the grasp and release points: where
   objects were picked and where they were put down.
4. **Backdrops.** Each fitted camera's background model (the per-pixel
   median, so the moving arm is gone) is saved as
   `locks/scene/<lock>_<camera>_backdrop.jpg`. The estimated table
   rectangle (workspace plus 10 cm) is projected into the camera with
   `antares.project` and compared with the dominant colour region of the
   backdrop (k-means on Lab, k=4, largest component): the honest check
   that the geometric table and the visual one agree.

The demo datasets, with their committed locks and cameras:

| dataset | table z | on-table grasps | lowest fingertip ever | per-episode min spread | tilt | workspace (cm) | picks / places |
|---|---|---|---|---|---|---|---|
| svla_so101_pickplace | -22.1 mm | 51 of 68 at -7.1 mm (p10 -8.7, median -6.6, p90 +66.1) | -23.7 mm | 16 mm | 0.6° | 24 x 33 x 12 | 68 / 68 |
| svla_so100_pickplace | -10.6 mm | 39 of 89 at +4.4 mm (p10 +2.1, median +74.3, p90 +95.9) | -5.7 mm | 17 mm | 1.4° | 27 x 31 x 14 | 89 / 89 |

Heights are in the base frame (z = 0 at the base body origin; the base
mesh's underside is at -2.4 mm). Both fits land where a table under the
base should be. The SO-100 grasps are bimodal: half close at the table
and half 7-9 cm up, at the container the objects are dropped into (its
releases sit at +53 to +88 mm). The SO-101's lowest-ever fingertip
(-23.7 mm, one episode) is 1.6 mm below its table estimate, so the
15 mm half-height is on the small side there.

Table plane against the backdrop image, per camera:

| dataset | camera | plane on table colour | region covered | IoU | note |
|---|---|---|---|---|---|
| svla_so101_pickplace | `up` | 0.85 | 0.64 | 0.57 | the far edge of the projected rectangle coincides with the real table edge; the colour region also takes in the near-white wall |
| svla_so101_pickplace | `side` | 0.96 | 0.55 | 0.54 | same; the real table is larger than the workspace rectangle, which caps the coverage |
| svla_so100_pickplace | `top` | 0.44 | 0.84 | 0.40 | the lighting gradient splits the black table into two colour clusters, so the region is the evenly lit half |

The numbers are what they are: a projected rectangle sized to the
workspace against a colour blob. The composite below is the better check.

Outputs: `locks/<lock>.scene.json` (the full fit), the backdrops, and a
compact `scene` entry in the lock (table height, normal, workspace box,
counts, backdrop paths; old locks load with none). `antares export` then
writes into `scene.xml` a translucent `<geom type="plane">` at the table
height sized to the workspace, and per camera a `2d` texture of its
backdrop on an unlit plane 3 m down the optical axis, sized to fill the
camera's view, so `python -m mujoco.viewer` looking through `up`, `side`
or `top` shows the real cell behind the sim arm (checked with MuJoCo's
offscreen renderer; the translucent table is what lets a top-down camera
see its own backdrop through it). The checker floor is dropped when a
table exists.

**Sim view.** `GET /api/scene/view?dataset=&camera=&episode=&frame=&lock=`
composites what a policy camera would see in the sim, with no OpenGL:
the backdrop, the table rectangle as a faint tint, and the sim arm's
hull polygons filled far to near with flat shading (nearer parts
lighter, a per-body step so neighbouring parts read as separate,
outlined). It returns `real_image` and `sim_image` (JPEG data URLs),
the IoU of the sim arm against the segmented real one, and the table
polygon. The UI's Camera section has a Scene block: "Fit scene from
logs", the table and workspace numbers, and a real + outline | sim view
| side by side toggle on the viewer; the 3D viewer draws the table as a
thin dark plane and the workspace as a faint box. Other routes:
`POST /api/scene/fit {dataset, lock}` (synchronous, a few seconds) and
`GET /api/scene?lock=`.

Caveats: the median background keeps whatever the arm did most, so the
SO-101 backdrops carry a ghost of its rest pose and the SO-100's the base;
the object half-height is a guess; the workspace is where the fingertips
went, not the table's extent; the grasp label is the command, not contact.

## Objects from the video and the residual

With the camera and the table plane fitted, anything that sits on the
table and shows up in the camera can be put in 3D: its pixel ray hits the
plane at the object's height. `antares objects-fit` finds the one object
a pick-and-place dataset moves, follows it through every frame of every
episode, checks the camera against the kinematics, and tries to weigh it
from how the arm sags while carrying it.

```sh
.venv/bin/antares objects-fit lerobot/svla_so101_pickplace --lock svla_so101_pickplace   # -> locks/<lock>.objects.json
.venv/bin/antares payload lerobot/svla_so101_pickplace --lock svla_so101_pickplace       # the per-episode mass table
```

How it works (`antares/objects.py`, `antares/payload.py`):

1. **Foreground.** Per frame, the camera's background model gives the
   moving pixels; the sim arm's silhouette at the logged joint angles,
   dilated by 14 px (hull slack, the real gripper reaching lower), is
   subtracted. What is left is what the arm moved: the object, its shadow,
   bits of the real arm the hulls miss (the SO-100's white parts, since it
   is simulated with the SO-101 model).
2. **Colour prior.** The object sits still before the grasp, so the
   largest blob of the first frames of each episode is the object; its Lab
   colour (median over episodes, spread floored) is the prior. Pixels off
   that colour are dropped (shadows, the white arm parts), and a blob has
   to be within 0.15-4x the prior's area (the SO-101's red cables are the
   right colour but too small).
3. **Tracking.** While the gripper holds something (the picks from
   `antares.task`, now also ending when the jaw shuts the rest of the way
   on nothing because the object slipped out) the object is between the
   jaws and mostly hidden: its position is the FK fingertip (`held`).
   Otherwise the blob nearest the last sighting is the object (`vision`);
   right after a release the search starts from where the jaws opened and
   is wider, since the object drops or slides. Not seen: `hidden`, last
   position carried forward.
4. **Pixel to metres.** The ray through the blob centroid meets the plane
   `z = table + 15 mm` (the scene's assumed half-height). The blob's
   pixels back-projected onto that plane give the footprint (oriented
   extent, median over the at-rest frames). About 6-9 s per episode.

The check that is built in: if the camera, the table and the tracking are
right, the gripper closes where the object was seen. Per episode, the
distance between the vision start pose (median before the grasp) and the
FK fingertip at the grasp:

| dataset | camera | object | footprint (IQR) | start poses: std, range (cm) | end poses: std (cm) | vision vs FK: median, IQR, worst | fixed offset (vision - FK) | spread about it |
|---|---|---|---|---|---|---|---|---|
| svla_so101_pickplace | `up` | a red 2x4-stud brick on a white table, carried into a shallow orange-rimmed tray | 3.9 x 1.8 cm (0.4 / 0.3) | 8.4 x 11.4; x 1..25, y -16..14 | 0.7 x 1.7 at (2.0, -15.9) | 2.1 / 0.7 / 3.4 cm | (+0.7, +0.2) cm | 1.8 cm |
| svla_so100_pickplace | `top` | a red cube on a black table, dropped into a white box | 3.9 x 3.5 cm (0.1 / 0.2) | 3.8 x 8.5; x 20..33, y -9..15 | 0.5 x 0.4 at (7.3, -11.9) | 1.4 / 0.7 / 2.5 cm | (+1.1, -0.3) cm | 0.8 cm |

50 of 50 episodes tracked on each. The vision start poses cover the same
area as `antares task`'s FK picks (SO-101: x 3..24, y -14..13; SO-100: x
19..33, y -8..14) and the two agree to about 2 cm, of which roughly 1 cm
is a fixed offset along +x: the camera puts the object a little further
out than the kinematics put the jaws (the blob centroid takes in the
object's front face from an oblique camera, and the camera fit is loose
along its optical axis). The end poses are tight (the container), and sit
7.6-7.9 cm from the FK release point: the object is let go 7-8 cm up and
drops, so that number is the drop height, not an error. A 2x4 brick is
3.2 x 1.6 cm and the SO-100 cube looks like 3.5 cm: the oblique `up`
camera adds the brick's front face to its footprint, the top-down `top`
camera gets the cube right.

**Cross-camera check (SO-101).** With `up` and `side` both fitted, the
object at rest is triangulated from the two pixel rays over 36 frames.
The rays pass within 0.5 cm of each other, but meet 3.0 cm from `up`'s
plane point and 2.6 cm *below* the object plane (1 cm below the table),
and `side`'s own plane intersection is 6.4 cm off in xy. The `side` fit
(IoU 0.58, the arm leaving the frame) is loose in depth, so the
single-camera plane intersection, validated by the FK agreement above, is
the better estimate; triangulation here measures the camera fits, not the
object.

**Payload mass from the residual.** A position servo settles short of its
command by load / gain, so a mass `m` at the fingertip adds a deflection
`-m g Jz / kp` on each load-bearing joint (`Jz`: fingertip height per
radian from the site Jacobian; `kp`: the lock's gain, N*m/rad). Lifting
also reverses the motion, and a deadband flips the tracking error with
direction, so the residual step across the *pick* (`P + D`) and across the
*release* (`-P + D`, the same down-then-up motion with the load leaving)
are measured separately: half the difference is the payload, half the sum
the motion term. Windows of 0.6 s either side of each event, medians,
`shoulder_lift` and `elbow_flex`.

| dataset | lock kp | shoulder: extra deflection (IQR), motion term | elbow: extra deflection (IQR), motion term | mass per encoder step | noise floor | combined mass (IQR) | verdict |
|---|---|---|---|---|---|---|---|
| svla_so101_pickplace | 10.5 | +0.69° (0.58), +0.38° | -0.27° (0.89), -0.18° | 13 g | 41-58 g | 22 g (69) over 50 picks | below the noise |
| svla_so100_pickplace | 1427 | +0.63° (0.52), +0.47° | +0.22° (1.10), +0.02° | 1.5 kg | 4-7 kg | 5.8 kg (6.9) over 49 picks | below the noise |

So: no. The residual does step across a pick, by about 0.6-0.7° on the
shoulder, repeatably (IQR 0.5°), but the same step shows on an arm whose
lock says `kp = 1427` and on one whose lock says `kp = 10.5`, so it is not
a stiffness-limited sag; the elbow's step has the opposite sign on the
SO-101; and the per-episode spread is 1.3-5x the median. The method's
resolution with the SO-101 lock is 13 g per encoder step and its noise
floor 40-60 g, both above a 30 g printed object; with the SO-100 lock the
resolution is 1.5 kg. The deflection is reported, the mass is marked as
not measured, and the export falls back to the assumed 30 g.
Assumptions, stated: quasi-static windows; torque = `kp (cmd - q)` with
the lock's `kp` (the mass scales 1:1 with it, and pick-and-place logs do
not pin `kp` down, see [Is a lock specific to its arm?](#is-a-lock-specific-to-its-arm));
a point mass at the `gripperframe` site; the sim's link masses are the
real ones; the gripper joint excluded; the motion term the same at pick
and release.

Outputs: `locks/<lock>.objects.json` (prior, per-episode summaries, the
start/end pose clouds, footprint, agreement numbers, cross-camera check,
the payload estimate), `locks/objects/<lock>_<camera>_tracks.npz` (the
per-frame tracks) and a compact `objects` entry in the lock (old locks
load with none). `antares export` then puts the object in `scene.xml` as a
free box with the vision footprint, the estimated mass (the assumed 30 g
when the estimate is below the noise, and the comment says so) at the mean
vision start pose, coloured like the object, with the start/end
distributions in `objects.json` next to it; the objects fit wins over
`--task`'s cube when both exist, and the task is still written with its
picks taken from vision (`"source": "vision"`;
`TaskDistribution.with_vision_picks`). The API: `POST /api/objects/fit
{dataset, lock, camera?}` (a job, progress per episode), `GET
/api/objects?lock=`, and `GET /api/objects/track?dataset=&episode=&lock=`
for the per-frame positions, source and pixel boxes. The UI's Objects
block under Scene has the fit button, footprint, mass verdict and the
vision-vs-FK number; the overlay viewer draws the object's box on the
frame (solid where seen, dashed while held or hidden) and the 3D viewer a
small box of the object's colour following the playhead.

Caveats: the object's height is the scene's assumed 15 mm half-height
(the camera cannot see it from above), so the end poses in a raised
container would be off; one object per dataset; a second object of the
same colour would be confused with it; the SO-100's end poses rely on the
cube being visible inside its box from above.

## Layout

| Path | What |
|---|---|
| `antares/data.py` | LeRobot dataset loader (tabular only) |
| `antares/validate.py` | Dataset checks shared by upload, `inspect` and the fleet scan |
| `antares/sim.py` | MuJoCo model, column-to-joint mapping, unit mapping, replay and rollout |
| `antares/robots.py` | Robot profiles: columns, joints, units, grippers, model file (`antares robots`) |
| `antares/factors.py` | What the fit can account for, and what it can't yet (`antares factors`) |
| `antares/menagerie.py` | Fetches non-vendored MuJoCo Menagerie models at a pinned commit |
| `antares/catalog.py` | Public datasets of other arms to test on (`antares datasets`) |
| `antares/calibrate.py` | CMA-ES parameter fit |
| `antares/sensor.py` | Per-joint command offset/scale fit |
| `antares/lerobot_calib.py` | The follower's LeRobot motor calibration as the arm's unit map; `inspect --calibration`, `/api/datasets/calibration-check` |
| `antares/gap.py` | Tracking-error metrics |
| `antares/lock.py` | Calibrated sim saved per robot (`locks/*.lock.json`) |
| `antares/export.py` | Lock -> standalone MJCF, sidecar, scene, zip |
| `antares/export_targets.py` | Lock -> Isaac Lab / ManiSkill / mjlab actuator configs and neutral JSON (`export --format`) |
| `antares/drift.py` | Sliding-window refits and trend statistics over a session |
| `antares/dr.py` | Domain-randomization ranges derived from refits (`locks/*.dr.json`) |
| `antares/check.py` | Pre-deploy drift gate (`antares check`, `/api/check`) |
| `antares/history.py` | Check history per lock across sessions (`locks/*.history.jsonl`, `antares history`, `/api/history`) |
| `antares/registry.py` | One id per physical robot, its lock lineage and active lock (`locks/registry.json`, `antares robots`, `@<robot>`, `/api/registry`) |
| `antares/diagnose.py` | Names the parameter change behind a drift (`check --explain`) |
| `antares/synth.py` | Synthetic recordings of an arm with known wear (`antares synth`) |
| `action.yml` | Composite GitHub Action running the gate in CI |
| `.github/workflows/` | Tests on every push (`ci.yml`) and an example weekly drift gate on the SO-101 demo lock (`robot-check.yml`) |
| `antares/events.py` | Contact events from the real-sim residual |
| `antares/compensate.py` | Lag compensation (shift, ILC), sim-validated |
| `antares/sag.py` | Residual vs gravity load and motion (`antares sag`) |
| `antares/specificity.py` | Cross-arm verdict, identifiability, swap and same-arm checks |
| `antares/synthetic.py` | Simulated arm datasets with known parameters |
| `antares/transfer.py` | Sim-to-sim policy transfer experiment (`antares transfer`) |
| `antares/policy.py` | State-based chunked policy trained on the logs; LeRobot checkpoint wrapper (`antares policy train`, `locks/*.policy.pt`) |
| `antares/shadow.py` | Open-loop shadow evaluation: the policy's commands rolled through each sim |
| `antares/closedloop.py` | Closed-loop evaluation: the policy drives each sim; grasp-point proxy success |
| `results/` | Robot-specificity reports (`antares specificity`), policy transfer report (`policy_transfer.{json,md}`) |
| `antares/cli.py` | The `antares` command line |
| `antares/server.py` | FastAPI backend for the UI |
| `examples/` | `randomize_env.py`: DR ranges in a training loop |
| `scripts/` | `sensor_synthetic.py` (synthetic sensor-fit check), `dr_compare.py` (derived vs hand-picked DR ranges) |
| `antares/fleet.py` | Hub discovery, fleet scan, fleet statistics |
| `antares/video.py` | Dataset video frames (episode -> file/timestamp, AV1 decode, frame cache) |
| `antares/silhouette.py` | Real arm mask: median background, colour model, static-arm region |
| `antares/project.py` | Sim arm silhouette: mesh hulls, pinhole camera, polygon fill |
| `antares/camera.py` | Fixed-camera fit (CMA-ES over look-at coordinates), `locks/*.camera.json` |
| `antares/videodelay.py` | How many frames a camera's video lags the joint log (`antares camera-delay`) |
| `antares/task.py` | Pick/place/start-pose distribution from the gripper and FK (`antares task`) |
| `antares/noise.py` | Joint-reading noise model measured while parked (`antares noise`) |

| `antares/scene.py` | Table, workspace and backdrops from the logs; sim-view composite (`locks/*.scene.json`, `locks/scene/`) |
| `antares/objects.py` | The manipulated object found and tracked in the video, vision vs FK, cross-camera check (`antares objects-fit`, `locks/*.objects.json`, `locks/objects/`) |
| `antares/payload.py` | Payload mass from the residual while the object is carried (`antares payload`) |
| `locks/` | Committed locks for the demo arms; `locks/fleet/` for the fleet scan |
| `web/` | React + three.js viewer |

## Status

Started as a weekend hackathon project. On the SO-101 and SO-100 demo
datasets, everything above works end to end from the CLI, the API and the
viewer: replay, shared and per-joint calibration, sensor calibration, robot
locks and MJCF export, drift scans and the pre-deploy gate with its cause
diagnosis, data-derived DR ranges, contact events, sim-only lag
compensation, the robot-specificity study, the fleet scan, and a learned
state policy judged in the stock and the calibrated sim.

Not yet measured on hardware: everything is judged against logged
trajectories, so lag compensation and DR are validated in the sim only.
Open questions: the fits are seed-sensitive and `kp`/`kv` are not
identifiable from pick-and-place logs (see "Is a lock specific to its
arm?"), and the elbow error is step-shaped, which the six servo
parameters do not model.
