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

## Layout

| Path | What |
|---|---|
| `antares/data.py` | LeRobot dataset loader (tabular only) |
| `antares/sim.py` | MuJoCo model, unit mapping, replay |
| `antares/calibrate.py` | CMA-ES parameter fit |
| `antares/gap.py` | Tracking-error metrics |
| `antares/lock.py` | Calibrated sim saved per robot as a lock file |
| `antares/export.py` | Lock -> standalone MJCF, sidecar, scene, zip |
| `antares/server.py` | FastAPI backend for the UI |
| `web/` | React + three.js viewer |

## Status

Weekend hackathon project. Data loading and the UI shell work; replay
and calibration are in progress.
