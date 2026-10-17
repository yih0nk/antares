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
| `antares/sim.py` | MuJoCo model, unit mapping, replay |
| `antares/calibrate.py` | CMA-ES parameter fit |
| `antares/gap.py` | Tracking-error metrics |
| `antares/lock.py` | Calibrated sim saved per robot as a lock file |
| `antares/fleet.py` | Hub discovery, fleet scan, fleet statistics |
| `antares/server.py` | FastAPI backend for the UI |
| `locks/` | Committed locks for the demo arms; `locks/fleet/` for the fleet scan |
| `web/` | React + three.js viewer |

## Status

Weekend hackathon project. Data loading and the UI shell work; replay
and calibration are in progress.
