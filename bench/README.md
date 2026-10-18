# Servo bench: a hobby servo as a one-joint robot

Two servos with the same part number behave measurably differently, and antares can tell them apart. This folder turns a Tower Pro SG92R on an Arduino into a robot antares calibrates like any arm (robot `sg92r` in `antares models`).

| File | What it does |
|---|---|
| `servo_log/servo_log.ino` | Arduino sketch: plays a fixed 60 s command program, logs command and pot reading every 20 ms |
| `capture.py` | Saves one run from the Arduino to CSV (`pip install pyserial`) |
| `pipeline.sh <dir>` | Imports, calibrates and demos every run in `<dir>` |
| `rehearse.sh [dir]` | The same pipeline on simulated servos, no hardware |

## Hardware

A stock SG92R has no position output: the three wires are ground (brown), +5 V (red) and the command signal (orange). antares needs to know where the shaft actually went, so add a fourth wire.

1. Undo the 4 bottom screws and lift off only the bottom cover.
2. Find the potentiometer's three pins on the board. With the servo powered and moving, the middle pin (wiper) reads a voltage that follows the shaft (about 0.3 to 2 V); the outer two don't move.
3. Solder a thin wire to the middle pin, tape it, route it out by the cable, close the case, add a dab of hot glue.
4. Stand the servo upright with a popsicle stick glued to the horn. The model assumes a vertical shaft, so gravity doesn't load it.

Wiring: brown to GND, red to 5V, orange to D9, feedback wire to A0. A second servo uses D10 and A1. Two servos together want a separate 5 V supply (1 A, with a 470 µF capacitor) with its ground tied to the Arduino's.

## Recording

Flash `servo_log.ino` (set `NUM_SERVOS` to 1 for a single servo). In the Serial Monitor at 115200 baud you should see `#ready`; type `g` and check the third column follows the second. Close the monitor, then:

```
python bench/capture.py recordings/servo_a_run1.csv
```

Record one servo at a time and name the files exactly like this, so `pipeline.sh` finds them:

- `servo_a_run1.csv`, `servo_a_run2.csv`, `servo_a_run3.csv`: servo A, nothing changed between runs. Runs 1 and 2 are fitted, run 3 is held out.
- `servo_b_run1.csv` to `servo_b_run3.csv`: servo B, same stick, same mounting.
- `servo_a_coin_run1.csv`: servo A with **three** coins taped at the stick's tip. In simulation one coin raised the error 85% but stayed under the drift limit; three are flagged.

## Turning recordings into the demo

```
bench/pipeline.sh recordings          # about 2 min per servo at 300 evaluations
antares bench live --lock recordings/locks/servo_a.lock.json
```

`pipeline.sh` prints the pot conversion for each servo (check the five holds sit on a line), the calibration before and after, then the demo table: each servo's held-out run through its own lock, the other servo's lock and the uncalibrated model, and the payload check. `bench live` opens a page on http://127.0.0.1:8010 with the real stick and both sim ghosts moving together; `--replay recordings/servo_a_run3.csv` plays a recording instead of the Arduino.

The datasets also open in the main viewer like any other: `antares serve`, then "+ add dataset" with the dataset folder's full path (for example `/Users/you/antares/recordings/ds_a`), and the Calibrate button fits the servo live.

## What the numbers mean

The pot conversion comes from the program's first 10 s (holds at 30, 60, 90, 120 and 150°), so "degrees" means where this servo settles when told that angle. A servo whose pulse mapping is a few degrees off is folded into the conversion, not reported as an error. The sketch reads the pot just before sending each command, which shows up as one frame (20 ms) of fitted latency on top of the servo's own.
