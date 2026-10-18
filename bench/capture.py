"""Record one 60 s run from the servo_log sketch into a CSV.

    pip install pyserial
    python bench/capture.py servo_a_run1.csv            # auto-detects the Arduino
    python bench/capture.py servo_a_run1.csv --port /dev/cu.usbmodem1101

The port search and the line reader are antares.servo_serial, shared with
``antares bench live``; run from a checkout, the script finds the package
next to it without an install.
"""
import argparse
import sys
from pathlib import Path

try:
    from antares.servo_serial import BenchSerialError, find_port, open_port, run_lines
except ImportError:  # a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from antares.servo_serial import BenchSerialError, find_port, open_port, run_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--port")
    args = ap.parse_args()

    try:
        port = args.port or find_port()
        lines = 0
        with open_port(port) as ser, open(args.out, "w") as f:
            for line in run_lines(ser):
                f.write(line + "\n")
                lines += 1
                if lines % 250 == 0:
                    print(f"  {lines // 50} s ...", flush=True)
    except BenchSerialError as e:
        sys.exit(f"error: {e}")
    print(f"Wrote {lines - 1} samples to {args.out} from {port}")


if __name__ == "__main__":
    main()
