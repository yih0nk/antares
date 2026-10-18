"""Record one 60 s run from the servo_log sketch into a CSV.

    pip install pyserial
    python capture.py servo_a_run1.csv            # auto-detects the Arduino
    python capture.py servo_a_run1.csv --port /dev/cu.usbmodem1101
"""
import argparse
import glob
import sys
import time

import serial


def find_port():
    ports = glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/ttyACM*")
    if not ports:
        sys.exit("No Arduino found; pass --port (Arduino IDE > Tools > Port shows it).")
    return ports[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--port")
    args = ap.parse_args()

    port = args.port or find_port()
    with serial.Serial(port, 115200, timeout=5) as ser:
        time.sleep(2.0)  # opening the port resets the Uno
        ser.reset_input_buffer()
        ser.write(b"g")
        lines = 0
        with open(args.out, "w") as f:
            while True:
                line = ser.readline().decode(errors="replace").strip()
                if not line:
                    sys.exit("Timed out waiting for data; is the sketch flashed?")
                if line.startswith("#done"):
                    break
                if line.startswith("#"):
                    continue
                f.write(line + "\n")
                lines += 1
                if lines % 250 == 0:
                    print(f"  {lines // 50} s ...", flush=True)
    print(f"Wrote {lines - 1} samples to {args.out} from {port}")


if __name__ == "__main__":
    main()
