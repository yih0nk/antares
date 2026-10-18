"""The bench sketch over USB: finding the Arduino and reading one run.

Shared by ``bench/capture.py`` (records a run to a CSV) and
:mod:`antares.servo_live` (steps the sims as the lines arrive), so the
two agree on where an Arduino shows up and how a run starts and ends.
Only pyserial is needed, and only once a port is opened, so
``capture.py`` runs on a laptop without the rest of antares' dependencies.
"""

from __future__ import annotations

import glob
import time
from collections.abc import Iterator

BAUD = 115200
#: Where the sketch's board shows up on macOS and Linux, in the order tried.
PORT_GLOBS = ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/ttyACM*", "/dev/ttyUSB*")
#: Seconds without a line before the run counts as dead.
READ_TIMEOUT = 5.0


class BenchSerialError(RuntimeError):
    """No Arduino, a port that will not open, or a sketch that stays silent."""


def find_port() -> str:
    """The first serial port that looks like an Arduino."""
    for pattern in PORT_GLOBS:
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    raise BenchSerialError("no Arduino found; pass --port (Arduino IDE > Tools > Port shows it)")


def open_port(port: str):
    """A pyserial handle on ``port`` (use as a context manager), after the
    reset that opening the port gives an Uno and with its boot chatter dropped."""
    try:
        import serial
    except ImportError:
        raise BenchSerialError("pyserial is not installed: pip install pyserial") from None
    try:
        ser = serial.Serial(port, BAUD, timeout=READ_TIMEOUT)
    except (OSError, serial.SerialException) as e:
        raise BenchSerialError(f"cannot open {port}: {e}") from None
    time.sleep(2.0)  # opening the port resets an Uno
    ser.reset_input_buffer()
    return ser


def run_lines(ser, start: bool = True) -> Iterator[str]:
    """The CSV lines of one run (header first), as the sketch prints them.

    With ``start`` the sketch is told to go (``g``). Comment lines (``#``)
    are dropped and ``#done`` ends the run; no line for :data:`READ_TIMEOUT`
    seconds raises :class:`BenchSerialError`.
    """
    if start:
        ser.write(b"g")
    while True:
        line = ser.readline().decode(errors="replace").strip()
        if not line:
            raise BenchSerialError(f"no data from the Arduino for {READ_TIMEOUT:g} s; is the sketch flashed?")
        if line.startswith("#done"):
            return
        if line.startswith("#"):
            continue
        yield line


__all__ = ["BAUD", "BenchSerialError", "PORT_GLOBS", "READ_TIMEOUT", "find_port", "open_port", "run_lines"]
