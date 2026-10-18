"""Live view of the servo bench: the real stick and its sim ghosts, in step.

``antares bench live`` reads the bench sketch's stream (or replays a
recorded CSV at real speed), converts each pot reading to degrees, and
steps two sims tick by tick with the same commands: the servo's lock and
the uncalibrated sg92r model. A small page (no build step, no extra
dependencies) draws the three sticks from above and the running error of
each ghost, so the audience sees the calibrated ghost stay on the real
stick while the default one lags and overshoots.

Server-sent events carry one JSON object per tick:
``{t, cmd, real, lock, default, err_lock, err_default}`` (degrees).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from antares import servo_bench as sb


class StepSim:
    """One sg92r sim advanced one bench tick at a time, through the same
    :class:`antares.closedloop.FrameStepper` path :meth:`antares.sim.Arm.rollout`
    takes in a batch (substeps, clipping and the lock's deadband).

    Locks fitted with a servo model (:attr:`antares.sim.SimParams.servo` other
    than ``position``) are refused: the stepper has no per-substep servo
    loop, so their ghost would not be the lock's model.
    """

    def __init__(self, params, fps: float = sb.FPS):
        from antares.sim import Arm

        if params.uses_servo_model:
            raise ValueError(
                f"the live view steps the position-actuator model only; this lock uses the "
                f"{params.servo!r} servo model"
            )
        self.arm = Arm(robot=sb.ROBOT)
        self.arm.apply(params)
        self.fps = float(fps)
        self.latency = int(self.arm.params.per_joint("latency_frames", 1)[0])
        self.pending: deque[float] = deque()
        self.stepper = None

    @property
    def started(self) -> bool:
        return self.stepper is not None

    def reset(self, q_deg: float, cmd_deg: float) -> None:
        from antares.closedloop import FrameStepper

        if self.stepper is not None:
            self.stepper.close()
        self.stepper = FrameStepper(self.arm, np.radians([q_deg]), self.fps)
        self.pending = deque([cmd_deg] * self.latency)

    def step(self, cmd_deg: float) -> float:
        """Send ``cmd_deg`` and advance one tick; returns the shaft angle (deg)."""
        if self.stepper is None:
            raise RuntimeError("reset() before step()")
        self.pending.append(cmd_deg)
        applied = self.pending.popleft()
        return float(np.degrees(self.stepper.step(np.radians([applied]))[0]))


def serial_ticks(port: str, servo: str, start: bool = True) -> Iterator[tuple[float, float, float]]:
    """``(t s, cmd deg, raw counts)`` from the bench sketch over USB.

    The port is opened here, in the caller's thread, so a missing board or
    pyserial fails before anything is served. With ``start`` it sends ``g``
    and restarts the program after each run, so the stick keeps moving for
    as long as the page is up.
    """
    from antares.servo_serial import BenchSerialError, open_port

    try:
        ser = open_port(port)
    except BenchSerialError as e:
        raise SystemExit(f"error: {e}") from None
    return _serial_ticks(ser, servo, start)


def _serial_ticks(ser, servo: str, start: bool) -> Iterator[tuple[float, float, float]]:
    from antares.servo_serial import BenchSerialError, run_lines

    with ser:
        t_base = 0.0
        while True:
            col = None
            last_t = 0.0
            for line in run_lines(ser, start=start):
                parts = line.split(",")
                if parts[0] == "t_ms":
                    if f"raw_{servo}" not in parts:
                        have = ", ".join(p.removeprefix("raw_") for p in parts if p.startswith("raw_"))
                        raise BenchSerialError(f"the sketch logs servo(s) {have}, not {servo}")
                    col = parts.index(f"raw_{servo}")
                    continue
                if col is None or len(parts) <= col:
                    continue
                last_t = float(parts[0]) / 1000.0
                yield t_base + last_t, float(parts[1]), float(parts[col])
            t_base += last_t + 0.02
            if not start:
                return


def csv_ticks(path: str | Path, servo: str, loop: bool = True) -> Iterator[tuple[float, float, float]]:
    """A recorded run played back at real speed (for rehearsing without hardware).

    The CSV is read and checked here, in the caller's thread; only the
    playback is lazy.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"error: no such replay CSV: {path}")
    try:
        run = sb.read_run(path)
    except ValueError as e:
        raise SystemExit(f"error: {e}") from None
    if servo not in run.raw:
        raise SystemExit(f"error: {path} has no raw_{servo} column (has {', '.join(run.raw)})")
    return _replay(run, servo, loop)


def _replay(run: sb.Run, servo: str, loop: bool) -> Iterator[tuple[float, float, float]]:
    t_base = 0.0
    while True:
        t0 = time.monotonic()
        for t, c, r in zip(run.t, run.cmd, run.raw[servo]):
            delay = t0 + t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            yield t_base + float(t), float(c), float(r)
        t_base += float(run.t[-1]) + 0.02
        if not loop:
            return


def dataset_candidates(lock, dataset: str | None = None, lock_path: str | Path | None = None) -> list[Path]:
    """Where the lock's dataset may be, in the order tried.

    ``calibrate`` records the dataset path as typed, relative to where it
    ran, and saves the lock under ``locks/`` there (bench/pipeline.sh runs
    inside the recordings folder: ``<dir>/locks/x.lock.json`` next to
    ``<dir>/ds_a``); so a relative ``lock.dataset`` is tried against the lock
    file's directory's parent first, then against the current directory.
    """
    if dataset:
        return [Path(dataset)]
    ds = Path(lock.dataset)
    if ds.is_absolute():
        return [ds]
    out = []
    if lock_path is not None:
        out.append(Path(lock_path).resolve().parent.parent / ds)
    cwd = Path.cwd() / ds
    if cwd not in out:
        out.append(cwd)
    return out


def pot_for_lock(lock, dataset: str | None = None, lock_path: str | Path | None = None) -> sb.PotFit:
    """The counts -> degrees line saved with the dataset the lock was fitted on.

    ``lock_path`` (where the lock file is) lets a relative ``lock.dataset``
    resolve from any directory; see :func:`dataset_candidates`.
    """
    candidates = dataset_candidates(lock, dataset, lock_path)
    for root in candidates:
        path = root / "meta" / "info.json"
        if path.is_file():
            break
    else:
        tried = " or ".join(str(c / "meta" / "info.json") for c in candidates)
        raise SystemExit(f"error: no {tried}; pass --dataset <the servo's imported dataset> for its pot conversion")
    bench = json.loads(path.read_text()).get("antares_bench")
    if not bench:
        raise SystemExit(f"error: {path} was not written by `antares bench import`")
    p = bench["pot"]
    return sb.PotFit(float(p["scale"]), float(p["offset"]), float(p["rms_deg"]), tuple(map(tuple, p["points"])))


class Broadcaster:
    """Fan ticks out to every open page."""

    def __init__(self) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, msg: dict) -> None:
        data = json.dumps(msg)
        with self._lock:
            for q in self._subs:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass


def run_ticks(ticks, pot: sb.PotFit, lock_params, default_params, out: Broadcaster, info: dict) -> None:
    """Step both sims with each tick and publish the three angles."""
    sims = {"lock": StepSim(lock_params), "default": StepSim(default_params)}
    sq = {"lock": 0.0, "default": 0.0}
    n = 0
    window: dict[str, deque] = {k: deque(maxlen=250) for k in sims}
    for t, cmd, raw in ticks:
        real = float(pot.to_deg(raw))
        if not sims["lock"].started:
            for s in sims.values():
                s.reset(real, cmd)
            angles = {k: real for k in sims}
        else:
            angles = {k: s.step(cmd) for k, s in sims.items()}
        n += 1
        msg = {"t": round(t, 3), "cmd": cmd, "real": round(real, 2)}
        for k, a in angles.items():
            e = a - real
            sq[k] += e * e
            window[k].append(e * e)
            msg[k] = round(a, 2)
            msg[f"err_{k}"] = round(float(np.sqrt(np.mean(window[k]))), 3)
            msg[f"rms_{k}"] = round(float(np.sqrt(sq[k] / n)), 3)
        info["ticks"] = n
        out.publish(msg)


PAGE = Path(__file__).resolve().parent / "assets" / "sg92r" / "live.html"


def live_info(lock) -> dict:
    """What ``/info`` reports: the lock's name and held-out score (``None``
    when the lock has none or it is not finite, so the JSON stays valid)."""
    from antares.check import _finite_or_none

    return {"lock": lock.name, "ticks": 0, "lock_score": _finite_or_none(getattr(lock.result, "eval_score", None))}


def serve(ticks, pot: sb.PotFit, lock, host: str = "127.0.0.1", port: int = 8010) -> int:
    """Serve the page until ctrl-c, or until the ticks end.

    The ticks are consumed on a worker thread; whatever stops it (the
    stream ending, the Arduino going quiet, a bad line) also stops the
    server, so the page never says "live" over a stick that stopped. A
    failure is raised as ``SystemExit`` with its message; a stream that
    simply ended returns 0. ``port=0`` picks a free port.
    """
    from antares.sim import Arm

    default_params = Arm(robot=sb.ROBOT).default_params()
    out = Broadcaster()
    info = live_info(lock)
    # Build both sims here so a lock the stepper cannot replay fails before serving.
    StepSim(lock.params)
    failure: list[BaseException] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:  # quiet
            pass

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/info":
                body = json.dumps(info).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q = out.subscribe()
                try:
                    while True:
                        try:
                            data = q.get(timeout=10)
                        except queue.Empty:
                            data = None
                        self.wfile.write((f"data: {data}\n\n" if data else ": ping\n\n").encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    out.unsubscribe(q)
            else:
                self.send_error(404)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True

    def work() -> None:
        try:
            run_ticks(ticks, pot, lock.params, default_params, out, info)
        except BaseException as e:  # noqa: BLE001 - whatever stopped the ticks stops the page
            failure.append(e)
        finally:
            server.shutdown()  # returns once serve_forever() has seen the request

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    host, port = server.server_address[:2]
    print(f"live view of lock {lock.name}: http://{host}:{port}  (ctrl-c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    if failure:
        e = failure[0]
        detail = e.code if isinstance(e, SystemExit) and isinstance(e.code, str) else f"{type(e).__name__}: {e}"
        raise SystemExit(f"error: the ticks stopped after {info['ticks']} tick(s): {detail}")
    print(f"the ticks ended after {info['ticks']} tick(s); stopping")
    return 0


__all__ = [
    "Broadcaster", "StepSim", "csv_ticks", "dataset_candidates", "live_info", "pot_for_lock", "run_ticks",
    "serial_ticks", "serve",
]
