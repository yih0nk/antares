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

import mujoco
import numpy as np

from antares import servo_bench as sb


class StepSim:
    """One sg92r sim advanced one bench tick at a time (what
    :meth:`antares.sim.Arm.rollout` does in a batch)."""

    def __init__(self, params, fps: float = sb.FPS):
        from antares.sim import Arm

        self.arm = Arm(robot=sb.ROBOT)
        self.arm.apply(params)
        self.latency = int(self.arm.params.per_joint("latency_frames", 1)[0])
        frame_dt = 1.0 / fps
        self.n_sub = max(1, int(round(frame_dt / self.arm._base_timestep)))
        self.dt = frame_dt / self.n_sub
        self.pending: deque[float] = deque()
        self.started = False

    def reset(self, q_deg: float, cmd_deg: float) -> None:
        m, d = self.arm.model, self.arm.data
        m.opt.timestep = self.dt
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm.qpos_ids[0]] = np.radians(q_deg)
        d.ctrl[self.arm.actuator_ids[0]] = np.radians(cmd_deg)
        mujoco.mj_forward(m, d)
        self.pending = deque([cmd_deg] * self.latency)
        self.started = True

    def step(self, cmd_deg: float) -> float:
        """Send ``cmd_deg`` and advance one tick; returns the shaft angle (deg)."""
        m, d = self.arm.model, self.arm.data
        self.pending.append(cmd_deg)
        applied = self.pending.popleft()
        lo, hi = self.arm.ctrl_range
        d.ctrl[self.arm.actuator_ids[0]] = float(np.clip(np.radians(applied), lo[0], hi[0]))
        mujoco.mj_step(m, d, self.n_sub)
        return float(np.degrees(d.qpos[self.arm.qpos_ids[0]]))


def serial_ticks(port: str, servo: str, start: bool = True) -> Iterator[tuple[float, float, float]]:
    """``(t s, cmd deg, raw counts)`` from the bench sketch over USB.

    With ``start`` it sends ``g`` and restarts the program after each run,
    so the stick keeps moving for as long as the page is up.
    """
    try:
        import serial
    except ImportError:
        raise SystemExit("error: live mode needs pyserial: pip install pyserial") from None
    with serial.Serial(port, 115200, timeout=5) as ser:
        time.sleep(2.0)  # opening the port resets an Uno
        ser.reset_input_buffer()
        col = None
        t_base = 0.0
        while True:
            if start:
                ser.write(b"g")
            last_t = 0.0
            while True:
                line = ser.readline().decode(errors="replace").strip()
                if not line:
                    raise SystemExit("error: no data from the Arduino for 5 s; is the sketch flashed?")
                if line.startswith("#done"):
                    break
                if line.startswith("#"):
                    continue
                parts = line.split(",")
                if parts[0] == "t_ms":
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
    """A recorded run played back at real speed (for rehearsing without hardware)."""
    run = sb.read_run(path)
    if servo not in run.raw:
        raise SystemExit(f"error: {path} has no raw_{servo} column")
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


def pot_for_lock(lock, dataset: str | None = None) -> sb.PotFit:
    """The counts -> degrees line saved with the dataset the lock was fitted on."""
    path = Path(dataset or lock.dataset) / "meta" / "info.json"
    if not path.is_file():
        raise SystemExit(f"error: no {path}; pass --dataset <the servo's imported dataset> for its pot conversion")
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


def serve(ticks, pot: sb.PotFit, lock, host: str = "127.0.0.1", port: int = 8010) -> None:
    from antares.sim import Arm

    default_params = Arm(robot=sb.ROBOT).default_params()
    out = Broadcaster()
    info = {"lock": lock.name, "ticks": 0, "lock_score": getattr(lock.result, "eval_score", None)}
    worker = threading.Thread(target=run_ticks, args=(ticks, pot, lock.params, default_params, out, info), daemon=True)

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
    worker.start()
    print(f"live view of lock {lock.name}: http://{host}:{port}  (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["Broadcaster", "StepSim", "csv_ticks", "pot_for_lock", "run_ticks", "serial_ticks", "serve"]
