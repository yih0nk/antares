"""A hobby servo on an Arduino as a one-joint robot.

The bench rig (``bench/servo_log``) drives one or two SG92R servos with the
same fixed 60 s command program and logs, every 20 ms, the whole-degree
command it sent and each servo's potentiometer reading (0..1023 counts,
from a wire soldered to the pot's wiper):

    t_ms,cmd_deg,raw_a[,raw_b]

This module turns those CSVs into LeRobot-layout datasets that every
antares command reads (robot ``sg92r``):

* :func:`fit_pot` maps counts to degrees from the program's first 10 s,
  five 2 s holds at 30/60/90/120/150 degrees: once settled the servo sits
  at its command, so a line through (median counts, command) per hold is
  the conversion. Degrees then mean "where this servo settles when told
  that angle", so a steady offset in the servo's own pulse mapping is
  folded into the conversion, not fitted.
* :func:`import_runs` cuts each run into 5 s episodes and writes the
  dataset; ``meta/info.json`` keeps the conversion and the source files.
* :func:`simulate_run` writes the CSV a servo with known parameters would
  have produced, so the whole path can be checked without hardware.

:func:`command_at` is the sketch's program; the two must stay identical.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from antares.data import Dataset, Episode

ROBOT = "sg92r"
ROBOT_TYPE = "sg92r_bench"
FPS = 50.0
RUN_SECONDS = 60.0
EPISODE_SECONDS = 5.0
JOINT = "shaft"
#: The calibration holds: (start s, end s, commanded degrees).
HOLDS: tuple[tuple[float, float, float], ...] = tuple((2.0 * i, 2.0 * i + 2.0, d) for i, d in enumerate((30, 60, 90, 120, 150)))
#: Part of each hold used for the conversion (after the move has settled).
HOLD_WINDOW = (1.0, 1.9)
LO_DEG, HI_DEG = 30.0, 150.0
_STEPS = (90, 130, 50, 110, 70, 140, 40, 100, 60, 120, 80, 135, 45, 95, 125, 55, 105, 65, 115, 90)


def command_at(t: float) -> float:
    """Commanded angle (degrees, before rounding) at ``t`` seconds into a run.

    Must match ``command_at`` in ``bench/servo_log/servo_log.ino``.
    """
    if t < 10.0:
        return float((30, 60, 90, 120, 150)[int(t / 2.0)])
    if t < 26.0:
        return float(_STEPS[int((t - 10.0) / 0.8) % len(_STEPS)])
    if t < 41.0:
        return 90.0 + 40.0 * math.sin(2.0 * math.pi * 0.5 * (t - 26.0))
    if t < 56.0:
        tau, f0, f1, dur = t - 41.0, 0.2, 2.0, 15.0
        return 90.0 + 30.0 * math.sin(2.0 * math.pi * (f0 * tau + 0.5 * (f1 - f0) / dur * tau * tau))
    return 90.0


def program(n_frames: int | None = None, fps: float = FPS) -> np.ndarray:
    """The whole-degree commands the sketch sends, one per frame."""
    n = int(round(RUN_SECONDS * fps)) if n_frames is None else n_frames
    return np.array([float(int(min(max(command_at(i / fps), LO_DEG), HI_DEG) + 0.5)) for i in range(n)])


@dataclass(frozen=True)
class Run:
    """One recorded run: time (s), command (deg) and pot counts per servo."""

    path: str
    t: np.ndarray
    cmd: np.ndarray
    raw: dict[str, np.ndarray]


def read_run(path: str | Path) -> Run:
    """Read a ``capture.py`` CSV (lines starting with ``#`` are skipped)."""
    path = Path(path)
    with path.open(newline="") as f:
        rows = [r for r in csv.reader(line for line in f if line.strip() and not line.startswith("#"))]
    if not rows or rows[0][:2] != ["t_ms", "cmd_deg"]:
        raise ValueError(f"{path}: expected a header starting t_ms,cmd_deg (from bench/capture.py)")
    header, body = rows[0], rows[1:]
    good = [r for r in body if len(r) == len(header)]
    if len(good) < len(body):
        print(f"warning: {path.name}: skipped {len(body) - len(good)} malformed line(s)")
    data = np.array(good, dtype=np.float64)
    if len(data) < 2:
        raise ValueError(f"{path}: no samples")
    raw = {h.removeprefix("raw_"): data[:, i] for i, h in enumerate(header) if h.startswith("raw_")}
    if not raw:
        raise ValueError(f"{path}: no raw_<servo> column")
    return Run(path=str(path), t=(data[:, 0] - data[0, 0]) / 1000.0, cmd=data[:, 1], raw=raw)


@dataclass(frozen=True)
class PotFit:
    """Counts -> degrees: ``deg = scale * counts + offset``."""

    scale: float
    offset: float
    #: RMS of the holds around the line, degrees.
    rms_deg: float
    #: (median counts, commanded degrees) per hold.
    points: tuple[tuple[float, float], ...]

    def to_deg(self, counts: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(counts, dtype=np.float64) + self.offset

    def to_dict(self) -> dict:
        return {"scale": self.scale, "offset": self.offset, "rms_deg": self.rms_deg,
                "points": [list(p) for p in self.points]}


def fit_pot(runs: Sequence[Run], servo: str) -> PotFit:
    """Fit the counts -> degrees line from the calibration holds of ``runs``."""
    xs, ys = [], []
    for run in runs:
        if servo not in run.raw:
            raise ValueError(f"{run.path}: no raw_{servo} column (has {', '.join(run.raw)})")
        for start, _, deg in HOLDS:
            sel = (run.t >= start + HOLD_WINDOW[0]) & (run.t < start + HOLD_WINDOW[1])
            if sel.sum() < 5:
                raise ValueError(f"{run.path}: too few samples in the {deg:g} degree hold; was the run cut short?")
            xs.append(float(np.median(run.raw[servo][sel])))
            ys.append(float(deg))
    x, y = np.array(xs), np.array(ys)
    if np.ptp(x) < 50:
        raise ValueError(
            f"servo {servo}: the pot reading barely moves across the holds ({np.ptp(x):.0f} counts); "
            "check the feedback wire is on the potentiometer's middle pin"
        )
    scale, offset = np.polyfit(x, y, 1)
    rms = float(np.sqrt(np.mean((scale * x + offset - y) ** 2)))
    return PotFit(float(scale), float(offset), rms, tuple(zip(xs, ys)))


def episodes_from_run(run: Run, servo: str, pot: PotFit, start_index: int = 0,
                      seconds: float = EPISODE_SECONDS) -> list[Episode]:
    """Cut one run into ``seconds``-long episodes (state in degrees)."""
    state = pot.to_deg(run.raw[servo])
    n = int(round(seconds * FPS))
    out = []
    for k in range(len(state) // n):
        s = slice(k * n, (k + 1) * n)
        out.append(Episode(index=start_index + k, fps=FPS, joint_names=(JOINT,),
                           state=state[s, None].copy(), action=run.cmd[s, None].copy()))
    return out


def timing_report(run: Run) -> dict:
    """How steady the 50 Hz ticks were (the dataset assumes exactly 20 ms)."""
    dt = np.diff(run.t) * 1000.0
    return {"frames": int(len(run.t)), "dt_ms_median": float(np.median(dt)),
            "dt_ms_max": float(dt.max()), "late_ticks": int((dt > 25.0).sum())}


def import_runs(paths: Sequence[str | Path], servo: str, out: str | Path,
                seconds: float = EPISODE_SECONDS) -> tuple[Path, PotFit, list[dict]]:
    """Write ``paths`` (one servo's runs) as one dataset under ``out``."""
    from antares.synth import write_local

    runs = [read_run(p) for p in paths]
    pot = fit_pot(runs, servo)
    episodes: list[Episode] = []
    timing = []
    for run in runs:
        episodes += episodes_from_run(run, servo, pot, start_index=len(episodes), seconds=seconds)
        timing.append({"file": Path(run.path).name, **timing_report(run)})
    out = Path(out)
    ds = Dataset(repo_id=str(out), robot_type=ROBOT_TYPE, fps=FPS, joint_names=(JOINT,), episodes=tuple(episodes))
    write_local(ds, out)
    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v3.0"
    info["source"] = "antares.servo_bench"
    info["antares_bench"] = {"servo": servo, "files": [Path(p).name for p in paths],
                             "pot": pot.to_dict(), "timing": timing}
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return out, pot, timing


def simulate_run(path: str | Path, params=None, *, servo: str = "a", pot_scale: float = 0.28,
                 pot_offset: float = -40.0, noise_counts: float = 1.0, seed: int = 0,
                 command_offset_deg: float = 0.0) -> Path:
    """Write the CSV a servo with ``params`` would have logged on the bench.

    ``pot_scale``/``pot_offset`` set where its pot sits (degrees per count
    and degrees at zero counts); ``command_offset_deg`` shifts where it
    settles for a given command, like a servo whose pulse mapping is off.
    """
    from antares.sim import Arm

    arm = Arm(robot=ROBOT)
    if params is not None:
        arm.apply(params)
    cmd = program()
    q0 = np.radians([cmd[0] + command_offset_deg])
    sim = np.degrees(arm.rollout(np.radians(cmd[:, None] + command_offset_deg), q0, FPS)[:, 0])
    # The sketch reads the pot before it sends each command, so row t pairs
    # command t with the position reached under the commands before it.
    # rollout's row t is after command t has acted for a frame, so the
    # logged row t is rollout's row t-1: a whole frame of apparent latency.
    sim = np.concatenate([sim[:1], sim[:-1]])
    rng = np.random.default_rng(seed)
    counts = np.clip(np.round((sim - pot_offset) / pot_scale + rng.normal(0, noise_counts, len(sim))), 0, 1023)
    path = Path(path)
    with path.open("w") as f:
        f.write(f"t_ms,cmd_deg,raw_{servo}\n")
        for i, (c, r) in enumerate(zip(cmd, counts)):
            f.write(f"{i * 20},{int(c)},{int(r)}\n")
    return path




# --------------------------------------------------------------------------
# The bench demo: same part number, different locks; a payload caught
# --------------------------------------------------------------------------


def load_lock(name_or_path: str | Path):
    """A lock by file path or by name in :data:`antares.lock.LOCK_DIR`."""
    from antares.lock import LOCK_DIR, LOCK_SUFFIX, RobotLock

    p = Path(name_or_path)
    if not p.is_file():
        p = LOCK_DIR / f"{name_or_path}{LOCK_SUFFIX}"
    return RobotLock.load(p)


def cross_scores(datasets: dict[str, Dataset], locks: dict[str, object]) -> dict[str, dict[str, float]]:
    """``{dataset: {lock: degrees}}`` plus each dataset on the uncalibrated
    servo (``"default"``): every recording replayed through every lock."""
    from antares.calibrate import default_simulate, score

    simulate = default_simulate(robot=ROBOT)
    params = {"default": simulate.default_params, **{k: v.params for k, v in locks.items()}}
    return {
        dname: {lname: float(score(p, list(ds.episodes), simulate).overall_deg) for lname, p in params.items()}
        for dname, ds in datasets.items()
    }


def payload_check(ds: Dataset, lock, ratio: float = 1.5):
    """:func:`antares.check.check` of ``lock`` on ``ds`` (the servo after a change)."""
    from antares.calibrate import default_simulate
    from antares.check import check

    return check(list(ds.episodes), lock, ratio=ratio, simulate=default_simulate(robot=ROBOT), dataset=ds.repo_id)


__all__ = [
    "FPS", "HOLDS", "JOINT", "PotFit", "ROBOT", "ROBOT_TYPE", "Run", "command_at", "cross_scores",
    "episodes_from_run", "fit_pot", "import_runs", "load_lock", "payload_check", "program", "read_run",
    "simulate_run", "timing_report",
]


# --------------------------------------------------------------------------
# `antares bench ...`
# --------------------------------------------------------------------------


def _pairs(items: Sequence[str], what: str) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise SystemExit(f"error: {what} must look like name=value, got {item!r}")
        out[key] = value
    return out


def cmd_import(args) -> int:
    out, pot, timing = import_runs(args.csv, args.servo, args.out, seconds=args.episode_seconds)
    n_eps = sum(t["frames"] for t in timing) // int(round(args.episode_seconds * FPS))
    print(f"wrote {out}: {len(timing)} run(s), {n_eps} episodes of {args.episode_seconds:g} s")
    print(f"pot: deg = {pot.scale:.4f} * counts + {pot.offset:.2f}  (holds fit to {pot.rms_deg:.2f} deg RMS)")
    for counts, deg in pot.points[:5]:
        print(f"  {deg:5.0f} deg  <-  {counts:6.1f} counts")
    for t in timing:
        flag = "" if t["late_ticks"] == 0 else f"  ({t['late_ticks']} ticks over 25 ms)"
        print(f"  {t['file']}: {t['frames']} frames, median tick {t['dt_ms_median']:.1f} ms{flag}")
    if pot.rms_deg > 3.0:
        print("warning: the holds do not sit on a line; check the feedback wire and that the servo reached each hold")
    return 0


def cmd_simulate(args) -> int:
    from antares.sim import Arm

    base = Arm(robot=ROBOT).default_params()
    changes = {k: (float(v),) for k, v in _pairs(args.set, "--set").items()}
    params = base.replace(**changes)
    path = simulate_run(args.out, params, servo=args.servo, pot_scale=args.pot_scale, pot_offset=args.pot_offset,
                        noise_counts=args.noise, seed=args.seed)
    print(f"wrote {path} (simulated servo {args.servo}: {', '.join(f'{k}={v[0]:g}' for k, v in changes.items()) or 'defaults'})")
    return 0


def cmd_demo(args) -> int:
    from antares.data import read_local

    servos = _pairs(args.servo, "--servo")
    locks = {k: load_lock(v) for k, v in _pairs(args.lock, "--lock").items()}
    datasets = {k: read_local(v, repo_id=v) for k, v in servos.items()}
    scores = cross_scores(datasets, locks)
    names = list(locks)
    print("Each recording replayed through each servo's lock (RMS tracking error, degrees):\n")
    width = max(8, *(len(n) for n in names))
    print("recording".ljust(12) + "default".rjust(10) + "".join(f"lock {n}".rjust(width + 6) for n in names))
    for d, row in scores.items():
        print(d.ljust(12) + f"{row['default']:10.3f}" + "".join(f"{row[n]:{width + 6}.3f}" for n in names))
    print()
    for d, row in scores.items():
        if d not in locks:
            continue
        own = row[d]
        others = {n: row[n] for n in names if n != d}
        line = f"servo {d}: own lock {own:.3f} deg, uncalibrated {row['default']:.3f} deg ({row['default'] / own:.1f}x worse)"
        if others:
            worst_other = min(others.values())
            line += f", best other lock {worst_other:.3f} deg ({worst_other / own:.1f}x worse)"
        print(line)
    if args.payload:
        key, _, path = args.payload.partition("=") if "=" in args.payload else ("a", "", args.payload)
        if key not in locks:
            raise SystemExit(f"error: --payload names servo {key!r} but there is no --lock {key}=...")
        res = payload_check(read_local(path, repo_id=path), locks[key], ratio=args.ratio)
        print()
        print(f"payload check, servo {key} with the coin ({path}): {res.score:.3f} deg vs its lock's "
              f"{res.reference:.3f} deg, limit {res.limit:.3f} -> {res.verdict.upper()}")
        if args.fresh:
            fres = payload_check(read_local(args.fresh, repo_id=args.fresh), locks[key], ratio=args.ratio)
            print(f"same servo, no coin ({args.fresh}): {fres.score:.3f} deg -> {fres.verdict.upper()}")
    return 0


def add_parser(sub) -> None:
    """Register ``antares bench`` on the CLI's subparsers."""
    p = sub.add_parser("bench", help="a hobby servo on an Arduino as a one-joint robot (import, simulate, demo, live)")
    bsub = p.add_subparsers(dest="bench_cmd", required=True)

    b = bsub.add_parser("import", help="turn bench CSVs (bench/capture.py) into a dataset for one servo")
    b.add_argument("csv", nargs="+", help="runs of the same servo")
    b.add_argument("--servo", default="a", help="which raw_<servo> column (default a)")
    b.add_argument("--out", required=True, help="dataset directory to write")
    b.add_argument("--episode-seconds", type=float, default=EPISODE_SECONDS)
    b.set_defaults(func=cmd_import)

    b = bsub.add_parser("simulate", help="write the CSV a servo with given parameters would log (no hardware)")
    b.add_argument("out")
    b.add_argument("--servo", default="a")
    b.add_argument("--set", action="append", default=[], metavar="PARAM=VALUE",
                   help="change a parameter from the sg92r model's defaults, e.g. --set armature=0.0005")
    b.add_argument("--pot-scale", type=float, default=0.28, help="degrees per pot count")
    b.add_argument("--pot-offset", type=float, default=-40.0, help="degrees at zero counts")
    b.add_argument("--noise", type=float, default=1.0, help="pot noise, counts (std)")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_simulate)

    b = bsub.add_parser("demo", help="every servo's recording through every lock, then the payload check")
    b.add_argument("--servo", action="append", required=True, metavar="NAME=DATASET",
                   help="a servo's held-out recording (e.g. a=datasets/servo_a_test); repeat per servo")
    b.add_argument("--lock", action="append", required=True, metavar="NAME=LOCK",
                   help="that servo's lock (name or path); same NAME as --servo")
    b.add_argument("--payload", metavar="NAME=DATASET", help="a recording of servo NAME with the coin taped on")
    b.add_argument("--fresh", metavar="DATASET", help="a recording of the same servo without the coin, for contrast")
    b.add_argument("--ratio", type=float, default=1.5, help="drift limit as a multiple of the lock's held-out score")
    b.set_defaults(func=cmd_demo)
