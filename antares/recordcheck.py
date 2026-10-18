"""Record a few fresh episodes with LeRobot, then run the drift gate on them.

One command for a technician at an arm: ``antares record-check`` drives
``lerobot-record`` to capture ``num_episodes`` episodes into a local
directory, then runs exactly what ``antares check <dir> --lock <name>``
runs on that directory and prints PASS or FAIL.

LeRobot's recorder is driven through its CLI, not imported: it needs
Python >= 3.12 and its own hardware extras, so it usually lives in a
separate environment from Antares. ``--recorder`` points at that
environment's ``lerobot-record`` when it is not on ``PATH``.

Verified against LeRobot 0.6.1 (PyPI, 2026-08-03) and the docs on
2026-10-17:

* recording guide, ``lerobot-record`` flags and keyboard controls:
  https://huggingface.co/docs/lerobot/il_robots#record-a-dataset
* ``DatasetRecordConfig`` (``--dataset.*`` fields, defaults, and that
  ``root`` is "Root directory where the dataset will be stored ... If None,
  defaults to $HF_LEROBOT_HOME/repo_id"):
  https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/dataset.py
* install extras (``core_scripts`` for ``lerobot-record``, ``feetech`` for
  SO-100/101 motors): https://huggingface.co/docs/lerobot/installation
* on-disk layout (``meta/info.json`` + ``data/**/*.parquet``, what
  :func:`antares.data.read_local` reads):
  https://huggingface.co/docs/lerobot/lerobot-dataset-v3

Flags used: ``--robot.type --robot.port --robot.id --teleop.type
--teleop.port --teleop.id --dataset.repo_id --dataset.root
--dataset.num_episodes --dataset.single_task --dataset.push_to_hub
--dataset.episode_time_s --dataset.reset_time_s``. The recorder appends a
date-time stamp to ``repo_id`` unless ``--dataset.no_stamp=true``; with an
explicit ``--dataset.root`` that only changes the name stored in the
metadata, never the directory, so this module leaves it alone.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

RECORDER = "lerobot-record"
INSTALL_HINT = (
    "install LeRobot with its recording scripts and Feetech motor support "
    "(Python >= 3.12): pip install 'lerobot[core_scripts,feetech]'  "
    "then pass --recorder /path/to/lerobot-record if it is not on PATH"
)
DEFAULT_ROBOT_TYPE = "so101_follower"
DEFAULT_TELEOP_TYPE = "so101_leader"
DEFAULT_EPISODES = 5
DEFAULT_TASK = "antares check"
RECORDINGS_DIR = Path("recordings")


class RecordError(RuntimeError):
    """Recording did not produce a dataset (missing recorder, non-zero exit,
    nothing written). The check never ran."""


@dataclass
class RecordOptions:
    """Everything ``lerobot-record`` needs; see :func:`build_record_command`."""

    robot_port: str
    robot_id: str
    robot_type: str = DEFAULT_ROBOT_TYPE
    teleop_type: str = DEFAULT_TELEOP_TYPE
    teleop_port: str | None = None
    teleop_id: str | None = None  # default: "<robot_id>_leader"
    num_episodes: int = DEFAULT_EPISODES
    task: str = DEFAULT_TASK
    root: Path | None = None  # default: recordings/<robot_id>/<YYYYMMDD-HHMMSS>
    repo_id: str | None = None  # default: local/antares-check-<robot_id>
    episode_time_s: float | None = None
    reset_time_s: float | None = None
    extra: list[str] = field(default_factory=list)  # passed through verbatim
    recorder: str | None = None  # path to lerobot-record; default: on PATH

    def resolved_teleop_id(self) -> str:
        return self.teleop_id or f"{self.robot_id}_leader"

    def resolved_repo_id(self) -> str:
        return self.repo_id or f"local/antares-check-{self.robot_id}"


def default_root(robot_id: str, now: datetime | None = None) -> Path:
    """``recordings/<robot_id>/<YYYYMMDD-HHMMSS>``: one directory per run."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return RECORDINGS_DIR / robot_id / stamp


def resolve_root(opts: RecordOptions, now: datetime | None = None) -> Path:
    return Path(opts.root) if opts.root is not None else default_root(opts.robot_id, now)


def build_record_command(opts: RecordOptions, root: Path | None = None, recorder: str = RECORDER) -> list[str]:
    """The ``lerobot-record`` argv for ``opts``. Pure.

    ``root`` defaults to :func:`resolve_root`; the dataset is never pushed
    to the Hub. Raises ``ValueError`` when a leader arm is the teleop device
    and no ``teleop_port`` was given.
    """
    if not opts.robot_port:
        raise ValueError("robot_port is required (the follower arm's serial port)")
    if not opts.robot_id:
        raise ValueError("robot_id is required (the LeRobot id the arm was calibrated under)")
    if opts.num_episodes < 1:
        raise ValueError("num_episodes must be >= 1")
    if opts.teleop_type.endswith("_leader") and not opts.teleop_port:
        raise ValueError(f"teleop_port is required for teleop type {opts.teleop_type!r}")
    root = root if root is not None else resolve_root(opts)

    cmd = [
        recorder,
        f"--robot.type={opts.robot_type}",
        f"--robot.port={opts.robot_port}",
        f"--robot.id={opts.robot_id}",
        f"--teleop.type={opts.teleop_type}",
    ]
    if opts.teleop_port:
        cmd.append(f"--teleop.port={opts.teleop_port}")
    cmd += [
        f"--teleop.id={opts.resolved_teleop_id()}",
        f"--dataset.repo_id={opts.resolved_repo_id()}",
        f"--dataset.root={root}",
        f"--dataset.num_episodes={int(opts.num_episodes)}",
        f"--dataset.single_task={opts.task}",
        "--dataset.push_to_hub=false",
    ]
    if opts.episode_time_s is not None:
        cmd.append(f"--dataset.episode_time_s={opts.episode_time_s:g}")
    if opts.reset_time_s is not None:
        cmd.append(f"--dataset.reset_time_s={opts.reset_time_s:g}")
    cmd += [str(a) for a in opts.extra]
    return cmd


def find_recorder(explicit: str | None = None) -> str | None:
    """Path of ``lerobot-record``: ``explicit`` if it is an executable file
    (or resolves on ``PATH``), else ``shutil.which``. None when absent."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        return shutil.which(explicit)
    return shutil.which(RECORDER)


def has_dataset(root: Path | str) -> bool:
    """Does ``root`` hold something :func:`antares.data.read_local` can read?"""
    root = Path(root)
    if not (root / "meta" / "info.json").is_file() or not (root / "data").is_dir():
        return False
    return any((root / "data").rglob("*.parquet"))


def record(
    opts: RecordOptions,
    *,
    dry_run: bool = False,
    run: Callable[..., Any] = subprocess.run,
    now: datetime | None = None,
) -> Path:
    """Run ``lerobot-record`` with the technician's terminal attached (so
    LeRobot's prompts and keyboard controls work) and return the dataset
    directory. ``dry_run`` only resolves the directory.

    Raises :class:`RecordError` when the recorder is missing (with the
    install hint), exits non-zero, or leaves no dataset behind.
    """
    root = resolve_root(opts, now)
    if dry_run:
        return root
    if has_dataset(root):
        raise RecordError(
            f"{root} already holds a dataset; pass --skip-record to check it, or another --root"
        )
    recorder = find_recorder(opts.recorder)
    if recorder is None:
        what = opts.recorder or RECORDER
        raise RecordError(f"{what}: not found; {INSTALL_HINT}")
    cmd = build_record_command(opts, root, recorder=recorder)
    root.parent.mkdir(parents=True, exist_ok=True)
    print(f"recording {opts.num_episodes} episode(s) -> {root}", file=sys.stderr, flush=True)
    print("  " + shlex.join(cmd), file=sys.stderr, flush=True)
    try:
        proc = run(cmd, check=False)  # inherit stdio
    except FileNotFoundError as e:
        raise RecordError(f"{cmd[0]}: not found; {INSTALL_HINT}") from e
    code = getattr(proc, "returncode", 0)
    if code:
        raise RecordError(f"{RECORDER} exited with status {code}; nothing was checked")
    if not has_dataset(root):
        raise RecordError(f"{RECORDER} finished but wrote no dataset under {root} (need meta/info.json + data/*.parquet)")
    return root


def record_and_check(
    opts: RecordOptions,
    lock: str | dict,
    *,
    check_kwargs: dict[str, Any] | None = None,
    dry_run: bool = False,
    skip_record: bool = False,
    run: Callable[..., Any] = subprocess.run,
    now: datetime | None = None,
):
    """Record (unless ``skip_record`` and ``opts.root`` already holds a
    dataset), then check the directory through the same path as ``antares
    check <dir>`` (``server.get_dataset`` + ``server.run_check``), so every
    check option behaves identically. Returns the
    :class:`~antares.check.CheckResult`, or ``None`` on ``dry_run``.

    ``check_kwargs`` are ``run_check``'s keywords (``threshold_deg``,
    ``ratio``, ``min_episodes``, ``last``). ``lock`` is a saved lock's name
    or its JSON dict. Loader and lock errors propagate as the
    ``HTTPException`` ``antares check`` prints.
    """
    if skip_record:
        root = resolve_root(opts, now)
        if not has_dataset(root):
            raise RecordError(f"--skip-record: no LeRobot dataset under {root} (need meta/info.json + data/*.parquet)")
    else:
        root = record(opts, dry_run=dry_run, run=run, now=now)
    if dry_run:
        return None

    from antares import server

    ds = server.get_dataset(str(root))
    return server.run_check(ds, lock, **(check_kwargs or {}))


def banner(res: Any, robot_id: str) -> str:
    """``PASS arm7: 1.067 deg (limit 1.582)`` / ``FAIL arm7: ...``."""
    if res.verdict == "insufficient_data":
        return f"FAIL {robot_id}: insufficient data, {res.n_episodes} episode(s), need {res.min_episodes}"
    word = "PASS" if res.passed else "FAIL"
    lim = f"{res.limit:.3f}" if res.limit is not None else "n/a"
    line = f"{word} {robot_id}: {res.score:.3f} deg (limit {lim})"
    if not res.passed and res.worst_joint:
        line += f", worst joint {res.worst_joint}"
    return line


__all__ = [
    "DEFAULT_EPISODES",
    "DEFAULT_ROBOT_TYPE",
    "DEFAULT_TASK",
    "DEFAULT_TELEOP_TYPE",
    "INSTALL_HINT",
    "RECORDER",
    "RecordError",
    "RecordOptions",
    "banner",
    "build_record_command",
    "default_root",
    "find_recorder",
    "has_dataset",
    "record",
    "record_and_check",
    "resolve_root",
]
