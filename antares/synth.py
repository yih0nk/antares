"""Synthetic robot logs: what an arm with known parameters would have recorded.

Used to test drift detection without a physical arm. Commands are smooth
random reaching motions in dataset units; the "measured" state is the
MuJoCo arm simulated with the given :class:`~antares.sim.SimParams`, mapped
back to dataset units, plus sensor noise. Because the true parameters are
known, a refit can be checked against them.

:func:`write_local` saves a dataset in LeRobot layout (``meta/info.json``
plus ``data/**.parquet``) so every command that takes a dataset can read it
with :func:`antares.data.read_local`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from antares.data import Dataset, Episode

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def command_trajectory(n_frames: int, fps: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random commands, (T, 6) in dataset units.

    Each body joint follows a sum of three low-frequency sinusoids around a
    random centre, kept inside [-80, 80] so the joint limits never clip;
    the gripper opens and closes inside [10, 90].
    """
    t = np.arange(n_frames) / fps
    action = np.empty((n_frames, len(JOINTS)), dtype=np.float64)
    for j in range(len(JOINTS) - 1):
        centre = rng.uniform(-30, 30)
        wave = np.zeros(n_frames)
        for _ in range(3):
            freq = rng.uniform(0.1, 0.8)
            wave += rng.uniform(5, 20) * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        action[:, j] = np.clip(centre + wave, -80, 80)
    freq = rng.uniform(0.1, 0.4)
    action[:, -1] = 50 + 40 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return action


def synthesize(
    params,
    n_episodes: int = 30,
    n_frames: int = 300,
    fps: float = 30.0,
    seed: int = 0,
    noise_units: float = 0.2,
    repo_id: str = "synthetic/arm",
    robot_type: str = "so101_follower",
    arm=None,
) -> Dataset:
    """Simulate ``n_episodes`` recordings of an arm whose true dynamics are ``params``.

    The sim starts each episode at rest at its first command, like a real
    arm that has settled before recording starts. ``noise_units`` is the
    standard deviation of Gaussian sensor noise on the measured state, in
    dataset units (100 units is half a joint's range).
    """
    if arm is None:
        from antares.sim import Arm

        arm = Arm()
    arm.apply(params)
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(n_episodes):
        action = command_trajectory(n_frames, fps, rng)
        commanded = Episode(index=i, fps=fps, joint_names=JOINTS, state=action, action=action)
        sim_q = arm.replay(commanded)
        state = arm.unit_map.to_units(sim_q) + rng.normal(0.0, noise_units, size=sim_q.shape)
        episodes.append(Episode(index=i, fps=fps, joint_names=JOINTS, state=state, action=action))
    return Dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        joint_names=JOINTS,
        episodes=tuple(episodes),
    )


def worn(params, changes: Mapping[str, float]):
    """``params`` with wear applied: ``latency_frames`` is shifted by its
    value, every other parameter is multiplied by its value. Per-joint
    fields are changed on every joint alike.

    ``worn(p, {"kp": 0.7, "frictionloss": 2.0, "latency_frames": 1})`` is a
    weaker, stickier arm that reacts one frame later.
    """
    known = params.field_names()
    out = {}
    for name, v in changes.items():
        if name not in known:
            raise ValueError(f"unknown parameter {name!r}; expected one of {sorted(known)}")
        if name == "latency_frames":
            new = np.maximum(params.per_joint(name) + int(round(v)), 0)
        else:
            new = params.per_joint(name) * float(v)
        out[name] = tuple(new.tolist()) if params.is_per_joint(name) else new[0].item()
    return params.replace(**out)


def parse_changes(spec: str, parse: Callable[[str], float] = float) -> dict[str, float]:
    """``"kp=0.7,frictionloss=2"`` -> ``{"kp": 0.7, "frictionloss": 2.0}``."""
    out = {}
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected name=value, got {item!r}")
        out[name.strip()] = parse(value)
    return out


def write_local(ds: Dataset, root: str | Path) -> Path:
    """Write ``ds`` in LeRobot layout under ``root``; readable by ``read_local``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    names = [f"{j}.pos" for j in ds.joint_names]
    info = {
        "robot_type": ds.robot_type,
        "fps": ds.fps,
        "total_episodes": len(ds.episodes),
        "total_frames": sum(len(e) for e in ds.episodes),
        "features": {
            "observation.state": {"dtype": "float64", "shape": [len(names)], "names": names},
            "action": {"dtype": "float64", "shape": [len(names)], "names": names},
        },
        "source": "antares.synth",
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    columns: dict[str, list] = {
        "episode_index": [],
        "frame_index": [],
        "observation.state": [],
        "action": [],
    }
    for ep in ds.episodes:
        n = len(ep)
        columns["episode_index"].extend([ep.index] * n)
        columns["frame_index"].extend(range(n))
        columns["observation.state"].extend(ep.state.tolist())
        columns["action"].extend(ep.action.tolist())
    path = root / "data" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return root


__all__ = ["JOINTS", "command_trajectory", "parse_changes", "synthesize", "worn", "write_local"]
