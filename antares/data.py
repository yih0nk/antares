"""Load real robot episodes from LeRobot-format datasets on the Hugging Face Hub.

Only the tabular part is used: per-frame commanded ``action`` and measured
``observation.state``. Videos are never downloaded.

SO-100/101 datasets store joint values normalized to each arm's own
calibrated range (body joints in [-100, 100], gripper in [0, 100]), not
radians. Mapping them into the simulator's joint space is the job of
:mod:`antares.sim`; this module keeps values in dataset units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Episode:
    index: int
    fps: float
    joint_names: tuple[str, ...]
    state: np.ndarray  # (T, J) measured joint positions, dataset units
    action: np.ndarray  # (T, J) commanded joint targets, dataset units

    def __len__(self) -> int:
        return len(self.state)

    @property
    def duration(self) -> float:
        return len(self) / self.fps


@dataclass(frozen=True)
class Dataset:
    repo_id: str
    robot_type: str
    fps: float
    joint_names: tuple[str, ...]
    episodes: tuple[Episode, ...]


def canonical_joint_name(name: str) -> str:
    """Map dataset-specific joint names onto one vocabulary.

    SO-100 datasets use ``main_shoulder_pan``; SO-101 use ``shoulder_pan.pos``.
    Both become ``shoulder_pan``.
    """
    name = name.removeprefix("main_")
    return name.removesuffix(".pos")


def episodes_from_columns(
    columns: dict[str, list], fps: float, joint_names: tuple[str, ...]
) -> list[Episode]:
    """Split flat per-frame columns into episodes ordered by frame index."""
    ep = np.asarray(columns["episode_index"])
    frame = np.asarray(columns["frame_index"])
    state = np.asarray(columns["observation.state"], dtype=np.float64)
    action = np.asarray(columns["action"], dtype=np.float64)

    episodes = []
    for idx in np.unique(ep):
        rows = np.flatnonzero(ep == idx)
        rows = rows[np.argsort(frame[rows], kind="stable")]
        episodes.append(
            Episode(
                index=int(idx),
                fps=fps,
                joint_names=joint_names,
                state=state[rows],
                action=action[rows],
            )
        )
    return episodes


def read_local(root: str | Path, repo_id: str = "local") -> Dataset:
    """Read a LeRobot dataset already on disk (``meta/info.json`` + ``data/**.parquet``)."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]
    from antares.validate import feature_names  # validate imports this module

    names = feature_names(features["observation.state"])
    if names is None or names != feature_names(features["action"]):
        raise ValueError("action and observation.state joints differ; cannot replay")
    joint_names = tuple(canonical_joint_name(n) for n in names)
    fps = float(info["fps"])

    wanted = ["episode_index", "frame_index", "observation.state", "action"]
    columns: dict[str, list] = {k: [] for k in wanted}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=wanted).to_pydict()
        for k in wanted:
            columns[k].extend(table[k])
    if not columns["episode_index"]:
        raise ValueError(f"no parquet data under {root / 'data'}")

    return Dataset(
        repo_id=repo_id,
        robot_type=info.get("robot_type", "unknown"),
        fps=fps,
        joint_names=joint_names,
        episodes=tuple(episodes_from_columns(columns, fps, joint_names)),
    )


def load_dataset(repo_id: str, cache_dir: str | Path | None = None) -> Dataset:
    """Download (tabular files only) and read a LeRobot dataset from the Hub."""
    from antares.hub import snapshot

    root = snapshot(
        repo_id,
        allow_patterns=["meta/info.json", "data/**/*.parquet"],
        cache_dir=cache_dir,
    )
    return read_local(root, repo_id=repo_id)
