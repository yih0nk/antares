"""Fleet scan: find every public SO-100/101 dataset on the Hub and calibrate each.

Every community LeRobot dataset was recorded on somebody's own arm, so each
one is a different physical robot. :func:`discover` searches the Hugging Face
Hub, fetches only ``meta/info.json`` for each hit and keeps the datasets the
simulator can replay (SO-100/101 robot type, the six canonical joints, 30 or
50 fps, enough episodes, a tabular part small enough to download). Every
rejected candidate keeps the reason it was dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from antares.data import canonical_joint_name
from antares.lock import LOCK_DIR

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

FLEET_LOCK_DIR: Path = LOCK_DIR / "fleet"
FLEET_REPORT_NAME = "fleet.json"

#: Hub search strings and tag filters tried in turn; hits are merged.
SEARCH_TERMS: tuple[str, ...] = ("so100", "so101", "so-100", "so-101", "so_100", "so_101", "so_arm")
TAG_FILTERS: tuple[str, ...] = ("so100", "so101")

#: Always inspected first and exempt from the one-dataset-per-owner rule.
PINNED: tuple[str, ...] = ("lerobot/svla_so101_pickplace", "lerobot/svla_so100_pickplace")

MIN_EPISODES = 15
ALLOWED_FPS: tuple[float, ...] = (30.0, 50.0)
MAX_DATA_BYTES = 200 * 2**20  # tabular part only; videos are never downloaded

# ``so100``/``so101`` in any spelling, plus LeRobot's newer unified
# ``so_follower`` name for the same arm.
_ROBOT_RE = re.compile(r"so[-_ ]?(?:arm)?[-_ ]?10[01]|^so[-_]?follower$")
# Datasets recorded in a simulator are not physical robots.
_SIM_RE = re.compile(r"(?:^|[^a-z])(?:sim|synthetic|synth|pybullet|isaac(?:sim|lab)?|mujoco|genesis|maniskill|gazebo)(?:[^a-z]|$)")


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One Hub dataset that was inspected, accepted or not."""

    repo_id: str
    accepted: bool
    reason: str = ""  # why it was rejected; empty when accepted
    robot_type: str = ""
    fps: float = 0.0
    n_episodes: int = 0
    n_frames: int = 0
    data_bytes: int = 0
    downloads: int = 0
    codebase_version: str = ""

    @property
    def owner(self) -> str:
        return self.repo_id.split("/", 1)[0] if "/" in self.repo_id else ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Candidate:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def accepted(candidates: Iterable[Candidate]) -> list[Candidate]:
    return [c for c in candidates if c.accepted]


def _names(feature: Any) -> list[str] | None:
    """``features[x]["names"]`` as a flat list; v2 datasets nest it under a key."""
    if not isinstance(feature, dict):
        return None
    names = feature.get("names")
    if isinstance(names, dict):
        if len(names) != 1:
            return None
        names = next(iter(names.values()))
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return None
    return names


def check_info(
    repo_id: str,
    info: dict[str, Any],
    data_bytes: int | None = None,
    downloads: int = 0,
) -> Candidate:
    """Decide from ``meta/info.json`` whether a dataset can join the fleet.

    ``data_bytes`` is the size of the ``data/**/*.parquet`` files when known;
    ``None`` skips the size check (the caller may fill it in afterwards).
    """
    robot_type = str(info.get("robot_type") or "")
    fps = float(info.get("fps") or 0)
    n_episodes = int(info.get("total_episodes") or 0)
    n_frames = int(info.get("total_frames") or 0)
    version = str(info.get("codebase_version") or "")

    def reject(reason: str) -> Candidate:
        return Candidate(
            repo_id=repo_id,
            accepted=False,
            reason=reason,
            robot_type=robot_type,
            fps=fps,
            n_episodes=n_episodes,
            n_frames=n_frames,
            data_bytes=int(data_bytes or 0),
            downloads=downloads,
            codebase_version=version,
        )

    sim = _SIM_RE.search(repo_id.lower())
    if sim:
        return reject(f"looks like a simulation dataset (name contains {sim.group(0).strip('-_ /.')!r})")
    if not re.match(r"v[23]\.", version):
        return reject(f"codebase_version {version or 'missing'} (need the v2/v3 parquet layout)")
    if not _ROBOT_RE.search(robot_type.lower()):
        return reject(f"robot_type {robot_type or 'missing'!r} is not an SO-100/101")
    features = info.get("features") or {}
    for key in ("action", "observation.state"):
        names = _names(features.get(key))
        if names is None:
            return reject(f"{key} has no joint names")
        if len(names) != 6:
            return reject(f"{key} has {len(names)} joints, not 6")
        canonical = tuple(canonical_joint_name(n) for n in names)
        if canonical != JOINTS:
            return reject(f"{key} joints {list(canonical)} are not the SO-101 joints")
    if fps not in ALLOWED_FPS:
        return reject(f"fps {fps:g} not in {sorted(int(f) for f in ALLOWED_FPS)}")
    if n_episodes < MIN_EPISODES:
        return reject(f"{n_episodes} episodes < {MIN_EPISODES}")
    if data_bytes is not None:
        if data_bytes <= 0:
            return reject("no data/*.parquet files")
        if data_bytes > MAX_DATA_BYTES:
            return reject(f"tabular data {data_bytes / 2**20:.0f} MB > {MAX_DATA_BYTES // 2**20} MB")
    return Candidate(
        repo_id=repo_id,
        accepted=True,
        robot_type=robot_type,
        fps=fps,
        n_episodes=n_episodes,
        n_frames=n_frames,
        data_bytes=int(data_bytes or 0),
        downloads=downloads,
        codebase_version=version,
    )


# --------------------------------------------------------------------------
# Hub access (injectable for tests)
# --------------------------------------------------------------------------


def hub_search(
    api: Any = None,
    search_terms: Sequence[str] = SEARCH_TERMS,
    tag_filters: Sequence[str] = TAG_FILTERS,
    per_query: int = 300,
) -> list[tuple[str, int]]:
    """``(repo_id, downloads)`` for every public, ungated hit, most downloaded first."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    hits: dict[str, int] = {}
    queries: list[dict[str, Any]] = [{"search": t} for t in search_terms]
    queries += [{"filter": t} for t in tag_filters]
    for q in queries:
        try:
            found = list(api.list_datasets(sort="downloads", limit=per_query, **q))
        except Exception:  # noqa: BLE001 - one failing query must not end discovery
            continue
        for d in found:
            if getattr(d, "gated", False) or getattr(d, "private", False) or getattr(d, "disabled", False):
                continue
            hits[d.id] = max(hits.get(d.id, 0), int(getattr(d, "downloads", 0) or 0))
    return sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))


def fetch_info(repo_id: str) -> dict[str, Any]:
    """Download only ``meta/info.json`` (cached by huggingface_hub) and parse it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
    return json.loads(Path(path).read_text())


def fetch_data_bytes(repo_id: str, api: Any = None) -> int:
    """Total size of ``data/**/*.parquet`` from the repo's file metadata."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    info = api.dataset_info(repo_id, files_metadata=True)
    total = 0
    for s in info.siblings or []:
        name = s.rfilename
        if name.startswith("data/") and name.endswith(".parquet"):
            total += int(s.size or 0)
    return total


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def discover(
    limit: int = 60,
    *,
    max_inspect: int | None = None,
    one_per_owner: bool = True,
    pinned: Sequence[str] = PINNED,
    api: Any = None,
    search: Callable[..., list[tuple[str, int]]] | None = None,
    fetch_info: Callable[[str], dict[str, Any]] | None = None,
    fetch_size: Callable[[str], int] | None = None,
    on_progress: Callable[[Candidate], None] | None = None,
) -> list[Candidate]:
    """Inspect Hub hits most-downloaded first until ``limit`` are accepted.

    Returns every inspected candidate, accepted or rejected with a reason,
    in inspection order. ``max_inspect`` (default ``5 * limit``) bounds the
    number of ``meta/info.json`` fetches. With ``one_per_owner`` a second
    dataset from the same account is rejected, since it is most likely the
    same physical arm recorded twice; ``pinned`` datasets come first and are
    exempt (the two demo datasets are different arms of one organisation).
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if max_inspect is None:
        max_inspect = 5 * limit
    search = search or (lambda: hub_search(api))
    get_info = fetch_info or globals()["fetch_info"]
    get_size = fetch_size or (lambda rid: fetch_data_bytes(rid, api))

    hits = search()
    downloads_of = dict(hits)
    order = [(rid, downloads_of.get(rid, 0)) for rid in pinned]
    order += [(rid, dl) for rid, dl in hits if rid not in set(pinned)]

    out: list[Candidate] = []
    owners: dict[str, str] = {}
    n_accepted = 0
    for repo_id, downloads in order:
        if n_accepted >= limit or len(out) >= max_inspect:
            break
        owner = repo_id.split("/", 1)[0] if "/" in repo_id else ""
        if one_per_owner and owner in owners and repo_id not in pinned:
            cand = Candidate(repo_id, False, f"same owner as {owners[owner]} (likely the same arm)", downloads=downloads)
        else:
            try:
                info = get_info(repo_id)
            except Exception as e:  # noqa: BLE001 - record and move on
                cand = Candidate(repo_id, False, f"no readable meta/info.json ({type(e).__name__})", downloads=downloads)
            else:
                cand = check_info(repo_id, info, None, downloads)
                if cand.accepted:
                    try:
                        size = get_size(repo_id)
                    except Exception as e:  # noqa: BLE001
                        cand = Candidate(repo_id, False, f"could not list files ({type(e).__name__})", downloads=downloads)
                    else:
                        cand = check_info(repo_id, info, size, downloads)
        if cand.accepted:
            n_accepted += 1
            owners.setdefault(owner, repo_id)
        out.append(cand)
        if on_progress is not None:
            on_progress(cand)
    return out


__all__ = [
    "ALLOWED_FPS",
    "FLEET_LOCK_DIR",
    "FLEET_REPORT_NAME",
    "JOINTS",
    "MAX_DATA_BYTES",
    "MIN_EPISODES",
    "Candidate",
    "accepted",
    "check_info",
    "discover",
    "fetch_data_bytes",
    "fetch_info",
    "hub_search",
]
