"""Can antares replay this dataset? One place that decides, for the fleet
scan, the API and the CLI alike.

:func:`validate_dataset_info` reads a LeRobot ``meta/info.json`` (no data
download) and answers with a :class:`Verdict`: hard ``reasons`` the dataset
cannot be used (not a single SO-100/101 arm, not the six canonical joints,
an fps the sim does not step at, too few episodes) and soft ``warnings``
(few episodes, a name that smells like a simulator, a missing codebase
version). :func:`validate_dataset` looks at loaded data: NaNs, and values
whose range says they are not LeRobot's normalized units.

Nothing here imports MuJoCo or touches the network.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from antares.data import Dataset, canonical_joint_name

JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER_INDEX = JOINTS.index("gripper")

#: Frame rates the replay steps at (dataset frame = whole number of 5 ms physics steps).
ALLOWED_FPS: tuple[float, ...] = (30.0, 50.0)
#: Fewest episodes a user-supplied dataset may have; the fleet scan wants more.
MIN_EPISODES = 5
#: Below this many episodes the fit still runs but the held-out split is thin.
RECOMMENDED_EPISODES = 15

#: LeRobot's normalized units: body joints in [-100, 100], gripper in [0, 100].
BODY_RANGE = (-120.0, 120.0)
GRIPPER_RANGE = (-5.0, 105.0)

# ``so100``/``so101`` in any spelling. ``so_follower`` is LeRobot's newer
# unified name for the SO-101 follower arm.
_SO100_RE = re.compile(r"so[-_ ]?(?:arm)?[-_ ]?100")
_SO101_RE = re.compile(r"so[-_ ]?(?:arm)?[-_ ]?101|^so[-_]?follower$")
_BIMANUAL_RE = re.compile(r"(?:^|[^a-z])(?:bi|bimanual|dual|dualarm|two_arm)(?:[^a-z]|$)")
# Datasets recorded in a simulator are not physical robots, and merges of
# several people's recordings are not one robot.
_SIM_RE = re.compile(
    r"(?:^|[^a-z])(?:synthetic|synth|pybullet|isaac(?:sim|lab)?|mujoco|genesis|maniskill|gazebo)(?:[^a-z]|$)"
    r"|sim(?:[^a-z]|$)"
)
_MERGED_RE = re.compile(r"(?:^|[^a-z])(?:merged?|combined|community|aggregated|converted)(?:[^a-z]|$)")
_VERSION_RE = re.compile(r"v[23]\.")


def robot_family(robot_type: str | None) -> str | None:
    """``"so100"`` / ``"so101"`` for a recognised single-arm robot type, else None.

    Bimanual rigs (``bi_so100_follower``) are *not* a family: antares
    simulates one six-joint arm.
    """
    rt = str(robot_type or "").strip().lower()
    if not rt or _BIMANUAL_RE.search(rt):
        return None
    if _SO100_RE.search(rt):
        return "so100"
    if _SO101_RE.search(rt):
        return "so101"
    return None


def sim_name_match(name: str) -> str | None:
    """The token that makes ``name`` look like a simulation dataset, if any."""
    m = _SIM_RE.search(str(name).lower())
    return m.group(0).strip("-_ /.") if m else None


def merged_name_match(name: str) -> str | None:
    """The token that makes ``name`` look like a merge of several datasets."""
    m = _MERGED_RE.search(str(name).lower())
    return m.group(0).strip("-_ /.") if m else None


def feature_names(feature: Any) -> list[str] | None:
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


@dataclass
class Verdict:
    """Outcome of a validation. ``ok`` is ``not reasons``."""

    ok: bool
    robot_type: str = ""
    robot_family: str | None = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fps: float = 0.0
    n_episodes: int = 0
    n_frames: int = 0
    joint_names: list[str] = field(default_factory=list)
    codebase_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Verdict:
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def summary(self) -> str:
        if self.ok:
            tail = f"; {len(self.warnings)} warning(s)" if self.warnings else ""
            return f"ok ({self.robot_family or 'unknown'}, {self.n_episodes} episodes @ {self.fps:g} fps){tail}"
        return "rejected: " + "; ".join(self.reasons)


def _finish(v: Verdict) -> Verdict:
    v.ok = not v.reasons
    return v


def validate_dataset_info(
    info: dict[str, Any],
    repo_id: str = "",
    *,
    min_episodes: int = MIN_EPISODES,
    recommended_episodes: int = RECOMMENDED_EPISODES,
) -> Verdict:
    """Judge a dataset from its ``meta/info.json`` alone.

    ``repo_id`` (or an upload name) is only used for the name heuristics,
    which produce warnings here; the fleet scan turns them into rejections
    itself.
    """
    robot_type = str(info.get("robot_type") or "")
    fps = float(info.get("fps") or 0)
    episodes_known = info.get("total_episodes") is not None
    n_episodes = int(info.get("total_episodes") or 0)
    n_frames = int(info.get("total_frames") or 0)
    version = str(info.get("codebase_version") or "")
    v = Verdict(
        ok=False,
        robot_type=robot_type,
        robot_family=robot_family(robot_type),
        fps=fps,
        n_episodes=n_episodes,
        n_frames=n_frames,
        codebase_version=version,
    )

    if version and not _VERSION_RE.match(version):
        v.reasons.append(f"codebase_version {version} (need the v2/v3 parquet layout)")
    elif not version:
        v.warnings.append("codebase_version missing from meta/info.json (expected v2.x or v3.x)")

    rt = robot_type.lower()
    if not robot_type:
        v.reasons.append("robot_type missing (expected so100, so101 or so_follower)")
    elif _BIMANUAL_RE.search(rt) and re.search(r"so[-_ ]?(?:arm)?[-_ ]?10[01]|so[-_]?follower", rt):
        v.reasons.append(f"robot_type {robot_type!r} is a bimanual rig; antares simulates one six-joint arm")
    elif v.robot_family is None:
        v.reasons.append(f"robot_type {robot_type!r} is not an SO-100/101")

    features = info.get("features") or {}
    names_seen: dict[str, list[str]] = {}
    for key in ("action", "observation.state"):
        names = feature_names(features.get(key))
        if names is None:
            v.reasons.append(f"{key} has no joint names")
            continue
        names_seen[key] = names
        if len(names) != 6:
            why = " (a bimanual recording: two arms)" if len(names) == 12 else ""
            v.reasons.append(f"{key} has {len(names)} joints, not 6{why}")
            continue
        canonical = tuple(canonical_joint_name(n) for n in names)
        if canonical != JOINTS:
            v.reasons.append(
                f"{key} joints {list(canonical)} are not the SO-101 joints {list(JOINTS)} (same names, same order)"
            )
    if len(names_seen) == 2 and names_seen["action"] != names_seen["observation.state"]:
        v.reasons.append("action and observation.state name different joints")
    if names_seen.get("action") and len(names_seen["action"]) == 6:
        v.joint_names = [canonical_joint_name(n) for n in names_seen["action"]]

    if not fps:
        v.reasons.append("fps missing")
    elif fps not in ALLOWED_FPS:
        v.reasons.append(f"fps {fps:g} not in {sorted(int(f) for f in ALLOWED_FPS)}")

    if not episodes_known:
        v.warnings.append("total_episodes missing from meta/info.json; the episode count is checked after loading")
    elif n_episodes < min_episodes:
        v.reasons.append(f"{n_episodes} episodes < {min_episodes}")
    elif n_episodes < recommended_episodes:
        v.warnings.append(
            f"only {n_episodes} episodes; calibration holds some out, {recommended_episodes}+ gives a steadier score"
        )

    for name in dict.fromkeys(n for n in (repo_id, robot_type) if n):
        tok = sim_name_match(name)
        if tok:
            v.warnings.append(
                f"looks like a simulation dataset (name contains {tok!r}); the lock would describe a simulator, not a robot"
            )
            break
    tok = merged_name_match(repo_id) if repo_id else None
    if tok:
        v.warnings.append(f"looks like a merge of several datasets (name contains {tok!r}); one lock cannot fit several arms")

    return _finish(v)


def _finite(ep_arrays: list[np.ndarray]) -> bool:
    return all(np.isfinite(a).all() for a in ep_arrays)


def validate_dataset(
    ds: Dataset,
    *,
    min_episodes: int = MIN_EPISODES,
    recommended_episodes: int = RECOMMENDED_EPISODES,
) -> Verdict:
    """Judge loaded data: joints, fps, episode count, NaNs and unit sanity.

    The unit check is a heuristic on the value range: body joints should stay
    inside about [-120, 120] and the gripper inside [-5, 105]; wider values
    look like degrees or raw encoder ticks, narrower ones (all within +-3.3)
    like radians. Either way replay through the unit map will be wrong, so
    it warns rather than rejects.
    """
    v = Verdict(
        ok=False,
        robot_type=str(ds.robot_type or ""),
        robot_family=robot_family(ds.robot_type),
        fps=float(ds.fps),
        n_episodes=len(ds.episodes),
        n_frames=int(sum(len(e) for e in ds.episodes)),
        joint_names=list(ds.joint_names),
    )
    if tuple(ds.joint_names) != JOINTS:
        v.reasons.append(f"joints {list(ds.joint_names)} are not the SO-101 joints {list(JOINTS)}")
    if v.fps not in ALLOWED_FPS:
        v.reasons.append(f"fps {v.fps:g} not in {sorted(int(f) for f in ALLOWED_FPS)}")

    usable = [ep for ep in ds.episodes if len(ep) >= 2 and _finite([ep.state, ep.action])]
    n_bad = sum(1 for ep in ds.episodes if not _finite([ep.state, ep.action]))
    n_short = sum(1 for ep in ds.episodes if len(ep) < 2)
    if n_bad:
        v.warnings.append(f"{n_bad} of {len(ds.episodes)} episodes contain NaN/inf values and are skipped")
    if n_short:
        v.warnings.append(f"{n_short} of {len(ds.episodes)} episodes are shorter than 2 frames and are skipped")
    if len(usable) < min_episodes:
        v.reasons.append(f"{len(usable)} usable episodes < {min_episodes}")
    elif len(usable) < recommended_episodes:
        v.warnings.append(
            f"only {len(usable)} usable episodes; calibration holds some out, {recommended_episodes}+ gives a steadier score"
        )

    if usable and len(ds.joint_names) == 6:
        state = np.concatenate([ep.state for ep in usable])
        action = np.concatenate([ep.action for ep in usable])
        body = np.concatenate([state[:, :GRIPPER_INDEX], action[:, :GRIPPER_INDEX]])
        grip = np.concatenate([state[:, GRIPPER_INDEX], action[:, GRIPPER_INDEX]])
        b_lo, b_hi = float(body.min()), float(body.max())
        g_lo, g_hi = float(grip.min()), float(grip.max())
        if b_lo < BODY_RANGE[0] or b_hi > BODY_RANGE[1]:
            v.warnings.append(
                f"body joints span [{b_lo:.0f}, {b_hi:.0f}]; values look like degrees/other units, not LeRobot's "
                "[-100, 100] normalized range: replay will be wrong"
            )
        elif max(abs(b_lo), abs(b_hi)) <= math.pi + 0.2 and g_hi <= math.pi + 0.2:
            v.warnings.append(
                f"all values lie within +-{math.pi + 0.2:.1f}; they look like radians, not LeRobot's normalized units: "
                "replay will be wrong"
            )
        if g_lo < GRIPPER_RANGE[0] or g_hi > GRIPPER_RANGE[1]:
            v.warnings.append(
                f"gripper spans [{g_lo:.0f}, {g_hi:.0f}]; expected LeRobot's [0, 100] range: replay will be wrong"
            )
    return _finish(v)


def merge_verdicts(info: Verdict, data: Verdict) -> Verdict:
    """The data verdict with the info verdict's extra warnings folded in."""
    out = Verdict.from_dict(data.to_dict())
    out.codebase_version = out.codebase_version or info.codebase_version
    counted = any(w.startswith("only ") for w in out.warnings)
    for w in info.warnings:
        if w.startswith("only ") and counted:
            continue  # the loaded data already said how many episodes are usable
        if w not in out.warnings:
            out.warnings.append(w)
    for r in info.reasons:
        if r not in out.reasons:
            out.reasons.append(r)
    return _finish(out)


__all__ = [
    "ALLOWED_FPS",
    "BODY_RANGE",
    "GRIPPER_RANGE",
    "JOINTS",
    "MIN_EPISODES",
    "RECOMMENDED_EPISODES",
    "Verdict",
    "feature_names",
    "merge_verdicts",
    "merged_name_match",
    "robot_family",
    "sim_name_match",
    "validate_dataset",
    "validate_dataset_info",
]
