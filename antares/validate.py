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
    """``"so100"`` / ``"so101"`` for a recognised single-arm SO robot type, the
    registry name of another robot whose ``robot_types`` match (``"aloha"``,
    ``"trossen_ai"``, ...), else None.

    Bimanual SO rigs (``bi_so100_follower``) are *not* a family: there is no
    two-arm SO model.
    """
    rt = str(robot_type or "").strip().lower()
    if not rt:
        return None
    from antares.robots import family_of

    other = family_of(rt)
    if other is not None:
        return other
    if _BIMANUAL_RE.search(rt):
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
    #: Registry model the dataset was checked against (when known).
    robot: str | None = None
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


def _robot_for(robot: str | None, family: str | None, repo_id: str = "") -> Any:
    """The registry entry a dataset is checked against: ``robot`` if given,
    else the catalog's pick for ``repo_id``, else its family's model, else
    the SO-101 (so an unknown arm is still told what was expected)."""
    from antares.catalog import robot_for_dataset
    from antares.robots import get_robot, model_for_family

    name = robot or robot_for_dataset(repo_id) or (model_for_family(family) if family else None)
    return get_robot(name)


def _joint_label(r: Any) -> str:
    return "SO-101" if r.name in ("so100", "so101") or r.name.startswith("so10") else (r.label or r.name)


def check_joint_names(names: list[str], r: Any) -> tuple[list[str] | None, str | None]:
    """Dataset joint names -> the robot's columns, or ``(None, reason)``.

    Names are canonicalized (``main_``/``.pos`` stripped) and then matched
    to the robot's columns directly or through its aliases, in order.
    """
    canonical = [canonical_joint_name(n) for n in names]
    n = len(r.joints)
    if len(canonical) != n:
        why = " (a bimanual recording: two arms)" if n == 6 and len(canonical) == 12 else ""
        return None, f"has {len(canonical)} joints, not {n}{why}"
    columns = [r.column_for(c) for c in canonical]
    if tuple(columns) != tuple(r.joints):
        return None, (
            f"joints {canonical} are not the {_joint_label(r)} joints {list(r.joints)} (same names, same order)"
        )
    return columns, None


def validate_dataset_info(
    info: dict[str, Any],
    repo_id: str = "",
    *,
    robot: str | None = None,
    min_episodes: int = MIN_EPISODES,
    recommended_episodes: int = RECOMMENDED_EPISODES,
) -> Verdict:
    """Judge a dataset from its ``meta/info.json`` alone.

    ``repo_id`` (or an upload name) is used for the name heuristics, which
    produce warnings here (the fleet scan turns them into rejections
    itself), and to look the dataset up in :mod:`antares.catalog`.
    ``robot`` names the registry model to check against; by default it
    follows ``robot_type``.
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
    from antares.catalog import robot_for_dataset

    chosen = robot or robot_for_dataset(repo_id)
    r = _robot_for(chosen, v.robot_family, repo_id)
    if chosen:
        v.robot = r.name
    elif v.robot_family is not None:
        v.robot = r.name
    if chosen:
        if not robot_type:
            v.warnings.append(f"robot_type missing; simulating it as {r.name}")
    elif not robot_type:
        v.reasons.append("robot_type missing (expected so100, so101, so_follower or a known arm; or pass a robot)")
    elif _BIMANUAL_RE.search(rt) and re.search(r"so[-_ ]?(?:arm)?[-_ ]?10[01]|so[-_]?follower", rt):
        v.reasons.append(f"robot_type {robot_type!r} is a bimanual rig; there is no two-arm SO model")
    elif v.robot_family is None:
        from antares.robots import robot_names

        v.reasons.append(
            f"robot_type {robot_type!r} is not an SO-100/101 or another arm antares has a model for "
            f"({', '.join(robot_names())})"
        )

    features = info.get("features") or {}
    names_seen: dict[str, list[str]] = {}
    columns: list[str] | None = None
    for key in ("action", "observation.state"):
        names = feature_names(features.get(key))
        if names is None:
            v.reasons.append(f"{key} has no joint names")
            continue
        names_seen[key] = names
        cols, why = check_joint_names(names, r)
        if why:
            v.reasons.append(f"{key} {why}")
        elif key == "action":
            columns = cols
    if len(names_seen) == 2 and names_seen["action"] != names_seen["observation.state"]:
        v.reasons.append("action and observation.state name different joints")
    if columns is not None:
        v.joint_names = [canonical_joint_name(n) for n in names_seen["action"]]

    allowed = r.fps
    if not fps:
        v.reasons.append("fps missing")
    elif allowed is not None and fps not in allowed:
        v.reasons.append(f"fps {fps:g} not in {sorted(int(f) for f in allowed)}")
    elif allowed is None and not 5 <= fps <= 500:
        v.reasons.append(f"fps {fps:g} outside 5..500")

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
    robot: str | None = None,
    min_episodes: int = MIN_EPISODES,
    recommended_episodes: int = RECOMMENDED_EPISODES,
) -> Verdict:
    """Judge loaded data: joints, fps, episode count, NaNs and unit sanity.

    The unit check is a heuristic on the value range of each column against
    the unit the robot expects it in. For LeRobot's normalized units, body
    joints should stay inside about [-120, 120] and the gripper inside
    [-5, 105]; wider values look like degrees or raw encoder ticks, narrower
    ones (all within +-3.3) like radians. Radian columns beyond about 2 pi
    look like degrees. A mismatch makes replay wrong, so it warns rather
    than rejects.
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
    from antares.catalog import robot_for_dataset

    chosen = robot or robot_for_dataset(ds.repo_id)
    r = _robot_for(chosen, v.robot_family, ds.repo_id)
    if chosen or v.robot_family is not None:
        v.robot = r.name
    cols, why = check_joint_names(list(ds.joint_names), r)
    if why:
        v.reasons.append(f"dataset {why}" if why.startswith("has ") else why)
    if r.fps is not None and v.fps not in r.fps:
        v.reasons.append(f"fps {v.fps:g} not in {sorted(int(f) for f in r.fps)}")

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

    if usable and cols is not None:
        state = np.concatenate([ep.state for ep in usable])
        action = np.concatenate([ep.action for ep in usable])
        v.warnings += unit_warnings(state, action, r)
    return _finish(v)


def unit_warnings(state: np.ndarray, action: np.ndarray, r: Any) -> list[str]:
    """Warnings for columns whose values do not look like the robot's units."""
    out: list[str] = []
    kinds = r.unit_kinds()
    akinds = r.unit_kinds(action=True)

    def cols(kind: str, grip: bool | None = None) -> np.ndarray:
        idx_s = [i for i, k in enumerate(kinds) if k == kind and (grip is None or (r.joints[i] in r.grippers) == grip)]
        idx_a = [i for i, k in enumerate(akinds) if k == kind and (grip is None or (r.joints[i] in r.grippers) == grip)]
        return np.concatenate([state[:, idx_s].ravel(), action[:, idx_a].ravel()])

    body = cols("range_m100_100")
    if body.size:
        b_lo, b_hi = float(body.min()), float(body.max())
        if b_lo < BODY_RANGE[0] or b_hi > BODY_RANGE[1]:
            hint = f" (if it was recorded in degrees, use robot {r.name}_deg)" if f"{r.name}_deg" in _robot_names() else ""
            out.append(
                f"body joints span [{b_lo:.0f}, {b_hi:.0f}]; values look like degrees/other units, not LeRobot's "
                f"[-100, 100] normalized range: replay will be wrong{hint}"
            )
        else:
            grip_all = cols("range_0_100")
            g_hi = float(grip_all.max()) if grip_all.size else 0.0
            if max(abs(b_lo), abs(b_hi)) <= math.pi + 0.2 and g_hi <= math.pi + 0.2:
                out.append(
                    f"all values lie within +-{math.pi + 0.2:.1f}; they look like radians, not LeRobot's normalized "
                    "units: replay will be wrong"
                )
    grip = cols("range_0_100")
    if grip.size:
        g_lo, g_hi = float(grip.min()), float(grip.max())
        if g_lo < GRIPPER_RANGE[0] or g_hi > GRIPPER_RANGE[1]:
            out.append(f"gripper spans [{g_lo:.0f}, {g_hi:.0f}]; expected LeRobot's [0, 100] range: replay will be wrong")
    unit = cols("range_0_1")
    if unit.size and (float(unit.min()) < -0.05 or float(unit.max()) > 1.05):
        out.append(f"normalized columns span [{unit.min():.2f}, {unit.max():.2f}]; expected 0..1: replay will be wrong")
    rad = cols("radians")
    if rad.size and float(np.abs(rad).max()) > 2 * math.pi + 0.5:
        out.append(f"radian columns reach {np.abs(rad).max():.0f}; they look like degrees: replay will be wrong")
    for kind in ("degrees", "degrees_raw"):
        deg = cols(kind, grip=False)
        if deg.size and float(np.abs(deg).max()) <= math.pi + 0.2:
            out.append(f"degree columns stay within +-{math.pi + 0.2:.1f}; they look like radians: replay will be wrong")
    return out


def _robot_names() -> list[str]:
    from antares.robots import robot_names

    return robot_names()


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
