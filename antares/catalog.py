"""Public datasets of arms other than the SO-100/101 that antares can replay.

Each :class:`CatalogEntry` names a Hub dataset, the registry robot that
simulates it (including the unit variant when ``robot_type`` alone cannot
tell, as for legacy Koch recordings) and what was checked about it.
:func:`robot_for_dataset` is how validation and the server pick a model for
a dataset whose ``robot_type`` is missing or ambiguous; ``antares datasets``
prints the list.

Facts were read from each dataset's ``meta/info.json`` and ``stats.json``
and from the recording code (2026-10-18); ``checked`` says how far each
entry has been taken.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CatalogEntry:
    repo_id: str
    robot: str
    robot_type: str
    fps: float
    episodes: int
    frames: int
    units: str
    notes: str
    #: ``"metadata"`` (info/stats read), ``"loaded"`` (data validated) or
    #: ``"calibrated"`` (a lock was fitted).
    checked: str = "metadata"

    def to_dict(self) -> dict:
        return asdict(self)


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        "lerobot/svla_so100_pickplace", "so101_deg", "so100",
        30, 50, 19631, "degrees (body joints); gripper 0..100",
        "Body joints span -90..124, outside LeRobot's -100..100 normalized range, so the recording is in "
        "degrees. Read as degrees the stock model's held-out gap is 2.61 deg (2.74 read as normalized).",
        "calibrated",
    ),
    CatalogEntry(
        "TrossenRoboticsCommunity/trossen_ai_stationary_block_transfer", "trossen_ai", "trossen_ai_stationary",
        30, 53, 31584, "radians; gripper carriage in metres",
        "Two WidowX AI arms, leader-follower teleop. Columns left_joint_0..6, right_joint_0..6. "
        "10 train / 5 held-out episodes, 150 evaluations: 2.152 -> 0.268 deg, latency 2 frames.",
        "calibrated",
    ),
    CatalogEntry(
        "lerobot/aloha_static_coffee", "aloha", "aloha",
        50, 50, 55000, "radians; grippers normalized 0..1",
        "Action is the leader arms. The gripper is normalized differently in action (leader joint) and "
        "state (follower fingers), so it is left out of the score. 0.606 -> 0.480 deg, latency 2; "
        "the uncalibrated sim already tracks it well (0.68 deg on episode 0).",
        "calibrated",
    ),
    CatalogEntry(
        "lerobot/aloha_mobile_cabinet", "aloha", "aloha",
        50, 85, 127500, "radians; grippers normalized 0..1",
        "Mobile ALOHA arms (the base is not simulated); larger than the static sets (~20 min download). "
        "1.231 -> 1.013 deg, latency 5; the waist barely improves.",
        "calibrated",
    ),
    CatalogEntry(
        "BotBarry/piper_pick_and_drop", "piper", "piper",
        30, 50, 30032, "radians; gripper jaw opening in metres",
        "Third-party Piper LeRobot integration; columns joint1..joint6, gripper. "
        "1.027 -> 0.346 deg, latency 2.",
        "calibrated",
    ),
    CatalogEntry(
        "lerobot/koch_pick_place_5_lego", "koch_deg", "unknown",
        30, 50, 37972, "degrees from the legacy calibration pose",
        "robot_type is 'unknown' and columns are motor_0..motor_5, so the catalog names the robot. "
        "Fits poorly so far (2.063 -> 1.793 deg): the data leaves the Menagerie joint limits (pitch to "
        "+44 deg against a 34 deg limit, wrist_roll to -251 deg), so the legacy calibration zero does not "
        "match the model's and the replay clips. Needs a per-joint zero/sign map before its numbers mean much.",
        "calibrated",
    ),
    CatalogEntry(
        "lerobot/koch_pick_place_1_lego", "koch_deg", "unknown",
        30, 50, 0, "degrees from the legacy calibration pose",
        "Same rig and format as koch_pick_place_5_lego, same joint-limit problem: the fit never beat "
        "the model's defaults (1.626 deg).",
        "calibrated",
    ),
)

_BY_ID = {e.repo_id: e for e in CATALOG}
#: Datasets given a robot at run time (``--robot`` on a dataset the catalog
#: doesn't list, or a different model for one it does).
_ASSIGNED: dict[str, str] = {}


def assign(repo_id: str, robot: str | None) -> None:
    """Replay ``repo_id`` on ``robot`` from now on (``None`` drops the
    assignment). Used by ``--robot`` so validation checks the dataset's
    columns against the model it will run on."""
    if robot is None:
        _ASSIGNED.pop(repo_id, None)
    else:
        _ASSIGNED[repo_id] = robot


def robot_for_dataset(repo_id: str | None) -> str | None:
    """The robot assigned to ``repo_id``, else the catalog's (``None`` if
    neither)."""
    key = str(repo_id or "")
    if key in _ASSIGNED:
        return _ASSIGNED[key]
    e = _BY_ID.get(key)
    return e.robot if e else None


def entries() -> list[dict]:
    return [e.to_dict() for e in CATALOG]


__all__ = ["CATALOG", "CatalogEntry", "assign", "entries", "robot_for_dataset"]
