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
        "TrossenRoboticsCommunity/trossen_ai_stationary_block_transfer", "trossen_ai", "trossen_ai_stationary",
        30, 53, 31584, "radians; gripper carriage in metres",
        "Two WidowX AI arms, leader-follower teleop. Columns left_joint_0..6, right_joint_0..6.",
    ),
    CatalogEntry(
        "TrossenRoboticsCommunity/trossen_ai_stationary_handover_cube", "trossen_ai", "bi_widowxai_follower",
        30, 52, 15283, "radians; gripper carriage in metres",
        "Newer Trossen LeRobot plugin: columns *_joint_0..5.pos and *_left_carriage_joint.pos.",
    ),
    CatalogEntry(
        "lerobot/aloha_static_coffee", "aloha", "aloha",
        50, 50, 55000, "radians; grippers normalized 0..1",
        "Action is the leader arms. The gripper is normalized differently in action (leader joint) and "
        "state (follower fingers), so it is left out of the score.",
    ),
    CatalogEntry(
        "lerobot/aloha_mobile_cabinet", "aloha", "aloha",
        50, 85, 127500, "radians; grippers normalized 0..1",
        "Mobile ALOHA arms (the base is not simulated); larger than the static sets.",
    ),
    CatalogEntry(
        "BotBarry/piper_pick_and_drop", "piper", "piper",
        30, 50, 30032, "radians; gripper jaw opening in metres",
        "Third-party Piper LeRobot integration; columns joint1..joint6, gripper.",
    ),
    CatalogEntry(
        "lerobot/koch_pick_place_5_lego", "koch_deg", "unknown",
        30, 50, 37972, "degrees from the legacy calibration pose",
        "robot_type is 'unknown' and columns are motor_0..motor_5, so the catalog names the robot. "
        "The 2024 calibration zero may not match the Menagerie model's; expect a sensor offset fit to matter.",
    ),
    CatalogEntry(
        "lerobot/koch_pick_place_1_lego", "koch_deg", "unknown",
        30, 50, 0, "degrees from the legacy calibration pose",
        "Same rig and format as koch_pick_place_5_lego (frames not checked).",
    ),
)

_BY_ID = {e.repo_id: e for e in CATALOG}


def robot_for_dataset(repo_id: str | None) -> str | None:
    """The robot the catalog assigns to ``repo_id`` (``None`` if not listed)."""
    e = _BY_ID.get(str(repo_id or ""))
    return e.robot if e else None


def entries() -> list[dict]:
    return [e.to_dict() for e in CATALOG]


__all__ = ["CATALOG", "CatalogEntry", "entries", "robot_for_dataset"]
