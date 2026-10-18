"""Which MuJoCo model simulates which robot.

Every vendored MJCF under ``antares/assets/<name>/`` has one
:class:`RobotModel` entry here: where the XML and its meshes live, what the
MJCF calls the six joints (in canonical order, see
:data:`antares.validate.JOINTS`) and how dataset units map into its joint
space. :class:`antares.sim.Arm` takes ``robot=<name>`` and looks the rest up.

A *family* is what a dataset says it was recorded on (``so100`` or
``so101``, from :func:`antares.validate.robot_family`); a *model* is what we
simulate it with. :data:`MODEL_FOR_FAMILY` maps one to the other. Both
families currently use the SO-101 model: on the SO-100 demo dataset the two
Menagerie models reach the same held-out error once they step with the same
integrator, and the SO-101 file replays 2.4x faster (see the README). The
``trs_so_arm100`` model stays available as ``robot="so100"``.

To add a robot: vendor its MJCF + meshes + LICENSE under ``assets/<name>/``,
add a :class:`RobotModel` below (map its joint names onto the canonical six
and give it a unit map if LeRobot's normalized units do not fit
``UnitMap.from_model``), and point the dataset family at it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from antares.validate import JOINTS, robot_family

ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"

#: The model used when nothing says otherwise (old locks, bare ``Arm()``).
DEFAULT_ROBOT = "so101"


def default_unit_map(model: Any) -> Any:
    """LeRobot units -> radians from the model's joint ranges (body joints
    -100..100 across the range, gripper 0..100)."""
    from antares.sim import UnitMap

    return UnitMap.from_model(model)


@dataclass(frozen=True)
class RobotModel:
    name: str
    model_path: Path
    mesh_dir: Path
    scene_path: Path
    #: The MJCF's joint names, in the order of the canonical six.
    joint_names: tuple[str, ...]
    description: str
    license: str = "Apache-2.0"
    unit_map_factory: Callable[[Any], Any] = default_unit_map

    def __post_init__(self) -> None:
        if len(self.joint_names) != len(JOINTS):
            raise ValueError(f"{self.name}: need {len(JOINTS)} joint names, got {len(self.joint_names)}")

    @property
    def canonical_joints(self) -> tuple[str, ...]:
        return JOINTS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "license": self.license,
            "model_path": str(self.model_path),
            "mesh_dir": str(self.mesh_dir),
            "joints": list(JOINTS),
            "mjcf_joint_names": list(self.joint_names),
        }


REGISTRY: dict[str, RobotModel] = {
    "so101": RobotModel(
        name="so101",
        model_path=ASSETS_DIR / "so101" / "so101.xml",
        mesh_dir=ASSETS_DIR / "so101" / "assets",
        scene_path=ASSETS_DIR / "so101" / "scene.xml",
        joint_names=JOINTS,
        description="The Robot Studio SO-101 (MuJoCo Menagerie robotstudio_so101); joints named like the datasets.",
    ),
    "so100": RobotModel(
        name="so100",
        model_path=ASSETS_DIR / "so100" / "so_arm100.xml",
        mesh_dir=ASSETS_DIR / "so100" / "assets",
        scene_path=ASSETS_DIR / "so100" / "scene.xml",
        joint_names=("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"),
        description="The Robot Studio SO-ARM100 v1.3 (MuJoCo Menagerie trs_so_arm100); Euler at 2 ms, white meshes.",
    ),
}

#: Dataset family -> model name. SO-100 datasets use the SO-101 model; see the
#: module docstring and the README for the measurement behind that.
MODEL_FOR_FAMILY: dict[str, str] = {"so101": "so101", "so100": "so101"}


def robot_names() -> list[str]:
    return list(REGISTRY)


def get_robot(name: str | None = None) -> RobotModel:
    """The registry entry for ``name`` (default :data:`DEFAULT_ROBOT`)."""
    key = (name or DEFAULT_ROBOT).strip().lower()
    try:
        return REGISTRY[key]
    except KeyError:
        raise ValueError(f"unknown robot {name!r}; known: {', '.join(REGISTRY)}") from None


def model_for_family(family: str | None) -> str:
    """Which model simulates datasets of ``family`` (unknown -> the default)."""
    return MODEL_FOR_FAMILY.get(family or "", DEFAULT_ROBOT)


def model_for_dataset(robot_type: str | None) -> str:
    """``robot_type`` from a dataset -> the model name to simulate it with."""
    return model_for_family(robot_family(robot_type))


__all__ = [
    "ASSETS_DIR",
    "DEFAULT_ROBOT",
    "MODEL_FOR_FAMILY",
    "REGISTRY",
    "RobotModel",
    "default_unit_map",
    "get_robot",
    "model_for_dataset",
    "model_for_family",
    "robot_names",
]
