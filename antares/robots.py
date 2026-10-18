"""Which MuJoCo model simulates which robot, and how its logs map onto it.

Every robot antares can simulate is one :class:`RobotModel`: where the MJCF
and its meshes live, the dataset columns it expects (in order), which MJCF
joint each column is, which columns are grippers, what units the columns
are logged in, and how its servos differ from the SO-101's (search bounds,
where the exporter writes parameters). :class:`antares.sim.Arm` takes
``robot=<name>`` and looks the rest up. Nothing else in antares needs to
know which arm it is working on.

A *family* is what a dataset says it was recorded on (``so100``, ``so101``,
``aloha``, ...; from :func:`antares.validate.robot_family`); a *model* is
what we simulate it with. :data:`MODEL_FOR_FAMILY` maps one to the other.
Both SO families use the SO-101 model: on the SO-100 demo dataset the two
Menagerie models reach the same held-out error once they step with the same
integrator, and the SO-101 file replays 2.4x faster (see the README). The
``trs_so_arm100`` model stays available as ``robot="so100"``.

The SO models are vendored under ``antares/assets/``; the others are
fetched from MuJoCo Menagerie at a pinned commit on first use (see
:mod:`antares.menagerie`).

To add a robot: write a TOML file (see :func:`robot_from_config` and
``robots/README.md``) and load it with ``ANTARES_ROBOTS=<file>`` or
:func:`load_robot_file`, or add a :class:`RobotModel` below.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from antares.menagerie import MenagerieModel
from antares.validate import ALLOWED_FPS, JOINTS, robot_family

ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"

#: The model used when nothing says otherwise (old locks, bare ``Arm()``).
DEFAULT_ROBOT = "so101"

#: Units a dataset column can be logged in. ``range_m100_100`` /
#: ``range_0_100`` / ``range_0_1`` span the joint's MJCF range (LeRobot's
#: normalized modes, ALOHA's normalized gripper); ``degrees`` is LeRobot's
#: DEGREES mode, zero at the middle of the range; ``degrees_raw`` and
#: ``radians`` have zero at the MJCF's zero; ``native`` is the joint's own
#: unit (radians or metres) with no change.
UNIT_KINDS = ("range_m100_100", "range_0_100", "range_0_1", "degrees", "degrees_raw", "radians", "native")

#: Where a robot's default (uncalibrated) parameters come from: the SO-101's
#: ``sts3215`` values (:class:`antares.sim.SimParams` defaults), or the
#: robot's own MJCF, joint by joint.
DEFAULT_PARAM_SOURCES = ("sts3215", "model")


def default_unit_map(model: Any) -> Any:
    """LeRobot units -> radians from the model's joint ranges (body joints
    -100..100 across the range, gripper 0..100)."""
    from antares.sim import UnitMap

    return UnitMap.from_model(model)


@dataclass(frozen=True)
class RobotModel:
    """One simulatable robot. Every field after ``description`` has a default
    that describes an SO-100/101, so the SO entries stay short."""

    name: str
    model_path: Path
    mesh_dir: Path
    scene_path: Path
    #: The MJCF's joint names, one per dataset column (same order as ``joints``).
    joint_names: tuple[str, ...]
    description: str
    license: str = "Apache-2.0"
    #: Builds the unit map from the compiled model; ``None`` uses ``units``.
    unit_map_factory: Callable[[Any], Any] | None = None
    #: Dataset columns after :func:`antares.data.canonical_joint_name`, in order.
    joints: tuple[str, ...] = JOINTS
    #: Columns that are grippers: left out of the score and the sensor fit.
    grippers: tuple[str, ...] = ("gripper",)
    #: Unit of each column (:data:`UNIT_KINDS`); ``None``: LeRobot's
    #: normalized ranges (-100..100, grippers 0..100).
    units: tuple[str, ...] | None = None
    #: Units of the commands when they differ from the measurements.
    action_units: tuple[str, ...] | None = None
    #: Extra factor on each column after its unit (e.g. -1 for a joint whose
    #: MJCF axis points the other way, 0.5 for a jaw opening driving one finger).
    unit_scale: tuple[float, ...] | None = None
    #: Other names a column goes by in datasets: ``{column: (alias, ...)}``.
    aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: Regexes on ``robot_type`` that name this robot's family (not for SO).
    robot_types: tuple[str, ...] = ()
    #: Search-bound overrides ``{factor: (lo, hi, scale)}`` for this robot's servos.
    bounds: Mapping[str, tuple[float, float, str]] = field(default_factory=dict)
    #: :data:`DEFAULT_PARAM_SOURCES`.
    default_params: str = "sts3215"
    #: Default class the exporter writes shared values into; ``None``: write
    #: every joint and actuator.
    export_class: str | None = "sts3215"
    #: Frame rates this model has been checked at; ``None``: any.
    fps: tuple[float, ...] | None = ALLOWED_FPS
    #: Fetched from Menagerie on first use (``None``: vendored files).
    menagerie: MenagerieModel | None = None
    #: Two arms in one model (one lock describes the pair).
    bimanual: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if not self.joints:
            raise ValueError(f"{self.name}: needs at least one joint")
        if len(self.joint_names) != len(self.joints):
            raise ValueError(f"{self.name}: need {len(self.joints)} joint names, got {len(self.joint_names)}")
        if len(set(self.joints)) != len(self.joints):
            raise ValueError(f"{self.name}: duplicate column names")
        for g in self.grippers:
            if g not in self.joints:
                raise ValueError(f"{self.name}: gripper {g!r} is not one of its joints")
        if self.unit_scale is not None and len(self.unit_scale) != len(self.joints):
            raise ValueError(f"{self.name}: unit_scale needs {len(self.joints)} entries")
        for attr in ("units", "action_units"):
            v = getattr(self, attr)
            if v is None:
                continue
            if len(v) != len(self.joints):
                raise ValueError(f"{self.name}: {attr} needs {len(self.joints)} entries, got {len(v)}")
            bad = [u for u in v if u not in UNIT_KINDS]
            if bad:
                raise ValueError(f"{self.name}: unknown unit(s) {bad}; known: {', '.join(UNIT_KINDS)}")
        if self.default_params not in DEFAULT_PARAM_SOURCES:
            raise ValueError(f"{self.name}: default_params must be one of {DEFAULT_PARAM_SOURCES}")

    @property
    def canonical_joints(self) -> tuple[str, ...]:
        return self.joints

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    @property
    def gripper_indices(self) -> tuple[int, ...]:
        return tuple(self.joints.index(g) for g in self.grippers)

    @property
    def body_indices(self) -> tuple[int, ...]:
        """Columns that are scored (everything but the grippers)."""
        return tuple(i for i, j in enumerate(self.joints) if j not in self.grippers)

    @property
    def scored_joints(self) -> tuple[str, ...]:
        return tuple(j for j in self.joints if j not in self.grippers)

    def unit_kinds(self, action: bool = False) -> tuple[str, ...]:
        if action and self.action_units is not None:
            return self.action_units
        if self.units is not None:
            return self.units
        return tuple("range_0_100" if j in self.grippers else "range_m100_100" for j in self.joints)

    def resolve(self) -> RobotModel:
        """This entry with its model files on disk (fetches Menagerie models)."""
        if self.menagerie is not None and not self.menagerie.is_fetched():
            from antares.menagerie import fetch

            fetch(self.menagerie)
        return self

    def column_for(self, dataset_name: str) -> str | None:
        """The column a dataset's (canonical) joint name is, via aliases."""
        if dataset_name in self.joints:
            return dataset_name
        for col, names in self.aliases.items():
            if dataset_name in names:
                return col
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label or self.name,
            "description": self.description,
            "license": self.license,
            "model_path": str(self.model_path),
            "mesh_dir": str(self.mesh_dir),
            "joints": list(self.joints),
            "mjcf_joint_names": list(self.joint_names),
            "grippers": list(self.grippers),
            "scored_joints": list(self.scored_joints),
            "units": list(self.unit_kinds()),
            "action_units": list(self.unit_kinds(action=True)),
            "unit_scale": None if self.unit_scale is None else list(self.unit_scale),
            "aliases": {k: list(v) for k, v in self.aliases.items()},
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "default_params": self.default_params,
            "export_class": self.export_class,
            "fps": None if self.fps is None else list(self.fps),
            "bimanual": self.bimanual,
            "source": "vendored" if self.menagerie is None else f"menagerie/{self.menagerie.directory}@{self.menagerie.commit[:12]}",
            "fetched": self.menagerie is None or self.menagerie.is_fetched(),
        }


def _menagerie_robot(name: str, directory: str, xml: str, **kw: Any) -> RobotModel:
    m = MenagerieModel(directory, xml)
    return RobotModel(
        name=name,
        model_path=m.model_path,
        mesh_dir=m.local_dir / "assets",
        scene_path=m.local_dir / "scene.xml",
        menagerie=m,
        **kw,
    )


def _two(prefix_a: str, prefix_b: str, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{prefix_a}{n}" for n in names) + tuple(f"{prefix_b}{n}" for n in names)





REGISTRY: dict[str, RobotModel] = {
    "so101": RobotModel(
        name="so101",
        model_path=ASSETS_DIR / "so101" / "so101.xml",
        mesh_dir=ASSETS_DIR / "so101" / "assets",
        scene_path=ASSETS_DIR / "so101" / "scene.xml",
        joint_names=JOINTS,
        description="The Robot Studio SO-101 (MuJoCo Menagerie robotstudio_so101); joints named like the datasets.",
        unit_map_factory=default_unit_map,
        label="SO-101",
    ),
    "so100": RobotModel(
        name="so100",
        model_path=ASSETS_DIR / "so100" / "so_arm100.xml",
        mesh_dir=ASSETS_DIR / "so100" / "assets",
        scene_path=ASSETS_DIR / "so100" / "scene.xml",
        joint_names=("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"),
        description="The Robot Studio SO-ARM100 v1.3 (MuJoCo Menagerie trs_so_arm100); Euler at 2 ms, white meshes.",
        unit_map_factory=default_unit_map,
        label="SO-100",
    ),
}

#: Dataset family -> model name. SO-100 datasets use the SO-101 model; see the
#: module docstring and the README for the measurement behind that.
MODEL_FOR_FAMILY: dict[str, str] = {"so101": "so101", "so100": "so101"}
for _name, _r in REGISTRY.items():
    if _r.robot_types:
        MODEL_FOR_FAMILY[_name] = _name


def robot_names() -> list[str]:
    return list(REGISTRY)


def get_robot(name: str | None = None) -> RobotModel:
    """The registry entry for ``name`` (default :data:`DEFAULT_ROBOT`)."""
    key = (name or DEFAULT_ROBOT).strip().lower()
    try:
        return REGISTRY[key]
    except KeyError:
        raise ValueError(f"unknown robot {name!r}; known: {', '.join(REGISTRY)}") from None


def family_of(robot_type: str | None) -> str | None:
    """A registered non-SO robot whose ``robot_types`` match ``robot_type``."""
    rt = str(robot_type or "").strip().lower()
    if not rt:
        return None
    for name, r in REGISTRY.items():
        if any(re.search(p, rt) for p in r.robot_types):
            return name
    return None


def model_for_family(family: str | None) -> str:
    """Which model simulates datasets of ``family`` (unknown -> the default)."""
    return MODEL_FOR_FAMILY.get(family or "", DEFAULT_ROBOT)


def model_for_dataset(robot_type: str | None) -> str:
    """``robot_type`` from a dataset -> the model name to simulate it with."""
    return model_for_family(robot_family(robot_type))


# --------------------------------------------------------------------------
# Robots from config files
# --------------------------------------------------------------------------


def robot_from_config(d: Mapping[str, Any], base_dir: str | Path = ".") -> RobotModel:
    """One ``[[robot]]`` table -> :class:`RobotModel`.

    Required: ``name``, ``joints`` (dataset columns), ``mjcf_joints``, and
    either ``model`` (an MJCF path, relative to the file) or ``menagerie =
    {directory, xml}``. Optional: ``grippers``, ``units``, ``action_units``, ``unit_scale``,
    ``aliases``, ``robot_types``, ``bounds``, ``mesh_dir``, ``description``,
    ``license``, ``fps``, ``bimanual``, ``label``. Defaults for an added
    robot: parameters read from its MJCF, per-joint export, any fps.
    """
    d = dict(d)
    base_dir = Path(base_dir)
    name = str(d.pop("name")).strip().lower()
    joints = tuple(d.pop("joints"))
    mjcf_joints = tuple(d.pop("mjcf_joints"))
    menagerie = d.pop("menagerie", None)
    if menagerie is not None:
        m = MenagerieModel(str(menagerie["directory"]), str(menagerie["xml"]),
                           str(menagerie.get("commit", MenagerieModel.__dataclass_fields__["commit"].default)))
        model_path, default_mesh = m.model_path, m.local_dir / "assets"
    else:
        m = None
        model_path = (base_dir / str(d.pop("model"))).resolve()
        default_mesh = model_path.parent / "assets"
    mesh_dir = Path(d.pop("mesh_dir")) if "mesh_dir" in d else default_mesh
    if not mesh_dir.is_absolute() and m is None:
        mesh_dir = (base_dir / mesh_dir).resolve()
    kw: dict[str, Any] = {}
    if "unit_scale" in d:
        kw["unit_scale"] = tuple(float(v) for v in d.pop("unit_scale"))
    for key in ("units", "action_units", "robot_types", "grippers"):
        if key in d:
            kw[key] = tuple(d.pop(key))
    if "fps" in d:
        fps = d.pop("fps")
        kw["fps"] = None if fps is None else tuple(float(f) for f in fps)
    if "aliases" in d:
        kw["aliases"] = {k: tuple(v) for k, v in d.pop("aliases").items()}
    if "bounds" in d:
        kw["bounds"] = {k: (float(v[0]), float(v[1]), str(v[2])) for k, v in d.pop("bounds").items()}
    r = RobotModel(
        name=name,
        model_path=model_path,
        mesh_dir=mesh_dir,
        scene_path=Path(d.pop("scene")) if "scene" in d else model_path,
        joint_names=mjcf_joints,
        description=str(d.pop("description", "")),
        license=str(d.pop("license", "see the model's own licence")),
        joints=joints,
        default_params=str(d.pop("default_params", "model")),
        export_class=d.pop("export_class", None),
        menagerie=m,
        bimanual=bool(d.pop("bimanual", False)),
        label=str(d.pop("label", name)),
        **{"fps": None, **kw},
    )
    if d:
        raise ValueError(f"{name}: unknown keys {sorted(d)}")
    return r


def register(robot: RobotModel, replace_existing: bool = False) -> RobotModel:
    if robot.name in REGISTRY and not replace_existing:
        raise ValueError(f"robot {robot.name!r} is already registered")
    if robot.name in ("so100", "so101"):
        raise ValueError(f"{robot.name!r} is built in")
    REGISTRY[robot.name] = robot
    if robot.robot_types:
        MODEL_FOR_FAMILY[robot.name] = robot.name
    return robot


def load_robot_file(path: str | Path, replace_existing: bool = True) -> list[RobotModel]:
    """Register every ``[[robot]]`` table in a TOML file."""
    import tomllib

    path = Path(path)
    data = tomllib.loads(path.read_text())
    return [register(robot_from_config(e, path.parent), replace_existing) for e in data.get("robot", [])]


def load_env_robots() -> list[RobotModel]:
    """Robots from the files in ``$ANTARES_ROBOTS`` (``os.pathsep``-separated)."""
    out: list[RobotModel] = []
    for p in filter(None, os.environ.get("ANTARES_ROBOTS", "").split(os.pathsep)):
        out += load_robot_file(p)
    return out


load_env_robots()


__all__ = [
    "ASSETS_DIR",
    "DEFAULT_ROBOT",
    "MODEL_FOR_FAMILY",
    "REGISTRY",
    "UNIT_KINDS",
    "RobotModel",
    "default_unit_map",
    "family_of",
    "get_robot",
    "load_robot_file",
    "model_for_dataset",
    "model_for_family",
    "register",
    "robot_from_config",
    "robot_names",
]
