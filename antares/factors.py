"""What antares accounts for: one named, documented entry per fitted factor.

Every quantity a calibration can move (servo stiffness, latency, a joint's
zero offset, ...) is a :class:`Factor` in :data:`REGISTRY`. The fit's search
spaces (:mod:`antares.calibrate`), the drift diagnosis (:mod:`antares.diagnose`),
trend scales (:mod:`antares.drift`), the MJCF exporter and the sensor fit
all read their lists and bounds from here, so a factor added once shows up
everywhere, and ``antares factors`` / ``GET /api/factors`` can show a user
what is modelled, what each factor explains, what data it needs and whether
it has been shown to be identifiable from real logs.

:data:`NOT_MODELLED` lists effects antares knows about but does not
simulate, so a residual the fit cannot explain can be pointed at them.

Users can add factors without touching this file: :func:`load_factor_file`
reads a TOML file of ``[[factor]]`` tables (see :class:`ModelFactor` for
config-only factors that scale or set a MuJoCo model field, or give
``apply = "module:function"`` for a Python one) and registers them.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Bumped when a built-in factor's meaning or bounds change.
REGISTRY_VERSION = 1

#: What a factor acts on, in the order the simulator applies them.
ACTS_ON = ("command", "timing", "actuator", "joint", "model")

#: How the search treats a factor's range: log10, linear, or rounded integer.
SCALES = ("log", "linear", "int")

#: Identifiability verdicts. ``yes``: real-data fits pin it down; ``partly``:
#: pinned on some arms only; ``no``: trades off with other factors;
#: ``untested``: no real-data study yet.
IDENTIFIABLE = ("yes", "partly", "no", "untested")


@dataclass(frozen=True)
class Factor:
    """One thing a calibration can account for.

    ``bounds`` is the shared-fit search range and ``per_joint_bounds`` the
    per-joint refinement's (``None``: same as ``bounds``). ``stages`` says
    which fits search it by default: ``"shared"`` (one value for all joints),
    ``"per_joint"`` (one per joint, after the shared fit) and ``"sensor"``
    (the command-map fit). ``drift`` is how :mod:`antares.diagnose` searches a
    change on top of a lock: ``("factor", r)`` multiplies by ``r**-1..r``,
    ``("shift", lo, hi)`` adds whole steps, ``None`` leaves it out.
    ``mjcf`` names where the exporter writes the value (``("joint", attr)``
    or ``("position", attr)``), ``None`` if it cannot live in an MJCF.
    """

    name: str
    label: str
    explains: str
    acts_on: str
    unit: str
    scale: str
    bounds: tuple[float, float]
    #: Value when the factor is not fitted; ``None``: the robot model's own value.
    neutral: float | None
    needs: str
    identifiable: str
    evidence: str
    stages: tuple[str, ...] = ("shared",)
    per_joint_bounds: tuple[float, float] | None = None
    drift: tuple[Any, ...] | None = None
    mjcf: tuple[str, str] | None = None
    #: ``"builtin"`` for the factors below, ``"model"``/``"plugin"`` for added ones.
    origin: str = "builtin"

    def __post_init__(self) -> None:
        if self.acts_on not in ACTS_ON:
            raise ValueError(f"{self.name}: acts_on {self.acts_on!r} not in {ACTS_ON}")
        if self.scale not in SCALES:
            raise ValueError(f"{self.name}: scale {self.scale!r} not in {SCALES}")
        if self.identifiable not in IDENTIFIABLE:
            raise ValueError(f"{self.name}: identifiable {self.identifiable!r} not in {IDENTIFIABLE}")
        lo, hi = (float(b) for b in self.bounds)
        if not lo < hi:
            raise ValueError(f"{self.name}: bounds must be lo < hi, got {self.bounds}")
        if self.scale == "log" and lo <= 0:
            raise ValueError(f"{self.name}: log-scale bounds must be positive")

    def space_entry(self, per_joint: bool = False) -> tuple[float, float, str]:
        """``(lo, hi, scale)`` as the calibration search space stores it."""
        lo, hi = self.per_joint_bounds if per_joint and self.per_joint_bounds else self.bounds
        return (lo, hi, self.scale)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bounds"] = list(self.bounds)
        d["per_joint_bounds"] = None if self.per_joint_bounds is None else list(self.per_joint_bounds)
        d["stages"] = list(self.stages)
        d["drift"] = None if self.drift is None else list(self.drift)
        d["mjcf"] = None if self.mjcf is None else list(self.mjcf)
        return d


# --------------------------------------------------------------------------
# Built-in factors. Order matters: the shared search vector follows it.
# --------------------------------------------------------------------------

_SPEC = (
    "robot-specificity study (results/robot_specificity.md): "
)

BUILTIN: tuple[Factor, ...] = (
    Factor(
        name="kp",
        label="Servo stiffness",
        explains="How hard the servo pulls toward its target: overshoot and how fast it closes a gap.",
        acts_on="actuator",
        unit="N*m/rad",
        scale="log",
        bounds=(10.0, 5000.0),  # SO-101 fits sat on the old 50 floor
        neutral=None,
        needs="moves with varied speed; any teleop log",
        identifiable="no",
        evidence=_SPEC + "refits of the same arm span x25 (SO-101) and x113 (SO-100); trades off with kv and latency",
        drift=("factor", 4.0),
        mjcf=("position", "kp"),
    ),
    Factor(
        name="kv",
        label="Servo velocity gain",
        explains="Damping the servo adds on its own: sluggish versus springy tracking.",
        acts_on="actuator",
        unit="N*m*s/rad",
        scale="log",
        bounds=(0.1, 50.0),
        neutral=None,
        needs="moves with varied speed; any teleop log",
        identifiable="no",
        evidence=_SPEC + "refits span x100; a higher kv can stand in for a frame of latency",
        drift=("factor", 4.0),
        mjcf=("position", "kv"),
    ),
    Factor(
        name="damping",
        label="Joint damping",
        explains="Viscous friction in the joint, growing with speed.",
        acts_on="joint",
        unit="N*m*s/rad",
        scale="log",
        bounds=(0.01, 10.0),
        neutral=None,
        needs="fast moves",
        identifiable="no",
        evidence=_SPEC + "least constrained direction on the SO-101 lock (condition number 1.1e5)",
        drift=("factor", 4.0),
        mjcf=("joint", "damping"),
    ),
    Factor(
        name="armature",
        label="Rotor inertia",
        explains="Inertia of the motor and gearbox as felt at the joint: lag when starting and stopping.",
        acts_on="joint",
        unit="kg*m^2",
        scale="log",
        bounds=(1e-4, 0.5),
        neutral=None,
        needs="sharp starts and stops",
        identifiable="partly",
        evidence=_SPEC + "x2 spread on the SO-101, x126 on the SO-100",
        drift=("factor", 4.0),
        mjcf=("joint", "armature"),
    ),
    Factor(
        name="frictionloss",
        label="Joint friction",
        explains="Dry friction: the arm stops short of slow or small targets.",
        acts_on="joint",
        unit="N*m",
        scale="log",
        bounds=(1e-3, 2.0),
        neutral=None,
        needs="slow moves and holds",
        identifiable="partly",
        evidence=_SPEC + "x11 spread on the SO-101, x2.4 on the SO-100",
        drift=("factor", 4.0),
        mjcf=("joint", "frictionloss"),
    ),
    Factor(
        name="latency_frames",
        label="Command latency",
        explains="The arm reacts whole frames after the command was logged.",
        acts_on="timing",
        unit="frames",
        scale="int",
        bounds=(0.0, 5.0),
        neutral=0.0,
        needs="any motion; fast moves pin it best",
        identifiable="yes",
        evidence=_SPEC + "clear minimum at 4 frames on the SO-101 (0.75 deg vs 1.61 deg at 3); "
        "the main per-arm difference across the 40-arm fleet",
        drift=("shift", -2, 3),
    ),
    Factor(
        name="forcerange",
        label="Torque limit",
        explains="Most torque the servo can give: the arm stalls or sags under load.",
        acts_on="actuator",
        unit="N*m",
        scale="log",
        bounds=(0.5, 30.0),  # the STS3215 stalls near 3
        neutral=None,
        needs="moves against gravity or a load",
        identifiable="untested",
        evidence="only searched in the per-joint refinement; no study yet",
        stages=("per_joint",),
    ),
    Factor(
        name="sensor_gain",
        label="Joint reading scale",
        explains="A joint's commands read a few percent too large or small (leader and follower calibrated apart).",
        acts_on="command",
        unit="x",
        scale="linear",
        bounds=(0.75, 1.33),
        neutral=1.0,
        needs="poses spread across each joint's range",
        identifiable="yes",
        evidence="PR #3: sensor fit took the SO-100 lock from 0.745 deg to 0.399 deg held-out",
        stages=("sensor",),
    ),
    Factor(
        name="sensor_bias",
        label="Joint zero offset",
        explains="A joint's commands read a constant angle off.",
        acts_on="command",
        unit="dataset units",
        scale="linear",
        bounds=(-25.0, 25.0),
        neutral=0.0,
        needs="still frames or varied poses",
        identifiable="yes",
        evidence="PR #3: sensor fit took the SO-100 lock from 0.745 deg to 0.399 deg held-out",
        stages=("sensor",),
    ),
)


@dataclass(frozen=True)
class Unmodelled:
    """An effect antares knows about but does not simulate."""

    name: str
    label: str
    symptom: str
    would_need: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


NOT_MODELLED: tuple[Unmodelled, ...] = (
    Unmodelled(
        "contact",
        "Contact and grasped objects",
        "error spikes while grasping or touching the table (see `antares events`)",
        "object poses and masses; the scene holds only the arm",
    ),
    Unmodelled(
        "payload",
        "Load in the gripper",
        "a joint sags while holding something",
        "logs with known payloads",
    ),
    Unmodelled(
        "backlash",
        "Gear backlash",
        "a step-shaped error when a joint reverses",
        "slow direction reversals",
    ),
    Unmodelled(
        "temperature",
        "Motor temperature",
        "drift within one session as the servos warm up",
        "motor temperature in the log (standard LeRobot logs do not record it)",
    ),
    Unmodelled(
        "cable_drag",
        "Cable drag",
        "a pose-dependent pull on the base or wrist",
        "still poses across the workspace",
    ),
    Unmodelled(
        "encoder_quantization",
        "Encoder resolution",
        "a noise floor of about one encoder tick",
        "nothing; it bounds the best possible score",
    ),
)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

REGISTRY: dict[str, Factor] = {f.name: f for f in BUILTIN}

#: ``apply(arm, values)`` hooks of added factors, by name. ``values`` is a
#: per-joint array (or one value); built-ins are applied by :class:`Arm`.
_APPLY: dict[str, Callable[[Any, np.ndarray], None]] = {}


def get(name: str) -> Factor:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown factor {name!r}; known: {', '.join(REGISTRY)}") from None


def names(stage: str | None = None, include: Iterable[str] = (), exclude: Iterable[str] = ()) -> tuple[str, ...]:
    """Registered factor names in order, optionally only those a ``stage``
    searches by default, plus ``include`` and minus ``exclude``."""
    inc, exc = set(include), set(exclude)
    for n in inc | exc:
        get(n)
    return tuple(
        n for n, f in REGISTRY.items() if n not in exc and (stage is None or stage in f.stages or n in inc)
    )


def is_extra(name: str) -> bool:
    """True for a registered factor that is not a built-in (has an apply hook)."""
    return name in _APPLY


def extras() -> tuple[str, ...]:
    return tuple(n for n in REGISTRY if n in _APPLY)


def apply_hook(name: str) -> Callable[[Any, np.ndarray], None]:
    return _APPLY[name]


def space(stage: str, per_joint: bool = False, factors: Sequence[str] | None = None,
          overrides: Mapping[str, tuple[float, float, str]] | None = None) -> dict[str, tuple[float, float, str]]:
    """Search-space bounds ``{name: (lo, hi, scale)}`` for ``stage``.

    ``factors`` picks the names (default: those whose ``stages`` include
    ``stage``); ``overrides`` replaces individual bounds, e.g. a robot whose
    servos are stiffer than the SO-101's.
    """
    picked = names(stage) if factors is None else tuple(factors)
    out = {}
    for n in picked:
        f = get(n)
        out[n] = tuple(overrides[n]) if overrides and n in overrides else f.space_entry(per_joint)
    return out


def drift_factors() -> dict[str, float]:
    """``{name: r}`` of the factors diagnosed as multiplicative changes."""
    return {n: float(f.drift[1]) for n, f in REGISTRY.items() if f.drift and f.drift[0] == "factor"}


def drift_shifts() -> dict[str, tuple[int, int]]:
    """``{name: (lo, hi)}`` of the factors diagnosed as whole-step shifts."""
    return {n: (int(f.drift[1]), int(f.drift[2])) for n, f in REGISTRY.items() if f.drift and f.drift[0] == "shift"}


def log_scaled() -> frozenset[str]:
    return frozenset(n for n, f in REGISTRY.items() if f.scale == "log")


def mjcf_attrs(tag: str) -> tuple[str, ...]:
    """Built-in factors the exporter writes on ``<joint>`` or ``<position>``."""
    return tuple(n for n, f in REGISTRY.items() if f.mjcf and f.mjcf[0] == tag)


def register(factor: Factor, apply: Callable[[Any, np.ndarray], None] | None = None, replace_existing: bool = False) -> Factor:
    """Add a factor. Non-built-ins need ``apply(arm, values)``."""
    if factor.name in REGISTRY and not replace_existing:
        raise ValueError(f"factor {factor.name!r} is already registered")
    if factor.name in {f.name for f in BUILTIN}:
        raise ValueError(f"{factor.name!r} is a built-in factor")
    if apply is None:
        raise ValueError(f"{factor.name}: an added factor needs an apply function")
    REGISTRY[factor.name] = factor
    _APPLY[factor.name] = apply
    return factor


def unregister(name: str) -> None:
    if name in {f.name for f in BUILTIN}:
        raise ValueError(f"{name!r} is a built-in factor")
    REGISTRY.pop(name, None)
    _APPLY.pop(name, None)


def describe(include_unmodelled: bool = True) -> dict[str, Any]:
    """The registry as JSON: what ``antares factors`` and ``/api/factors`` show."""
    out: dict[str, Any] = {
        "version": REGISTRY_VERSION,
        "factors": [f.to_dict() for f in REGISTRY.values()],
    }
    if include_unmodelled:
        out["not_modelled"] = [u.to_dict() for u in NOT_MODELLED]
    return out


# --------------------------------------------------------------------------
# Config-defined factors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFactor:
    """A factor that sets or scales one MuJoCo ``MjModel`` array.

    ``target`` is an attribute of ``MjModel`` (``"dof_damping"``,
    ``"actuator_gainprm"``, ``"body_mass"``) or ``"opt.<field>"``
    (``"opt.gravity"``), optionally with a column (``"actuator_gainprm:0"``).
    Per joint, ``dof_*``/``jnt_*``/``actuator_*`` rows are picked by the
    robot's joints and actuators; any other array (or ``per_joint = false``)
    is changed as a whole. ``mode = "scale"`` multiplies the model's own
    value, ``"set"`` replaces it. The original values are kept, so applying
    a new value never compounds.
    """

    target: str
    mode: str = "scale"
    per_joint: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("scale", "set"):
            raise ValueError(f"mode must be 'scale' or 'set', got {self.mode!r}")

    def _locate(self, model: Any) -> tuple[Any, str, int | None]:
        target, _, col = self.target.partition(":")
        obj, attr = model, target
        if target.startswith("opt."):
            obj, attr = model.opt, target[4:]
        if not hasattr(obj, attr):
            raise ValueError(f"MjModel has no field {self.target!r}")
        return obj, attr, (int(col) if col else None)

    def rows(self, arm: Any, attr: str) -> np.ndarray | None:
        if not self.per_joint:
            return None
        if attr.startswith("dof_"):
            return np.asarray(arm.dof_ids)
        if attr.startswith("jnt_"):
            return np.asarray(arm.joint_ids)
        if attr.startswith("actuator_"):
            return np.asarray(arm.actuator_ids)
        return None

    def __call__(self, arm: Any, values: np.ndarray) -> None:
        obj, attr, col = self._locate(arm.model)
        arr = getattr(obj, attr)
        key = ("model_factor", self.target)
        base = arm._factor_base.get(key)
        if base is None:
            base = np.array(arr, copy=True)
            arm._factor_base[key] = base
        values = np.asarray(values, dtype=np.float64)
        rows = self.rows(arm, attr)
        new = np.array(base, copy=True)
        if rows is None:
            v = float(np.mean(values))
            sel = (slice(None), col) if col is not None and new.ndim > 1 else slice(None)
            new[sel] = base[sel] * v if self.mode == "scale" else v
        else:
            v = values if values.ndim else np.full(len(rows), float(values))
            if col is not None:
                new[rows, col] = base[rows, col] * v if self.mode == "scale" else v
            else:
                new[rows] = base[rows] * v if self.mode == "scale" else v
        arr[...] = new
        import mujoco

        mujoco.mj_setConst(arm.model, arm.data)


def _resolve(ref: str) -> Callable[[Any, np.ndarray], None]:
    mod, _, fn = ref.partition(":")
    if not mod or not fn:
        raise ValueError(f"apply must look like 'module:function', got {ref!r}")
    return getattr(importlib.import_module(mod), fn)


def factor_from_config(d: Mapping[str, Any]) -> tuple[Factor, Callable[[Any, np.ndarray], None]]:
    """One ``[[factor]]`` table -> ``(Factor, apply)``.

    Required keys: ``name``, ``bounds``, and either ``target`` (a
    :class:`ModelFactor`) or ``apply`` (``"module:function"``). Everything
    else has a default; ``identifiable`` starts as ``"untested"``.
    """
    d = dict(d)
    name = str(d.pop("name"))
    if "target" in d and "apply" in d:
        raise ValueError(f"{name}: give target or apply, not both")
    per_joint = bool(d.pop("per_joint", False))
    if "target" in d:
        fn: Callable[[Any, np.ndarray], None] = ModelFactor(
            target=str(d.pop("target")), mode=str(d.pop("mode", "scale")), per_joint=per_joint
        )
        origin = "model"
    elif "apply" in d:
        fn = _resolve(str(d.pop("apply")))
        origin = "plugin"
    else:
        raise ValueError(f"{name}: needs target (an MjModel field) or apply (module:function)")
    bounds = tuple(float(b) for b in d.pop("bounds"))
    scale = str(d.pop("scale", "linear"))
    neutral = float(d.pop("neutral", 1.0 if scale == "log" else 0.0))
    stages = tuple(d.pop("stages", ("per_joint",) if per_joint else ("shared",)))
    drift = d.pop("drift", None)
    factor = Factor(
        name=name,
        label=str(d.pop("label", name)),
        explains=str(d.pop("explains", "")),
        acts_on=str(d.pop("acts_on", "model")),
        unit=str(d.pop("unit", "")),
        scale=scale,
        bounds=bounds,  # type: ignore[arg-type]
        neutral=neutral,
        needs=str(d.pop("needs", "")),
        identifiable=str(d.pop("identifiable", "untested")),
        evidence=str(d.pop("evidence", "added by the user; no study yet")),
        stages=stages,
        drift=None if drift is None else tuple(drift),
        origin=origin,
    )
    if d:
        raise ValueError(f"{name}: unknown keys {sorted(d)}")
    return factor, fn


def load_factor_file(path: str | Path, replace_existing: bool = True) -> list[Factor]:
    """Register every ``[[factor]]`` in a TOML file; returns them in order."""
    import tomllib

    data = tomllib.loads(Path(path).read_text())
    out = []
    for entry in data.get("factor", []):
        factor, fn = factor_from_config(entry)
        out.append(register(factor, fn, replace_existing=replace_existing))
    return out


def parse_selection(spec: str | None, stage: str = "shared") -> tuple[str, ...] | None:
    """``"+gravity,-forcerange"`` / ``"kp,kv,latency_frames"`` -> factor names.

    Entries with ``+``/``-`` add to or drop from ``stage``'s defaults; a
    plain list replaces them. ``None`` or empty means the defaults.
    """
    if not spec or not spec.strip():
        return None
    items = [s.strip() for s in spec.split(",") if s.strip()]
    if all(s[0] in "+-" for s in items):
        inc = [s[1:] for s in items if s[0] == "+"]
        exc = [s[1:] for s in items if s[0] == "-"]
        return names(stage, include=inc, exclude=exc)
    if any(s[0] in "+-" for s in items):
        raise ValueError("mix of +/- and plain factor names; use one form")
    for s in items:
        get(s)
    return tuple(items)


__all__ = [
    "BUILTIN",
    "Factor",
    "IDENTIFIABLE",
    "ModelFactor",
    "NOT_MODELLED",
    "REGISTRY",
    "REGISTRY_VERSION",
    "Unmodelled",
    "apply_hook",
    "describe",
    "drift_factors",
    "drift_shifts",
    "extras",
    "factor_from_config",
    "get",
    "is_extra",
    "load_factor_file",
    "log_scaled",
    "mjcf_attrs",
    "names",
    "parse_selection",
    "register",
    "space",
    "unregister",
]
