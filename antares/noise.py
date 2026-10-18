"""How noisy the arm's joint readings are, measured while it holds still.

Policies are usually trained with Gaussian noise of a guessed size added to
the joint readings. The logs say what the real readings look like: while
the command for a joint has not moved for a while, the servo is parked and
whatever the measured position still does is sensor noise (plus the servo
hunting around its target). :func:`estimate` pools those stretches per
joint and fits

* ``scale``: the standard deviation about each stretch's mean, radians,
* ``acf1``: the lag-1 autocorrelation, so the noise can be replayed with its
  real colour as an AR(1) process rather than white noise,
* ``quantum``: the smallest step between readings (the encoder resolution
  as it shows up in the dataset), radians, and ``origin``: one value on
  that lattice, so rounding lands on the values the real readings take
  (the dataset's lattice is a multiple of the step in dataset units, which
  the unit map's offset shifts off a multiple of ``quantum`` in radians).

:class:`NoiseModel` adds noise of that size and colour to simulated joint
readings and rounds them to the same steps. Saved next to the locks as
``locks/<name>.noise.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from antares import __version__
from antares.data import Episode
from antares.lock import JsonFile, now_iso, slug
from antares.sim import JOINTS, N_JOINTS, UnitMap, load_model

NOISE_SUFFIX = ".noise.json"

#: A joint counts as parked once its command has stayed within
#: ``STILL_TOL`` dataset units for ``STILL_FRAMES`` frames.
STILL_FRAMES = 15
STILL_TOL = 0.3


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` runs of True."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def still_mask(cmd: np.ndarray, frames: int = STILL_FRAMES, tol: float = STILL_TOL) -> np.ndarray:
    """(T,) True where ``cmd`` (one joint, dataset units) has spanned at most
    ``tol`` over the last ``frames`` frames, including this one."""
    x = np.asarray(cmd, dtype=np.float64)
    T = x.size
    out = np.zeros(T, dtype=bool)
    if T < frames:
        return out
    win = np.lib.stride_tricks.sliding_window_view(x, frames)
    out[frames - 1 :] = (win.max(axis=1) - win.min(axis=1)) <= tol
    return out


def quantum(values: np.ndarray) -> float:
    """Smallest step between distinct readings of one joint (same units), or NaN."""
    v = np.unique(np.asarray(values, dtype=np.float64))
    d = np.diff(v)
    d = d[d > 1e-4]  # float32 storage jitters a lattice by ~1e-5
    return float(d.min()) if d.size else float("nan")


def lattice_origin(values: np.ndarray, step: float) -> float:
    """A value on the lattice of ``step`` that ``values`` (one joint) lie on.

    The first reading, corrected by the median offset of every reading from
    the lattice through it (so float32 jitter and the odd outlier do not
    move it). NaN when ``step`` is not a positive number.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if not (np.isfinite(step) and step > 0) or v.size == 0:
        return float("nan")
    r = (v - v[0] + step / 2.0) % step - step / 2.0  # offset from the lattice through v[0], in [-step/2, step/2)
    return float(v[0] + np.median(r))


@dataclass
class NoiseModel(JsonFile):
    """Per-joint AR(1) reading noise plus quantization, in radians."""

    FILE_SUFFIX: ClassVar[str] = NOISE_SUFFIX

    scale: np.ndarray  # (6,) stationary std, rad
    acf1: np.ndarray  # (6,) lag-1 autocorrelation in [0, 1)
    quantum: np.ndarray  # (6,) reading step, rad (NaN: do not round)
    n_frames: np.ndarray  # (6,) parked frames each estimate used
    origin: np.ndarray | None = None  # (6,) a reading on each joint's lattice, rad (None: 0)
    name: str = ""
    dataset: str = ""
    created_at: str = field(default_factory=now_iso)
    antares_version: str = __version__

    def __post_init__(self) -> None:
        if self.origin is None:
            self.origin = np.zeros(N_JOINTS)
        for k in ("scale", "acf1", "quantum", "n_frames", "origin"):
            v = np.asarray(getattr(self, k), dtype=np.float64).reshape(-1)
            if v.shape != (N_JOINTS,):
                raise ValueError(f"{k} needs {N_JOINTS} values, got {v.shape[0]}")
            setattr(self, k, v)
        self.origin = np.nan_to_num(self.origin)

    @property
    def file_stem(self) -> str:
        return slug(self.name)

    def sample(self, T: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """(T,6) noise with each joint's std and lag-1 correlation, radians.

        ``out[t] = rho * out[t-1] + eps[t]`` from a stationary draw at
        ``t = 0``, run as a prefix scan that doubles its span each pass
        (``log2(T)`` vectorised passes rather than ``T`` steps).
        """
        rng = rng if rng is not None else np.random.default_rng()
        T = int(T)
        rho = np.clip(np.nan_to_num(self.acf1), 0.0, 0.999)
        sd = np.nan_to_num(self.scale)
        out = rng.normal(size=(T, N_JOINTS)) * sd * np.sqrt(1.0 - rho**2)
        if T == 0:
            return out
        out[0] = rng.normal(size=N_JOINTS) * sd
        k = 1
        while k < T:
            # after this pass out[t] = sum_{s > t-2k} rho^(t-s) eps[s]; the right
            # side is evaluated before the add, so the overlap is fine
            out[k:] += rho**k * out[:-k]
            k *= 2
        return out

    def quantize(self, q: np.ndarray) -> np.ndarray:
        """Round each joint to the nearest value on its reading lattice
        (``origin + n * quantum``; joints with no step pass through)."""
        q = np.asarray(q, dtype=np.float64)
        step, o = self.quantum, self.origin
        ok = np.isfinite(step) & (step > 0)
        out = q.copy()
        out[..., ok] = o[ok] + np.round((q[..., ok] - o[ok]) / step[ok]) * step[ok]
        return out

    def apply(self, q: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Simulated readings ``q`` (T,6) rad -> what the real sensors would report."""
        q = np.asarray(q, dtype=np.float64)
        return self.quantize(q + self.sample(q.shape[0], rng))

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        def lst(v):
            return [None if not np.isfinite(x) else float(x) for x in v]

        return {
            "name": self.name,
            "dataset": self.dataset,
            "joints": list(JOINTS),
            "scale_rad": lst(self.scale),
            "acf1": lst(self.acf1),
            "quantum_rad": lst(self.quantum),
            "origin_rad": [float(x) for x in self.origin],
            "n_frames": [int(x) for x in self.n_frames],
            "created_at": self.created_at,
            "antares_version": self.antares_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NoiseModel:
        def arr(v):
            return np.array([np.nan if x is None else float(x) for x in v])

        return cls(
            scale=arr(d["scale_rad"]),
            acf1=arr(d["acf1"]),
            quantum=arr(d["quantum_rad"]),
            n_frames=np.asarray(d["n_frames"], dtype=np.float64),
            origin=arr(d["origin_rad"]) if d.get("origin_rad") is not None else None,
            name=str(d.get("name", "")),
            dataset=str(d.get("dataset", "")),
            created_at=str(d.get("created_at", "")),
            antares_version=str(d.get("antares_version", "")),
        )


def estimate(
    episodes: Sequence[Episode],
    name: str = "",
    dataset: str = "",
    unit_map: UnitMap | None = None,
    frames: int = STILL_FRAMES,
    tol: float = STILL_TOL,
) -> NoiseModel:
    """Fit a :class:`NoiseModel` from the parked stretches of ``episodes``.

    Each parked stretch is centred on its own mean (the servo may settle a
    little off its target; that offset is not noise), then pooled per joint.
    A joint with no parked stretch of at least 3 frames gets NaN.
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("no episodes")
    unit_map = unit_map if unit_map is not None else UnitMap.from_model(load_model())
    scale = np.full(N_JOINTS, np.nan)
    acf1 = np.full(N_JOINTS, np.nan)
    step = np.full(N_JOINTS, np.nan)
    origin = np.full(N_JOINTS, np.nan)
    n = np.zeros(N_JOINTS)
    for j in range(N_JOINTS):
        dev, lag0, lag1 = [], 0.0, 0.0
        for ep in episodes:
            x = np.asarray(ep.state[:, j], dtype=np.float64) * unit_map.scale[j]
            for s, e in _runs(still_mask(ep.action[:, j], frames, tol)):
                if e - s < 3:
                    continue
                d = x[s:e] - x[s:e].mean()
                dev.append(d)
                lag0 += float(np.dot(d, d))
                lag1 += float(np.dot(d[:-1], d[1:]))
        readings = np.concatenate([ep.state[:, j] for ep in episodes])
        step_u = quantum(readings)
        step[j] = step_u * abs(unit_map.scale[j])
        origin[j] = unit_map.scale[j] * lattice_origin(readings, step_u) + unit_map.offset[j]
        if dev:
            pooled = np.concatenate(dev)
            n[j] = pooled.size
            scale[j] = float(np.sqrt(np.mean(pooled**2)))
            acf1[j] = lag1 / lag0 if lag0 > 0 else 0.0
    return NoiseModel(scale=scale, acf1=acf1, quantum=step, n_frames=n, origin=origin, name=name, dataset=dataset)
