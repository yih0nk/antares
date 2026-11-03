"""A learned policy trained on the dataset's own logs, to put in the loop.

The calibration story stops at "the sim matches the robot". The consequence
worth showing is that a *policy* behaves differently in the calibrated sim
than in the stock one, so the stock sim would misjudge it. This module
supplies the policy; :mod:`antares.shadow` and :mod:`antares.closedloop`
run it through both sims.

**The policy.** A state-based chunked policy, ACT-like without images: the
input is the last ``K`` measured joint states (dataset units, normalized by
the training split's statistics), the output a chunk of the next ``H``
commanded actions. Two more inputs stand in for what the images would tell
an ACT policy, because a pick-and-place policy that does not know where the
object is cannot be run closed-loop at all: the episode's **grasp point**
(the fingertip position at the operator's first pick, from the gripper
command and FK as :mod:`antares.task` reads it, 3 numbers in the base
frame) and the **time since the episode started** (seconds). Both are
read off the real episode, and both sims get the same values, so what
differs between the sims is only the dynamics. It is a small MLP trained
with MSE on the lock's training episodes, with Gaussian noise on the
observed states so that the sim's own states in closed loop do not throw
it, and early-stopped on the lock's held-out episodes, which it therefore
never fits either. Training takes seconds on a laptop CPU; ``torch`` is an
optional extra (``pip install -e ".[policy]"``) and is imported lazily, so
the rest of Antares runs without it.

**Baselines.** A policy that cannot beat "repeat what the arm is doing" on
held-out episodes is not a policy, so every training run reports two
trivial baselines on the same windows: *hold state* (command the last
measured state for the whole chunk, which is what a state-only policy can
do without learning anything) and *repeat command* (repeat the previous
logged command, which a deployed policy does not even have).

**Units.** Everything the policy sees or emits is in dataset units (LeRobot's
normalized -100..100 / 0..100), exactly what a LeRobot policy consumes and
produces, so a checkpoint from the Hub can be dropped in through
:func:`load_lerobot_policy` with the same :class:`Policy` interface. Errors
are reported in dataset units and in degrees through the arm's
:class:`~antares.sim.UnitMap`.

The saved file is ``locks/<lock>.policy.pt`` (weights, normalization,
``K``/``H``, dataset, split, metrics) with a ``.policy.json`` sidecar holding
the same metadata, so the API can list policies without importing torch.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from antares import __version__
from antares.data import Episode
from antares.events import GripperThresholds
from antares.lock import LOCK_DIR, now_iso
from antares.robots import get_robot
from antares.sim import GRIPPER_INDEX, JOINTS, N_JOINTS, UnitMap, load_model
from antares.task import EmptyLevel, Grasp, Tip, dataset_thresholds, empty_level, find_grasps

#: Observation history (frames) and action chunk (frames) of the default policy.
K_OBS = 2
H_CHUNK = 10

#: Gaussian noise added to the observed states while training (dataset
#: units; about a third of a degree on the body joints).
OBS_NOISE_UNITS = 1.0

POLICY_SUFFIX = ".policy.pt"
POLICY_META_SUFFIX = ".policy.json"

#: Joints the degree metrics average over (the gripper is not simulated).
BODY_JOINTS = tuple(j for j in JOINTS if j != "gripper")
_BODY_IDX = [JOINTS.index(j) for j in BODY_JOINTS]


def _torch():
    """Import torch on first use, with a pointer to the extra when missing."""
    try:
        import torch
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the policy needs torch: pip install -e '.[policy]' (CPU wheels are enough)"
        ) from e
    return torch


# --------------------------------------------------------------------------
# Goals from the logs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GraspContext:
    """What every episode's grasp read-out shares: the dataset's gripper
    thresholds and empty-jaw level (:mod:`antares.task`), the fingertip FK and
    the unit map the real states are read through."""

    thresholds: GripperThresholds
    empty: EmptyLevel
    tip: Tip
    unit_map: UnitMap

    @classmethod
    def for_dataset(cls, episodes: Sequence[Episode], robot: str | None = None, unit_map: UnitMap | None = None) -> GraspContext:
        from antares.sim import Arm

        thr = dataset_thresholds(episodes)
        model = load_model(get_robot(robot).model_path)
        # The robot's own units (so101_deg reads degrees), not the MJCF's normalized default.
        if unit_map is None:
            unit_map = Arm(robot=robot).unit_map
        return cls(thr, empty_level(episodes, thr), Tip(model), unit_map)


def real_grasp(ep: Episode, ctx: GraspContext, unit_map: UnitMap | None = None) -> Grasp | None:
    """The episode's first pick (a close that held something), else its first
    close, else None when the gripper never closes. ``unit_map`` defaults to
    the context's."""
    grasps = find_grasps(ep, ctx.thresholds, ctx.tip, unit_map if unit_map is not None else ctx.unit_map, ctx.empty)
    picks = [g for g in grasps if g.kind == "pick"]
    if picks:
        return picks[0]
    return grasps[0] if grasps else None


def episode_goals(episodes: Sequence[Episode], ctx: GraspContext) -> dict[int, np.ndarray | None]:
    """Episode index -> grasp point (base frame, metres), None when there is none."""
    out: dict[int, np.ndarray | None] = {}
    for ep in episodes:
        g = real_grasp(ep, ctx)
        out[int(ep.index)] = None if g is None else np.asarray(g.xyz, dtype=np.float64)
    return out


@dataclass(frozen=True)
class StepContext:
    """What a policy may know besides the joint states at one frame: the
    frame index and rate (so time since the start), and the goal (grasp
    point, metres, base frame) when the task has one."""

    frame: int
    fps: float
    goal: np.ndarray | None = None

    @property
    def seconds(self) -> float:
        return float(self.frame) / float(self.fps)


# --------------------------------------------------------------------------
# Windows and normalization
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Features:
    """Which extra inputs the policy takes besides the ``K`` states.

    ``time`` is off by default: on the demo data a clock made the policy
    worse on held-out episodes and in closed loop (it learns *when* the
    operator closed rather than *where*), see the README.
    """

    goal: bool = True
    time: bool = False

    @property
    def n_extra(self) -> int:
        return 3 * int(self.goal) + int(self.time)

    def extras(self, ctx: StepContext) -> np.ndarray:
        parts: list[float] = []
        if self.goal:
            if ctx.goal is None:
                raise ValueError("this policy needs a goal (grasp point) and the episode has none")
            parts += [float(v) for v in np.asarray(ctx.goal, dtype=np.float64).reshape(3)]
        if self.time:
            parts.append(ctx.seconds)
        return np.asarray(parts, dtype=np.float64)

    def to_dict(self) -> dict[str, bool]:
        return {"goal": self.goal, "time": self.time}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Features:
        d = d or {}
        return cls(bool(d.get("goal", True)), bool(d.get("time", False)))


@dataclass(frozen=True)
class Windows:
    """Every frame of a set of episodes as one training window (dataset units).

    ``obs[n]`` is the ``k`` measured states up to and including the frame
    (the first frame repeated before the episode starts); ``extras[n]`` the
    goal and time features; ``target[n]`` the ``h`` commands from the frame
    on (the last command repeated past the end); ``prev_action[n]`` the
    command of the frame before (the frame's own at the start), for the
    repeat-command baseline; ``episode[n]`` the episode index.
    """

    obs: np.ndarray  # (N,k,6)
    extras: np.ndarray  # (N,E)
    target: np.ndarray  # (N,h,6)
    prev_action: np.ndarray  # (N,6)
    episode: np.ndarray  # (N,)

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def windows(
    episodes: Sequence[Episode],
    k: int = K_OBS,
    h: int = H_CHUNK,
    features: Features = Features(),
    goals: dict[int, np.ndarray | None] | None = None,
) -> Windows:
    """Build :class:`Windows`. Episodes without a goal are left out when the
    policy takes one; ``goals`` defaults to none (so ``features.goal`` needs
    them passed in)."""
    if k < 1 or h < 1:
        raise ValueError("k and h must be >= 1")
    obs, ext, tgt, prev, idx = [], [], [], [], []
    for ep in episodes:
        T = len(ep)
        if T == 0:
            continue
        goal = (goals or {}).get(int(ep.index))
        if features.goal and goal is None:
            continue
        t = np.arange(T)
        oi = np.clip(t[:, None] - np.arange(k)[::-1][None, :], 0, T - 1)  # (T,k)
        ai = np.clip(t[:, None] + np.arange(h)[None, :], 0, T - 1)  # (T,h)
        obs.append(np.asarray(ep.state, dtype=np.float64)[oi])
        tgt.append(np.asarray(ep.action, dtype=np.float64)[ai])
        prev.append(np.asarray(ep.action, dtype=np.float64)[np.maximum(t - 1, 0)])
        e = np.stack([features.extras(StepContext(int(f), ep.fps, goal)) for f in t]) if features.n_extra else np.zeros((T, 0))
        ext.append(e)
        idx.append(np.full(T, int(ep.index)))
    if not obs:
        raise ValueError("no usable frames (no episodes, or none with a grasp point)")
    return Windows(np.concatenate(obs), np.concatenate(ext), np.concatenate(tgt), np.concatenate(prev), np.concatenate(idx))


@dataclass(frozen=True)
class Normalizer:
    """Per-joint mean and standard deviation of states and actions (dataset
    units), and per-feature for the extras."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    extra_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extra_std: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        for name in ("state_mean", "state_std", "action_mean", "action_std", "extra_mean", "extra_std"):
            v = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if name.startswith(("state", "action")) and v.shape[0] != N_JOINTS:
                raise ValueError(f"{name} needs {N_JOINTS} values, got {v.shape[0]}")
            if name.endswith("std"):
                v = np.where(v > 1e-6, v, 1.0)
            object.__setattr__(self, name, v)
        if self.extra_mean.shape != self.extra_std.shape:
            raise ValueError("extra_mean and extra_std must have the same shape")

    @classmethod
    def from_windows(cls, w: Windows) -> Normalizer:
        s = w.obs[:, -1, :]
        a = w.target[:, 0, :]
        return cls(s.mean(0), s.std(0), a.mean(0), a.std(0), w.extras.mean(0), w.extras.std(0))

    @classmethod
    def from_episodes(cls, episodes: Sequence[Episode]) -> Normalizer:
        s = np.concatenate([np.asarray(e.state, dtype=np.float64) for e in episodes])
        a = np.concatenate([np.asarray(e.action, dtype=np.float64) for e in episodes])
        return cls(s.mean(0), s.std(0), a.mean(0), a.std(0))

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "extra_mean": self.extra_mean.tolist(),
            "extra_std": self.extra_std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Normalizer:
        return cls(
            d["state_mean"], d["state_std"], d["action_mean"], d["action_std"],
            d.get("extra_mean", []), d.get("extra_std", []),
        )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _deg_per_unit(unit_map: UnitMap | None) -> np.ndarray:
    """Degrees per dataset unit, per joint (the unit map's scale is rad/unit)."""
    if unit_map is None:
        unit_map = UnitMap.from_model(load_model())
    return np.degrees(np.abs(np.asarray(unit_map.scale, dtype=np.float64)))


def action_errors(pred: np.ndarray, target: np.ndarray, unit_map: UnitMap | None = None) -> dict[str, Any]:
    """Chunk-prediction errors in dataset units and degrees.

    ``pred``/``target`` are ``(N,H,6)``. ``mse_units`` is the mean squared
    error over every joint and chunk step (what training minimizes);
    ``rmse_deg`` the RMS error over the body joints in degrees;
    ``first_rmse_deg`` the same for the first action of the chunk, the one a
    receding-horizon controller executes; ``per_joint_deg`` per joint over
    the chunk (the gripper in degrees too, for completeness).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"pred and target must both be (N,H,6), got {pred.shape} and {target.shape}")
    dpu = _deg_per_unit(unit_map)
    err = pred - target  # (N,H,6)
    mse_units = float(np.mean(err**2))
    mse_joint = np.mean(err**2, axis=(0, 1))  # (6,)
    per_joint_deg = {j: float(np.sqrt(mse_joint[i]) * dpu[i]) for i, j in enumerate(JOINTS)}
    body_deg2 = (mse_joint * dpu**2)[_BODY_IDX]
    first_deg2 = (np.mean(err[:, 0, :] ** 2, axis=0) * dpu**2)[_BODY_IDX]
    return {
        "mse_units": mse_units,
        "rmse_units": float(math.sqrt(mse_units)),
        "rmse_deg": float(np.sqrt(np.mean(body_deg2))),
        "first_rmse_deg": float(np.sqrt(np.mean(first_deg2))),
        "gripper_rmse_units": float(np.sqrt(mse_joint[GRIPPER_INDEX])),
        "per_joint_deg": per_joint_deg,
        "n_windows": int(pred.shape[0]),
    }


def baseline_predictions(w: Windows, h: int) -> dict[str, np.ndarray]:
    """The two trivial chunk predictors on a set of windows (see the module docstring)."""
    hold = np.repeat(w.obs[:, -1:, :], h, axis=1)
    repeat = np.repeat(w.prev_action[:, None, :], h, axis=1)
    return {"hold_state": hold, "repeat_command": repeat}


# --------------------------------------------------------------------------
# The policy interface
# --------------------------------------------------------------------------


@runtime_checkable
class Policy(Protocol):
    """What the evaluators need: a history length, a chunk length, and ``act``.

    ``act`` takes the last ``k`` observed joint states ``(k,6)`` in dataset
    units plus the frame's :class:`StepContext` and returns the next ``h``
    commands ``(h,6)`` in dataset units. ``reset`` is called at the start
    of every episode.
    """

    k: int
    h: int

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray, ctx: StepContext) -> np.ndarray: ...


def build_mlp(n_in: int, h: int, hidden: int, layers: int) -> Any:
    """``n_in -> hidden x layers -> h*6`` MLP with GELU (a ``torch.nn.Sequential``)."""
    torch = _torch()
    nn = torch.nn
    dims = [n_in] + [hidden] * layers
    mods: list[Any] = []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [nn.Linear(a, b), nn.GELU()]
    mods.append(nn.Linear(dims[-1], h * N_JOINTS))
    return nn.Sequential(*mods)


class ChunkedPolicy:
    """The trained state-based chunked policy (see the module docstring).

    ``meta`` carries everything the file records besides the weights:
    ``k``, ``h``, ``hidden``, ``layers``, ``features``, ``dataset``,
    ``lock``, the split, the metrics and the timestamp.
    """

    def __init__(
        self,
        module: Any,
        norm: Normalizer,
        k: int,
        h: int,
        features: Features = Features(),
        meta: dict[str, Any] | None = None,
    ):
        self.module = module
        self.norm = norm
        self.k = int(k)
        self.h = int(h)
        self.features = features
        self.meta: dict[str, Any] = dict(meta or {})
        self.module.eval()

    @property
    def needs_goal(self) -> bool:
        return self.features.goal

    def reset(self) -> None:  # stateless
        return None

    def act(self, obs: np.ndarray, ctx: StepContext) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (self.k, N_JOINTS):
            raise ValueError(f"obs must be ({self.k},{N_JOINTS}), got {obs.shape}")
        return self.act_batch(obs[None], self.features.extras(ctx)[None])[0]

    def act_batch(self, obs: np.ndarray, extras: np.ndarray | None = None) -> np.ndarray:
        """``(N,k,6)`` observations and ``(N,E)`` extras -> ``(N,h,6)`` chunks, dataset units."""
        torch = _torch()
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim != 3 or obs.shape[1:] != (self.k, N_JOINTS):
            raise ValueError(f"obs must be (N,{self.k},{N_JOINTS}), got {obs.shape}")
        n_extra = self.features.n_extra
        extras = np.zeros((len(obs), 0)) if extras is None else np.asarray(extras, dtype=np.float64).reshape(len(obs), -1)
        if extras.shape[1] != n_extra:
            raise ValueError(f"extras must be (N,{n_extra}), got {extras.shape}")
        x = ((obs - self.norm.state_mean) / self.norm.state_std).reshape(len(obs), -1)
        if n_extra:
            x = np.concatenate([x, (extras - self.norm.extra_mean) / self.norm.extra_std], axis=1)
        with torch.no_grad():
            y = self.module(torch.as_tensor(x, dtype=torch.float32)).numpy()
        y = y.reshape(len(x), self.h, N_JOINTS).astype(np.float64)
        return y * self.norm.action_std + self.norm.action_mean

    # -- files ----------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        torch = _torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**self.meta, "k": self.k, "h": self.h, "features": self.features.to_dict(), "norm": self.norm.to_dict()}
        torch.save({"state_dict": self.module.state_dict(), "meta": meta}, path)
        if path.name.endswith(POLICY_SUFFIX):
            sidecar = path.with_name(path.name[: -len(POLICY_SUFFIX)] + POLICY_META_SUFFIX)
        else:
            sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> ChunkedPolicy:
        torch = _torch()
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        meta = dict(blob["meta"])
        k, h = int(meta["k"]), int(meta["h"])
        feats = Features.from_dict(meta.get("features"))
        module = build_mlp(k * N_JOINTS + feats.n_extra, h, int(meta.get("hidden", 256)), int(meta.get("layers", 3)))
        module.load_state_dict(blob["state_dict"])
        return cls(module, Normalizer.from_dict(meta["norm"]), k, h, feats, meta)


class LoggedPolicy:
    """Replays an episode's logged commands regardless of what it observes.

    The control: a closed-loop run with this policy must reproduce
    :meth:`antares.sim.Arm.replay` exactly, which pins the evaluators'
    timing conventions to the calibration's.
    """

    needs_goal = False

    def __init__(self, episode: Episode, k: int = K_OBS, h: int = H_CHUNK):
        self.episode = episode
        self.k, self.h = int(k), int(h)
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def act(self, obs: np.ndarray, ctx: StepContext | None = None) -> np.ndarray:
        T = len(self.episode)
        t = self.t if ctx is None else int(ctx.frame)
        idx = np.clip(np.arange(t, t + self.h), 0, T - 1)
        self.t += 1
        return np.asarray(self.episode.action, dtype=np.float64)[idx]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainProgress:
    epoch: int
    epochs: int
    train_mse: float
    eval_mse: float
    best_eval_mse: float
    best_epoch: int
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {"type": "progress", **{k: (float(v) if isinstance(v, float) else v) for k, v in self.__dict__.items()}}


@dataclass
class TrainResult:
    policy: ChunkedPolicy
    metrics: dict[str, Any]
    history: list[tuple[int, float, float]] = field(default_factory=list)  # (epoch, train_mse, eval_mse)
    epochs_run: int = 0
    best_epoch: int = 0
    elapsed_s: float = 0.0
    stopped: bool = False


def train_policy(
    train: Sequence[Episode],
    eval: Sequence[Episode],
    k: int = K_OBS,
    h: int = H_CHUNK,
    hidden: int = 256,
    layers: int = 3,
    epochs: int = 200,
    patience: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    features: Features = Features(),
    obs_noise_units: float = OBS_NOISE_UNITS,
    ctx: GraspContext | None = None,
    unit_map: UnitMap | None = None,
    on_progress: Callable[[TrainProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    meta: dict[str, Any] | None = None,
) -> TrainResult:
    """Fit the chunked policy on ``train``, early-stopping on ``eval``.

    Normalization statistics come from ``train`` only. Goals (grasp points)
    are read from each episode through ``ctx`` (built from ``train + eval``
    when not given); episodes without one are dropped from both splits when
    ``features.goal``. The weights with the best eval MSE are kept.
    ``metrics`` holds the policy's and both baselines' errors on each split
    (:func:`action_errors`), plus ``beats_hold_state`` /
    ``beats_repeat_command`` on the eval split. ``should_stop`` is polled
    every epoch (a cancelled job keeps the best weights so far).
    """
    torch = _torch()
    train, eval = list(train), list(eval)
    if not train:
        raise ValueError("no training episodes")
    if not eval:
        raise ValueError("no eval episodes (early stopping needs a held-out split)")
    goals: dict[int, np.ndarray | None] = {}
    if features.goal:
        if ctx is None:
            ctx = GraspContext.for_dataset(train + eval)
        goals = episode_goals(train + eval, ctx)
        if unit_map is None:
            unit_map = ctx.unit_map  # report degrees in the dataset's own units
    wtr = windows(train, k, h, features, goals)
    wev = windows(eval, k, h, features, goals)
    norm = Normalizer.from_windows(wtr)
    n_in = k * N_JOINTS + features.n_extra

    def enc(w: Windows) -> tuple[Any, Any, Any]:
        xs = ((w.obs - norm.state_mean) / norm.state_std).reshape(len(w), -1)
        es = (w.extras - norm.extra_mean) / norm.extra_std if features.n_extra else np.zeros((len(w), 0))
        ys = ((w.target - norm.action_mean) / norm.action_std).reshape(len(w), -1)
        return (
            torch.as_tensor(xs, dtype=torch.float32),
            torch.as_tensor(es, dtype=torch.float32),
            torch.as_tensor(ys, dtype=torch.float32),
        )

    xtr, etr, ytr = enc(wtr)
    xev, eev, yev = enc(wev)
    # observation noise in normalized units, per state column (repeated k times)
    noise_scale = torch.as_tensor(np.tile(obs_noise_units / norm.state_std, k), dtype=torch.float32)

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    module = build_mlp(n_in, h, hidden, layers)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = torch.nn.MSELoss()

    best_state = copy.deepcopy(module.state_dict())
    best_eval, best_epoch = math.inf, 0
    history: list[tuple[int, float, float]] = []
    t0 = time.perf_counter()
    stopped = False
    epochs_run = 0
    n = len(xtr)
    xev_full = torch.cat([xev, eev], dim=1)
    for epoch in range(1, epochs + 1):
        module.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = xtr[idx]
            if obs_noise_units > 0:
                xb = xb + torch.randn(xb.shape, generator=gen) * noise_scale
            opt.zero_grad()
            loss = loss_fn(module(torch.cat([xb, etr[idx]], dim=1)), ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        module.eval()
        with torch.no_grad():
            ev = float(loss_fn(module(xev_full), yev))
        tr = tot / n
        epochs_run = epoch
        if ev < best_eval - 1e-9:
            best_eval, best_epoch = ev, epoch
            best_state = copy.deepcopy(module.state_dict())
        history.append((epoch, tr, ev))
        if on_progress is not None:
            on_progress(TrainProgress(epoch, epochs, tr, ev, best_eval, best_epoch, time.perf_counter() - t0))
        if should_stop is not None and should_stop():
            stopped = True
            break
        if epoch - best_epoch >= patience:
            break
    module.load_state_dict(best_state)
    module.eval()

    policy = ChunkedPolicy(module, norm, k, h, features)
    pred_tr = policy.act_batch(wtr.obs, wtr.extras)
    pred_ev = policy.act_batch(wev.obs, wev.extras)
    base_tr = baseline_predictions(wtr, h)
    base_ev = baseline_predictions(wev, h)
    metrics: dict[str, Any] = {
        "policy": {"train": action_errors(pred_tr, wtr.target, unit_map), "eval": action_errors(pred_ev, wev.target, unit_map)},
        "baselines": {
            name: {
                "train": action_errors(base_tr[name], wtr.target, unit_map),
                "eval": action_errors(base_ev[name], wev.target, unit_map),
            }
            for name in base_tr
        },
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "best_eval_mse_normalized": float(best_eval),
    }
    pe = metrics["policy"]["eval"]["rmse_deg"]
    metrics["beats_hold_state"] = bool(pe < metrics["baselines"]["hold_state"]["eval"]["rmse_deg"])
    metrics["beats_repeat_command"] = bool(pe < metrics["baselines"]["repeat_command"]["eval"]["rmse_deg"])
    elapsed = time.perf_counter() - t0
    used_tr = sorted({int(i) for i in wtr.episode})
    used_ev = sorted({int(i) for i in wev.episode})
    policy.meta = {
        **(meta or {}),
        "kind": "chunked_mlp",
        "k": k,
        "h": h,
        "hidden": hidden,
        "layers": layers,
        "features": features.to_dict(),
        "n_params": int(sum(p.numel() for p in module.parameters())),
        "train_episodes": used_tr,
        "eval_episodes": used_ev,
        "dropped_episodes": sorted({int(e.index) for e in train + eval} - set(used_tr) - set(used_ev)),
        "n_train_windows": int(len(wtr)),
        "n_eval_windows": int(len(wev)),
        "epochs": epochs,
        "patience": patience,
        "lr": lr,
        "batch_size": batch_size,
        "obs_noise_units": float(obs_noise_units),
        "seed": seed,
        "metrics": metrics,
        "history": [[int(e), float(a), float(b)] for e, a, b in history],
        "elapsed_s": float(elapsed),
        "stopped": stopped,
        "created_at": now_iso(),
        "antares_version": __version__,
    }
    return TrainResult(policy, metrics, history, epochs_run, best_epoch, elapsed, stopped)


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def policy_path(name: str, lock_dir: Path | str | None = None) -> Path:
    return Path(lock_dir if lock_dir is not None else LOCK_DIR) / f"{name}{POLICY_SUFFIX}"


def save_policy(policy: ChunkedPolicy, name: str, lock_dir: Path | str | None = None) -> Path:
    """``locks/<name>.policy.pt`` plus its ``.policy.json`` sidecar."""
    policy.meta["name"] = name
    return policy.save(policy_path(name, lock_dir))


def load_policy(name: str, lock_dir: Path | str | None = None) -> ChunkedPolicy:
    p = policy_path(name, lock_dir)
    if not p.is_file():
        raise FileNotFoundError(f"no policy named {name!r} ({p})")
    return ChunkedPolicy.load(p)


def policy_meta(name: str, lock_dir: Path | str | None = None) -> dict[str, Any] | None:
    """The sidecar of a saved policy (no torch needed), or None."""
    side = policy_path(name, lock_dir).with_name(f"{name}{POLICY_META_SUFFIX}")
    if side.is_file():
        try:
            return json.loads(side.read_text())
        except ValueError:
            return None
    return None


def list_policies(lock_dir: Path | str | None = None, lock: str | None = None) -> list[dict[str, Any]]:
    """Every saved policy's sidecar (optionally only those trained for ``lock``),
    newest first. A policy named after a lock belongs to it even if the
    sidecar predates the ``lock`` field."""
    d = Path(lock_dir if lock_dir is not None else LOCK_DIR)
    if not d.is_dir():
        return []
    out = []
    for side in sorted(d.glob(f"*{POLICY_META_SUFFIX}")):
        try:
            meta = json.loads(side.read_text())
        except (OSError, ValueError):
            continue
        name = side.name[: -len(POLICY_META_SUFFIX)]
        meta.setdefault("name", name)
        weights = side.with_name(f"{name}{POLICY_SUFFIX}")
        meta["path"] = str(weights)
        meta["exists"] = weights.is_file()
        if lock is not None and meta.get("lock") != lock and name != lock:
            continue
        out.append(meta)
    out.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
    return out


# --------------------------------------------------------------------------
# LeRobot checkpoints
# --------------------------------------------------------------------------


class LeRobotPolicy:
    """A LeRobot ``PreTrainedPolicy`` behind the :class:`Policy` interface.

    LeRobot policies keep their own observation history and action queue:
    ``select_action(batch)`` takes one frame and returns one action, so the
    wrapper reports ``k = 1`` and ``h = 1`` (the chunk-shift variant of the
    closed loop then has nothing to shift; ACT/diffusion queue their own
    chunks internally). ``batch`` is ``{"observation.state": (1,6) float32}``
    plus whatever ``extra_inputs(ctx)`` returns, which is where rendered
    camera images go for an image policy (see the README: the fitted camera
    and scene are what make that render match the real one). Such a policy
    does not need the grasp point: the images carry it.

    Dataset units in, dataset units out, like the LeRobot runtime.
    """

    needs_goal = False

    def __init__(
        self,
        policy: Any,
        device: str = "cpu",
        state_key: str = "observation.state",
        extra_inputs: Callable[[StepContext], dict[str, Any]] | None = None,
    ):
        self.policy = policy
        self.device = device
        self.state_key = state_key
        self.extra_inputs = extra_inputs
        self.k = 1
        self.h = 1

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def act(self, obs: np.ndarray, ctx: StepContext) -> np.ndarray:
        torch = _torch()
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != N_JOINTS:
            raise ValueError(f"obs must be (k,{N_JOINTS}), got {obs.shape}")
        batch = {self.state_key: torch.as_tensor(obs[-1][None], dtype=torch.float32, device=self.device)}
        if self.extra_inputs is not None:
            batch.update(self.extra_inputs(ctx))
        with torch.no_grad():
            a = self.policy.select_action(batch)
        a = np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, dtype=np.float64).reshape(-1, N_JOINTS)
        return a[:1]


def load_lerobot_policy(repo_id: str, device: str = "cpu", **kwargs: Any) -> LeRobotPolicy:
    """Load a LeRobot checkpoint (Hub id or local directory) as a :class:`Policy`.

    Needs the ``lerobot`` package (not installed by any Antares extra; follow
    LeRobot's own install instructions). The policy class is picked from the
    checkpoint's config, as ``lerobot.policies.factory`` does. ``kwargs`` go
    to :class:`LeRobotPolicy` (``extra_inputs`` for image policies).
    """
    try:
        from lerobot.configs.policies import PreTrainedConfig  # type: ignore
        from lerobot.policies.factory import get_policy_class  # type: ignore
    except ImportError as e:
        raise ImportError(
            "load_lerobot_policy needs the lerobot package (pip install lerobot); "
            "the state-based ChunkedPolicy needs only torch"
        ) from e
    cfg = PreTrainedConfig.from_pretrained(repo_id)
    cls = get_policy_class(cfg.type)
    policy = cls.from_pretrained(repo_id, config=cfg)
    policy.eval()
    to = getattr(policy, "to", None)
    if callable(to):
        to(device)
    return LeRobotPolicy(policy, device=device, **kwargs)


__all__ = [
    "BODY_JOINTS",
    "H_CHUNK",
    "K_OBS",
    "OBS_NOISE_UNITS",
    "POLICY_META_SUFFIX",
    "POLICY_SUFFIX",
    "ChunkedPolicy",
    "Features",
    "GraspContext",
    "LeRobotPolicy",
    "LoggedPolicy",
    "Normalizer",
    "Policy",
    "StepContext",
    "TrainProgress",
    "TrainResult",
    "Windows",
    "action_errors",
    "baseline_predictions",
    "build_mlp",
    "episode_goals",
    "list_policies",
    "load_lerobot_policy",
    "load_policy",
    "policy_meta",
    "policy_path",
    "real_grasp",
    "save_policy",
    "train_policy",
    "windows",
]
