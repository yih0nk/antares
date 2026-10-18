"""Does calibrating the sim help a policy transfer? A sim-to-sim experiment.

Antares's headline number is a replay metric: how much of the tracking gap
between the simulator and one robot's logs the fit closes. The claim a user
cares about is one step further: a policy trained in the calibrated sim
should work better on that robot than one trained in the default sim.
Nothing with a real arm can run in CI, so this module tests the claim
sim-to-sim: a MuJoCo arm with *hidden* true parameters stands in for the
real robot, a lock is fitted to logs synthesized from it (the fit never sees
the truth), policies are trained in several sims and all of them are scored
on the hidden arm.

Task
    Joint-space reaching at the dataset frame rate: from a random start
    pose, reach a random target pose (both uniform inside +-``task_range``
    dataset units on the five body joints, gripper held) within ``horizon``
    frames. Success is every body joint within ``tol_deg`` of the target for
    the last ``k_frames`` frames; the mean final error (degrees, last
    ``k_frames`` frames) is reported alongside.

Policy
    A memoryless controller that observes joint positions, velocities and
    the target and emits a position *command* in dataset units:
    ``u = t + kp*(t - q) - kd*v + 0.1*MLP(q, t)`` per joint, with per-joint
    ``kp``/``kd`` and a one-hidden-layer tanh network for pose-dependent
    feedforward. ``u = t`` (all parameters zero) is the untrained policy.
    Reaching a soft, laggy servo with that structure means learning how far
    past the target to command and how much to lead, which is exactly what
    the calibrated parameters change.

Training
    CMA-ES (:mod:`cma`) on the policy parameters, objective = mean tracking
    error over a fixed batch of start/target pairs plus the mean final
    error, minus a success bonus. The batch depends only on the training
    seed, so every variant trained with seed ``s`` sees the same tasks. For
    domain-randomized variants a fresh parameter set per task is drawn each
    generation (shared by the generation's candidates), and the returned
    policy is the final CMA-ES mean rather than the noisy best sample.

The simulator is injected (:class:`Simulator`); :class:`MuJoCoSim` wraps
:class:`antares.sim.Arm` for frame-by-frame closed-loop control with the
lock's latency, and tests use a toy linear arm.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from antares import __version__
from antares.sim import GRIPPER_INDEX, JOINTS, N_JOINTS, SimParams

#: The five body joints the policy drives and the metrics score.
N_BODY = N_JOINTS - 1
BODY_JOINTS = tuple(j for j in JOINTS if j != "gripper")

#: Observation scaling: positions in hundredths of the joint range,
#: velocities in units of 30 dataset units per frame (a fast move peaks
#: near 10 units per frame, so the scaled velocity stays well inside +-1).
_POS_SCALE = 100.0
_VEL_SCALE = 30.0
_CMD_CLIP = 1.1  # commands may overshoot the normalized range a little
#: The feedforward head is in tenths of the normalized range (10 dataset
#: units), so CMA-ES's initial step moves it by about a degree, like the
#: corrections it has to find, instead of tens of degrees.
_FF_SCALE = 0.1

VARIANTS: tuple[str, ...] = ("default", "lock", "lock_dr", "oracle", "default_pm30")
VARIANT_LABELS: dict[str, str] = {
    "default": "default sim",
    "lock": "fitted lock",
    "lock_dr": "fitted lock + data-derived DR",
    "oracle": "oracle (true parameters)",
    "default_pm30": "default sim + hand-picked +-30% DR",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Task:
    """One reaching problem in dataset units: ``start`` -> ``target``, (6,) each.

    The gripper entry of ``target`` equals ``start``'s; it is commanded but
    never scored.
    """

    start: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", np.asarray(self.start, dtype=np.float64).reshape(N_JOINTS))
        object.__setattr__(self, "target", np.asarray(self.target, dtype=np.float64).reshape(N_JOINTS))

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.tolist(), "target": self.target.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(start=np.asarray(d["start"]), target=np.asarray(d["target"]))


def sample_tasks(n: int, seed: int, task_range: float = 70.0, gripper: tuple[float, float] = (20.0, 80.0)) -> list[Task]:
    """``n`` start/target pairs, uniform inside ``+-task_range`` on the body
    joints and ``gripper`` on the gripper (held through the episode).

    The sequence is a function of ``seed`` alone, so train and held-out sets
    are reproducible and the first ``k`` tasks of a longer draw are the
    shorter draw.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if task_range <= 0 or task_range > 100:
        raise ValueError("task_range must be in (0, 100] dataset units")
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n):
        start = np.empty(N_JOINTS)
        target = np.empty(N_JOINTS)
        start[:N_BODY] = rng.uniform(-task_range, task_range, N_BODY)
        target[:N_BODY] = rng.uniform(-task_range, task_range, N_BODY)
        g = rng.uniform(*gripper)
        start[GRIPPER_INDEX] = g
        target[GRIPPER_INDEX] = g
        tasks.append(Task(start, target))
    return tasks


# --------------------------------------------------------------------------
# Simulators
# --------------------------------------------------------------------------


class Simulator(Protocol):
    """Frame-by-frame closed-loop arm in dataset units.

    ``deg_per_unit`` converts dataset units to degrees per joint, (6,).
    ``set_params`` switches the dynamics (no-op for a toy); ``reset`` places
    the arm at rest at ``q0`` and returns the first observation; ``step``
    applies one command for one frame and returns the measured positions.
    """

    deg_per_unit: np.ndarray

    def set_params(self, params: SimParams) -> None: ...

    def reset(self, q0_units: np.ndarray) -> np.ndarray: ...

    def step(self, cmd_units: np.ndarray) -> np.ndarray: ...


class MuJoCoSim:
    """:class:`antares.sim.Arm` driven one dataset frame at a time.

    Follows the replay convention of :meth:`antares.sim.Arm.rollout`, where
    the command logged at frame ``s`` is in force at frame ``s + latency``.
    In closed loop the command sent after observing frame ``t`` can first
    act on the step producing frame ``t + 1``, so a command sent at ``t``
    drives the step producing frame ``t + max(latency, 1)``; latency 0 and 1
    are the same closed-loop arm.
    """

    def __init__(self, params: SimParams | None = None, fps: float = 30.0, arm: Any | None = None):
        import mujoco

        from antares.sim import Arm

        self._mujoco = mujoco
        self.arm = arm if arm is not None else Arm()
        self.fps = float(fps)
        self.frame_dt = 1.0 / self.fps
        self.n_sub = max(1, int(round(self.frame_dt / self.arm._base_timestep)))
        self.deg_per_unit = np.degrees(self.arm.unit_map.scale)
        self._ctrl_lo, self._ctrl_hi = self.arm.ctrl_range
        self._delay = np.zeros(N_JOINTS, dtype=np.int64)
        self._buf = np.zeros((1, N_JOINTS))
        self._q0_ctrl = np.zeros(N_JOINTS)
        self._t = 0
        self.set_params(params if params is not None else self.arm.params)

    def set_params(self, params: SimParams) -> None:
        self.arm.apply(params)
        self._delay = np.maximum(params.per_joint("latency_frames") - 1, 0)
        self._buf = np.zeros((int(self._delay.max()) + 1, N_JOINTS))

    def reset(self, q0_units: np.ndarray) -> np.ndarray:
        m, d = self.arm.model, self.arm.data
        m.opt.timestep = self.frame_dt / self.n_sub
        q0 = self.arm.unit_map.to_rad(q0_units)
        self._mujoco.mj_resetData(m, d)
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        self._q0_ctrl = np.clip(q0, self._ctrl_lo, self._ctrl_hi)
        d.ctrl[:] = self._q0_ctrl
        self._mujoco.mj_forward(m, d)
        self._buf[:] = self._q0_ctrl
        self._t = 0
        return self.arm.unit_map.to_units(d.qpos)

    def step(self, cmd_units: np.ndarray) -> np.ndarray:
        m, d = self.arm.model, self.arm.data
        ctrl = np.clip(self.arm.unit_map.to_rad(cmd_units), self._ctrl_lo, self._ctrl_hi)
        L = self._buf.shape[0]
        t = self._t
        self._buf[t % L] = ctrl
        src = t - self._delay
        applied = np.where(src >= 0, self._buf[src % L, np.arange(N_JOINTS)], self._q0_ctrl)
        d.ctrl[:] = applied
        self._mujoco.mj_step(m, d, self.n_sub)
        self._t = t + 1
        return self.arm.unit_map.to_units(d.qpos)


class ToyArm:
    """A first-order servo per joint with latency and a constant sag, in
    dataset units; MuJoCo-free stand-in for tests.

    ``q += rate * (u_delayed - q) - sag`` per frame, where ``sag`` is a
    steady offset (gravity on a soft servo) the policy must command past.
    """

    def __init__(self, rate: float = 0.3, latency: int = 2, sag: float | np.ndarray = 0.0, deg_per_unit: float = 1.0):
        self.rate = float(rate)
        self.latency = int(latency)
        self.sag = np.broadcast_to(np.asarray(sag, dtype=np.float64), (N_JOINTS,)).copy()
        self.deg_per_unit = np.full(N_JOINTS, float(deg_per_unit))
        self._q = np.zeros(N_JOINTS)
        self._hist: list[np.ndarray] = []

    def set_params(self, params: SimParams) -> None:  # the toy has no SimParams
        return None

    def reset(self, q0_units: np.ndarray) -> np.ndarray:
        self._q = np.asarray(q0_units, dtype=np.float64).copy()
        self._hist = [self._q.copy()]
        return self._q.copy()

    def step(self, cmd_units: np.ndarray) -> np.ndarray:
        self._hist.append(np.asarray(cmd_units, dtype=np.float64).copy())
        u = self._hist[max(len(self._hist) - 1 - self.latency, 0)]
        self._q = self._q + self.rate * (u - self._q) - self.sag
        return self._q.copy()


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Policy:
    """``u = t + kp*(t-q) - kd*v + 0.1*(W2 tanh(W1 [q, t] + b1) + b2)`` on the body joints.

    ``theta`` holds, in order, ``kp`` (5), ``kd`` (5), ``W1`` (hidden x 10),
    ``b1`` (hidden), ``W2`` (5 x hidden), ``b2`` (5). All zero is ``u = t``.
    """

    hidden: int
    theta: np.ndarray

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=np.float64).reshape(-1)
        n = self.n_params(self.hidden)
        if theta.shape[0] != n:
            raise ValueError(f"policy with hidden={self.hidden} needs {n} parameters, got {theta.shape[0]}")
        object.__setattr__(self, "theta", theta)
        h = self.hidden
        i = 0
        kp, i = theta[i : i + N_BODY], i + N_BODY
        kd, i = theta[i : i + N_BODY], i + N_BODY
        w1, i = theta[i : i + h * 2 * N_BODY].reshape(h, 2 * N_BODY), i + h * 2 * N_BODY
        b1, i = theta[i : i + h], i + h
        w2, i = theta[i : i + N_BODY * h].reshape(N_BODY, h), i + N_BODY * h
        b2 = theta[i : i + N_BODY]
        object.__setattr__(self, "_parts", (kp, kd, w1, b1, w2, b2))

    @staticmethod
    def n_params(hidden: int) -> int:
        if hidden < 1:
            raise ValueError("hidden must be >= 1")
        return 2 * N_BODY + hidden * 2 * N_BODY + hidden + N_BODY * hidden + N_BODY

    @classmethod
    def zeros(cls, hidden: int = 6) -> Policy:
        """The untrained policy: command the target directly."""
        return cls(hidden=hidden, theta=np.zeros(cls.n_params(hidden)))

    def act(self, q_units: np.ndarray, v_units_per_frame: np.ndarray, target_units: np.ndarray) -> np.ndarray:
        kp, kd, w1, b1, w2, b2 = self._parts
        qn = q_units[:N_BODY] / _POS_SCALE
        tn = target_units[:N_BODY] / _POS_SCALE
        vn = v_units_per_frame[:N_BODY] / _VEL_SCALE
        h = np.tanh(w1 @ np.concatenate((qn, tn)) + b1)
        out = tn + kp * (tn - qn) - kd * vn + _FF_SCALE * (w2 @ h + b2)
        cmd = np.empty(N_JOINTS)
        cmd[:N_BODY] = _POS_SCALE * np.clip(out, -_CMD_CLIP, _CMD_CLIP)
        cmd[GRIPPER_INDEX] = target_units[GRIPPER_INDEX]
        return cmd

    def to_dict(self) -> dict[str, Any]:
        return {"hidden": int(self.hidden), "theta": self.theta.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Policy:
        return cls(hidden=int(d["hidden"]), theta=np.asarray(d["theta"], dtype=np.float64))


# --------------------------------------------------------------------------
# Rollout and metrics
# --------------------------------------------------------------------------


def rollout(sim: Simulator, policy: Policy, task: Task, horizon: int) -> np.ndarray:
    """Run ``policy`` on ``task`` for ``horizon`` frames; positions ``(horizon+1, 6)`` in units."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    q = sim.reset(task.start)
    out = np.empty((horizon + 1, N_JOINTS))
    out[0] = q
    v = np.zeros(N_JOINTS)
    for t in range(horizon):
        cmd = policy.act(q, v, task.target)
        q_next = sim.step(cmd)
        v = q_next - q
        q = q_next
        out[t + 1] = q
    return out


def episode_metrics(
    q_units: np.ndarray, task: Task, deg_per_unit: np.ndarray, tol_deg: float, k_frames: int
) -> tuple[bool, float, float]:
    """``(success, final_err_deg, mean_err_deg)`` of one trajectory.

    Errors are absolute, in degrees, over the five body joints. ``success``
    is every joint within ``tol_deg`` on each of the last ``k_frames``
    frames; ``final_err_deg`` is the mean error over those frames and
    joints; ``mean_err_deg`` the mean over the whole trajectory.
    """
    q = np.asarray(q_units, dtype=np.float64)
    if k_frames < 1 or k_frames > q.shape[0]:
        raise ValueError("k_frames must be in [1, n_frames]")
    err = np.abs(q[:, :N_BODY] - task.target[:N_BODY]) * np.asarray(deg_per_unit)[:N_BODY]
    tail = err[-k_frames:]
    success = bool(np.all(tail <= tol_deg))
    return success, float(tail.mean()), float(err.mean())


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean of ``values``."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(lo), float(hi)


@dataclass(frozen=True, eq=False)
class EvalResult:
    """Per-task outcomes of one policy on one set of tasks, with summaries."""

    success: np.ndarray  # (n,) bool
    final_err_deg: np.ndarray  # (n,)
    mean_err_deg: np.ndarray  # (n,)
    tol_deg: float
    k_frames: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "success", np.asarray(self.success, dtype=bool).reshape(-1))
        object.__setattr__(self, "final_err_deg", np.asarray(self.final_err_deg, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "mean_err_deg", np.asarray(self.mean_err_deg, dtype=np.float64).reshape(-1))
        n = self.success.shape[0]
        if self.final_err_deg.shape[0] != n or self.mean_err_deg.shape[0] != n:
            raise ValueError("per-task arrays must have the same length")

    @property
    def n_tasks(self) -> int:
        return int(self.success.shape[0])

    @property
    def success_rate(self) -> float:
        return float(self.success.mean()) if self.n_tasks else float("nan")

    @property
    def success_ci(self) -> tuple[float, float]:
        return bootstrap_ci(self.success.astype(np.float64))

    @property
    def final_err(self) -> float:
        return float(self.final_err_deg.mean()) if self.n_tasks else float("nan")

    @property
    def final_err_ci(self) -> tuple[float, float]:
        return bootstrap_ci(self.final_err_deg)

    @property
    def mean_err(self) -> float:
        return float(self.mean_err_deg.mean()) if self.n_tasks else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tasks": self.n_tasks,
            "tol_deg": float(self.tol_deg),
            "k_frames": int(self.k_frames),
            "success_rate": self.success_rate,
            "success_ci": list(self.success_ci),
            "final_err_deg": self.final_err,
            "final_err_ci": list(self.final_err_ci),
            "mean_err_deg": self.mean_err,
            "per_task": {
                "success": [bool(s) for s in self.success],
                "final_err_deg": self.final_err_deg.tolist(),
                "mean_err_deg": self.mean_err_deg.tolist(),
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalResult:
        p = d["per_task"]
        return cls(
            success=np.asarray(p["success"], dtype=bool),
            final_err_deg=np.asarray(p["final_err_deg"], dtype=np.float64),
            mean_err_deg=np.asarray(p["mean_err_deg"], dtype=np.float64),
            tol_deg=float(d["tol_deg"]),
            k_frames=int(d["k_frames"]),
        )


def evaluate(
    policy: Policy,
    sim: Simulator,
    tasks: Sequence[Task],
    horizon: int = 60,
    tol_deg: float = 2.0,
    k_frames: int = 5,
    params_per_task: Sequence[SimParams] | None = None,
) -> EvalResult:
    """Run ``policy`` on every task in ``sim``; ``params_per_task`` (optional)
    switches the sim's parameters before each task, for evaluating a policy
    in the randomized sim it was trained in."""
    if params_per_task is not None and len(params_per_task) != len(tasks):
        raise ValueError("params_per_task must have one entry per task")
    succ, fin, mean = [], [], []
    for i, task in enumerate(tasks):
        if params_per_task is not None:
            sim.set_params(params_per_task[i])
        q = rollout(sim, policy, task, horizon)
        s, f, m = episode_metrics(q, task, sim.deg_per_unit, tol_deg, k_frames)
        succ.append(s)
        fin.append(f)
        mean.append(m)
    return EvalResult(np.array(succ, dtype=bool), np.array(fin), np.array(mean), tol_deg, k_frames)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainResult:
    policy: Policy
    history: list[tuple[int, float]]  # (evaluations so far, best cost this generation)
    n_evaluations: int
    elapsed_s: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "history": [[int(i), float(c)] for i, c in self.history],
            "n_evaluations": int(self.n_evaluations),
            "elapsed_s": float(self.elapsed_s),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainResult:
        return cls(
            policy=Policy.from_dict(d["policy"]),
            history=[(int(i), float(c)) for i, c in d.get("history", [])],
            n_evaluations=int(d["n_evaluations"]),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            seed=int(d.get("seed", 0)),
        )


def episode_cost(
    sim: Simulator, policy: Policy, tasks: Sequence[Task], horizon: int, tol_deg: float, k_frames: int, bonus_deg: float
) -> float:
    """Training objective: mean error + final error (degrees), minus
    ``bonus_deg`` times the success rate, averaged over ``tasks``."""
    total = 0.0
    for task in tasks:
        q = rollout(sim, policy, task, horizon)
        s, f, m = episode_metrics(q, task, sim.deg_per_unit, tol_deg, k_frames)
        total += m + f - (bonus_deg if s else 0.0)
    return total / len(tasks)


def train_policy(
    sim: Simulator,
    tasks: Sequence[Task],
    horizon: int = 60,
    seed: int = 0,
    generations: int = 100,
    popsize: int | None = None,
    sigma0: float = 0.1,
    hidden: int = 6,
    tol_deg: float = 2.0,
    k_frames: int = 5,
    bonus_deg: float = 2.0,
    dr: Any | None = None,
    nominal: SimParams | None = None,
    on_generation: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """Fit a :class:`Policy` to ``tasks`` in ``sim`` with separable CMA-ES.

    The covariance is kept diagonal (``CMA_diagonal``): with around a
    hundred parameters and a few thousand evaluations, per-coordinate step
    sizes adapt in time where a full covariance would not, and the policy's
    parameters are already in meaningful, roughly independent units.

    ``dr`` (an :class:`antares.dr.DRRanges`) randomizes the sim: each
    generation draws one parameter set per task, applied before that task
    for every candidate of the generation (common random numbers). With
    ``dr`` the returned policy is the final CMA-ES mean; without it the best
    candidate ever evaluated. ``nominal`` is written back into the sim at
    the end so the caller finds it where it left it. Without ``dr`` the
    untrained policy is scored first and counts as one evaluation, so the
    result is never worse than ``u = t`` on the training tasks.
    """
    import cma

    from antares.dr import sample as dr_sample

    tasks = list(tasks)
    if not tasks:
        raise ValueError("train_policy needs at least one task")
    if generations < 1:
        raise ValueError("generations must be >= 1")
    n = Policy.n_params(hidden)
    opts: dict[str, Any] = {
        "seed": int(np.random.default_rng(seed).integers(1, 2**31 - 1)),
        "verbose": -9,
        "verb_log": 0,
        "verb_disp": 0,
        "CMA_diagonal": True,
    }
    if popsize is not None:
        opts["popsize"] = int(popsize)
    es = cma.CMAEvolutionStrategy(np.zeros(n), sigma0, opts)
    dr_rng = np.random.default_rng(seed + 7919)
    t0 = time.perf_counter()
    history: list[tuple[int, float]] = []
    best_x = np.zeros(n)
    # CMA-ES never scores its own start point; score the untrained policy
    # once so the returned policy is never worse than it on the training tasks.
    if dr is None:
        best_cost = episode_cost(sim, Policy(hidden, best_x), tasks, horizon, tol_deg, k_frames, bonus_deg)
        n_evals = 1
    else:
        best_cost = math.inf
        n_evals = 0
    for g in range(generations):
        X = es.ask()
        params_per_task = dr_sample(dr, dr_rng, len(tasks)) if dr is not None else None
        costs = []
        for x in X:
            policy = Policy(hidden, x)
            if params_per_task is None:
                c = episode_cost(sim, policy, tasks, horizon, tol_deg, k_frames, bonus_deg)
            else:
                total = 0.0
                for task, p in zip(tasks, params_per_task):
                    sim.set_params(p)
                    q = rollout(sim, policy, task, horizon)
                    s, f, m = episode_metrics(q, task, sim.deg_per_unit, tol_deg, k_frames)
                    total += m + f - (bonus_deg if s else 0.0)
                c = total / len(tasks)
            costs.append(float(c))
            n_evals += 1
        es.tell(X, costs)
        gen_best = min(costs)
        if gen_best < best_cost:
            best_cost = gen_best
            best_x = np.array(X[int(np.argmin(costs))], dtype=np.float64)
        history.append((n_evals, float(gen_best)))
        if on_generation is not None:
            on_generation(g, float(gen_best))
    theta = np.asarray(es.mean, dtype=np.float64) if dr is not None else best_x
    if nominal is not None:
        sim.set_params(nominal)
    return TrainResult(Policy(hidden, theta), history, n_evals, time.perf_counter() - t0, int(seed))


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferConfig:
    """Everything a run depends on, so the report can say what it ran.

    ``arms`` maps a hidden-arm name to its true :class:`SimParams` as a
    dict. The fit never sees them; see :func:`arms_from_locks`.
    """

    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    variants: tuple[str, ...] = VARIANTS
    seeds: tuple[int, ...] = (0, 1, 2)
    # task
    horizon: int = 60
    fps: float = 30.0
    tol_deg: float = 2.0
    k_frames: int = 5
    task_range: float = 70.0
    n_train_tasks: int = 24
    n_eval_tasks: int = 64
    eval_seed: int = 2024
    # policy / CMA-ES
    hidden: int = 6
    generations: int = 150
    popsize: int | None = None
    sigma0: float = 0.1
    bonus_deg: float = 2.0
    # stand-in calibration (what `antares calibrate --seeds` and `antares dr` would do)
    n_episodes: int = 50
    n_frames: int = 300
    noise_units: float = 0.3
    n_train: int = 20
    n_eval: int = 10
    max_evaluations: int = 150
    n_seeds: int = 3
    dr_n_fits: int = 8
    dr_n_train: int = 12
    dr_max_evaluations: int = 60
    pm_frac: float = 0.3
    # execution
    workers: int | None = None
    quick: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.variants) - set(VARIANTS)
        if unknown:
            raise ValueError(f"unknown variants {sorted(unknown)}; expected a subset of {list(VARIANTS)}")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        object.__setattr__(self, "variants", tuple(self.variants))
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        object.__setattr__(self, "arms", {str(k): dict(v) for k, v in self.arms.items()})

    @classmethod
    def quick_config(cls, arms: dict[str, dict[str, Any]], **overrides: Any) -> TransferConfig:
        """Tiny budgets: finishes in well under a minute; what the tests run."""
        base = dict(
            arms=arms,
            seeds=(0,),
            horizon=30,
            n_train_tasks=4,
            n_eval_tasks=8,
            hidden=4,
            generations=3,
            popsize=6,
            n_episodes=4,
            n_frames=60,
            n_train=2,
            n_eval=2,
            max_evaluations=6,
            n_seeds=1,
            dr_n_fits=2,
            dr_n_train=2,
            dr_max_evaluations=4,
            workers=1,
            quick=True,
        )
        base.update(overrides)
        return cls(**base)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "arms": {k: dict(v) for k, v in self.arms.items()},
            "variants": list(self.variants),
            "seeds": list(self.seeds),
        }
        for name in (
            "horizon", "fps", "tol_deg", "k_frames", "task_range", "n_train_tasks", "n_eval_tasks", "eval_seed",
            "hidden", "generations", "popsize", "sigma0", "bonus_deg",
            "n_episodes", "n_frames", "noise_units", "n_train", "n_eval", "max_evaluations", "n_seeds",
            "dr_n_fits", "dr_n_train", "dr_max_evaluations", "pm_frac", "workers", "quick",
        ):  # fmt: skip
            d[name] = getattr(self, name)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TransferConfig:
        kwargs = dict(d)
        kwargs["variants"] = tuple(kwargs.get("variants", VARIANTS))
        kwargs["seeds"] = tuple(kwargs.get("seeds", (0, 1, 2)))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in kwargs.items() if k in known})


def arms_from_locks(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    """``{arm name: lock file}`` -> ``{arm name: SimParams dict}`` (the lock's params)."""
    out = {}
    for name, path in paths.items():
        lock = json.loads(Path(path).read_text())
        out[name] = SimParams.from_dict(lock["params"]).to_dict()
    return out


DEFAULT_ARM_LOCKS: dict[str, str] = {
    "so101": "locks/svla_so101_pickplace.lock.json",
    "so100": "locks/svla_so100_pickplace.lock.json",
}


@dataclass(frozen=True)
class HiddenArmFit:
    """What Antares learned about a hidden arm from its synthesized logs."""

    name: str
    truth: SimParams
    fitted: SimParams
    fit_eval_score: float  # held-out tracking RMSE of the fitted lock, deg
    default_eval_score: float  # same for SimParams()
    truth_eval_score: float  # same for the true parameters (sensor noise floor)
    dr: dict[str, Any]  # DRRanges.to_dict()
    pm: dict[str, Any]  # uniform_ranges(SimParams(), pm_frac).to_dict()
    seed_params: list[dict[str, Any]] = field(default_factory=list)  # every CMA-ES seed's params
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "truth": self.truth.to_dict(),
            "fitted": self.fitted.to_dict(),
            "fit_eval_score": _json_float(self.fit_eval_score),
            "default_eval_score": _json_float(self.default_eval_score),
            "truth_eval_score": _json_float(self.truth_eval_score),
            "dr": self.dr,
            "pm": self.pm,
            "seed_params": list(self.seed_params),
            "elapsed_s": float(self.elapsed_s),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HiddenArmFit:
        return cls(
            name=str(d["name"]),
            truth=SimParams.from_dict(d["truth"]),
            fitted=SimParams.from_dict(d["fitted"]),
            fit_eval_score=_parse_float(d["fit_eval_score"]),
            default_eval_score=_parse_float(d["default_eval_score"]),
            truth_eval_score=_parse_float(d["truth_eval_score"]),
            dr=dict(d["dr"]),
            pm=dict(d["pm"]),
            seed_params=list(d.get("seed_params", [])),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
        )


def fit_hidden_arm(name: str, truth: SimParams, cfg: TransferConfig, seed: int = 1) -> HiddenArmFit:
    """Synthesize teleop-style logs from the hidden arm and fit a lock and DR
    ranges to them, exactly as ``antares calibrate --seeds`` and ``antares
    dr`` would; the fit starts from the MJCF defaults and never sees ``truth``."""
    from antares.calibrate import calibrate, default_simulate, score, split_episodes
    from antares.dr import DRRanges, bootstrap_fits, uniform_ranges
    from antares.synth import synthesize

    t0 = time.perf_counter()
    ds = synthesize(
        truth,
        n_episodes=cfg.n_episodes,
        n_frames=cfg.n_frames,
        fps=cfg.fps,
        seed=seed,
        noise_units=cfg.noise_units,
        repo_id=f"synthetic/{name}",
    )
    train, eval_ = split_episodes(ds.episodes, cfg.n_train, cfg.n_eval, seed=0)
    simulate = default_simulate()
    res = calibrate(
        train, eval_, simulate=simulate, max_evaluations=cfg.max_evaluations, seed=0, n_seeds=cfg.n_seeds
    )
    fits = bootstrap_fits(
        train,
        res.params,
        n_fits=cfg.dr_n_fits,
        n_train=cfg.dr_n_train,
        max_evaluations=cfg.dr_max_evaluations,
        seed=0,
        simulate=simulate,
    )
    dr = DRRanges.from_fits(fits, res.params, dataset=ds.repo_id, lock=f"{name}-fit")
    pm = uniform_ranges(SimParams(), cfg.pm_frac)
    truth_score = score(truth, eval_, simulate).overall_deg if eval_ else float("nan")
    seed_params = [dict(s["params"]) for s in res.stages if s.get("name") == "shared"]
    return HiddenArmFit(
        name=name,
        truth=truth,
        fitted=res.params,
        fit_eval_score=float(res.eval_score),
        default_eval_score=float(res.baseline_eval_score),
        truth_eval_score=float(truth_score),
        dr=dr.to_dict(),
        pm=pm.to_dict(),
        seed_params=seed_params,
        elapsed_s=time.perf_counter() - t0,
    )


def variant_sim(variant: str, fit: HiddenArmFit) -> tuple[SimParams, dict[str, Any] | None]:
    """``(nominal params, DR ranges dict or None)`` the variant trains in."""
    if variant == "default":
        return SimParams(), None
    if variant == "lock":
        return fit.fitted, None
    if variant == "lock_dr":
        return fit.fitted, fit.dr
    if variant == "oracle":
        return fit.truth, None
    if variant == "default_pm30":
        return SimParams(), fit.pm
    raise ValueError(f"unknown variant {variant!r}")


@dataclass(frozen=True)
class SeedResult:
    """One trained policy: scored in its training sim and on the hidden arm."""

    seed: int
    train: TrainResult
    in_train_sim: EvalResult
    on_hidden: EvalResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "train": self.train.to_dict(),
            "in_train_sim": self.in_train_sim.to_dict(),
            "on_hidden": self.on_hidden.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeedResult:
        return cls(
            seed=int(d["seed"]),
            train=TrainResult.from_dict(d["train"]),
            in_train_sim=EvalResult.from_dict(d["in_train_sim"]),
            on_hidden=EvalResult.from_dict(d["on_hidden"]),
        )


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


@dataclass(frozen=True)
class VariantResult:
    name: str
    nominal: SimParams
    randomized: bool
    seeds: list[SeedResult]

    def _stat(self, which: str, attr: str) -> tuple[float, float]:
        return _mean_sd([getattr(getattr(s, which), attr) for s in self.seeds])

    @property
    def train_success(self) -> tuple[float, float]:
        return self._stat("in_train_sim", "success_rate")

    @property
    def hidden_success(self) -> tuple[float, float]:
        return self._stat("on_hidden", "success_rate")

    @property
    def train_final_err(self) -> tuple[float, float]:
        return self._stat("in_train_sim", "final_err")

    @property
    def hidden_final_err(self) -> tuple[float, float]:
        return self._stat("on_hidden", "final_err")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": VARIANT_LABELS.get(self.name, self.name),
            "nominal": self.nominal.to_dict(),
            "randomized": bool(self.randomized),
            "summary": {
                "train_success": list(self.train_success),
                "hidden_success": list(self.hidden_success),
                "train_final_err_deg": list(self.train_final_err),
                "hidden_final_err_deg": list(self.hidden_final_err),
            },
            "seeds": [s.to_dict() for s in self.seeds],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VariantResult:
        return cls(
            name=str(d["name"]),
            nominal=SimParams.from_dict(d["nominal"]),
            randomized=bool(d.get("randomized", False)),
            seeds=[SeedResult.from_dict(s) for s in d.get("seeds", [])],
        )


@dataclass(frozen=True)
class ArmResult:
    fit: HiddenArmFit
    variants: dict[str, VariantResult]
    untrained: EvalResult | None = None  # ``u = t`` on the hidden arm, for reference

    @property
    def name(self) -> str:
        return self.fit.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit": self.fit.to_dict(),
            "variants": {k: v.to_dict() for k, v in self.variants.items()},
            "untrained": None if self.untrained is None else self.untrained.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArmResult:
        u = d.get("untrained")
        return cls(
            fit=HiddenArmFit.from_dict(d["fit"]),
            variants={k: VariantResult.from_dict(v) for k, v in d.get("variants", {}).items()},
            untrained=None if u is None else EvalResult.from_dict(u),
        )


@dataclass(frozen=True)
class TransferReport:
    config: TransferConfig
    arms: dict[str, ArmResult]
    elapsed_s: float = 0.0
    created_at: str = field(default_factory=_now_iso)
    antares_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "arms": {k: v.to_dict() for k, v in self.arms.items()},
            "elapsed_s": float(self.elapsed_s),
            "created_at": self.created_at,
            "antares_version": self.antares_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TransferReport:
        return cls(
            config=TransferConfig.from_dict(d["config"]),
            arms={k: ArmResult.from_dict(v) for k, v in d.get("arms", {}).items()},
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            created_at=str(d.get("created_at", "")),
            antares_version=str(d.get("antares_version", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> TransferReport:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path, markdown: str | Path | None = None) -> Path:
        """Write the JSON report to ``path`` and the markdown next to it
        (``markdown``, default: same stem with ``.md``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        md = Path(markdown) if markdown is not None else path.with_suffix(".md")
        md.write_text(self.to_markdown())
        return path

    # -- markdown ---------------------------------------------------------

    def headline_rows(self) -> list[dict[str, Any]]:
        """One row per (variant, arm) for the summary table."""
        rows = []
        for variant in self.config.variants:
            for arm_name, arm in self.arms.items():
                v = arm.variants.get(variant)
                if v is None:
                    continue
                rows.append(
                    {
                        "variant": variant,
                        "label": VARIANT_LABELS.get(variant, variant),
                        "arm": arm_name,
                        "train_success": v.train_success,
                        "hidden_success": v.hidden_success,
                        "train_final_err": v.train_final_err,
                        "hidden_final_err": v.hidden_final_err,
                        "n_seeds": len(v.seeds),
                    }
                )
        return rows

    def to_markdown(self) -> str:
        cfg = self.config
        lines = ["## Does calibration help a policy transfer? (sim-to-sim)", ""]
        lines.append(
            f"Joint-space reaching on a hidden MuJoCo arm. Success = all five body joints within "
            f"{cfg.tol_deg:g} deg of the target for the last {cfg.k_frames} of {cfg.horizon} frames at "
            f"{cfg.fps:g} fps; final error = mean absolute joint error over those frames. "
            f"{len(cfg.seeds)} training seed(s) per variant ({', '.join(map(str, cfg.seeds))}), "
            f"{cfg.n_train_tasks} training tasks, {cfg.n_eval_tasks} held-out tasks (seed {cfg.eval_seed}), "
            f"{cfg.generations} CMA-ES generations, hidden size {cfg.hidden} "
            f"({Policy.n_params(cfg.hidden)} policy parameters). Values are mean +- sd over seeds."
        )
        lines.append("")
        lines.append("| variant | hidden arm | success in train sim | success on hidden arm | final error on hidden arm (deg) |")
        lines.append("|---|---|---|---|---|")
        for r in self.headline_rows():
            lines.append(
                f"| {r['label']} | {r['arm']} | {_pct(r['train_success'])} | {_pct(r['hidden_success'])} | "
                f"{_deg(r['hidden_final_err'])} |"
            )
        lines.append("")
        for arm_name, arm in self.arms.items():
            fit = arm.fit
            lines.append(f"### Hidden arm `{arm_name}`")
            lines.append("")
            lines.append(
                f"Lock fitted to {cfg.n_train} synthesized episodes ({cfg.n_frames} frames, noise "
                f"{cfg.noise_units:g} units), best of {cfg.n_seeds} CMA-ES seed(s) x {cfg.max_evaluations} evaluations; "
                f"held-out tracking RMSE: default {_f(fit.default_eval_score)} deg, fitted {_f(fit.fit_eval_score)} deg, "
                f"true parameters {_f(fit.truth_eval_score)} deg. DR ranges from {cfg.dr_n_fits} refits x "
                f"{cfg.dr_n_train} episodes x {cfg.dr_max_evaluations} evaluations."
            )
            lines.append("")
            names = [n for n in SimParams.field_names() if n != "forcerange"]
            lines.append("| | " + " | ".join(names) + " |")
            lines.append("|---|" + "---|" * len(names))
            lines.append("| truth | " + " | ".join(_param(n, getattr(fit.truth, n)) for n in names) + " |")
            lines.append("| fitted | " + " | ".join(_param(n, getattr(fit.fitted, n)) for n in names) + " |")
            dr = fit.dr.get("ranges", {})
            if dr:
                lines.append(
                    "| DR lo | " + " | ".join(_param(n, dr[n]["lo"]) if n in dr else "" for n in names) + " |"
                )
                lines.append(
                    "| DR hi | " + " | ".join(_param(n, dr[n]["hi"]) if n in dr else "" for n in names) + " |"
                )
            lines.append("")
            if arm.untrained is not None:
                u = arm.untrained
                lines.append(
                    f"Untrained policy (`u = target`) on the hidden arm: success {100 * u.success_rate:.0f}% "
                    f"[{100 * u.success_ci[0]:.0f}, {100 * u.success_ci[1]:.0f}], "
                    f"final error {u.final_err:.2f} deg."
                )
                lines.append("")
            lines.append("| variant | seed | success in train sim [95% CI] | success on hidden arm [95% CI] | final error on hidden arm (deg) [95% CI] | CMA-ES evaluations | train time (s) |")
            lines.append("|---|---|---|---|---|---|---|")
            for variant in cfg.variants:
                v = arm.variants.get(variant)
                if v is None:
                    continue
                for s in v.seeds:
                    a, b = s.in_train_sim, s.on_hidden
                    lines.append(
                        f"| {VARIANT_LABELS.get(variant, variant)} | {s.seed} | {_pct_ci(a)} | {_pct_ci(b)} | "
                        f"{_deg_ci(b)} | {s.train.n_evaluations} | {s.train.elapsed_s:.0f} |"
                    )
            lines.append("")
        lines.append(
            f"Run: antares {self.antares_version}, {self.created_at}, {self.elapsed_s / 60:.1f} min wall clock"
            f"{' (quick budgets)' if cfg.quick else ''}."
        )
        lines.append("")
        return "\n".join(lines)


def _f(v: float) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{v:.2f}"


def _pct(ms: tuple[float, float]) -> str:
    m, s = ms
    if not math.isfinite(m):
        return "n/a"
    return f"{100 * m:.0f}% +- {100 * s:.0f}"


def _deg(ms: tuple[float, float]) -> str:
    m, s = ms
    if not math.isfinite(m):
        return "n/a"
    return f"{m:.2f} +- {s:.2f}"


def _pct_ci(e: EvalResult) -> str:
    lo, hi = e.success_ci
    return f"{100 * e.success_rate:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def _deg_ci(e: EvalResult) -> str:
    lo, hi = e.final_err_ci
    return f"{e.final_err:.2f} [{lo:.2f}, {hi:.2f}]"


def _param(name: str, v: Any) -> str:
    if name == "latency_frames":
        return str(int(round(float(v))))
    return f"{float(v):.4g}"


def _json_float(v: float) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def _parse_float(v: Any) -> float:
    return float("nan") if v is None else float(v)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _train_job(job: dict[str, Any]) -> dict[str, Any]:
    """Train one (arm, variant, seed) and score it in its sim and on the hidden
    arm. A plain-dict interface so it crosses a process boundary."""
    from antares.dr import DRRanges
    from antares.dr import sample as dr_sample

    cfg = TransferConfig.from_dict(job["config"])
    truth = SimParams.from_dict(job["truth"])
    nominal = SimParams.from_dict(job["nominal"])
    dr = DRRanges.from_dict(job["dr"]) if job.get("dr") is not None else None
    seed = int(job["seed"])

    sim = MuJoCoSim(nominal, fps=cfg.fps)
    train_tasks = sample_tasks(cfg.n_train_tasks, seed=1000 + seed, task_range=cfg.task_range)
    eval_tasks = sample_tasks(cfg.n_eval_tasks, seed=cfg.eval_seed, task_range=cfg.task_range)
    tr = train_policy(
        sim,
        train_tasks,
        horizon=cfg.horizon,
        seed=seed,
        generations=cfg.generations,
        popsize=cfg.popsize,
        sigma0=cfg.sigma0,
        hidden=cfg.hidden,
        tol_deg=cfg.tol_deg,
        k_frames=cfg.k_frames,
        bonus_deg=cfg.bonus_deg,
        dr=dr,
        nominal=nominal,
    )
    params_per_task = dr_sample(dr, np.random.default_rng(cfg.eval_seed), len(eval_tasks)) if dr is not None else None
    in_train = evaluate(tr.policy, sim, eval_tasks, cfg.horizon, cfg.tol_deg, cfg.k_frames, params_per_task)
    sim.set_params(truth)
    on_hidden = evaluate(tr.policy, sim, eval_tasks, cfg.horizon, cfg.tol_deg, cfg.k_frames)
    return {
        "arm": job["arm"],
        "variant": job["variant"],
        "result": SeedResult(seed, tr, in_train, on_hidden).to_dict(),
    }


def _fit_job(job: dict[str, Any]) -> dict[str, Any]:
    cfg = TransferConfig.from_dict(job["config"])
    fit = fit_hidden_arm(job["arm"], SimParams.from_dict(job["truth"]), cfg, seed=int(job["seed"]))
    return {"arm": job["arm"], "fit": fit.to_dict()}


def _run_jobs(fn: Callable[[dict], dict], jobs: list[dict], workers: int, log: Callable[[str], None]) -> list[dict]:
    if workers <= 1 or len(jobs) <= 1:
        out = []
        for i, job in enumerate(jobs):
            out.append(fn(job))
            log(f"  [{i + 1}/{len(jobs)}] {_job_name(job)}")
        return out
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ctx = multiprocessing.get_context("spawn")
    out = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as pool:
        futures = {pool.submit(fn, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures)):
            out.append(fut.result())
            log(f"  [{i + 1}/{len(jobs)}] {_job_name(futures[fut])}")
    return out


def _job_name(job: dict) -> str:
    parts = [str(job.get("arm", ""))]
    if "variant" in job:
        parts.append(str(job["variant"]))
    if "seed" in job and "variant" in job:
        parts.append(f"seed {job['seed']}")
    return " / ".join(parts)


def run_experiment(cfg: TransferConfig, log: Callable[[str], None] | None = None) -> TransferReport:
    """Fit each hidden arm, train every variant x seed, score everything.

    Training jobs run in ``cfg.workers`` processes (default: all cores);
    ``workers=1`` runs inline. Returns the :class:`TransferReport`.
    """
    log = log or (lambda s: None)
    if not cfg.arms:
        raise ValueError("config needs at least one hidden arm")
    t0 = time.perf_counter()
    workers = cfg.workers if cfg.workers is not None else (os.cpu_count() or 1)
    cfg_dict = cfg.to_dict()

    log(f"fitting {len(cfg.arms)} hidden arm(s)")
    fit_jobs = [
        {"arm": name, "truth": truth, "config": cfg_dict, "seed": i + 1}
        for i, (name, truth) in enumerate(cfg.arms.items())
    ]
    fits = {r["arm"]: HiddenArmFit.from_dict(r["fit"]) for r in _run_jobs(_fit_job, fit_jobs, workers, log)}

    train_jobs = []
    for name in cfg.arms:
        fit = fits[name]
        for variant in cfg.variants:
            nominal, dr = variant_sim(variant, fit)
            for seed in cfg.seeds:
                train_jobs.append(
                    {
                        "arm": name,
                        "variant": variant,
                        "seed": seed,
                        "truth": fit.truth.to_dict(),
                        "nominal": nominal.to_dict(),
                        "dr": dr,
                        "config": cfg_dict,
                    }
                )
    log(f"training {len(train_jobs)} policies on {workers} worker(s)")
    results = _run_jobs(_train_job, train_jobs, workers, log)

    eval_tasks = sample_tasks(cfg.n_eval_tasks, seed=cfg.eval_seed, task_range=cfg.task_range)
    sim = MuJoCoSim(fps=cfg.fps)
    arms: dict[str, ArmResult] = {}
    for name in cfg.arms:
        fit = fits[name]
        variants: dict[str, VariantResult] = {}
        for variant in cfg.variants:
            nominal, dr = variant_sim(variant, fit)
            seeds = sorted(
                (SeedResult.from_dict(r["result"]) for r in results if r["arm"] == name and r["variant"] == variant),
                key=lambda s: s.seed,
            )
            variants[variant] = VariantResult(variant, nominal, dr is not None, seeds)
        sim.set_params(fit.truth)
        untrained = evaluate(Policy.zeros(cfg.hidden), sim, eval_tasks, cfg.horizon, cfg.tol_deg, cfg.k_frames)
        arms[name] = ArmResult(fit=fit, variants=variants, untrained=untrained)
    return TransferReport(config=cfg, arms=arms, elapsed_s=time.perf_counter() - t0)


__all__ = [
    "BODY_JOINTS",
    "DEFAULT_ARM_LOCKS",
    "N_BODY",
    "VARIANTS",
    "VARIANT_LABELS",
    "ArmResult",
    "EvalResult",
    "HiddenArmFit",
    "MuJoCoSim",
    "Policy",
    "SeedResult",
    "Simulator",
    "Task",
    "ToyArm",
    "TrainResult",
    "TransferConfig",
    "TransferReport",
    "VariantResult",
    "arms_from_locks",
    "bootstrap_ci",
    "episode_cost",
    "episode_metrics",
    "evaluate",
    "fit_hidden_arm",
    "rollout",
    "run_experiment",
    "sample_tasks",
    "train_policy",
    "variant_sim",
]
