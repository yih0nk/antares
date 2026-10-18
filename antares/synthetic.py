"""Synthetic arm datasets with known parameters.

The cloud dev environment cannot reach the Hugging Face Hub, and checks
like :mod:`antares.specificity` are easier to trust once they have been run
on arms whose true parameters are known. :func:`make_dataset` drives the
MuJoCo arm with smooth random commands under a given :class:`SimParams`
and records the response (plus sensor noise) in dataset units, so the
result loads anywhere a real :class:`antares.data.Dataset` does.
"""

from __future__ import annotations

import numpy as np

from antares.data import Dataset, Episode
from antares.sim import GRIPPER_INDEX, JOINTS, Arm, SimParams


def commands(T: int, fps: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth multi-sine joint targets in dataset units, (T, 6).

    Body joints stay inside +-60 of 100, the gripper inside 10..90, so the
    targets never touch the joint limits.
    """
    t = np.arange(T) / fps
    out = np.zeros((T, len(JOINTS)))
    for j in range(len(JOINTS)):
        for _ in range(3):
            f = rng.uniform(0.1, 1.2)
            out[:, j] += rng.uniform(5, 20) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        out[:, j] += rng.uniform(-15, 15)
    out[:, :] = np.clip(out, -60, 60)
    out[:, GRIPPER_INDEX] = 50 + 0.65 * out[:, GRIPPER_INDEX]
    return out


def make_dataset(
    repo_id: str,
    params: SimParams,
    n_episodes: int = 30,
    T: int = 300,
    fps: float = 30.0,
    noise_units: float = 0.3,
    seed: int = 0,
    arm: Arm | None = None,
) -> Dataset:
    """Episodes of ``params``'s simulated response to random commands.

    ``state`` is the simulated joint trajectory converted back to dataset
    units plus Gaussian noise of ``noise_units`` (about 0.3 deg for the body
    joints at the default 0.3).
    """
    rng = np.random.default_rng(seed)
    arm = arm or Arm()
    arm.apply(params)
    episodes = []
    for i in range(n_episodes):
        action = commands(T, fps, rng)
        seed_ep = Episode(index=i, fps=fps, joint_names=JOINTS, state=action, action=action)
        q = arm.replay(seed_ep)
        state = arm.unit_map.to_units(q) + rng.normal(0.0, noise_units, size=q.shape)
        state[0] = action[0]
        episodes.append(Episode(index=i, fps=fps, joint_names=JOINTS, state=state, action=action))
    return Dataset(
        repo_id=repo_id,
        robot_type="synthetic_so101",
        fps=fps,
        joint_names=JOINTS,
        episodes=tuple(episodes),
    )


__all__ = ["commands", "make_dataset"]
