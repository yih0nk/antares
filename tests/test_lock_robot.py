"""A lock records which simulator model it was fit on; old locks mean SO-101."""

from __future__ import annotations

import numpy as np

from antares.calibrate import CalibrationResult
from antares.data import Episode
from antares.lock import RobotLock
from antares.sim import SimParams


def _result() -> CalibrationResult:
    return CalibrationResult.from_dict(
        {
            "params": SimParams(kp=100.0).to_dict(),
            "baseline": SimParams().to_dict(),
            "train_score": 1.0,
            "eval_score": 1.0,
            "baseline_train_score": 2.0,
            "baseline_eval_score": 2.0,
            "n_evaluations": 1,
            "elapsed_s": 0.1,
        }
    )


def test_lock_records_its_robot_model_and_defaults_to_so101():
    ep = Episode(index=0, fps=30.0, joint_names=("a",), state=np.zeros((2, 1)), action=np.zeros((2, 1)))
    lock = RobotLock.create("x/y", "so100", [ep], _result())
    assert lock.robot == "so101" and lock.to_dict()["robot"] == "so101"
    other = RobotLock.create("x/y", "so100", [ep], _result(), robot="so100")
    assert RobotLock.from_dict(other.to_dict()).robot == "so100"
    legacy = {k: v for k, v in lock.to_dict().items() if k != "robot"}
    assert RobotLock.from_dict(legacy).robot == "so101"
