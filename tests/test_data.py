import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from antares.data import canonical_joint_name, episodes_from_columns, read_local


def test_canonical_joint_name():
    assert canonical_joint_name("main_shoulder_pan") == "shoulder_pan"
    assert canonical_joint_name("shoulder_pan.pos") == "shoulder_pan"
    assert canonical_joint_name("gripper") == "gripper"


def test_episodes_split_and_sorted_by_frame():
    cols = {
        "episode_index": [1, 0, 0, 1, 0],
        "frame_index": [1, 2, 0, 0, 1],
        "observation.state": [[11.0], [2.0], [0.0], [10.0], [1.0]],
        "action": [[11.5], [2.5], [0.5], [10.5], [1.5]],
    }
    eps = episodes_from_columns(cols, fps=30.0, joint_names=("j",))
    assert [e.index for e in eps] == [0, 1]
    np.testing.assert_array_equal(eps[0].state[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(eps[1].action[:, 0], [10.5, 11.5])
    assert eps[0].duration == pytest.approx(0.1)


def test_read_local(tmp_path):
    names = ["shoulder_pan.pos", "gripper.pos"]
    info = {
        "robot_type": "so100_follower",
        "fps": 30,
        "features": {
            "action": {"names": names},
            "observation.state": {"names": names},
        },
    }
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    table = pa.table({
        "episode_index": [0, 0],
        "frame_index": [0, 1],
        "observation.state": [[0.0, 1.0], [2.0, 3.0]],
        "action": [[0.5, 1.5], [2.5, 3.5]],
    })
    pq.write_table(table, tmp_path / "data" / "chunk-000" / "file-000.parquet")

    ds = read_local(tmp_path)
    assert ds.joint_names == ("shoulder_pan", "gripper")
    assert ds.robot_type == "so100_follower"
    assert len(ds.episodes) == 1
    assert ds.episodes[0].state.shape == (2, 2)
