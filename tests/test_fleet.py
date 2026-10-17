"""Tests for antares.fleet. The Hub is never touched: search, info and size
lookups are injected."""

from __future__ import annotations

import pytest

from antares import fleet
from antares.fleet import Candidate, check_info, discover

SO101_NAMES = [f"{j}.pos" for j in fleet.JOINTS]
SO100_NAMES = [f"main_{j}" for j in fleet.JOINTS]


def info(**over):
    base = {
        "codebase_version": "v3.0",
        "robot_type": "so101_follower",
        "fps": 30,
        "total_episodes": 50,
        "total_frames": 12000,
        "features": {
            "action": {"names": list(SO101_NAMES)},
            "observation.state": {"names": list(SO101_NAMES)},
        },
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# check_info
# --------------------------------------------------------------------------


def test_check_info_accepts_so101_and_so100_naming():
    c = check_info("a/b", info(), data_bytes=10_000, downloads=7)
    assert c.accepted and c.reason == ""
    assert (c.robot_type, c.fps, c.n_episodes, c.n_frames) == ("so101_follower", 30.0, 50, 12000)
    assert c.data_bytes == 10_000 and c.downloads == 7 and c.owner == "a"
    old = info(
        codebase_version="v2.1",
        robot_type="so100",
        features={"action": {"names": SO100_NAMES}, "observation.state": {"names": SO100_NAMES}},
    )
    assert check_info("a/b", old, 1).accepted
    nested = info(features={"action": {"names": {"motors": SO100_NAMES}}, "observation.state": {"names": {"motors": SO100_NAMES}}})
    assert check_info("a/b", nested, 1).accepted
    for rt in ("so-arm101", "SO_ARM100", "so100_follower", "so-101"):
        assert check_info("a/b", info(robot_type=rt), 1).accepted, rt


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"robot_type": "koch"}, "robot_type"),
        ({"robot_type": ""}, "robot_type"),
        ({"codebase_version": "v1.6"}, "codebase_version"),
        ({"fps": 60}, "fps 60"),
        ({"total_episodes": 14}, "14 episodes"),
        ({"features": {"action": {"names": SO101_NAMES[:5]}, "observation.state": {"names": SO101_NAMES}}}, "5 joints"),
        ({"features": {"action": {"names": SO101_NAMES}, "observation.state": {"names": list(reversed(SO101_NAMES))}}}, "not the SO-101 joints"),
        ({"features": {"action": {"shape": [6]}, "observation.state": {"names": SO101_NAMES}}}, "no joint names"),
        ({"features": {}}, "no joint names"),
    ],
)
def test_check_info_rejects_with_reason(over, needle):
    c = check_info("a/b", info(**over), data_bytes=1)
    assert not c.accepted
    assert needle in c.reason


def test_check_info_size_rules():
    assert "no data" in check_info("a/b", info(), data_bytes=0).reason
    big = check_info("a/b", info(), data_bytes=fleet.MAX_DATA_BYTES + 1)
    assert "MB" in big.reason and not big.accepted
    assert check_info("a/b", info(), data_bytes=None).accepted  # unknown size: decided later


def test_candidate_round_trip():
    c = check_info("a/b", info(), 5, 3)
    back = Candidate.from_dict({**c.to_dict(), "extra": 1})
    assert back == c


# --------------------------------------------------------------------------
# discover
# --------------------------------------------------------------------------


def fake_hub(infos: dict, sizes: dict | None = None):
    sizes = sizes or {}
    hits = [(rid, 1000 - i) for i, rid in enumerate(infos)]

    def search():
        return hits

    def get_info(rid):
        v = infos[rid]
        if isinstance(v, Exception):
            raise v
        return v

    def get_size(rid):
        v = sizes.get(rid, 1_000_000)
        if isinstance(v, Exception):
            raise v
        return v

    return search, get_info, get_size


def test_discover_filters_and_records_reasons():
    infos = {
        "alice/so101_a": info(),
        "bob/koch_b": info(robot_type="koch"),
        "carol/so100_c": info(robot_type="so100"),
        "dave/nope": FileNotFoundError("meta/info.json"),
        "erin/huge": info(),
        "alice/so101_again": info(),
        "frank/so101_f": info(),
    }
    sizes = {"erin/huge": fleet.MAX_DATA_BYTES * 3}
    seen = []
    cands = discover(limit=10, pinned=(), search=fake_hub(infos, sizes)[0], fetch_info=fake_hub(infos, sizes)[1],
                     fetch_size=fake_hub(infos, sizes)[2], on_progress=seen.append)
    assert [c.repo_id for c in cands] == list(infos)
    assert seen == cands
    ok = [c.repo_id for c in fleet.accepted(cands)]
    assert ok == ["alice/so101_a", "carol/so100_c", "frank/so101_f"]
    by = {c.repo_id: c for c in cands}
    assert "robot_type" in by["bob/koch_b"].reason
    assert "FileNotFoundError" in by["dave/nope"].reason
    assert "MB" in by["erin/huge"].reason
    assert "same owner as alice/so101_a" in by["alice/so101_again"].reason
    assert by["alice/so101_a"].downloads == 1000 and by["alice/so101_a"].data_bytes == 1_000_000


def test_discover_stops_at_limit_and_max_inspect():
    infos = {f"u{i}/so101_{i}": info() for i in range(10)}
    search, get_info, get_size = fake_hub(infos)
    cands = discover(limit=3, pinned=(), search=search, fetch_info=get_info, fetch_size=get_size)
    assert len(fleet.accepted(cands)) == 3 and len(cands) == 3
    cands = discover(limit=10, max_inspect=4, pinned=(), search=search, fetch_info=get_info, fetch_size=get_size)
    assert len(cands) == 4
    with pytest.raises(ValueError):
        discover(limit=0, pinned=(), search=search, fetch_info=get_info, fetch_size=get_size)


def test_discover_size_lookup_failure_is_a_rejection():
    infos = {"a/x": info(), "b/y": info()}
    search, get_info, _ = fake_hub(infos)

    def get_size(rid):
        if rid == "a/x":
            raise ConnectionError("boom")
        return 10

    cands = discover(limit=5, pinned=(), search=search, fetch_info=get_info, fetch_size=get_size)
    assert [c.accepted for c in cands] == [False, True]
    assert "ConnectionError" in cands[0].reason


def test_discover_allows_several_per_owner_when_asked():
    infos = {"a/x": info(), "a/y": info()}
    search, get_info, get_size = fake_hub(infos)
    cands = discover(limit=5, one_per_owner=False, pinned=(), search=search, fetch_info=get_info, fetch_size=get_size)
    assert all(c.accepted for c in cands)


def test_hub_search_merges_sorts_and_skips_gated():
    class D:
        def __init__(self, id, downloads, gated=False):
            self.id, self.downloads, self.gated, self.private, self.disabled = id, downloads, gated, False, False

    class Api:
        calls = 0

        def list_datasets(self, sort, limit, search=None, filter=None):
            Api.calls += 1
            if filter == "so101":
                raise RuntimeError("hub hiccup")
            return [D("a/x", 5), D("b/y", 50, gated=True), D("c/z", 9)]

    hits = fleet.hub_search(Api(), search_terms=("so100",), tag_filters=("so100", "so101"))
    assert hits == [("c/z", 9), ("a/x", 5)]
    assert Api.calls == 3


def test_check_info_accepts_so_follower_and_rejects_simulated():
    assert check_info("a/b", info(robot_type="so_follower"), 1).accepted
    assert not check_info("a/b", info(robot_type="so_sensor_follower"), 1).accepted
    for rid in ("u/so100_pybullet_pickup", "u/pick-the-block-synthetic-512", "u/0916_isaacsim_so101", "u/so101_bench_sim_v2"):
        c = check_info(rid, info(), 1)
        assert not c.accepted and "simulation" in c.reason, rid
    assert check_info("u/so101_simple_task", info(), 1).accepted  # "simple" is not "sim"
    assert "simulation" in check_info("u/so101_newsim_place", info(), 1).reason
    for rid in ("u/community-v1-so101-converted", "u/Full_Merged_Final_SO101", "u/so100_combined"):
        c = check_info(rid, info(), 1)
        assert not c.accepted and "merge" in c.reason, rid


def test_discover_pinned_come_first_and_skip_owner_rule():
    infos = {"org/arm_b": info(), "x/so101": info(), "org/arm_a": info()}
    search, get_info, get_size = fake_hub(infos)
    cands = discover(limit=5, pinned=("org/arm_a", "org/arm_b"), search=search,
                     fetch_info=get_info, fetch_size=get_size)
    assert [c.repo_id for c in cands] == ["org/arm_a", "org/arm_b", "x/so101"]
    assert all(c.accepted for c in cands)
    assert cands[0].downloads == 998  # downloads still come from the Hub listing
    # a pinned dataset missing from the search is still inspected
    cands = discover(limit=5, pinned=("ghost/arm",), search=search, fetch_info=get_info, fetch_size=get_size)
    assert cands[0].repo_id == "ghost/arm" and not cands[0].accepted
