"""Tests for antares.fleet. The Hub is never touched: search, info and size
lookups are injected."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from antares import fleet
from antares.data import Dataset, Episode
from antares.fleet import Candidate, FleetReport, RobotEntry, check_info, discover, plan_split, scan

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
    many = check_info("a/b", info(), data_bytes=1, data_files=fleet.MAX_DATA_FILES + 1)
    assert "parquet files" in many.reason and not many.accepted
    assert check_info("a/b", info(), data_bytes=1, data_files=fleet.MAX_DATA_FILES).accepted


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
        return v if isinstance(v, tuple) else (v, 3)

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
    sizes = {"erin/huge": fleet.MAX_DATA_BYTES * 3, "frank/so101_f": (5_000_000, 1801)}
    seen = []
    cands = discover(limit=10, pinned=(), search=fake_hub(infos, sizes)[0], fetch_info=fake_hub(infos, sizes)[1],
                     fetch_size=fake_hub(infos, sizes)[2], on_progress=seen.append)
    assert [c.repo_id for c in cands] == list(infos)
    assert seen == cands
    ok = [c.repo_id for c in fleet.accepted(cands)]
    assert ok == ["alice/so101_a", "carol/so100_c"]
    by = {c.repo_id: c for c in cands}
    assert "robot_type" in by["bob/koch_b"].reason
    assert "FileNotFoundError" in by["dave/nope"].reason
    assert "MB" in by["erin/huge"].reason
    assert "1801 parquet files" in by["frank/so101_f"].reason
    assert "same owner as alice/so101_a" in by["alice/so101_again"].reason
    assert by["alice/so101_a"].downloads == 1000 and by["alice/so101_a"].data_bytes == 1_000_000
    assert by["alice/so101_a"].data_files == 3


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
        return 10, 1

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


# --------------------------------------------------------------------------
# scan (toy simulator, no MuJoCo, no network)
# --------------------------------------------------------------------------


def toy_response(p, action, fps):
    """First-order lag with a pure delay; dataset units treated as radians."""
    dt = 1.0 / fps
    tau = 20.0 * (p.kv + p.damping) / p.kp + p.armature
    alpha = dt / (tau + dt)
    out = np.empty_like(action)
    out[0] = action[0]
    for t in range(1, len(action)):
        target = action[max(t - p.latency_frames, 0)]
        out[t] = out[t - 1] + alpha * (target - out[t - 1])
    return out


def toy_simulate(params, ep):
    return ep.state, toy_response(params, ep.action, ep.fps)


def make_dataset(repo_id, n=12, T=40, fps=30.0, robot_type="so101_follower", seed=0, true=None):
    from antares.sim import SimParams

    true = true or SimParams(kp=300.0, kv=3.0, damping=1.0, armature=0.05, frictionloss=0.05, latency_frames=2)
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    eps = []
    for i in range(n):
        action = np.zeros((T, 6))
        for j in range(6):
            action[:, j] = rng.uniform(-0.5, 0.5) * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t + rng.uniform(0, 6))
        eps.append(Episode(index=i, fps=fps, joint_names=fleet.JOINTS, state=toy_response(true, action, fps), action=action))
    return Dataset(repo_id=repo_id, robot_type=robot_type, fps=fps, joint_names=fleet.JOINTS, episodes=tuple(eps))


def cand(repo_id, accepted=True, **kw):
    return Candidate(repo_id=repo_id, accepted=accepted, reason="" if accepted else "nope",
                     robot_type="so101_follower", fps=30.0, n_episodes=12, **kw)


def test_plan_split():
    assert plan_split(50, 20, 10) == (20, 10)
    assert plan_split(30, 20, 10) == (20, 10)
    assert plan_split(15, 20, 10) == (10, 5)
    assert plan_split(16, 20, 10) == (11, 5)
    assert plan_split(2, 20, 10) == (1, 1)
    with pytest.raises(ValueError):
        plan_split(1, 20, 10)


def test_lock_name():
    assert fleet.lock_name("Some-User/so101 arm!") == "some-user__so101_arm"
    assert fleet.lock_name("plain") == "plain"


def test_load_with_retries_only_retries_network_errors():
    calls = []

    def flaky(rid):
        calls.append(rid)
        if len(calls) < 3:
            raise ConnectionError("down")
        return "ok"

    waits = []
    assert fleet.load_with_retries("a/b", flaky, retries=3, wait_s=1.0, sleep=waits.append) == "ok"
    assert waits == [1.0, 2.0]

    def bad(rid):
        raise ValueError("corrupt")

    with pytest.raises(ValueError):
        fleet.load_with_retries("a/b", bad, retries=3, sleep=waits.append)
    assert waits == [1.0, 2.0]  # no sleeps for a non-network error

    def always_down(rid):
        raise OSError("down")

    with pytest.raises(OSError):
        fleet.load_with_retries("a/b", always_down, retries=2, wait_s=0.5, sleep=waits.append)
    assert waits[-1] == 0.5


def test_scan_calibrates_each_robot_and_tolerates_failures(tmp_path):
    datasets = {
        "alice/so101_a": make_dataset("alice/so101_a", seed=1),
        "carol/so101_c": make_dataset("carol/so101_c", n=15, seed=3),
    }

    def load(rid):
        if rid == "bob/broken":
            raise ValueError("no parquet data")
        return datasets[rid]

    events = []
    cands = [cand("alice/so101_a", data_bytes=123, downloads=9), cand("bob/broken"),
             cand("zed/koch", accepted=False), cand("carol/so101_c")]
    rep = scan(cands, n_train=20, n_eval=10, max_evaluations=12, time_budget_s=600,
               on_progress=events.append, lock_dir=tmp_path / "fleet", load=load, simulate=toy_simulate, retries=1)

    assert [r.dataset for r in rep.robots] == ["alice/so101_a", "carol/so101_c"]
    assert rep.failures == [{"dataset": "bob/broken", "error": "ValueError: no parquet data"}]
    assert rep.rejected == [{"dataset": "zed/koch", "reason": "nope"}]
    assert rep.skipped == []
    a = rep.robots[0]
    assert (a.n_train, a.n_eval) == (8, 4)  # 12 episodes: a third held out
    assert rep.robots[1].n_train == 10 and rep.robots[1].n_eval == 5
    assert a.robot_type == "so101_follower" and a.fps == 30.0 and a.n_episodes == 12
    assert a.data_bytes == 123 and a.downloads == 9
    assert a.n_evaluations == 12 and a.lock == "alice__so101_a"
    assert set(a.params) >= {"kp", "kv", "latency_frames"}
    assert a.calibrated_eval <= a.baseline_eval
    assert a.improvement == pytest.approx(1 - a.calibrated_eval / a.baseline_eval)
    assert a.latency_ms == pytest.approx(a.params["latency_frames"] / 30.0 * 1000)
    assert set(a.per_joint_after) == set(fleet.JOINTS)

    # locks on disk, loadable by the lock module, digest over the train split
    from antares.lock import RobotLock

    lock = RobotLock.load(tmp_path / "fleet" / "alice__so101_a.lock.json")
    assert lock.dataset == "alice/so101_a" and lock.params.to_dict() == a.params
    assert sorted(p.name for p in (tmp_path / "fleet").glob("*.lock.json")) == ["alice__so101_a.lock.json", "carol__so101_c.lock.json"]

    # report saved and round-trips
    saved = json.loads((tmp_path / "fleet" / "fleet.json").read_text())
    assert saved["summary"]["n_robots"] == 2 and saved["summary"]["n_failed"] == 1
    assert saved["settings"]["n_candidates"] == 3
    back = FleetReport.load(tmp_path / "fleet" / "fleet.json")
    assert [r.dataset for r in back.robots] == [r.dataset for r in rep.robots]
    assert back.robots[0].params == a.params
    assert back.to_dict()["robots"] == rep.to_dict()["robots"]

    types_ = [e["type"] for e in events]
    assert types_[0] == "start" and types_[-1] == "done"
    assert types_.count("robot") == 2 and types_.count("failed") == 1 and "fit" in types_
    assert events[-1]["summary"]["n_robots"] == 2


def test_scan_stops_when_the_budget_is_spent(tmp_path, monkeypatch):
    ds = make_dataset("alice/so101_a", seed=1)
    clock = {"t": 0.0}

    def monotonic():
        clock["t"] += 100.0  # every call burns 100 s
        return clock["t"]

    monkeypatch.setattr(fleet.time, "monotonic", monotonic)
    cands = [cand("alice/so101_a"), cand("bob/so101_b"), cand("carol/so101_c")]
    rep = scan(cands, max_evaluations=2, time_budget_s=250, lock_dir=tmp_path, load=lambda rid: ds, simulate=toy_simulate)
    assert [r.dataset for r in rep.robots] == ["alice/so101_a"]
    assert [s["dataset"] for s in rep.skipped] == ["bob/so101_b", "carol/so101_c"]
    assert all(s["reason"] == "time budget exhausted" for s in rep.skipped)
    assert rep.elapsed_s > 0


def test_scan_rejects_wrong_joints_and_unusable_episodes(tmp_path):
    ds = make_dataset("a/x", seed=2)
    wrong = Dataset(repo_id="a/x", robot_type="so101", fps=30.0, joint_names=tuple(reversed(fleet.JOINTS)), episodes=ds.episodes)
    bad_ep = Episode(index=99, fps=30.0, joint_names=fleet.JOINTS, state=np.full((5, 6), np.nan), action=np.zeros((5, 6)))
    short_ep = Episode(index=98, fps=30.0, joint_names=fleet.JOINTS, state=np.zeros((1, 6)), action=np.zeros((1, 6)))
    dirty = Dataset(repo_id="b/y", robot_type="so101", fps=30.0, joint_names=fleet.JOINTS, episodes=ds.episodes + (bad_ep, short_ep))
    loads = {"a/x": wrong, "b/y": dirty}
    rep = scan([cand("a/x"), cand("b/y")], max_evaluations=2, lock_dir=tmp_path, load=lambda rid: loads[rid], simulate=toy_simulate)
    assert [f["dataset"] for f in rep.failures] == ["a/x"] and "joints" in rep.failures[0]["error"]
    assert rep.robots[0].dataset == "b/y" and rep.robots[0].n_episodes == 12  # the two unusable episodes dropped


def entry(dataset, n_episodes=50, fps=30.0, **params):
    p = {"kp": 1000.0, "kv": 3.0, "damping": 0.5, "armature": 0.03, "frictionloss": 0.05, "latency_frames": 2}
    p.update(params)
    return RobotEntry(dataset, "so101", n_episodes, fps, 20, 10, 5.0, 2.0, 0.6, p, fleet.lock_name(dataset))


def test_robust_z():
    z = fleet.robust_z(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert z[2] == 0 and z[4] > 2.5 and abs(z[0]) < 2.5
    assert fleet.robust_z(np.array([5.0, 5.0, 5.0])).tolist() == [0.0, 0.0, 0.0]
    assert fleet.robust_z(np.array([])).size == 0
    # MAD of zero with some spread falls back to the IQR, then to the std
    z = fleet.robust_z(np.array([1.0, 1.0, 1.0, 2.0, 9.0]))
    assert z[-1] > 0 and np.isfinite(z).all()
    z = fleet.robust_z(np.array([2.0] * 8 + [5.0]))
    assert z[-1] > 2.5 and z[0] == 0


def test_fleet_stats_and_outliers():
    robots = [entry(f"u{i}/arm", kp=1000.0 * (1 + 0.05 * i), latency_frames=2) for i in range(8)]
    robots.append(entry("odd/arm", kp=20000.0, latency_frames=5, fps=50.0))
    stats = fleet.fleet_stats(robots)
    assert set(stats) == set(fleet.STAT_PARAMS)
    kp = stats["kp"]
    assert kp["scale"] == "log" and kp["n"] == 9
    assert kp["q25"] <= kp["median"] <= kp["q75"] and kp["iqr"] == pytest.approx(kp["q75"] - kp["q25"])
    assert kp["min"] == 1000.0 and kp["max"] == 20000.0
    assert kp["z"]["odd/arm"] > 2.5 and abs(kp["z"]["u4/arm"]) < 1
    assert stats["latency_ms"]["scale"] == "linear"
    assert stats["latency_ms"]["z"]["odd/arm"] > 2.5  # 100 ms vs 66.7 ms for all the rest (std fallback)
    assert stats["damping"]["z"]["odd/arm"] == 0.0  # no spread at all

    outliers = fleet.find_outliers(robots, stats)
    assert {(o["dataset"], o["param"]) for o in outliers} == {("odd/arm", "kp"), ("odd/arm", "latency_frames")}
    kp_o = next(o for o in outliers if o["param"] == "kp")
    assert kp_o["value"] == 20000.0 and "x the fleet median" in kp_o["note"]
    lat_o = next(o for o in outliers if o["param"] == "latency_frames")
    assert "5 frames (100 ms)" in lat_o["note"]
    assert abs(outliers[0]["z"]) >= abs(outliers[-1]["z"])
    assert fleet.find_outliers(robots, stats, threshold=1e9) == []
    assert fleet.fleet_stats([]) == {}


def test_pick_cross_robots_prefers_demo_and_needs_full_split():
    robots = [entry("a/x"), entry("b/y", n_episodes=29), entry("lerobot/svla_so100_pickplace"), entry("c/z"), entry("d/w")]
    chosen = fleet.pick_cross_robots(robots, n=3, prefer=fleet.PINNED)
    assert [r.dataset for r in chosen] == ["lerobot/svla_so100_pickplace", "a/x", "c/z"]
    assert len(fleet.pick_cross_robots(robots, n=10)) == 4  # b/y cannot take the 20 + 10 split


def test_cross_summary():
    m = {"rows": ["a", "b", "c"], "cols": ["la", "lb", "lc", "default"],
         "scores": [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 5.0, 4.0], [3.0, 3.0, 1.0, 2.0]]}
    s = fleet.cross_summary(m)
    assert s["n"] == 3 and s["mean_diagonal"] == 1.0
    assert s["mean_off_diagonal"] == pytest.approx(18 / 6)
    assert s["mean_default"] == pytest.approx(10 / 3)
    assert s["off_over_diagonal"] == pytest.approx(3.0) and s["default_over_diagonal"] == pytest.approx(10 / 3)
    assert s["other_lock_beats_default"] == pytest.approx(3 / 6)
    assert s["own_lock_is_best"] == 3
    with pytest.raises(ValueError):
        fleet.cross_summary({"rows": ["a"], "scores": [[1.0]]})


def test_fleet_cross_uses_injected_scorer_and_summarises(tmp_path):
    robots = [entry("lerobot/svla_so101_pickplace"), entry("a/x"), entry("b/y"), entry("c/z", n_episodes=12)]
    datasets = {r.dataset: make_dataset(r.dataset, n=30) for r in robots}
    calls = []

    def cross_fn(dss, locks, n_train, n_eval, seed, lock_dir):
        calls.append(([d.repo_id for d in dss], list(locks), n_train, n_eval, seed, lock_dir))
        k = len(dss)
        return {"rows": [d.repo_id for d in dss], "cols": list(locks) + ["default"],
                "scores": [[1.0 if i == j else 2.0 for j in range(k)] + [3.0] for i in range(k)], "per_joint": []}

    out = fleet.fleet_cross(robots, lock_dir=tmp_path, n=6, load=lambda rid: datasets[rid], cross_fn=cross_fn)
    assert calls[0][0] == ["lerobot/svla_so101_pickplace", "a/x", "b/y"]  # c/z too small
    assert calls[0][1] == ["lerobot__svla_so101_pickplace", "a__x", "b__y"]
    assert calls[0][2:] == (20, 10, 0, tmp_path)
    assert out["locks"] == calls[0][1]
    assert out["summary"]["mean_diagonal"] == 1.0 and out["summary"]["mean_off_diagonal"] == 2.0
    assert fleet.fleet_cross(robots[:1], lock_dir=tmp_path, load=lambda rid: datasets[rid], cross_fn=cross_fn) is None


def test_scan_finalizes_with_stats_outliers_and_cross(tmp_path):
    datasets = {f"u{i}/arm": make_dataset(f"u{i}/arm", n=30, seed=i) for i in range(3)}

    def cross_fn(dss, locks, n_train, n_eval, seed, lock_dir):
        assert all((lock_dir / f"{l}.lock.json").is_file() for l in locks)
        k = len(dss)
        return {"rows": [d.repo_id for d in dss], "cols": list(locks) + ["default"],
                "scores": [[1.0 if i == j else 2.0 for j in range(k)] + [3.0] for i in range(k)], "per_joint": []}

    rep = scan([cand(r) for r in datasets], max_evaluations=3, lock_dir=tmp_path, load=lambda rid: datasets[rid],
               simulate=toy_simulate, cross_n=2, cross_fn=cross_fn)
    assert len(rep.robots) == 3
    assert set(rep.stats) == set(fleet.STAT_PARAMS) and rep.stats["kp"]["n"] == 3
    assert isinstance(rep.outliers, list)
    assert rep.cross["summary"]["n"] == 2 and rep.cross["rows"] == ["u0/arm", "u1/arm"]
    saved = json.loads((tmp_path / "fleet.json").read_text())
    assert saved["cross"]["summary"]["off_over_diagonal"] == 2.0 and "kp" in saved["stats"]

    # a failing cross matrix is recorded, not raised
    def bad_cross(*a, **k):
        raise RuntimeError("no sim")

    rep = scan([cand(r) for r in datasets], max_evaluations=2, lock_dir=tmp_path, load=lambda rid: datasets[rid],
               simulate=toy_simulate, cross_n=2, cross_fn=bad_cross)
    assert len(rep.robots) == 3 and rep.cross is None
    assert rep.failures[-1]["dataset"] == "<cross matrix>" and "no sim" in rep.failures[-1]["error"]

    # cross disabled
    rep = scan([cand("u0/arm")], max_evaluations=2, lock_dir=tmp_path, load=lambda rid: datasets[rid],
               simulate=toy_simulate, cross_n=0)
    assert rep.cross is None and rep.stats["kp"]["n"] == 1


def test_robot_entry_nan_round_trip():
    e = RobotEntry("a/x", "so101", 10, 30.0, 6, 4, float("nan"), 1.0, float("nan"), {"latency_frames": 3}, "a__x")
    d = e.to_dict()
    assert d["baseline_eval"] is None and d["latency_ms"] == pytest.approx(100.0)
    back = RobotEntry.from_dict(json.loads(json.dumps(d)))
    assert math.isnan(back.baseline_eval) and back.calibrated_eval == 1.0
