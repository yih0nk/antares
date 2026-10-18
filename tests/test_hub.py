"""Hub reads fall back to the local cache when the network misbehaves."""

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from antares import hub


class ProxyError(Exception):
    pass


def hub_response(status_code: int, from_hub: bool = True):
    """What ``huggingface_hub`` attaches to its HTTP errors: the Hub's own
    answers carry a request id, a captive portal's do not."""
    headers = {"x-request-id": "abc123"} if from_hub else {"content-type": "text/html"}
    return SimpleNamespace(status_code=status_code, headers=headers, request=None)


def fake_download(behaviour, error=None):
    calls = []

    def fn(repo_id, filename, local_files_only=False, **kwargs):
        calls.append(local_files_only)
        if local_files_only:
            if behaviour == "cache-miss":
                raise FileNotFoundError("not cached")
            return f"/cache/{repo_id}/{filename}"
        if error is not None:
            raise error
        if behaviour == "missing":
            raise RepositoryNotFoundError(repo_id, response=hub_response(404))
        raise ProxyError("403 Forbidden")

    return fn, calls


def test_proxy_error_falls_back_to_the_cache(monkeypatch):
    fn, calls = fake_download("cached")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fn)
    assert hub.download("a/b", "meta/info.json") == Path("/cache/a/b/meta/info.json")
    assert calls == [False, True]


def test_missing_repo_is_not_retried(monkeypatch):
    fn, calls = fake_download("missing")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fn)
    with pytest.raises(RepositoryNotFoundError):
        hub.download("a/b", "meta/info.json")
    assert calls == [False]


@pytest.mark.parametrize("error", [
    GatedRepoError("gated", response=hub_response(403)),
    RevisionNotFoundError("no such revision", response=hub_response(404)),
    EntryNotFoundError("no such file"),
])
def test_other_hub_answers_are_not_retried(monkeypatch, error):
    fn, calls = fake_download("missing", error)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fn)
    with pytest.raises(type(error)):
        hub.download("a/b", "meta/info.json")
    assert calls == [False]


def test_a_401_falls_back_to_the_cache(monkeypatch):
    """huggingface_hub reports any 401 as RepositoryNotFoundError, which is
    what a captive portal answering every request with 401 looks like: the
    cached copy is used, and only a cache miss raises as the Hub answered."""
    fn, calls = fake_download("missing", RepositoryNotFoundError("401", response=hub_response(401, from_hub=False)))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fn)
    assert hub.download("a/b", "meta/info.json") == Path("/cache/a/b/meta/info.json")
    assert calls == [False, True]

    fn, calls = fake_download("cache-miss", RepositoryNotFoundError("401", response=hub_response(401)))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fn)
    with pytest.raises(RepositoryNotFoundError, match="401"):
        hub.download("a/b", "meta/info.json")
    assert calls == [False, True]


def test_a_not_found_without_the_hubs_request_id_falls_back_to_the_cache(monkeypatch):
    fn, calls = fake_download("missing", RepositoryNotFoundError("portal", response=hub_response(404, from_hub=False)))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fn)
    assert hub.download("a/b", "meta/info.json") == Path("/cache/a/b/meta/info.json")
    assert calls == [False, True]


def test_a_local_cache_miss_is_not_a_hub_answer(monkeypatch):
    """LocalEntryNotFoundError is an EntryNotFoundError in huggingface_hub 2.x,
    but it means the network failed and the cache missed."""
    fn, calls = fake_download("cache-miss", LocalEntryNotFoundError("no network, not cached"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fn)
    with pytest.raises(hub.HubUnavailable, match="LocalEntryNotFoundError"):
        hub.download("a/b", "meta/info.json")
    assert calls == [False, True]


def test_fleet_fetch_info_reads_through_the_hub_wrapper(monkeypatch, tmp_path):
    from antares import fleet

    info = tmp_path / "info.json"
    info.write_text('{"robot_type": "so101_follower", "fps": 30}')
    calls = []

    def fn(repo_id, filename, local_files_only=False, **kwargs):
        calls.append(local_files_only)
        return str(info)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fn)
    monkeypatch.setenv("ANTARES_OFFLINE", "1")
    assert fleet.fetch_info("a/b") == {"robot_type": "so101_follower", "fps": 30}
    assert calls == [True]


def test_video_meta_falls_back_to_episode_lengths_when_offline(monkeypatch, tmp_path):
    """An offline cache miss on meta/episodes is a VideoUnavailable like a
    dataset without the table, so the segments come from the episode lengths."""
    from antares import video
    from antares.data import Dataset, Episode

    info = tmp_path / "info.json"
    info.write_text('{"fps": 30, "features": {"observation.images.up": {"dtype": "video", "shape": [480, 640, 3]}}}')

    def download(repo_id, filename, **kwargs):
        if filename == "meta/info.json":
            return info
        raise hub.HubUnavailable(f"{repo_id}/{filename} is not in the local Hugging Face cache and offline mode is on")

    monkeypatch.setattr(hub, "download", download)
    import numpy as np

    eps = tuple(
        Episode(index=i, fps=30.0, joint_names=("a",), state=np.zeros((n, 1)), action=np.zeros((n, 1)))
        for i, n in ((0, 30), (1, 60))
    )
    ds = Dataset(repo_id="a/b", robot_type="so101_follower", fps=30.0, joint_names=("a",), episodes=eps)
    segs = video.episode_segments("a/b", "up", dataset=ds)
    assert segs == video.segments_from_episodes([(0, 30), (1, 60)], 30.0)
    with pytest.raises(video.VideoUnavailable, match="offline mode is on"):
        video.video_file("a/b", "up")


def test_cache_miss_names_the_network_error(monkeypatch):
    fn, _ = fake_download("cache-miss")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fn)
    with pytest.raises(hub.HubUnavailable, match="ProxyError"):
        hub.download("a/b", "meta/info.json")


def test_offline_mode_never_touches_the_network(monkeypatch):
    fn, calls = fake_download("cached")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fn)
    monkeypatch.setenv("ANTARES_OFFLINE", "1")
    hub.download("a/b", "meta/info.json")
    assert calls == [True]


def test_snapshot_falls_back_to_the_cache(monkeypatch):
    calls = []

    def fn(repo_id, local_files_only=False, **kwargs):
        calls.append(local_files_only)
        if not local_files_only:
            raise ProxyError("403")
        return f"/cache/{repo_id}"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fn)
    assert hub.snapshot("a/b", allow_patterns=["x"]) == "/cache/a/b"
    assert calls == [False, True]


def test_serve_offline_sets_the_environment(monkeypatch):
    import uvicorn

    from antares import cli

    monkeypatch.delenv("ANTARES_OFFLINE", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    cli.main(["serve", "--offline"])
    import os

    assert os.environ["ANTARES_OFFLINE"] == "1"
    monkeypatch.delenv("ANTARES_OFFLINE")
