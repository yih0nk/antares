"""Hub reads fall back to the local cache when the network misbehaves."""

from pathlib import Path

import huggingface_hub
import pytest

from antares import hub


class ProxyError(Exception):
    pass


class RepositoryNotFoundError(Exception):
    pass


def fake_download(behaviour):
    calls = []

    def fn(repo_id, filename, local_files_only=False, **kwargs):
        calls.append(local_files_only)
        if local_files_only:
            if behaviour == "cache-miss":
                raise FileNotFoundError("not cached")
            return f"/cache/{repo_id}/{filename}"
        if behaviour == "missing":
            raise RepositoryNotFoundError(repo_id)
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

