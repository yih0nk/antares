"""Hugging Face Hub downloads that keep working offline.

``huggingface_hub`` falls back to its cache when the network refuses a
connection, but not when something answers with an error (a proxy, a
captive portal) or hangs. Every Hub read in Antares goes through
:func:`download` / :func:`snapshot`, which retry from the local cache on any
error that is not "this file or repo does not exist".

``ANTARES_OFFLINE=1`` (set by ``antares serve --offline``) skips the network
entirely and reads only the cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class HubUnavailable(RuntimeError):
    """The Hub could not be reached and the file is not in the local cache."""


def offline() -> bool:
    return os.environ.get("ANTARES_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _is_missing(exc: Exception) -> bool:
    """The Hub answered that the repo, file or revision does not exist."""
    name = type(exc).__name__
    if name == "LocalEntryNotFoundError":  # a cache miss, not an answer from the Hub
        return False
    return "NotFound" in name or "GatedRepo" in name or "Revision" in name


def _cached(fn, what: str, first: Exception | None, **kwargs: Any) -> str:
    try:
        return fn(local_files_only=True, **kwargs)
    except Exception as e:  # noqa: BLE001 - any cache miss
        why = f"the Hub could not be reached ({type(first).__name__}: {first})" if first else "offline mode is on"
        raise HubUnavailable(f"{what} is not in the local Hugging Face cache and {why}") from (first or e)


def download(repo_id: str, filename: str, **kwargs: Any) -> Path:
    """``hf_hub_download`` of one dataset file, falling back to the cache."""
    from huggingface_hub import hf_hub_download

    kwargs.setdefault("repo_type", "dataset")
    what = f"{repo_id}/{filename}"
    if offline():
        return Path(_cached(hf_hub_download, what, None, repo_id=repo_id, filename=filename, **kwargs))
    try:
        return Path(hf_hub_download(repo_id, filename, **kwargs))
    except Exception as e:  # noqa: BLE001
        if _is_missing(e):
            raise
        return Path(_cached(hf_hub_download, what, e, repo_id=repo_id, filename=filename, **kwargs))


def snapshot(repo_id: str, **kwargs: Any) -> str:
    """``snapshot_download`` of a dataset, falling back to the cache."""
    from huggingface_hub import snapshot_download

    kwargs.setdefault("repo_type", "dataset")
    if offline():
        return _cached(snapshot_download, repo_id, None, repo_id=repo_id, **kwargs)
    try:
        return snapshot_download(repo_id, **kwargs)
    except Exception as e:  # noqa: BLE001
        if _is_missing(e):
            raise
        return _cached(snapshot_download, repo_id, e, repo_id=repo_id, **kwargs)
