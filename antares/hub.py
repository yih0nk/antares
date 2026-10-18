"""Hugging Face Hub downloads that keep working offline.

``huggingface_hub`` falls back to its cache when the network refuses a
connection, but not when something answers with an error (a proxy, a
captive portal) or hangs. Every Hub read in Antares goes through
:func:`download` / :func:`snapshot`, which retry from the local cache on any
error that is not "this file or repo does not exist".

A captive portal answering 401 to everything looks, to ``huggingface_hub``,
like a repo that does not exist (it maps any 401 to
``RepositoryNotFoundError``). Such a "not found" is only trusted when the
cache cannot answer instead: a 401, or a response without the Hub's own
request id, is retried from the cache first and re-raised as it came if
the cache misses.

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


def _not_found_types() -> tuple[type[Exception], ...]:
    """The exceptions ``huggingface_hub`` raises when the Hub answers that a
    repo, revision or file does not exist (``GatedRepoError`` is a
    ``RepositoryNotFoundError``). ``EntryNotFoundError`` became
    ``RemoteEntryNotFoundError`` in huggingface_hub 2.x; both are taken
    when present."""
    from huggingface_hub import errors

    names = ("RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError", "RemoteEntryNotFoundError")
    return tuple(t for n in names if isinstance(t := getattr(errors, n, None), type))


def _is_missing(exc: Exception) -> bool:
    """The Hub answered that the repo, file or revision does not exist."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    if isinstance(exc, LocalEntryNotFoundError):  # a cache miss, not an answer from the Hub
        return False
    return isinstance(exc, _not_found_types())


def _portal_like(exc: Exception) -> bool:
    """A "not found" that may not have come from the Hub at all: a 401 (a
    captive portal or proxy challenge, which ``huggingface_hub`` reports as
    ``RepositoryNotFoundError``), or a response without the Hub's request id."""
    response = getattr(exc, "response", None)
    if response is None:
        return False
    if getattr(response, "status_code", None) == 401:
        return True
    headers = getattr(response, "headers", None) or {}
    return not any(k.lower() == "x-request-id" for k in headers)


def _cached(fn, what: str, first: Exception | None, **kwargs: Any) -> str:
    try:
        return fn(local_files_only=True, **kwargs)
    except Exception as e:  # noqa: BLE001 - any cache miss
        why = f"the Hub could not be reached ({type(first).__name__}: {first})" if first else "offline mode is on"
        raise HubUnavailable(f"{what} is not in the local Hugging Face cache and {why}") from (first or e)


def _fallback(fn, what: str, first: Exception, **kwargs: Any) -> str:
    """The cached copy after a network error ``first``. A genuine "does not
    exist" from the Hub is re-raised; one that may be a portal's answer is
    re-raised only when the cache has no copy either."""
    if _is_missing(first):
        if not _portal_like(first):
            raise first
        try:
            return fn(local_files_only=True, **kwargs)
        except Exception:  # noqa: BLE001 - any cache miss
            raise first from None
    return _cached(fn, what, first, **kwargs)


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
        return Path(_fallback(hf_hub_download, what, e, repo_id=repo_id, filename=filename, **kwargs))


def snapshot(repo_id: str, **kwargs: Any) -> str:
    """``snapshot_download`` of a dataset, falling back to the cache."""
    from huggingface_hub import snapshot_download

    kwargs.setdefault("repo_type", "dataset")
    if offline():
        return _cached(snapshot_download, repo_id, None, repo_id=repo_id, **kwargs)
    try:
        return snapshot_download(repo_id, **kwargs)
    except Exception as e:  # noqa: BLE001
        return _fallback(snapshot_download, repo_id, e, repo_id=repo_id, **kwargs)
