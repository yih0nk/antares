"""Fetch robot models from MuJoCo Menagerie on first use instead of vendoring them.

Only the SO-100/101 models ship inside the package (``antares/assets``).
Other arms name a directory and an MJCF file in Menagerie at a pinned
commit; :func:`fetch` downloads that MJCF, every file it ``<include>``s, the
meshes and textures they reference and the directory's LICENSE, into a
cache (``$ANTARES_MODEL_CACHE``, default ``~/.cache/antares/models``). A
pinned commit means a lock fitted today replays on the same geometry later.
"""

from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: The Menagerie commit every non-vendored model is pinned to.
MENAGERIE_COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
RAW_URL = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}/{path}"

_INCLUDE_RE = re.compile(r"<include\b[^>]*\bfile=\"([^\"]+)\"")
_ASSET_RE = re.compile(r"<(mesh|texture|hfield|skin)\b[^>]*\bfile=\"([^\"]+)\"")
_COMPILER_RE = re.compile(r"<compiler\b[^>]*>")
_DIR_ATTR_RE = re.compile(r"\b(meshdir|texturedir|assetdir)=\"([^\"]*)\"")


def cache_root() -> Path:
    return Path(os.environ.get("ANTARES_MODEL_CACHE", Path.home() / ".cache" / "antares" / "models"))


@dataclass(frozen=True)
class MenagerieModel:
    """``directory/xml`` in Menagerie at ``commit``."""

    directory: str
    xml: str
    commit: str = MENAGERIE_COMMIT

    @property
    def local_dir(self) -> Path:
        return cache_root() / "menagerie" / self.commit[:12] / self.directory

    @property
    def model_path(self) -> Path:
        return self.local_dir / self.xml

    def is_fetched(self) -> bool:
        return (self.local_dir / ".complete").is_file()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310 - fixed host
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.replace(dest)


def asset_dirs(xml: str) -> dict[str, str]:
    """``meshdir``/``texturedir``/``assetdir`` from the ``<compiler>`` tag."""
    m = _COMPILER_RE.search(xml)
    return dict(_DIR_ATTR_RE.findall(m.group(0))) if m else {}


def referenced_files(xml: str, dirs: dict[str, str]) -> tuple[list[str], list[str]]:
    """``(includes, assets)``: relative paths an MJCF text needs, assets
    resolved against its compiler directories."""
    includes = _INCLUDE_RE.findall(xml)
    assets = []
    for tag, f in _ASSET_RE.findall(xml):
        base = dirs.get("texturedir" if tag == "texture" else "meshdir", dirs.get("assetdir", ""))
        assets.append(str(Path(base) / f) if base else f)
    return includes, assets


def fetch(model: MenagerieModel, force: bool = False) -> Path:
    """Download ``model`` into the cache if needed; returns the MJCF path."""
    if model.is_fetched() and not force:
        return model.model_path

    def url(rel: str) -> str:
        return RAW_URL.format(commit=model.commit, path=f"{model.directory}/{rel}")

    root = model.local_dir
    seen: set[str] = set()
    queue = [model.xml]
    dirs: dict[str, str] = {}
    assets: list[str] = []
    while queue:
        rel = queue.pop(0)
        if rel in seen:
            continue
        seen.add(rel)
        dest = root / rel
        _download(url(rel), dest)
        text = dest.read_text()
        dirs = {**asset_dirs(text), **dirs}  # the top-level compiler tag wins
        inc, ass = referenced_files(text, dirs)
        queue += [str(Path(rel).parent / i) if Path(rel).parent != Path(".") else i for i in inc]
        assets += ass
    for rel in dict.fromkeys(assets):
        if not (root / rel).is_file() or force:
            _download(url(rel), root / rel)
    try:
        _download(url("LICENSE"), root / "LICENSE")
    except OSError:
        pass
    (root / ".complete").write_text(model.commit + "\n")
    return model.model_path


__all__ = ["MENAGERIE_COMMIT", "MenagerieModel", "cache_root", "fetch", "referenced_files"]
