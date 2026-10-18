"""Frames from a LeRobot dataset's fixed cameras, decoded on demand.

A v3 dataset stores every camera as AV1 mp4 files under
``videos/<camera>/chunk-XXX/file-XXX.mp4`` with the episodes concatenated;
``meta/episodes`` says which file each episode is in and at what timestamp
it starts. Only the one video file an episode lives in is downloaded (with
``hf_hub_download``, so the Hub cache dedupes), and decoded frames are kept
under ``.cache/antares/frames/`` so a refit or the overlay viewer never
decodes the same frame twice.

Decoding needs PyAV (``pip install "antares[video]"``); it ships dav1d.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from antares.data import Dataset
from antares.lock import slug

FRAME_CACHE: Path = Path(os.environ.get("ANTARES_FRAME_CACHE", ".cache/antares/frames"))
VIDEO_PREFIX = "observation.images."


class VideoUnavailable(RuntimeError):
    """A video dependency is missing or the dataset has no such camera."""


def _require_av():
    try:
        import av  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise VideoUnavailable(
            "video decoding needs PyAV: pip install 'antares[video]' (av, opencv-python-headless, scipy)"
        ) from e
    return av


# --------------------------------------------------------------------------
# Dataset metadata
# --------------------------------------------------------------------------


def _is_local(repo_id: str) -> bool:
    return os.path.isdir(repo_id)


def _hub_file(repo_id: str, filename: str) -> Path:
    if _is_local(repo_id):
        p = Path(repo_id) / filename
        if not p.is_file():
            raise VideoUnavailable(f"{repo_id} has no {filename}")
        return p
    from huggingface_hub.utils import EntryNotFoundError

    from antares.hub import HubUnavailable, download

    try:
        return download(repo_id, filename)
    except EntryNotFoundError as e:
        raise VideoUnavailable(f"{repo_id} has no {filename}") from e
    except HubUnavailable as e:
        # Offline (or behind a portal) with no cached copy: the same outcome
        # as a dataset without the file, so the callers' fallbacks apply.
        raise VideoUnavailable(str(e)) from e


def dataset_info(repo_id: str) -> dict[str, Any]:
    """``meta/info.json`` of a Hub (or local) LeRobot dataset."""
    return json.loads(_hub_file(repo_id, "meta/info.json").read_text())


def camera_name(key: str) -> str:
    """``observation.images.up`` -> ``up`` (a bare name passes through)."""
    return key[len(VIDEO_PREFIX) :] if key.startswith(VIDEO_PREFIX) else key


def camera_key(name: str) -> str:
    """``up`` -> ``observation.images.up`` (a full key passes through)."""
    return name if name.startswith(VIDEO_PREFIX) else VIDEO_PREFIX + name


def is_fixed_camera(name: str) -> bool:
    """Wrist/gripper cameras move with the arm and cannot be fitted as a static pose."""
    n = camera_name(name).lower()
    return not any(tag in n for tag in ("wrist", "gripper", "hand", "ego"))


@dataclass(frozen=True)
class CameraInfo:
    name: str  # short name, e.g. "up"
    key: str  # feature key, e.g. "observation.images.up"
    width: int
    height: int
    fps: float
    fixed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fixed": self.fixed,
        }


def cameras_from_info(info: dict[str, Any]) -> list[CameraInfo]:
    """Every video feature in ``info.json`` with its size and whether it is fixed."""
    out = []
    fps = float(info.get("fps", 30))
    for key, feat in info.get("features", {}).items():
        if feat.get("dtype") != "video":
            continue
        shape = feat.get("shape") or [0, 0, 3]
        vinfo = feat.get("info") or {}
        out.append(
            CameraInfo(
                name=camera_name(key),
                key=key,
                width=int(vinfo.get("video.width", shape[1])),
                height=int(vinfo.get("video.height", shape[0])),
                fps=float(vinfo.get("video.fps", fps)),
                fixed=is_fixed_camera(key),
            )
        )
    return out


def list_cameras(repo_id: str) -> list[CameraInfo]:
    return cameras_from_info(dataset_info(repo_id))


def find_camera(repo_id: str, camera: str) -> CameraInfo:
    cams = {c.name: c for c in list_cameras(repo_id)}
    name = camera_name(camera)
    if name not in cams:
        raise VideoUnavailable(f"{repo_id} has no camera {camera!r}; cameras: {sorted(cams)}")
    return cams[name]


# --------------------------------------------------------------------------
# Episode -> (video file, time offset)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """Where one episode's frames are in a camera's video files."""

    episode: int
    chunk: int
    file: int
    from_ts: float  # seconds into the file where frame 0 of the episode is
    to_ts: float
    length: int  # frames


def _segments_from_meta(repo_id: str, key: str, info: dict[str, Any]) -> dict[int, Segment] | None:
    """Read ``meta/episodes`` (v3). None when the dataset has no such table."""
    import pyarrow.parquet as pq

    pattern = info.get("episodes_path") or "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    try:
        path = _hub_file(repo_id, pattern.format(chunk_index=0, file_index=0))
    except VideoUnavailable:
        return None
    wanted = ["episode_index", "length", f"videos/{key}/chunk_index", f"videos/{key}/file_index",
              f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp"]
    table = pq.read_table(path)
    if any(c not in table.schema.names for c in wanted):
        return None
    cols = table.select(wanted).to_pydict()
    segs = {}
    for i, ep in enumerate(cols["episode_index"]):
        segs[int(ep)] = Segment(
            episode=int(ep),
            chunk=int(cols[wanted[2]][i]),
            file=int(cols[wanted[3]][i]),
            from_ts=float(cols[wanted[4]][i]),
            to_ts=float(cols[wanted[5]][i]),
            length=int(cols["length"][i]),
        )
    return segs


def segments_from_episodes(lengths: Sequence[tuple[int, int]], fps: float) -> dict[int, Segment]:
    """Fallback: episodes concatenated in index order into one file (chunk 0, file 0)."""
    segs, t = {}, 0.0
    for ep, n in sorted(lengths):
        segs[ep] = Segment(episode=ep, chunk=0, file=0, from_ts=t, to_ts=t + n / fps, length=n)
        t += n / fps
    return segs


def episode_segments(repo_id: str, camera: str, dataset: Dataset | None = None) -> dict[int, Segment]:
    """Episode index -> :class:`Segment` for ``camera``."""
    info = dataset_info(repo_id)
    key = camera_key(camera)
    segs = _segments_from_meta(repo_id, key, info)
    if segs is not None:
        return segs
    if dataset is None:
        from antares.data import load_dataset, read_local

        dataset = read_local(repo_id, repo_id=repo_id) if _is_local(repo_id) else load_dataset(repo_id)
    return segments_from_episodes([(ep.index, len(ep)) for ep in dataset.episodes], dataset.fps)


def video_file(repo_id: str, camera: str, chunk: int = 0, file: int = 0) -> Path:
    """Download (or find in the Hub cache) one video file of ``camera``."""
    info = dataset_info(repo_id)
    pattern = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    rel = pattern.format(video_key=camera_key(camera), chunk_index=chunk, file_index=file)
    return _hub_file(repo_id, rel)


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def decode_at(path: Path, times: Sequence[float], fps: float) -> np.ndarray:
    """Decode the frames nearest to ``times`` (seconds) from one file.

    One seek to the earliest time, then a forward decode picking each wanted
    frame as the first decoded frame whose presentation time reaches it
    (minus half a frame, so timestamp jitter does not skip a frame).
    Returns (N,H,W,3) uint8 RGB in the order of ``times``.
    """
    av = _require_av()
    order = np.argsort(times, kind="stable")
    targets = [float(times[i]) for i in order]
    half = 0.5 / float(fps)
    out: list[np.ndarray | None] = [None] * len(targets)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(max(0.0, targets[0] - half) / stream.time_base), stream=stream, backward=True)
        j = 0
        last = None
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            last = frame
            while j < len(targets) and frame.time >= targets[j] - half:
                out[j] = frame.to_ndarray(format="rgb24")
                j += 1
            if j >= len(targets):
                break
        # Past the end of the file: repeat the last frame rather than fail.
        if j < len(targets) and last is not None:
            arr = last.to_ndarray(format="rgb24")
            while j < len(targets):
                out[j] = arr
                j += 1
    if any(o is None for o in out):
        raise VideoUnavailable(f"could not decode frames at {targets[:3]}... from {path}")
    result = np.empty(len(targets), dtype=object)
    for k, i in enumerate(order):
        result[i] = out[k]
    return np.stack(list(result))


def _cache_dir(repo_id: str, camera: str) -> Path:
    return FRAME_CACHE / slug(repo_id) / camera_name(camera)


def frames(
    repo_id: str,
    camera: str,
    episode: int,
    indices: Sequence[int],
    dataset: Dataset | None = None,
    cache: bool = True,
) -> np.ndarray:
    """Frames ``indices`` (frame_index within ``episode``) of ``camera`` as (N,H,W,3) RGB uint8.

    Frames already decoded are read from the on-disk cache; the rest are
    decoded in one pass over the video and cached.
    """
    indices = [int(i) for i in indices]
    cdir = _cache_dir(repo_id, camera) / f"ep{episode:05d}"
    out: dict[int, np.ndarray] = {}
    missing: list[int] = []
    for i in indices:
        p = cdir / f"f{i:06d}.npy"
        if cache and p.is_file():
            try:
                out[i] = np.load(p)
                continue
            except (OSError, ValueError):
                pass
        missing.append(i)
    if missing:
        segs = episode_segments(repo_id, camera, dataset)
        if episode not in segs:
            raise VideoUnavailable(f"episode {episode} not in {repo_id}'s {camera} video index")
        seg = segs[episode]
        cam = find_camera(repo_id, camera)
        bad = [i for i in missing if i < 0 or i >= seg.length]
        if bad:
            raise VideoUnavailable(f"frames {bad[:5]} out of range for episode {episode} ({seg.length} frames)")
        path = video_file(repo_id, camera, seg.chunk, seg.file)
        uniq = sorted(set(missing))
        times = [seg.from_ts + i / cam.fps for i in uniq]
        decoded = decode_at(path, times, cam.fps)
        if cache:
            cdir.mkdir(parents=True, exist_ok=True)
        for i, arr in zip(uniq, decoded):
            out[i] = arr
            if cache:
                np.save(cdir / f"f{i:06d}.npy", arr)
    return np.stack([out[i] for i in indices])


def frame_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode an RGB frame as JPEG bytes (OpenCV)."""
    import cv2

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(frame[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return bytes(buf)


__all__ = [
    "FRAME_CACHE",
    "CameraInfo",
    "Segment",
    "VideoUnavailable",
    "camera_key",
    "camera_name",
    "cameras_from_info",
    "dataset_info",
    "decode_at",
    "episode_segments",
    "find_camera",
    "frame_jpeg",
    "frames",
    "is_fixed_camera",
    "list_cameras",
    "segments_from_episodes",
    "video_file",
]
