"""How late a camera's video is relative to the joint log.

A policy trained on images and joint states assumes both describe the same
instant. If the camera pipeline adds a few frames of delay, the image a
policy sees at frame ``t`` shows the arm at ``t - d``, and a sim that
renders the camera at the current state teaches it the wrong pairing.

The delay is measurable without any vision model: the share of pixels that
change between consecutive frames rises and falls with how fast the joints
move. :func:`video_delay` cross-correlates the two over a range of lags and
refines the peak to a fraction of a frame; :func:`camera_delay` runs that
on several episodes of a dataset camera (frames from :mod:`antares.video`)
and reports the median.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from antares.data import Dataset, Episode

#: Decode frames at 1/DOWNSAMPLE of the video's resolution per axis.
DOWNSAMPLE = 4
#: Frames decoded per call when reading a whole episode.
CHUNK = 128


def grey_small(rgb: np.ndarray, k: int = DOWNSAMPLE) -> np.ndarray:
    """(N, H, W, 3) uint8 RGB -> (N, H//k, W//k) float32 grey, block-averaged."""
    x = np.asarray(rgb, dtype=np.float32).mean(axis=-1)
    n, h, w = x.shape
    h, w = h - h % k, w - w % k
    return x[:, :h, :w].reshape(n, h // k, k, w // k, k).mean(axis=(2, 4))


def video_delay(frames: np.ndarray, q: np.ndarray, max_lag: int = 10) -> tuple[float, np.ndarray]:
    """Frames by which the video lags the joint log (positive: video is late).

    ``frames`` (T, H, W) grey and ``q`` (T, J) joint angles, frame for frame.
    Picture change (the share of pixels that change by well over the sensor
    noise between frames) and joint speed ``sum |q[t] - q[t-1]|`` are
    cross-correlated over ``-max_lag..max_lag``; the peak is refined with a
    parabola through its neighbours. Returns ``(lag, correlation per lag)``.
    """
    f = np.asarray(frames, dtype=np.float32)
    d = np.abs(np.diff(f, axis=0))
    level = max(8.0, 4.0 * 1.4826 * float(np.median(d)))  # well above sensor noise
    motion = (d > level).mean(axis=(1, 2))
    speed = np.abs(np.diff(np.asarray(q, dtype=np.float64), axis=0)).sum(axis=1)
    m = (motion - motion.mean()) / (motion.std() + 1e-12)
    s = (speed - speed.mean()) / (speed.std() + 1e-12)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array(
        [np.mean(m[lag:] * s[: s.size - lag]) if lag >= 0 else np.mean(m[: m.size + lag] * s[-lag:]) for lag in lags]
    )
    i = int(np.argmax(corr))
    lag = float(lags[i])
    if 0 < i < len(corr) - 1:
        a, b, c = corr[i - 1], corr[i], corr[i + 1]
        den = a - 2 * b + c
        if den < 0:
            lag += 0.5 * (a - c) / den
    return lag, corr


@dataclass(frozen=True)
class CameraDelay:
    camera: str
    fps: float
    per_episode: dict[int, float]  # episode -> lag in frames
    peak_corr: dict[int, float]  # episode -> correlation at the peak

    @property
    def frames(self) -> float:
        return float(np.median(list(self.per_episode.values())))

    @property
    def ms(self) -> float:
        return 1000.0 * self.frames / self.fps

    def to_dict(self) -> dict:
        return {
            "camera": self.camera,
            "fps": self.fps,
            "delay_frames": self.frames,
            "delay_ms": self.ms,
            "per_episode": {str(k): v for k, v in self.per_episode.items()},
            "peak_corr": {str(k): v for k, v in self.peak_corr.items()},
        }


def episode_lag(episode: Episode, grey: np.ndarray, max_lag: int = 10) -> tuple[float, float]:
    """``(lag, peak correlation)`` for one episode's grey frames."""
    T = min(len(episode), len(grey))
    lag, corr = video_delay(grey[:T], episode.state[:T, :5], max_lag)
    return lag, float(corr.max())


def camera_delay(
    dataset: Dataset,
    camera: str,
    episodes: Sequence[int],
    max_lag: int = 10,
    frames_fn=None,
) -> CameraDelay:
    """Median video delay of ``camera`` over ``episodes`` of ``dataset``.

    ``frames_fn(episode, indices) -> (N,H,W,3) RGB`` defaults to decoding the
    dataset's own video with :func:`antares.video.frames` (no frame cache:
    every frame of the episode is read once). Frames are fetched ``CHUNK`` at
    a time and shrunk straight away, so a long episode never sits in memory
    at full resolution.
    """
    from antares import video

    if frames_fn is None:

        def frames_fn(ep: Episode, idx: list[int]) -> np.ndarray:
            return video.frames(dataset.repo_id, camera, ep.index, idx, dataset=dataset, cache=False)

    by_index = {e.index: e for e in dataset.episodes}
    lags, peaks = {}, {}
    for i in episodes:
        ep = by_index[int(i)]
        idx = list(range(len(ep)))
        grey = np.concatenate([grey_small(frames_fn(ep, idx[k : k + CHUNK])) for k in range(0, len(idx), CHUNK)])
        lags[ep.index], peaks[ep.index] = episode_lag(ep, grey, max_lag)
    if not lags:
        raise ValueError("no episodes")
    return CameraDelay(camera=camera, fps=float(dataset.fps), per_episode=lags, peak_corr=peaks)
