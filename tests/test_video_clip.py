"""One episode cut out of a dataset video and re-encoded for the browser."""

from __future__ import annotations

import numpy as np
import pytest

av = pytest.importorskip("av")

from antares import video  # noqa: E402

FPS = 30.0
LENGTHS = [12, 15]  # two episodes back to back in one file


def _source(path, W=66, H=50, dropped=()):
    """A file whose frame k is filled with grey level 8*k, so a decoded frame
    names its index; frames in ``dropped`` are missing (a timestamp gap)."""
    from fractions import Fraction

    with av.open(str(path), "w") as out:
        s = out.add_stream("libx264", rate=int(FPS))
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"crf": "0", "preset": "ultrafast"}
        for k in range(sum(LENGTHS)):
            if k in dropped:
                continue
            f = av.VideoFrame.from_ndarray(np.full((H, W, 3), 8 * k, np.uint8), format="rgb24")
            f.pts, f.time_base = k, Fraction(1, int(FPS))
            for p in s.encode(f):
                out.mux(p)
        for p in s.encode():
            out.mux(p)


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    src = tmp_path / "src.mp4"
    _source(src)
    monkeypatch.setattr(video, "CLIP_CACHE", tmp_path / "clips")
    monkeypatch.setattr(video, "episode_segments", lambda repo, cam, ds=None: video.segments_from_episodes(list(enumerate(LENGTHS)), FPS))
    monkeypatch.setattr(video, "find_camera", lambda repo, cam: video.CameraInfo("up", video.camera_key("up"), 66, 50, FPS, True))
    monkeypatch.setattr(video, "video_file", lambda repo, cam, chunk=0, file=0: src)
    return tmp_path


def _grey_levels(path):
    with av.open(str(path)) as c:
        return [float(f.to_ndarray(format="rgb24").mean()) for f in c.decode(video=0)]


@pytest.mark.parametrize("fmt,codec,tol", [("mp4", "h264", 3), ("webm", "vp9", 8)])
def test_clip_starts_at_the_episodes_first_frame(stubbed, fmt, codec, tol):
    clip = video.episode_clip("fake/ds", "up", 1, fmt=fmt)
    assert clip.suffix == f".{fmt}"
    levels = _grey_levels(clip)
    assert len(levels) == LENGTHS[1]
    want = [8 * (LENGTHS[0] + i) for i in range(LENGTHS[1])]
    assert np.allclose(levels, want, atol=tol)
    with av.open(str(clip)) as c:
        st = c.streams.video[0]
        assert st.codec_context.name == codec and (st.width, st.height) == (66, 50)


def test_clip_is_cached(stubbed):
    a = video.episode_clip("fake/ds", "up", 0)
    mtime = a.stat().st_mtime_ns
    assert video.episode_clip("fake/ds", "up", 0) == a and a.stat().st_mtime_ns == mtime
    assert not list(a.parent.glob("*.tmp.mp4"))


def test_datasets_with_the_same_name_get_their_own_clips(stubbed):
    a = video.episode_clip("alice/so100_test", "up", 0)
    b = video.episode_clip("bob/so100_test", "up", 0)
    assert a != b and a.is_file() and b.is_file()


def test_unknown_episode_or_format_is_refused(stubbed):
    with pytest.raises(video.VideoUnavailable):
        video.episode_clip("fake/ds", "up", 7)
    with pytest.raises(ValueError):
        video.episode_clip("fake/ds", "up", 0, fmt="gif")
