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
    monkeypatch.setattr(video, "_CLIP_INDEX", {})
    monkeypatch.setattr(video, "_CLIP_LOCKS", {})
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


def test_dropped_source_frames_take_the_next_frame_like_the_camera_panel(stubbed, monkeypatch):
    # Episode 1's own frame 0 is missing, and so are frames 4-6: each wanted
    # frame gets the next decoded frame (decode_at's choice, which the Camera
    # panel shows), never the previous episode's last frame.
    src = stubbed / "gappy.mp4"
    first = LENGTHS[0]
    _source(src, dropped={first, first + 4, first + 5, first + 6})
    monkeypatch.setattr(video, "video_file", lambda repo, cam, chunk=0, file=0: src)
    levels = _grey_levels(video.episode_clip("fake/ds", "up", 1))
    want = [8 * (first + (1 if i == 0 else 7 if 4 <= i <= 6 else i)) for i in range(LENGTHS[1])]
    assert len(levels) == LENGTHS[1] and np.allclose(levels, want, atol=3)
    panel = video.decode_at(src, [first / FPS + i / FPS for i in range(LENGTHS[1])], FPS)
    assert np.allclose([f.mean() for f in panel], levels, atol=3)


def test_clip_metadata_is_read_once_per_clip(stubbed, monkeypatch):
    reads = {"segments": 0, "camera": 0}
    segments, camera = video.episode_segments, video.find_camera

    def count_segments(*a, **k):
        reads["segments"] += 1
        return segments(*a, **k)

    def count_camera(*a, **k):
        reads["camera"] += 1
        return camera(*a, **k)

    monkeypatch.setattr(video, "episode_segments", count_segments)
    monkeypatch.setattr(video, "find_camera", count_camera)
    clip = video.episode_clip("fake/ds", "up", 0)
    assert reads == {"segments": 1, "camera": 1}
    # Every byte-range request of the browser is another call.
    for _ in range(3):
        assert video.episode_clip("fake/ds", "up", 0) == clip
    assert reads == {"segments": 1, "camera": 1}
    # A re-recorded source (new size or mtime) is noticed and re-resolved.
    src = stubbed / "src.mp4"
    _source(src, dropped={0})
    assert video.episode_clip("fake/ds", "up", 0) != clip
    assert reads == {"segments": 2, "camera": 2}


def test_concurrent_requests_encode_a_clip_once(stubbed, monkeypatch):
    import threading
    import time

    encode = video._encode_clip
    calls = []

    def slow_encode(*a, **k):
        calls.append(threading.get_ident())
        time.sleep(0.3)  # long enough for the second request to arrive
        return encode(*a, **k)

    monkeypatch.setattr(video, "_encode_clip", slow_encode)
    results, errors = [], []

    def request():
        try:
            results.append(video.episode_clip("fake/ds", "up", 1))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=request) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 2 and results[0] == results[1] and results[0].is_file()
    assert len(calls) == 1
    assert not list(results[0].parent.glob("*.tmp.mp4"))


def test_unreadable_source_is_a_decode_error(stubbed, monkeypatch):
    src = stubbed / "garbage.mp4"
    src.write_bytes(b"not a video" * 100)
    monkeypatch.setattr(video, "video_file", lambda repo, cam, chunk=0, file=0: src)
    with pytest.raises(video.VideoDecodeError, match="could not re-encode"):
        video.episode_clip("fake/ds", "up", 0)
    assert not list((stubbed / "clips").rglob("*.mp4"))


def test_datasets_with_the_same_name_get_their_own_clips(stubbed):
    a = video.episode_clip("alice/so100_test", "up", 0)
    b = video.episode_clip("bob/so100_test", "up", 0)
    assert a != b and a.is_file() and b.is_file()


def test_unknown_episode_or_format_is_refused(stubbed):
    with pytest.raises(video.VideoNotFound):
        video.episode_clip("fake/ds", "up", 7)
    with pytest.raises(ValueError):
        video.episode_clip("fake/ds", "up", 0, fmt="gif")
