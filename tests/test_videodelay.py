import numpy as np
import pytest

from antares.data import Dataset, Episode
from antares.videodelay import CameraDelay, camera_delay, grey_small, video_delay


def _blob_video(q, delay, H=48, W=64, seed=0):
    """A bright square that follows joints 0 (x) and 1 (y), ``delay`` frames late."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(40, 120, (H, W)).astype(np.float32)
    out = []
    for t in range(len(q)):
        f = bg.copy()
        x = int((q[max(0, t - delay), 0] + 1) / 2 * (W - 8))
        y = int((q[max(0, t - delay), 1] + 1) / 2 * (H - 8))
        f[y : y + 8, x : x + 8] = 220
        out.append(np.clip(f + rng.normal(0, 3, f.shape), 0, 255))
    return np.asarray(out, dtype=np.uint8)


def _jerky(T=240, seed=0):
    rng = np.random.default_rng(seed)
    q, cur = np.zeros((T, 6)), np.zeros(6)
    for s in range(0, T, 30):
        tgt = rng.uniform(-0.9, 0.9, 6)
        for k in range(min(30, T - s)):
            a = min(1.0, k / 10)
            q[s + k] = cur + (tgt - cur) * (3 * a * a - 2 * a**3)
        cur = tgt
    q[:, 2:] = 0.0  # only joints 0 and 1 move the blob
    return q


@pytest.mark.parametrize("delay", [0, 3])
def test_video_delay_finds_the_lag(delay):
    q = _jerky()
    lag, corr = video_delay(_blob_video(q, delay), q)
    assert lag == pytest.approx(delay, abs=0.5)
    assert corr.shape == (21,)


def test_grey_small_block_averages():
    rgb = np.zeros((2, 9, 10, 3), dtype=np.uint8)
    rgb[:, :4, :4] = 200
    g = grey_small(rgb, k=4)
    assert g.shape == (2, 2, 2)
    assert g[0, 0, 0] == pytest.approx(200.0)
    assert g[0, 1, 1] == pytest.approx(0.0)


def test_camera_delay_takes_the_median_over_episodes():
    eps, videos = [], {}
    for i, d in enumerate([2, 2, 5]):
        q = _jerky(seed=i)
        eps.append(Episode(index=i, fps=30.0, joint_names=tuple("abcdef"), state=q, action=q))
        grey = _blob_video(q, d, seed=i)
        videos[i] = np.repeat(np.repeat(grey, 4, axis=1), 4, axis=2)[..., None].repeat(3, axis=-1)
    ds = Dataset(repo_id="x", robot_type="so101", fps=30.0, joint_names=tuple("abcdef"), episodes=tuple(eps))
    calls = []

    def frames_fn(ep, idx):
        calls.append(len(idx))
        return videos[ep.index][idx]

    res = camera_delay(ds, "front", [0, 1, 2], frames_fn=frames_fn)
    assert res.per_episode[2] == pytest.approx(5, abs=0.5)
    assert res.frames == pytest.approx(2, abs=0.5)
    assert res.ms == pytest.approx(1000 * res.frames / 30.0)
    assert max(calls) <= 128  # decoded in chunks
    d = res.to_dict()
    assert d["camera"] == "front" and set(d["per_episode"]) == {"0", "1", "2"}


def test_camera_delay_needs_episodes():
    ds = Dataset(repo_id="x", robot_type="so101", fps=30.0, joint_names=(), episodes=())
    with pytest.raises(ValueError):
        camera_delay(ds, "front", [], frames_fn=lambda ep, idx: None)
    assert CameraDelay("c", 30.0, {0: 3.0}, {0: 0.5}).ms == pytest.approx(100.0)
