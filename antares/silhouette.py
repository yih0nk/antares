"""The real arm's silhouette in a fixed camera, from the frames alone.

The camera does not move and the arm does, so the per-pixel median over
frames spread across several episodes is the background (table, wall,
objects that sit still); whatever differs from it is the arm, the object it
carries, and its shadow. Three refinements make that usable:

* **Contrast sign.** The arm is either dark on a light table (SO-101, black
  prints) or light on a dark one (SO-100, orange and white prints).
  :func:`arm_appearance` learns which from the moving pixels and the
  luminance test then ignores shadows, which have the other sign.
* **Colour model.** A Lab histogram of moving pixels against background
  pixels gives a per-pixel "looks like the arm" probability. It recovers the
  arm where background subtraction cannot see it: the static base and pan
  servo, and the rest pose the arm sits in long enough to leak into the
  median.
* **Static-arm region.** Colour alone would also pick up static clutter of
  the arm's colour (a cardboard box, a sticker). So colour-model pixels only
  count where they lie inside the *static arm* region (arm-coloured parts of
  the background that the moving arm touches in most frames, i.e. the base)
  or right next to moving pixels, and of the resulting connected components
  only those that overlap the moving arm survive.

Everything is OpenCV morphology and connected components; no learned
segmenter. The result is a per-frame uint8 mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

BINS = 16
L_MAX = 100.0
AB_RANGE = 64.0  # a and b clipped to [-64, 64)


def _cv2():
    import cv2

    return cv2


def to_lab(frames: np.ndarray) -> np.ndarray:
    """RGB uint8 (..,H,W,3) -> float32 CIE Lab with L in 0..100, a/b centred at 0."""
    cv2 = _cv2()
    frames = np.asarray(frames)
    single = frames.ndim == 3
    if single:
        frames = frames[None]
    out = np.empty(frames.shape, dtype=np.float32)
    for i, f in enumerate(frames):
        lab = cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 0] *= L_MAX / 255.0
        lab[..., 1:] -= 128.0
        out[i] = lab
    return out[0] if single else out


def background(frames: np.ndarray) -> np.ndarray:
    """Per-pixel median over frames (RGB uint8) -> RGB uint8 background."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[0] < 3:
        raise ValueError("need at least three frames (N,H,W,3) to build a background")
    return np.median(frames, axis=0).astype(np.uint8)


def _clean(mask: np.ndarray, open_px: int, close_px: int) -> np.ndarray:
    cv2 = _cv2()
    m = mask.astype(np.uint8)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    cv2 = _cv2()
    if px <= 0:
        return mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k)


def lab_bins(lab: np.ndarray) -> np.ndarray:
    """Lab float (..,3) -> flat histogram bin index (..,) in [0, BINS**3)."""
    l = np.clip(lab[..., 0] / L_MAX * BINS, 0, BINS - 1).astype(np.int64)
    a = np.clip((lab[..., 1] + AB_RANGE) / (2 * AB_RANGE) * BINS, 0, BINS - 1).astype(np.int64)
    b = np.clip((lab[..., 2] + AB_RANGE) / (2 * AB_RANGE) * BINS, 0, BINS - 1).astype(np.int64)
    return (l * BINS + a) * BINS + b


@dataclass(frozen=True)
class Appearance:
    """What the arm looks like, learned from the frames."""

    sign: int  # +1: arm brighter than its background, -1: darker
    arm_l: float  # median L of moving pixels
    bg_l: float  # median L of the background
    diff_l: float  # |dL| threshold for the moving test
    diff_c: float  # chroma distance threshold for the moving test
    p_arm: np.ndarray = field(default_factory=lambda: np.zeros(BINS**3, dtype=np.float32))  # P(arm | Lab bin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sign": self.sign,
            "arm_l": self.arm_l,
            "bg_l": self.bg_l,
            "diff_l": self.diff_l,
            "diff_c": self.diff_c,
            "p_arm": self.p_arm.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Appearance:
        return cls(
            sign=int(d["sign"]),
            arm_l=float(d["arm_l"]),
            bg_l=float(d["bg_l"]),
            diff_l=float(d["diff_l"]),
            diff_c=float(d["diff_c"]),
            p_arm=np.asarray(d.get("p_arm", np.zeros(BINS**3)), dtype=np.float32),
        )

    def moving(self, lab: np.ndarray, bg_lab: np.ndarray) -> np.ndarray:
        """Signed background difference: brighter/darker than the background on the arm's side, or a colour change."""
        dl = lab[..., 0] - bg_lab[..., 0]
        dc = np.linalg.norm(lab[..., 1:] - bg_lab[..., 1:], axis=-1)
        return (self.sign * dl > self.diff_l) | (dc > self.diff_c)

    def arm_coloured(self, lab: np.ndarray, p: float = 0.7) -> np.ndarray:
        return self.p_arm[lab_bins(lab)] > p


def arm_appearance(
    frames: np.ndarray, bg: np.ndarray, diff_l: float = 25.0, diff_c: float = 18.0
) -> Appearance:
    """Learn the contrast sign and the colour model from a stack of frames."""
    lab = to_lab(frames)
    bg_lab = to_lab(bg)
    dl = lab[..., 0] - bg_lab[None, ..., 0]
    dc = np.linalg.norm(lab[..., 1:] - bg_lab[None, ..., 1:], axis=-1)
    moving = (np.abs(dl) > diff_l) | (dc > diff_c)
    bg_l = float(np.median(bg_lab[..., 0]))
    if moving.sum() < 50:
        return Appearance(sign=-1, arm_l=bg_l, bg_l=bg_l, diff_l=diff_l, diff_c=diff_c)
    arm_l = float(np.median(lab[..., 0][moving]))
    sign = 1 if arm_l >= bg_l else -1
    # Re-run the moving test with the sign so shadows do not train the arm colour.
    moving = (sign * dl > diff_l) | (dc > diff_c)
    bins = lab_bins(lab)
    h_arm = np.bincount(bins[moving], minlength=BINS**3).astype(np.float64)
    h_bg = np.bincount(bins[~moving], minlength=BINS**3).astype(np.float64)
    h_bg += np.bincount(lab_bins(bg_lab).ravel(), minlength=BINS**3)
    # Equal priors: the question is "is this colour more arm than background".
    h_arm /= max(1.0, h_arm.sum())
    h_bg /= max(1.0, h_bg.sum())
    eps = 1e-6
    p_arm = ((h_arm + eps) / (h_arm + h_bg + 2 * eps)).astype(np.float32)
    # Colours never seen at all are not evidence of anything.
    p_arm[(h_arm == 0) & (h_bg == 0)] = 0.0
    return Appearance(sign=sign, arm_l=arm_l, bg_l=bg_l, diff_l=diff_l, diff_c=diff_c, p_arm=p_arm)


@dataclass(frozen=True)
class BackgroundModel:
    bg: np.ndarray  # (H,W,3) RGB uint8
    appearance: Appearance
    static_arm: np.ndarray  # (H,W) uint8: arm-coloured background the moving arm touches most of the time

    @classmethod
    def fit(
        cls,
        frames: np.ndarray,
        open_px: int = 3,
        close_px: int = 9,
        touch_frac: float = 0.5,
        touch_px: int = 6,
    ) -> BackgroundModel:
        """Median background, colour model, and the static-arm region.

        The static-arm region is every connected component of the
        arm-coloured background that the (dilated) moving mask overlaps in at
        least ``touch_frac`` of the frames: the base is touched by the arm in
        every frame, a same-coloured box only when the arm passes over it.
        """
        cv2 = _cv2()
        bg = background(frames)
        ap = arm_appearance(frames, bg)
        bg_lab = to_lab(bg)
        prior_bg = _clean(ap.arm_coloured(bg_lab), open_px, close_px)
        n, labels, _, _ = cv2.connectedComponentsWithStats(prior_bg, connectivity=8)
        touched = np.zeros(n, dtype=np.int64)
        for f in frames:
            mv = _dilate(_clean(ap.moving(to_lab(f), bg_lab), open_px, close_px), touch_px).astype(bool)
            touched += np.bincount(labels[mv], minlength=n) > 0
        touched[0] = 0
        keep = touched >= touch_frac * len(frames)
        static_arm = keep[labels].astype(np.uint8)
        return cls(bg=bg, appearance=ap, static_arm=static_arm)

    def mask(self, frame: np.ndarray, **kw: Any) -> np.ndarray:
        return arm_mask(frame, self.bg, self.appearance, self.static_arm, **kw)

    def to_dict(self) -> dict[str, Any]:
        return {"appearance": self.appearance.to_dict()}


def arm_mask(
    frame: np.ndarray,
    bg: np.ndarray,
    appearance: Appearance,
    static_arm: np.ndarray | None = None,
    open_px: int = 3,
    close_px: int = 9,
    near_px: int = 12,
    min_component_frac: float = 0.002,
    use_prior: bool = True,
) -> np.ndarray:
    """One frame -> (H,W) uint8 mask of the arm.

    ``moving`` is the signed background difference. With ``use_prior``,
    arm-coloured pixels that lie in the static-arm region or within
    ``near_px`` of moving pixels are added. Of the connected components of
    the result, the one holding most moving pixels is the arm, and any other
    component that is mostly moving and not tiny is kept too (a jaw cut off
    by occlusion).
    """
    cv2 = _cv2()
    ap = appearance
    lab = to_lab(frame)
    bg_lab = to_lab(bg)
    moving = _clean(ap.moving(lab, bg_lab), open_px, close_px)
    if use_prior:
        prior = _clean(ap.arm_coloured(lab), open_px, close_px).astype(bool)
        allowed = _dilate(moving, near_px).astype(bool)
        if static_arm is not None:
            allowed |= static_arm.astype(bool)
        union = np.maximum(moving, (prior & allowed).astype(np.uint8))
        union = _clean(union, 0, close_px)
    else:
        union = moving
    n, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    if n <= 1:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    moving_count = np.bincount(labels[moving.astype(bool)], minlength=n)
    moving_count[0] = 0
    keep = np.zeros(n, dtype=bool)
    best = int(np.argmax(moving_count))
    if moving_count[best] > 0:
        keep[best] = True
    min_area = min_component_frac * frame.shape[0] * frame.shape[1]
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if lbl != best and area >= min_area and moving_count[lbl] >= 0.5 * area:
            keep[lbl] = True
    return keep[labels].astype(np.uint8)


def masks(frames: np.ndarray, model: BackgroundModel, **kw: Any) -> np.ndarray:
    return np.stack([model.mask(f, **kw) for f in frames])


def mask_polygons(mask: np.ndarray, epsilon_px: float = 1.5, min_area: float = 20.0) -> list[list[list[float]]]:
    """Outer contours of a mask as simplified polygons (for drawing in the UI)."""
    cv2 = _cv2()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, epsilon_px, True).reshape(-1, 2)
        polys.append([[float(x), float(y)] for x, y in approx])
    return polys


def overlay(frame: np.ndarray, mask: np.ndarray, color: Sequence[int] = (91, 140, 255), alpha: float = 0.45) -> np.ndarray:
    """Tint ``mask`` over ``frame`` (RGB uint8) for a quick visual check."""
    out = frame.astype(np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.asarray(color, dtype=np.float32)
    return out.astype(np.uint8)


__all__ = [
    "Appearance",
    "BackgroundModel",
    "arm_appearance",
    "arm_mask",
    "background",
    "lab_bins",
    "mask_polygons",
    "masks",
    "overlay",
    "to_lab",
]
