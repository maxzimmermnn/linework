"""Stage 2: raw edge map -> finished line drawing. Fast, runs on every slider move.

All pixel-sized parameters are in *detail-resolution pixels* so a setting looks
the same in the preview and in the full-size export. Internally we work at 2x
detail resolution (supersampling) for smooth, anti-aliased strokes.
"""

from __future__ import annotations

import cv2
import numpy as np

SUPERSAMPLE = 2

DEFAULTS = {
    # stage 1 (model)
    "model": "lineart",
    "detail": 1024,             # short side the model sees, px
    "texture_smoothing": 0.0,   # 0..10 edge-preserving blur before detection
    "local_contrast": 0.0,      # 0..4 CLAHE before detection
    "canny_low": 80,
    "canny_high": 180,
    "shadow_level": 0.5,        # 0.2..0.9 Shadow shapes: higher = more black
    # stage 2 (styling)
    "faint_cutoff": 0.15,       # 0..0.9  edges weaker than this disappear
    "solid_at": 0.75,           # 0.1..1  edges stronger than this are fully black
    "darkness": 1.0,            # 0.3..3  >1 darkens mid-tone lines (gamma)
    "style": "soft",            # soft | hard | pen
    "threshold": 0.5,           # 0.05..0.95  hard/pen: where a line starts
    "close_gaps": 0.0,          # 0..4 px  join nearly touching strokes
    "despeckle": 0,             # 0..2000  remove blobs smaller than this (px² at 1000px)
    "line_weight": 0.0,         # -2..4 px  thin (-) or thicken (+) strokes (soft/hard)
    "pen_width": 2.0,           # 0.5..10 px  stroke width in pen mode
    "smoothing": 0.5,           # 0..4 px  rounds jagged stroke edges
    "output_size": "original",  # original | detail
    "background": "white",      # white | transparent
    "line_color": "#000000",
}


def _disk(r: float) -> np.ndarray:
    k = max(1, int(round(r)) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _remove_specks(ink: np.ndarray, min_area: float) -> np.ndarray:
    if min_area < 1:
        return ink
    mask = (ink > 0.08).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return ink * keep[labels]


def _round_off(ink: np.ndarray, sigma: float, crisp: bool) -> np.ndarray:
    if sigma <= 0:
        return ink
    ink = cv2.GaussianBlur(ink, (0, 0), sigma)
    if crisp:  # re-threshold the blurred mask: rounder strokes, ~1px anti-aliased edge
        ink = ((ink - 0.5) * max(1.0, 2.5 * sigma) + 0.5).clip(0, 1)
    return ink


def stylize(edges: np.ndarray, p: dict, scale: float = SUPERSAMPLE) -> np.ndarray:
    """edges (detail res, 0..1) -> ink coverage map at scale x detail res.

    Exports use SUPERSAMPLE. The live preview passes a smaller scale (>= 1) so it works at
    about the size it is shown: same strokes, much less work at high Detail."""
    p = {**DEFAULTS, **p}
    s = scale
    h, w = edges.shape
    e = edges if s == 1 else cv2.resize(edges, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC).clip(0, 1)

    lo, hi = float(p["faint_cutoff"]), float(p["solid_at"])
    hi = max(hi, lo + 0.01)
    ink = ((e - lo) / (hi - lo)).clip(0, 1)
    if p["darkness"] != 1:
        ink = ink ** (1.0 / float(p["darkness"]))

    style = p["style"]
    if style in ("hard", "pen"):
        ink = (ink >= float(p["threshold"])).astype(np.float32)

    if p["close_gaps"] > 0:
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, _disk(p["close_gaps"] * s))

    ref = (min(ink.shape) / 1000.0) ** 2
    ink = _remove_specks(ink, float(p["despeckle"]) * ref)

    if style == "pen":
        from skimage.morphology import skeletonize

        skel = skeletonize(ink > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(1 - skel, cv2.DIST_L2, 5)
        radius = max(0.5, float(p["pen_width"]) * s / 2)
        ink = (radius + 0.5 - dist).clip(0, 1).astype(np.float32)
    else:
        lw = float(p["line_weight"]) * s  # radius change in work px (slider = detail px per side)
        if abs(lw) >= 0.25:
            if style == "hard":
                m = (ink > 0.5).astype(np.uint8)
                if lw > 0:
                    d = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
                    ink = (lw + 0.5 - d).clip(0, 1).astype(np.float32)
                else:
                    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
                    ink = (d - (-lw) + 0.5).clip(0, 1).astype(np.float32)
            else:
                op = cv2.dilate if lw > 0 else cv2.erode
                ink = op(ink, _disk(abs(lw)))

    ink = _round_off(ink, float(p["smoothing"]) * s / 2, crisp=style in ("hard", "pen"))
    return ink.astype(np.float32)


def resize_ink(ink: np.ndarray, size_wh: tuple[int, int], crisp: bool) -> np.ndarray:
    w, h = size_wh
    if (w, h) == (ink.shape[1], ink.shape[0]):
        return ink
    scale = w / ink.shape[1]
    if scale < 1:
        return cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA)
    out = cv2.resize(ink, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    if crisp and scale > 1.5:  # keep hard strokes hard after enlarging
        out = ((out - 0.5) * scale + 0.5).clip(0, 1)
    return out


def hex_to_rgb(c: str) -> tuple[int, int, int]:
    c = (c or "#000000").strip()
    if c.startswith("rgb"):
        vals = [float(v) for v in c[c.index("(") + 1 : c.index(")")].split(",")[:3]]
        return tuple(int(round(v)) for v in vals)
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))


def compose(ink: np.ndarray, p: dict) -> np.ndarray:
    """ink 0..1 -> uint8 RGB (white paper) or RGBA (transparent, lines only)."""
    p = {**DEFAULTS, **p}
    col = np.array(hex_to_rgb(p["line_color"]), dtype=np.float32)
    a = ink[..., None]
    if p["background"] == "transparent":
        rgb = np.broadcast_to(col, ink.shape + (3,))
        return np.dstack([rgb, ink * 255]).round().astype(np.uint8)
    return (255 * (1 - a) + col * a).round().clip(0, 255).astype(np.uint8)


def overlay_on_photo(rgb_photo: np.ndarray, ink: np.ndarray, p: dict, photo_opacity=0.35) -> np.ndarray:
    """Preview helper: lines drawn over a faded version of the photo."""
    h, w = ink.shape
    photo = cv2.resize(rgb_photo, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    faded = 255 - (255 - photo) * photo_opacity
    col = np.array(hex_to_rgb(p.get("line_color", "#000000")), dtype=np.float32)
    a = ink[..., None]
    return (faded * (1 - a) + col * a).clip(0, 255).astype(np.uint8)


def output_wh(rgb: np.ndarray, edges: np.ndarray, size: str) -> tuple[int, int]:
    """Output (w, h): the photo's size for "original", else detail resolution."""
    if size == "original":
        return rgb.shape[1], rgb.shape[0]
    return edges.shape[1], edges.shape[0]


def render(rgb: np.ndarray, edges: np.ndarray, p: dict, size: str | None = None):
    """Full stage 2. Returns (image uint8, ink float) at the requested output size."""
    p = {**DEFAULTS, **p}
    ink = stylize(edges, p)
    wh = output_wh(rgb, edges, size or p["output_size"])
    ink = resize_ink(ink, wh, crisp=p["style"] in ("hard", "pen"))
    return compose(ink, p), ink
