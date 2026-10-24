"""Ink map -> SVG. Filled outlines of every stroke, so it opens as editable paths
in Illustrator and as a crisp layer in Photoshop.

Traces the 0.5 iso-line of the anti-aliased ink map (marching squares, sub-pixel),
simplifies it, then draws smooth cubic curves through the remaining points. The
viewBox is the ink map's own pixel grid, stretched to the output size, so the SVG
lines up with the photo exactly like the PNG does. Width/height are given in
inches at the photo's ppi: Photoshop reads "px" in SVGs as points (1/72 in),
which would make the drawing the wrong size when placed on the photo.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import find_contours

from . import postprocess

SIMPLIFY = 0.35  # max deviation when dropping points, in ink-map px (2x detail res)


def _smooth_path(pts: np.ndarray) -> str:
    """Closed Catmull-Rom spline through pts (N x 2) as SVG cubic Bezier segments."""
    p0 = np.roll(pts, 1, axis=0)
    p2 = np.roll(pts, -1, axis=0)
    p3 = np.roll(pts, -2, axis=0)
    c1 = pts + (p2 - p0) / 6
    c2 = p2 - (p3 - pts) / 6
    segs = np.hstack([c1, c2, p2]).round(2)
    body = " ".join("C{:g} {:g} {:g} {:g} {:g} {:g}".format(*s) for s in segs)
    return f"M{pts[0, 0]:.2f} {pts[0, 1]:.2f}{body}Z"


def trace(ink: np.ndarray, simplify: float = SIMPLIFY) -> list[str]:
    """ink (0..1, any resolution) -> list of closed SVG path strings in ink pixel coords."""
    padded = np.pad(ink.astype(np.float32), 1)  # closes contours that touch the border
    paths = []
    for c in find_contours(padded, 0.5):
        # (row, col) in padded index space -> (x, y) with pixel i covering [i, i+1]
        xy = c[:, ::-1] - 0.5
        poly = cv2.approxPolyDP(xy.astype(np.float32).reshape(-1, 1, 2), simplify, True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        paths.append(_smooth_path(poly.astype(np.float64)))
    return paths


def to_svg(ink: np.ndarray, size_wh: tuple[int, int], p: dict, dpi: float = 72.0) -> str:
    """Full SVG document. size_wh = output size in px, dpi = the photo's ppi; ink is the stylize() map."""
    p = {**postprocess.DEFAULTS, **p}
    h, w = ink.shape
    ow, oh = size_wh
    r, g, b = postprocess.hex_to_rgb(p["line_color"])
    bg = "" if p["background"] == "transparent" else f'<rect width="{w}" height="{h}" fill="#fff"/>\n'
    d = "\n".join(trace(ink))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ow / dpi:.5f}in" height="{oh / dpi:.5f}in" '
        f'viewBox="0 0 {w} {h}" preserveAspectRatio="none">\n'
        f"{bg}"
        f'<path fill="#{r:02x}{g:02x}{b:02x}" fill-rule="evenodd" d="{d}"/>\n'
        "</svg>\n"
    )

