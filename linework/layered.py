"""Layered PSD: the photo, a hidden white "Paper" layer and the lines on top.

Same pixel size, resolution and colour profile as the photo, so it opens in
Photoshop ready to trace, and anything placed into it later lines up too.
"""

from __future__ import annotations

import struct

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.psd.image_resources import ImageResource

from . import postprocess

RESOLUTION_INFO = 1005  # Photoshop image resource ids
ICC_PROFILE = 1039


def _resolution(dpi: float) -> bytes:
    # hRes (16.16 fixed), hResUnit=ppi, widthUnit=inches, then the same for vertical
    fixed = int(round(dpi * 65536))
    return struct.pack(">IHHIHH", fixed, 1, 1, fixed, 1, 1)


def to_psd(rgb: np.ndarray, edges: np.ndarray, p: dict, dpi: float = 72.0, icc: bytes | None = None) -> PSDImage:
    """Lines are always rendered transparent at photo size, whatever the Output settings say."""
    p = {**postprocess.DEFAULTS, **p, "background": "transparent"}
    lines, _ = postprocess.render(rgb, edges, p, size="original")
    h, w = rgb.shape[:2]

    psd = PSDImage.new("RGBA", (w, h))  # RGBA so the lines get real layer transparency, not a mask
    res = psd._record.image_resources  # psd-tools has no public setter for these
    res[RESOLUTION_INFO] = ImageResource(key=RESOLUTION_INFO, data=_resolution(dpi))
    if icc:
        res[ICC_PROFILE] = ImageResource(key=ICC_PROFILE, data=icc)

    paper = PixelLayer.frompil(Image.new("RGB", (w, h), "white"), psd, "Paper")
    paper.visible = False
    for layer in (
        PixelLayer.frompil(Image.fromarray(rgb), psd, "Photo"),
        paper,
        PixelLayer.frompil(Image.fromarray(lines, "RGBA"), psd, "Lines"),
    ):
        psd.append(layer)
    psd._record.header.channels = 3  # plain RGB file, else Photoshop shows an extra "Alpha 1" channel
    return psd
