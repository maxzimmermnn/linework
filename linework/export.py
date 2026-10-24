"""Write a finished drawing to disk as PNG, SVG or layered PSD. Shared by the app, batch and CLI."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import detect, layered, postprocess, vector

FORMATS = ("png", "svg", "psd")


def check(fmt: str, p: dict) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    if fmt == "svg" and {**postprocess.DEFAULTS, **p}["style"] == "soft":
        raise ValueError("SVG needs the Hard or Pen ink style (SVG has no grey tones).")


def free_path(dest: Path) -> Path:
    """dest, or dest with a time (then counter) suffix if that name is taken. Never overwrite."""
    if not dest.exists():
        return dest
    stamped = dest.with_name(f"{dest.stem}_{time.strftime('%H%M%S')}{dest.suffix}")
    n = 2
    while stamped.exists():
        stamped = dest.with_name(f"{dest.stem}_{time.strftime('%H%M%S')}_{n}{dest.suffix}")
        n += 1
    return stamped


def save(fmt: str, src, rgb: np.ndarray, edges: np.ndarray, p: dict, folder: Path) -> tuple[Path, tuple[int, int]]:
    """Render and write one drawing into folder. Returns (path, (w, h) in px)."""
    p = {**postprocess.DEFAULTS, **p}
    check(fmt, p)
    folder.mkdir(parents=True, exist_ok=True)
    dest = free_path(folder / Path(detect.output_name(src, p["model"])).with_suffix(f".{fmt}"))
    if fmt == "png":
        img, _ = postprocess.render(rgb, edges, p)
        Image.fromarray(img).save(dest, compress_level=4)
        return dest, (img.shape[1], img.shape[0])
    if fmt == "svg":
        wh = postprocess.output_wh(rgb, edges, p["output_size"])
        dest.write_text(vector.to_svg(postprocess.stylize(edges, p), wh, p, detect.image_info(src)["dpi"]), encoding="utf-8")
        return dest, wh
    layered.to_psd(rgb, edges, p, **detect.image_info(src)).save(dest)  # always photo size
    return dest, (rgb.shape[1], rgb.shape[0])
