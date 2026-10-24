"""Command line / batch use.

    python -m linework.cli photo.jpg                     # -> photo_lines_lineart.png next to it
    python -m linework.cli ~/Shoot/*.jpg -o ~/Lines --preset "Clean outlines"
    python -m linework.cli folder/ --model teed --detail 1280 --style hard --transparent
    python -m linework.cli photo.jpg --style pen --svg      # vector paths for Illustrator
    python -m linework.cli photo.jpg --psd                  # photo + lines as Photoshop layers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from . import detect, layered, postprocess, presets, vector
from .models import registry

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif"}


def expand(inputs: list[str]) -> list[Path]:
    out = []
    for s in inputs:
        p = Path(s).expanduser()
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        elif p.exists():
            out.append(p)
        else:
            print(f"skip (not found): {p}", file=sys.stderr)
    return out


def process_file(path: Path, params: dict, out_dir: Path | None, fmt: str = "png") -> Path:
    rgb = detect.load_image(path)
    edges = detect.detect(rgb, params)
    dest = (out_dir or path.parent) / detect.output_name(path, params["model"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "svg":
        dest = dest.with_suffix(".svg")
        wh = postprocess.output_wh(rgb, edges, params["output_size"])
        svg = vector.to_svg(postprocess.stylize(edges, params), wh, params, detect.image_info(path)["dpi"])
        dest.write_text(svg, encoding="utf-8")
        return dest
    if fmt == "psd":
        dest = dest.with_suffix(".psd")
        layered.to_psd(rgb, edges, params, **detect.image_info(path)).save(dest)
        return dest
    img, _ = postprocess.render(rgb, edges, params)
    Image.fromarray(img).save(dest, optimize=False, compress_level=4)
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Turn photos into black line drawings.")
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("-o", "--out", help="output folder (default: next to each input)")
    ap.add_argument("--preset", help="preset name (see app) or path to a .json preset")
    ap.add_argument("--model", choices=list(registry.MODELS))
    ap.add_argument("--detail", type=int)
    ap.add_argument("--style", choices=["soft", "hard", "pen"])
    ap.add_argument("--transparent", action="store_true", help="lines on transparent background")
    ap.add_argument("--detail-size", action="store_true", help="output at detail resolution instead of original size")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--svg", action="store_true", help="save vector SVG instead of PNG (hard/pen styles)")
    fmt.add_argument("--psd", action="store_true", help="save a layered PSD (photo + lines) instead of PNG")
    a = ap.parse_args(argv)

    params = dict(postprocess.DEFAULTS)
    if a.preset:
        params.update(presets.load(a.preset))
    for k in ("model", "detail", "style"):
        if getattr(a, k) is not None:
            params[k] = getattr(a, k)
    if a.transparent:
        params["background"] = "transparent"
    if a.detail_size:
        params["output_size"] = "detail"

    if a.svg and params["style"] == "soft":
        ap.error("--svg needs --style hard or pen (SVG has no grey tones)")

    files = expand(a.inputs)
    out_dir = Path(a.out).expanduser() if a.out else None
    print(f"{len(files)} image(s), model={params['model']}, device={registry.DEVICE}")
    for i, f in enumerate(files, 1):
        dest = process_file(f, params, out_dir, "svg" if a.svg else "psd" if a.psd else "png")
        print(f"[{i}/{len(files)}] {f.name} -> {dest}")


if __name__ == "__main__":
    main()
