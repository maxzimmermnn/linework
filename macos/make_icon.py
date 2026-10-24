"""Draws the app icon: ink line art (hills and a sun) on a paper tile, macOS icon grid.

    .venv/bin/python macos/make_icon.py   # rewrites macos/AppIcon.icns (committed, used by build_app.sh)
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 4  # supersampling
N = 1024 * S
INK = (28, 26, 24, 255)


def stroke(d: ImageDraw.ImageDraw, pts, w0: float, w1: float):
    """Polyline whose width tapers from w0 to w1 (pen-like), with round joints and caps."""
    pts = np.asarray(pts, float) * S
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    for (x, y), t in zip(pts, seg / seg[-1]):
        r = (w0 + (w1 - w0) * t) * S / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=INK)
    for (a, b), t in zip(zip(pts[:-1], pts[1:]), seg[:-1] / seg[-1]):
        d.line([tuple(a), tuple(b)], fill=INK, width=int((w0 + (w1 - w0) * t) * S))


def curve(p0, p1, p2, p3, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def draw() -> Image.Image:
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))

    # tile on the standard grid (824 px with ~185 px corners in a 1024 canvas) plus a soft shadow
    box = [100 * S, 100 * S, 924 * S, 924 * S]
    shadow = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle([box[0], box[1] + 12 * S, box[2], box[3] + 12 * S], 185 * S, fill=(0, 0, 0, 90))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18 * S)))
    tile = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle(box, 185 * S, fill=(248, 245, 238, 255))
    img.alpha_composite(tile)

    d = ImageDraw.Draw(img)
    a = np.array
    # sun: an open circle, like a quick pen loop
    th = np.linspace(np.radians(-60), np.radians(265), 160)
    stroke(d, np.c_[640 + 92 * np.cos(th), 330 + 92 * np.sin(th)], 24, 14)
    # far hill, ending where the near hill hides it
    stroke(d, curve(a([190, 640]), a([280, 470]), a([400, 460]), a([461, 598])), 16, 24)
    # near hill, the main gesture
    stroke(d, curve(a([300, 760]), a([470, 520]), a([680, 470]), a([840, 700])), 34, 18)
    # ground line
    stroke(d, curve(a([210, 790]), a([400, 770]), a([620, 800]), a([820, 780])), 14, 22)

    return img.resize((1024, 1024), Image.LANCZOS)


def main():
    icon = draw()
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for s in (16, 32, 128, 256, 512):
            icon.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
            icon.resize((s * 2, s * 2), Image.LANCZOS).save(iconset / f"icon_{s}x{s}@2x.png")
        out = Path(__file__).with_name("AppIcon.icns")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
