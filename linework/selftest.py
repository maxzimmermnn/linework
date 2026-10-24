"""Download every model, run it on a sample photo, write a contact sheet.

    python -m linework.selftest            # all models
    python -m linework.selftest lineart hed

Result: output/selftest.png plus a pass/fail line per model. A model "passes"
when its edge map is neither empty nor solid (i.e. weights loaded correctly).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import detect, postprocess
from .models import registry


def sample_photo() -> np.ndarray:
    try:
        from skimage import data

        return data.astronaut()
    except Exception:  # synthetic fallback
        img = np.full((512, 512, 3), 230, np.uint8)
        cv2.circle(img, (256, 256), 150, (40, 60, 90), -1)
        cv2.rectangle(img, (60, 60), (200, 160), (180, 40, 40), -1)
        return img


def main(keys=None):
    keys = keys or list(registry.MODELS)
    rgb = sample_photo()
    tiles, ok_all = [], True
    print(f"device: {registry.DEVICE}")
    for k in keys:
        p = {**postprocess.DEFAULTS, "model": k, "detail": 768}
        t = time.time()
        try:
            e = detect.detect(rgb, p)
            frac = float((e > 0.5).mean())
            ok = 0.002 < frac < 0.5
            msg = f"line pixels {frac:.1%}"
            img, _ = postprocess.render(rgb, e, p, size="detail")
        except Exception as ex:
            ok, msg, img = False, f"ERROR {ex}", np.full((768, 768, 3), 255, np.uint8)
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {k:<17} {time.time() - t:5.1f}s  {msg}")
        tile = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
        tile = np.ascontiguousarray(tile[..., :3][..., ::-1])
        cv2.putText(tile, k, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
    out = Path(__file__).resolve().parents[1] / "output" / "selftest.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"contact sheet: {out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
