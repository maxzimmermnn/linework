"""Stage 1: photo -> raw edge strength map (the slow, model part). Cached."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .models import registry

try:  # iPhone HEIC support, optional
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif"}

# Parameters that change the model output (anything else is post-processing).
DETECT_KEYS = ("model", "detail", "texture_smoothing", "local_contrast", "canny_low", "canny_high")


def load_image(path_or_img) -> np.ndarray:
    """Any image file -> uint8 RGB, EXIF rotation applied, alpha flattened on white."""
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float32)
        arr = (arr / max(arr.max(), 1) * 255).astype(np.uint8)
        im = Image.fromarray(arr).convert("RGB")
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def image_info(path) -> dict:
    """Resolution (ppi, 72 if missing) and ICC profile of the file, for exports that keep them."""
    try:
        with Image.open(path) as im:
            dpi = float(im.info.get("dpi", (72, 72))[0] or 72)
            return {"dpi": dpi if dpi > 1 else 72.0, "icc": im.info.get("icc_profile")}
    except Exception:
        return {"dpi": 72.0, "icc": None}


def resize_short_side(rgb: np.ndarray, short: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    k = short / min(h, w)
    nh, nw = max(8, round(h * k)), max(8, round(w * k))
    interp = cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, (nw, nh), interpolation=interp)


def prepare_input(rgb: np.ndarray, p: dict) -> np.ndarray:
    """Resize to detail resolution and apply optional input clean-up."""
    x = resize_short_side(rgb, int(p["detail"]))
    s = float(p.get("texture_smoothing", 0))
    if s > 0:
        # Edge-preserving smoothing: flattens skin, grain and fabric, keeps contours.
        d = max(3, int(s * 2) | 1)
        for _ in range(1 + int(s // 4)):
            x = cv2.bilateralFilter(x, d, sigmaColor=10 + s * 12, sigmaSpace=1 + s)
    c = float(p.get("local_contrast", 0))
    if c > 0:
        lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
        tiles = max(2, min(16, int(min(x.shape[:2]) / 96)))
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=c, tileGridSize=(tiles, tiles)).apply(lab[:, :, 0])
        x = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return x


class EdgeCache:
    """Small LRU so moving post-processing sliders never re-runs the model. Bounded by entries
    and by bytes: one edge map at Detail 4096 is ~80 MB."""

    def __init__(self, size: int = 12, max_bytes: int | None = None):
        self.size = size
        self.max_bytes = max_bytes if max_bytes is not None else int(registry.MEMORY_GB * 2**30 / 32)
        self.data: OrderedDict = OrderedDict()

    def key(self, image_id: str, p: dict) -> str:
        sub = {k: p.get(k) for k in DETECT_KEYS}
        if p.get("model") != "canny":
            sub.pop("canny_low"), sub.pop("canny_high")
        return image_id + repr(sorted(sub.items()))

    def get(self, k):
        if k in self.data:
            self.data.move_to_end(k)
            return self.data[k]

    def put(self, k, v):
        self.data[k] = v
        self.data.move_to_end(k)
        while len(self.data) > 1 and (
            len(self.data) > self.size or sum(a.nbytes for a in self.data.values()) > self.max_bytes
        ):
            self.data.popitem(last=False)


CACHE = EdgeCache()


def image_id(rgb: np.ndarray) -> str:
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    return hashlib.sha1(small.tobytes() + str(rgb.shape).encode()).hexdigest()


_LOCK = threading.Lock()  # the UI and the watch folder thread share the model and the cache


def detect(rgb: np.ndarray, p: dict, progress=None, img_id: str | None = None) -> np.ndarray:
    """Returns float32 edge map at detail resolution (0..1, 1 = line)."""
    k = CACHE.key(img_id or image_id(rgb), p)
    with _LOCK:
        hit = CACHE.get(k)
        if hit is not None:
            return hit
        x = prepare_input(rgb, p)
        edges = registry.run_model(p["model"], x, p, progress)
        CACHE.put(k, edges)
        return edges


def output_name(src: str | Path | None, model: str) -> str:
    stem = Path(src).stem if src else "image"
    return f"{stem}_lines_{model}.png"
