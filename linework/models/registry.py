"""Model catalogue, weight download and loading.

Every detector returns an *edge strength* map: float32, HxW, 0 = nothing,
1 = strongest line. All styling happens later in linework.postprocess.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported MPS ops fall back to CPU

import ssl
import sys
import warnings

warnings.filterwarnings("ignore", message=r".*torch\.jit.*")
import json
import logging
import struct
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

logging.getLogger("dinov2").setLevel(logging.ERROR)  # vendored DINOv2: "xFormers not available"

WEIGHTS_DIR = Path(os.environ.get("LINEWORK_WEIGHTS", Path(__file__).resolve().parents[2] / "weights"))

HF = "https://huggingface.co"
GH = "https://raw.githubusercontent.com"


@dataclass
class ModelInfo:
    key: str
    label: str
    blurb: str
    filename: str | None = None          # local file name inside WEIGHTS_DIR
    urls: list[str] = field(default_factory=list)  # tried in order
    size_mb: float = 0


MODELS: dict[str, ModelInfo] = {
    m.key: m
    for m in [
        ModelInfo(
            "lineart", "Lineart (realistic)",
            "Closest to a hand inked drawing. Best starting point for photos.",
            "sk_model.pth", [f"{HF}/lllyasviel/Annotators/resolve/main/sk_model.pth"], 17,
        ),
        ModelInfo(
            "lineart_coarse", "Lineart (coarse)",
            "Bolder, simplified strokes. Drops fine texture.",
            "sk_model2.pth", [f"{HF}/lllyasviel/Annotators/resolve/main/sk_model2.pth"], 17,
        ),
        ModelInfo(
            "contour", "Contour",
            "Bold, even outlines like a hand inked contour drawing. Clean on faces and objects, busy on foliage.",
            "contour.safetensors", [
                f"{HF}/OzzyGT/contour/resolve/main/contour.safetensors",
                f"{HF}/spaces/njsaunders/informativedrawings/resolve/main/contour.pth",
            ], 17,
        ),
        ModelInfo(
            "anyline", "Anyline (MistoLine TEED)",
            "TEED retrained for line art (the Anyline ComfyUI node). Crisp contours plus inner detail.",
            "MTEED.pth", [f"{HF}/TheMistoAI/MistoLine/resolve/main/Anyline/MTEED.pth"], 0.3,
        ),
        ModelInfo(
            "teed", "TEED",
            "Tiny and fast. Thin, precise outlines, little texture.",
            "teed_biped_5.pth", [
                f"{GH}/xavysp/TEED/main/checkpoints/BIPED/5/5_model.pth",
                f"{HF}/fal-ai/teed/resolve/main/5_model.pth",
            ], 0.3,
        ),
        ModelInfo(
            "pidinet", "PiDiNet",
            "Clean, soft object contours. Ignores most surface detail.",
            "table5_pidinet.pth", [
                f"{GH}/hellozhuo/pidinet/master/trained_models/table5_pidinet.pth",
                f"{HF}/lllyasviel/Annotators/resolve/main/table5_pidinet.pth",
            ], 3,
        ),
        ModelInfo(
            "hed", "HED (soft)",
            "Thick, soft, painterly edges. Nice for loose sketch bases.",
            "ControlNetHED.pth", [f"{HF}/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"], 29,
        ),
        ModelInfo(
            "depth", "Depth outlines",
            "Only the outer shapes of people and objects, found from depth. Ignores texture, light and "
            "patterns. Little to see in close-ups and flat scenes.",
            "depth_anything_v2_vits.pth",
            [f"{HF}/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"], 99,
        ),
        ModelInfo(
            "lineart_standard", "Lineart standard (no AI)",
            "Classic filter that traces every dark line. Very detailed, instant, no download.",
        ),
        ModelInfo(
            "xdog", "Shadow shapes (no AI)",
            "Flat black shadow shapes, like a woodcut or stencil, for blocking in the darks. Instant, no download.",
        ),
        ModelInfo(
            "canny", "Canny (no AI)",
            "Classic edge filter. Thin and mechanical, reacts to texture. No download.",
        ),
    ]
}

DEFAULT_MODEL = "lineart"
FILTERS = ("canny", "lineart_standard", "xdog")      # classic image filters, no network
LINEART = ("lineart", "lineart_coarse", "contour")   # Informative Drawings generator, three trained styles


# ------------------------------------------------------------------ device --
def pick_device() -> torch.device:
    forced = os.environ.get("LINEWORK_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = pick_device()
# Half precision on Apple GPUs: 2-3x less memory and faster, drawings unchanged (IoU >= 0.997
# against float32 for every model). CPU stays float32, half convolutions are slow there.
HALF = DEVICE.type == "mps" and os.environ.get("LINEWORK_PRECISION", "half") != "full"


def _memory_gb() -> float:
    forced = os.environ.get("LINEWORK_MEMORY_GB")  # act like a Mac with this much RAM (testing)
    if forced:
        return float(forced)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 16.0


MEMORY_GB = _memory_gb()
# Highest Detail the UI offers. Lineart above ~3000 holds several GB of activations in RAM.
MAX_DETAIL = 4096 if MEMORY_GB >= 24 else 3072 if MEMORY_GB >= 12 else 2048


# ---------------------------------------------------------------- download --
def _ssl_context():
    try:
        import certifi  # python.org builds on macOS often lack system certs

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def weight_path(key: str) -> Path | None:
    info = MODELS[key]
    return WEIGHTS_DIR / info.filename if info.filename else None


def is_downloaded(key: str) -> bool:
    p = weight_path(key)
    return p is None or p.exists()


def download(key: str, progress: Callable[[float, str], None] | None = None) -> Path | None:
    info = MODELS[key]
    dest = weight_path(key)
    if dest is None or dest.exists():
        return dest
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in info.urls:
        tmp = dest.with_suffix(".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "linework/1.0"})
            with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress(done / total, f"Downloading {info.label} model")
            tmp.rename(dest)
            return dest
        except Exception as e:  # try next mirror
            errors.append(f"{url}: {e}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download weights for {info.label}:\n" + "\n".join(errors))


# ------------------------------------------------------------------ loading --
_loaded: dict[str, torch.nn.Module] = {}
_lock = threading.Lock()


_SAFETENSORS_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
                       "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8}


def _torch_load(path: Path) -> dict:
    """State dict from a PyTorch checkpoint or a .safetensors file (told apart by content, so
    mirrors in either format can share a file name). Neither path can run code from the file."""
    raw = path.read_bytes()
    if raw[8:9] != b"{":
        return torch.load(path, map_location="cpu", weights_only=True)
    n = struct.unpack("<Q", raw[:8])[0]  # safetensors: header length, JSON header, raw tensors
    header, base = json.loads(raw[8 : 8 + n]), 8 + n
    out = {}
    for k, v in header.items():
        if k != "__metadata__":
            a, b = v["data_offsets"]
            t = torch.frombuffer(bytearray(raw[base + a : base + b]), dtype=_SAFETENSORS_DTYPES[v["dtype"]])
            out[k] = t.reshape(v["shape"])
    return out


def load(key: str, progress=None) -> torch.nn.Module | None:
    if key in FILTERS:
        return None
    with _lock:
        if key in _loaded:
            return _loaded[key]
        path = download(key, progress)
        if key in LINEART:
            from .small_archs import LineartGenerator

            net = LineartGenerator(3, 1, 3)
            net.load_state_dict(_torch_load(path))
        elif key == "hed":
            from .small_archs import ControlNetHED_Apache2

            net = ControlNetHED_Apache2()
            net.load_state_dict(_torch_load(path))
        elif key in ("teed", "anyline"):
            from .teed.ted import TED

            net = TED()
            net.load_state_dict(_torch_load(path))
        elif key == "depth":
            from .depth_anything_v2.dpt import DepthAnythingV2

            net = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
            net.load_state_dict(_torch_load(path))
        elif key == "pidinet":
            from .pidinet_arch import pidinet

            net = pidinet()
            sd = _torch_load(path)
            sd = sd.get("state_dict", sd)
            net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
        else:
            raise KeyError(key)
        net = net.float().eval().to(DEVICE)
        if HALF:
            net = net.half()
        _loaded[key] = net
        return net


# ---------------------------------------------------------------- inference --
def _pad_to(img: np.ndarray, multiple: int):
    h, w = img.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return img, h, w


def _dtype(net: torch.nn.Module) -> torch.dtype:
    return next(net.parameters()).dtype


def _to_tensor(img: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(DEVICE, dtype)


def _sigmoid_mean(outs, h, w) -> np.ndarray:
    maps = [cv2.resize(o.detach().float().cpu().numpy()[0, 0], (w, h), interpolation=cv2.INTER_LINEAR) for o in outs]
    return (1.0 / (1.0 + np.exp(-np.mean(maps, axis=0)))).astype(np.float32)


def _lift_haze(e: np.ndarray) -> np.ndarray:
    # TEED, Anyline and HED leave an even grey haze on empty areas (TEED ~0.15-0.3).
    # Lift it to 0 so the same "Ignore faint lines" setting works for every model.
    # Runs on the whole stitched map, never per tile, or tiles would get different floors.
    floor = float(np.clip(np.percentile(e, 5), 0, 0.5))
    return ((e - floor) / (1 - floor)).clip(0, 1).astype(np.float32)


def _infer(key: str, net, rgb: np.ndarray) -> np.ndarray:
    """One network pass on rgb (uint8 HxWx3, any size) -> float32 HxW, 1 = line, before haze lifting."""
    img, h, w = _pad_to(rgb, 64)
    dt = _dtype(net)
    if key in LINEART:  # network draws white paper (1) with black ink (0)
        out = net(_to_tensor(img, dt) / 255.0)[0, 0].float().cpu().numpy()[:h, :w]
        return (1.0 - out).clip(0, 1).astype(np.float32)
    if key in ("hed", "teed", "anyline"):
        return _sigmoid_mean(net(_to_tensor(img, dt)), img.shape[0], img.shape[1])[:h, :w]
    if key == "pidinet":
        out = net(_to_tensor(img[:, :, ::-1].copy(), dt) / 255.0)[-1]  # BGR input
        return out[0, 0].float().cpu().numpy()[:h, :w].clip(0, 1).astype(np.float32)
    raise KeyError(key)


# ------------------------------------------------------------------ tiling --
# Above TILE_ABOVE pixels the model runs in pieces so memory stays flat. Untiled GPU use is
# ~0.5-0.9 GB per megapixel in half precision (1.5-2 in float32), and the GPU shares RAM with
# macOS and other apps, so tile above what a quarter of the RAM holds: ~4.4 MP (Detail ~1900)
# on a 16 GB Mac, 9 MP (the cap, Detail ~2600) from 32 GB. Both tiled paths reproduce the
# untiled result exactly (tests/test_pipeline.py).
TILE_ABOVE = int(min(9.0, MEMORY_GB * 0.25 / (0.9 if HALF else 2.0)) * 1e6)
TILE = 1536 if MEMORY_GB >= 12 else 1024
MARGIN = 192   # tile output this close to a cut edge differs (too little context, PiDiNet worst): dropped
OVERLAP = 448  # MARGIN dropped on each side + a 64 px crossfade; TILE - OVERLAP must be a multiple of 64


def _starts(n: int) -> list[int]:
    # Steps of TILE - OVERLAP (a multiple of 64) keep every tile on the networks' downsampling
    # grid; an offset that isn't would change the output everywhere in that tile. The last
    # tile is just shorter, and its edge padding then matches the untiled run exactly.
    starts = [0]
    while starts[-1] + TILE < n:
        starts.append(starts[-1] + TILE - OVERLAP)
    return starts


def _feather(h: int, w: int, top: bool, bottom: bool, left: bool, right: bool) -> np.ndarray:
    """Blend weight for one tile, on sides that have a neighbour: 0 for MARGIN px, then a
    linear ramp to 1. Neighbouring ramps sum to exactly 1 across the overlap."""
    def ramp(n, lo, hi):
        r = np.ones(n, np.float32)
        x = ((np.arange(OVERLAP, dtype=np.float32) + 0.5 - MARGIN) / (OVERLAP - 2 * MARGIN)).clip(0, 1)
        if lo:
            r[:OVERLAP] = x
        if hi:
            r[-OVERLAP:] = np.minimum(r[-OVERLAP:], x[::-1])
        return r
    return np.outer(ramp(h, top, bottom), ramp(w, left, right))


def _infer_tiled(key: str, net, rgb: np.ndarray, progress=None) -> np.ndarray:
    """Overlapping tiles, blended. Only for networks without whole-image normalisation."""
    h, w = rgb.shape[:2]
    ys, xs = _starts(h), _starts(w)
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    n, i = len(ys) * len(xs), 0
    for y in ys:
        for x in xs:
            i += 1
            if progress:
                progress(i / n, f"Finding lines (part {i} of {n})")
            th, tw = min(TILE, h - y), min(TILE, w - x)
            wt = _feather(th, tw, y > 0, y + th < h, x > 0, x + tw < w)
            acc[y : y + th, x : x + tw] += _infer(key, net, rgb[y : y + th, x : x + tw]) * wt
            wsum[y : y + th, x : x + tw] += wt
    return acc / np.maximum(wsum, 1e-6)


# Lineart normalises every layer over the whole image (InstanceNorm), so plain tiles
# come out with different line density. Instead run it layer by layer: each convolution
# is computed tile by tile on the GPU, the full-size activations live in RAM (in the
# network's precision, so half on Apple GPUs), and every norm sees the whole image.
# Smaller pieces keep peak RAM down on 8-16 GB Macs (Detail 3072: 4.9 GB instead of 7.4 GB)
# for ~20% more time.
STREAM_TILE = 1024 if MEMORY_GB >= 32 else 512  # in each layer's own pixels
NORM_CHANNELS = 16 if MEMORY_GB >= 32 else 8     # channels per float32 statistics chunk
STREAM_HALO = 8     # context around a tile, enough for the 7x7 convs; must be even


def _conv_streamed(mod: torch.nn.Module, x: torch.Tensor, scale: float) -> torch.Tensor:
    """mod (convs + padding, no norms) on CPU tensor x, tile by tile. scale = output/input size."""
    _, _, h, w = x.shape
    out = None
    for y0 in range(0, h, STREAM_TILE):
        for x0 in range(0, w, STREAM_TILE):
            y1, x1 = min(y0 + STREAM_TILE, h), min(x0 + STREAM_TILE, w)
            ys, xs = max(0, y0 - STREAM_HALO), max(0, x0 - STREAM_HALO)
            ye, xe = min(h, y1 + STREAM_HALO), min(w, x1 + STREAM_HALO)
            t = mod(x[:, :, ys:ye, xs:xe].to(DEVICE)).cpu()
            if out is None:
                out = torch.empty((1, t.shape[1], int(h * scale), int(w * scale)), dtype=t.dtype)
            oy, ox = int((y0 - ys) * scale), int((x0 - xs) * scale)
            th, tw = int((y1 - y0) * scale), int((x1 - x0) * scale)
            out[:, :, int(y0 * scale) : int(y0 * scale) + th, int(x0 * scale) : int(x0 * scale) + tw] = t[
                :, :, oy : oy + th, ox : ox + tw
            ]
    if DEVICE.type == "mps":
        torch.mps.empty_cache()  # GPU memory counts against the same RAM as the activations
    return out


def _norm_inplace(x: torch.Tensor, norm: torch.nn.InstanceNorm2d, relu: bool = True) -> torch.Tensor:
    # float32 statistics, a few channels at a time: exact sums, small temporary copies
    k = NORM_CHANNELS
    for c in range(0, x.shape[1], k):
        xc = x[:, c : c + k]
        f = xc.float()  # the same tensor when x is float32
        var, mean = torch.var_mean(f, dim=(2, 3), unbiased=False, keepdim=True)
        f.sub_(mean).div_(torch.sqrt(var + norm.eps))
        if norm.affine:
            f.mul_(norm.weight.detach().float().cpu()[None, c : c + k, None, None])
            f.add_(norm.bias.detach().float().cpu()[None, c : c + k, None, None])
        if relu:
            f.clamp_(min=0)
        if f.data_ptr() != xc.data_ptr():
            xc.copy_(f)
    return x


def _lineart_streamed(net, rgb: np.ndarray, progress=None) -> np.ndarray:
    img, h, w = _pad_to(rgb, 64)  # same padding as the untiled run, so the norm statistics match
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(_dtype(net)) / 255.0
    steps, done = 12, 0

    def step(mod, x, scale):
        nonlocal done
        done += 1
        if progress:
            progress(done / steps, f"Finding lines (step {done} of {steps})")
        return _conv_streamed(mod, x, scale)

    m0, m1, m3 = net.model0, net.model1, net.model3
    x = _norm_inplace(step(m0[:2], x, 1), m0[2])
    for i in (0, 3):  # two stride-2 downsampling convs
        x = _norm_inplace(step(m1[i], x, 0.5), m1[i + 1])
    for block in net.model2:  # residual blocks
        cb = block.conv_block
        y = _norm_inplace(step(cb[0:2], x, 1), cb[2])
        x += _norm_inplace(step(cb[4:6], y, 1), cb[6], relu=False)
        del y
    for i in (0, 3):  # two transposed convs back up to full size
        x = _norm_inplace(step(m3[i], x, 2), m3[i + 1])
    out = step(net.model4, x, 1)[0, 0].float().numpy()[:h, :w]
    return (1.0 - out).clip(0, 1).astype(np.float32)


# ------------------------------------------------------------ depth outlines --
DEPTH_RES = 770  # short side the depth network sees, whatever the Detail (it works on the whole image)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _thin(g: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Non-maximum suppression: keep only the crest of each edge across its gradient direction."""
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    out = np.zeros_like(g)
    for lo, hi, dy, dx in [(0, 22.5, 0, 1), (157.5, 180, 0, 1), (22.5, 67.5, 1, 1), (67.5, 112.5, 1, 0), (112.5, 157.5, 1, -1)]:
        m = (ang >= lo) & (ang < hi) & (g >= np.roll(g, (dy, dx), (0, 1))) & (g >= np.roll(g, (-dy, -dx), (0, 1)))
        out[m] = g[m]
    return out


def _depth_outlines(net, rgb: np.ndarray) -> np.ndarray:
    """Depth Anything V2 depth -> lines where the depth jumps (silhouettes, overlaps)."""
    h, w = rgb.shape[:2]
    s = DEPTH_RES / min(h, w)
    x = cv2.resize(rgb, (max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    t = torch.from_numpy(((x / 255.0 - _MEAN) / _STD).astype(np.float32).transpose(2, 0, 1))[None]
    d = net(t.to(DEVICE, _dtype(net)))[0].float().cpu().numpy()
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)  # relative inverse depth, 1 = near
    gx, gy = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    # Relative jump per 1/1000 of the image: the same lines at any Detail, distant edges count too.
    # Silhouettes measure ~1-6, flat surfaces ~0.01-0.04.
    g = np.hypot(gx, gy) / (d + 0.05) * (min(h, w) / 1000)
    g = np.minimum(_thin(cv2.GaussianBlur(g, (0, 0), 1.0), gx, gy), 1.0)
    g = cv2.dilate(g, np.ones((3, 3), np.uint8))  # 1 px crest -> a stroke of ~2-3 px
    return cv2.GaussianBlur(g, (0, 0), 0.7).clip(0, 1).astype(np.float32)


def _shadow_shapes(rgb: np.ndarray, level: float) -> np.ndarray:
    """XDoG (Winnemoeller et al. 2012): sharpened difference of Gaussians with a soft threshold.
    Everything darker than `level` becomes ink; higher level = more black."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    sigma, k, p, phi = 0.8, 1.6, 20.0, 10.0
    s = (1 + p) * cv2.GaussianBlur(g, (0, 0), sigma) - p * cv2.GaussianBlur(g, (0, 0), sigma * k)
    paper = np.where(s >= level, 1.0, 1 + np.tanh(phi * (s - level)))
    return (1 - paper).clip(0, 1).astype(np.float32)


@torch.inference_mode()
def _run(key: str, rgb: np.ndarray, opts: dict, progress=None) -> np.ndarray:
    """rgb: uint8 HxWx3 at detection resolution. Returns float32 HxW edge map."""
    if key == "canny":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lo, hi = int(opts.get("canny_low", 80)), int(opts.get("canny_high", 180))
        return (cv2.Canny(gray, min(lo, hi), max(lo, hi), L2gradient=True) / 255.0).astype(np.float32)

    if key == "lineart_standard":
        # ComfyUI "Lineart Standard": dark detail = blurred image minus image
        x = rgb.astype(np.float32)
        g = cv2.GaussianBlur(x, (0, 0), float(opts.get("std_sigma", 2.0)))
        inten = np.min(g - x, axis=2).clip(0, 255)
        nz = inten[inten > 3]
        inten /= max(16.0, float(np.median(nz)) if nz.size else 16.0)
        return (inten * 0.6).clip(0, 1).astype(np.float32)

    if key == "xdog":
        return _shadow_shapes(rgb, float(opts.get("shadow_level", 0.5)))

    net = load(key, progress)
    if key == "depth":
        return _depth_outlines(net, rgb)
    tiled = opts.get("tiled")
    if tiled is None:
        tiled = rgb.shape[0] * rgb.shape[1] > TILE_ABOVE
    if not tiled:
        e = _infer(key, net, rgb)
    elif key in LINEART:
        e = _lineart_streamed(net, rgb, progress)
    else:
        e = _infer_tiled(key, net, rgb, progress)
    return _lift_haze(e) if key in ("hed", "teed", "anyline") else e


def run_model(key: str, rgb: np.ndarray, opts: dict | None = None, progress=None) -> np.ndarray:
    opts = opts or {}
    try:
        return _run(key, rgb, opts, progress)
    except RuntimeError as e:
        # Rare MPS problems (unsupported op / out of memory): retry on CPU once, in float32.
        global DEVICE, HALF
        if DEVICE.type != "cpu" and "download" not in str(e).lower():
            print(f"[linework] {DEVICE} failed ({e}); falling back to CPU", file=sys.stderr)
            DEVICE, HALF = torch.device("cpu"), False
            with _lock:
                for k in list(_loaded):
                    _loaded[k] = _loaded[k].to(DEVICE).float()
            return _run(key, rgb, opts, progress)
        raise
    finally:
        if DEVICE.type == "mps":
            torch.mps.empty_cache()  # give the GPU's cached memory back to macOS and other apps
