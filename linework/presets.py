"""Built-in starting points plus user presets saved as JSON in ./presets/ (or $LINEWORK_PRESETS)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .postprocess import DEFAULTS

USER_DIR = Path(os.environ.get("LINEWORK_PRESETS", Path(__file__).resolve().parents[1] / "presets"))

BUILTIN: dict[str, dict] = {
    "Default": {},
    "Clean outlines": {
        "model": "lineart", "detail": 768, "faint_cutoff": 0.3, "solid_at": 0.7,
        "style": "hard", "threshold": 0.45, "despeckle": 300, "smoothing": 0.8,
    },
    "Detailed ink": {
        "model": "lineart", "detail": 1536, "faint_cutoff": 0.12, "solid_at": 0.65,
        "darkness": 1.3, "style": "soft", "despeckle": 60,
    },
    "Bold & simple": {
        "model": "lineart_coarse", "detail": 640, "texture_smoothing": 4, "faint_cutoff": 0.3,
        "style": "pen", "threshold": 0.4, "despeckle": 400, "pen_width": 2.5, "smoothing": 1.0,
    },
    "Fine pen (uniform)": {
        "model": "anyline", "detail": 1280, "faint_cutoff": 0.2, "style": "pen",
        "threshold": 0.4, "despeckle": 150, "pen_width": 1.2, "smoothing": 0.6,
    },
    "Contours only": {
        "model": "pidinet", "detail": 1024, "texture_smoothing": 3, "faint_cutoff": 0.35,
        "solid_at": 0.8, "style": "hard", "threshold": 0.5, "despeckle": 500, "line_weight": 0.5,
    },
    "Loose sketch": {
        "model": "hed", "detail": 768, "faint_cutoff": 0.25, "solid_at": 0.95,
        "darkness": 0.8, "style": "soft", "line_weight": -0.5, "despeckle": 100,
    },
}


def _slug(name: str) -> str:
    return re.sub(r"[^\w\- ]+", "", name).strip() or "preset"


def names() -> list[str]:
    user = sorted(p.stem for p in USER_DIR.glob("*.json")) if USER_DIR.exists() else []
    return list(BUILTIN) + [n for n in user if n not in BUILTIN]


def load(name_or_path: str) -> dict:
    if name_or_path in BUILTIN:
        return {**DEFAULTS, **BUILTIN[name_or_path]}
    p = Path(name_or_path).expanduser()
    if not p.exists():
        p = USER_DIR / f"{_slug(name_or_path)}.json"
    data = json.loads(p.read_text())
    return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}


def save(name: str, params: dict) -> str:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    name = _slug(name)
    data = {k: params[k] for k in DEFAULTS if k in params}
    (USER_DIR / f"{name}.json").write_text(json.dumps(data, indent=2))
    return name
