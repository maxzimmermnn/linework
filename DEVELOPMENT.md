# Developing Linework

Linework was built for an artist whose first step on every piece is tracing a photo in Photoshop. ComfyUI's line models gave the right look but were too fiddly for daily use, so Linework wraps them in a small local app: drop a photo in, tune a few sliders with a live preview, save lines that line up 1:1 with the photo. The main user is not a programmer, which shapes the UI rules below.

Everything runs locally and offline after the first model download. Primary target is an Apple Silicon Mac (MPS); CPU and CUDA should work too. `start.command` and the app bundle are macOS only.

## Setup and commands

```bash
./start.command                                   # macOS: creates/updates .venv, then starts the app in the browser
./macos/build_app.sh                              # macOS: builds dist/Linework.app (see "Mac app" below)
# or by hand:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m linework.desktop              # the app's native window, without building the bundle
.venv/bin/python app.py                           # UI at http://127.0.0.1:7860
.venv/bin/python -m pytest -q                     # fast tests, no model downloads
.venv/bin/python -m linework.selftest [model ...] # downloads models, contact sheet in output/selftest.png
.venv/bin/python -m linework.cli photo.jpg --psd  # CLI, see linework/cli.py for options
```

Env vars: `LINEWORK_DEVICE` (force `cpu`/`mps`/`cuda`), `LINEWORK_WEIGHTS` (weights folder), `LINEWORK_OUTPUT` (output folder), `LINEWORK_PRESETS` (user presets folder), `LINEWORK_NO_BROWSER=1`, `LINEWORK_ICON` (window/Dock icon in desktop mode).

## Layout

```
app.py                     Gradio UI (Gradio 6: theme/css are passed to launch(), not Blocks())
start.command              macOS double-click launcher (browser)
macos/                     Mac app: build_app.sh, launcher.sh, setup_env.sh (shared with start.command), icon
linework/
  detect.py                Stage 1: load image, resize to "detail", pre-filter, run model, LRU cache
  postprocess.py           Stage 2: levels, styles, despeckle, weight, smoothing, compose. DEFAULTS live here
  export.py                writes PNG / SVG / PSD with never-overwrite naming; used by app, batch and CLI
  vector.py                SVG export (traces the stage 2 ink map)
  layered.py               Layered PSD export
  presets.py               Built-in presets + user presets in ./presets/*.json
  desktop.py               native window (pywebview) around the UI, used by the app bundle
  cli.py, selftest.py
  models/
    registry.py            model catalogue, download with mirrors, loading, device, inference, tiling
    small_archs.py         Lineart generator, HED          (vendored, see THIRD_PARTY_NOTICES.md)
    teed/, pidinet_arch.py TEED / Anyline, PiDiNet networks (vendored)
tests/test_pipeline.py
weights/ output/ presets/  created at runtime, gitignored
```

## How it works

Two stages so sliders feel instant:

1. **Detect** (slow, cached). The photo is resized so its short side = `detail` px, optionally bilateral-smoothed (`texture_smoothing`) and CLAHE'd (`local_contrast`), padded to a multiple of 64 and run through the model. Output: float32 edge-strength map, 0..1, at detail resolution. Cache key = image hash + the params in `detect.DETECT_KEYS`; no other param may change stage 1.
2. **Stylize** (fast). Upsampled 2x (`SUPERSAMPLE`) for anti-aliased strokes, then levels (`faint_cutoff`, `solid_at`), gamma (`darkness`), style (`soft` grey / `hard` threshold / `pen` = skeletonize, then redraw at a fixed radius), `close_gaps`, `despeckle` (area normalised to a 1000 px short side), `line_weight`, `smoothing`. Then resized to the output size (photo size by default) and composed onto white or transparent.

All pixel-valued sliders are in detail-resolution pixels, so preview and export match. The preview is only downscaled for display.

Model outputs are normalised so one set of defaults works everywhere: Lineart networks draw 1 = paper, so `_infer` inverts them; TEED, Anyline and HED leave a grey haze that `_lift_haze` removes (5th percentile floor).

### Large images (tiling)

Untiled MPS inference needs about 1.5-2 GB per megapixel the model sees. Above `registry.TILE_ABOVE` (9 MP, beyond detail ~2560 on a typical photo) models run in pieces, and the result is identical to the untiled run:

- `_infer_tiled`: overlapping 1536 px tiles. Tile offsets must be multiples of 64 (the networks' downsampling grid; an unaligned tile differs everywhere, not just at its edges). Output within 192 px of a cut edge is wrong for lack of context (PiDiNet worst, HED 128, Lineart 64) and gets zero weight; neighbours crossfade over 64 px.
- Lineart uses InstanceNorm over the whole image, so tiles cannot be normalised independently, and statistics from a downscaled pass made drawings ~25% lighter. `_lineart_streamed` runs it layer by layer instead: each convolution is computed tile by tile on the GPU, full activations live in RAM, every norm sees the whole image.
- `_lift_haze` runs after stitching, never per tile.

Detail 4096 on a 19 MP photo: Lineart ~15 s, ~6 GB GPU, ~11 GB RAM. Pen styling at that size is ~5 s per slider move (skeletonize on an 82 MP map).

### Exports

All formats go through `export.save()`, which picks a free file name (time suffix, then a counter) so nothing is overwritten, in the UI, Batch and the CLI alike.

- **PNG**: stage 2 at photo size.
- **SVG** (`vector.py`, hard/pen only): marching squares at the 0.5 level of the anti-aliased ink map, simplified, Catmull-Rom curves, one even-odd filled path. ~1 s; potrace's pure Python port took 35 s. The viewBox is the ink map grid stretched to the output size; width/height are in inches at the photo's ppi because Photoshop reads SVG `px` as points (1/72 in).
- **PSD** (`layered.py`, psd-tools): Photo, hidden white Paper, transparent Lines; photo's ppi and ICC profile carried over. psd-tools stores layer alpha as a layer *mask* unless the document is RGBA, so the PSD is built as RGBA and the header is set back to 3 channels before saving (otherwise Photoshop shows an "Alpha 1" channel).

Both were checked in Photoshop 2026 (layer structure, resolution, profile; the placed SVG lands on the PSD's Lines layer pixel for pixel). With Photoshop installed, exports can be inspected headlessly via ExtendScript: `osascript -e 'tell application "Adobe Photoshop 2026" to do javascript (read POSIX file "check.jsx")'`. Setting `activeLayer` to a hidden layer from a script makes it visible.

## Mac app

`macos/build_app.sh` assembles `dist/Linework.app` without any Python: the code is copied into `Contents/Resources/src`, `Contents/MacOS/Linework` is `macos/launcher.sh`, the icon is the committed `macos/AppIcon.icns` (drawn by `macos/make_icon.py`), and the bundle is ad-hoc signed. At runtime:

- The Python environment, models and presets live in `~/Library/Application Support/Linework`, drawings in `~/Pictures/Linework`, the log in `~/Library/Logs/Linework.log`. Nothing may run from `~/Documents`, `~/Desktop` or `~/Downloads`: an app reading those makes macOS block it behind a privacy prompt (the bundle reading its own files is fine).
- `setup_env.sh` creates the environment and reinstalls whenever `requirements.txt` changes (sha256 stamp in the venv). `start.command` uses the same script for `.venv`.
- The launcher `exec`s Python, so the process keeps the bundle's Dock icon and name. That only works when the hand-off is quick: after a multi-minute first-run setup macOS has given up on the launch and Python shows up as "python". So after setup the launcher runs `open -n` on the bundle and exits. (A backgrounded `sleep; open` does not work: it is killed together with the launcher.)
- `linework/desktop.py` opens a pywebview (WebKit) window with a "Starting…" page, starts Gradio in the background (`app.launch(block=False)`), then loads the UI. It renames the bundle in Python's own Info dictionary to "Linework" so the menu bar does not say "Python". Closing the window or Cmd+Q ends the process.

Checked on this setup: first start from scratch, relaunch after setup, Dock identity (`lsappinfo` shows "Linework" with the bundle id), UI loads in the window, quit via Apple Event frees the port. Not checked by eye yet: the icon in the Dock and menu bar text; drag-and-drop of photos into the window.

## Conventions

- **New parameter:** add it to `postprocess.DEFAULTS`, to `PARAM_KEYS` in `app.py` and to the `controls` list there in the same order (an assert checks the length). If it affects the model input, add it to `detect.DETECT_KEYS`.
- **New model:** add a `ModelInfo` to `registry.MODELS` (download URLs tried in order, GitHub raw mirrors first when they exist) and branches in `load()` and `_infer()` returning a 0..1 map with 1 = line. If weights fail to load, fix the module to match the state_dict, not the weights. Check its license (see THIRD_PARTY_NOTICES.md), check tiled vs untiled output, run the selftest.
- UI copy is for a non-technical artist: plain labels ("Ignore faint lines", not "low threshold"), no em dashes in user-facing text.
- Never overwrite user files; uploaded originals are never modified.
- Stay local: no telemetry, `share=False`, bound to 127.0.0.1.

## Status

Verified on an Apple Silicon Mac: all six downloadable models load and run on MPS; tiled output is identical to untiled for every model; PNG/SVG/PSD export from the UI, Batch and CLI; PSD and SVG in Photoshop 2026.

Not verified: `start.command` on a clean machine (venv creation, Gatekeeper prompt); SVG in Illustrator; presets were tuned on small test images and still need tuning on real 20+ MP photos.

## Roadmap

- Tune `presets.py` and `postprocess.DEFAULTS` on real photos, ideally against a hand-traced reference.
- "Watch folder" mode: drop photos into a folder, drawings appear next to them.
- Optional extra models (Lineart Anime, DexiNed, a manga line model) if a different look is needed.
