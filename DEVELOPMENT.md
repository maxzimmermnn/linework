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

Env vars: `LINEWORK_DEVICE` (force `cpu`/`mps`/`cuda`), `LINEWORK_PRECISION=full` (float32 on Apple GPUs), `LINEWORK_MEMORY_GB` (act like a Mac with that much RAM; tiling, Detail limit and cache follow it), `LINEWORK_WEIGHTS` (weights folder), `LINEWORK_OUTPUT` (output folder), `LINEWORK_PRESETS` (user presets folder), `LINEWORK_NO_BROWSER=1`, `LINEWORK_ICON` (window/Dock icon in desktop mode).

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
  watch.py                 watch folder: polls a folder, draws new photos next to them (UI and cli --watch)
  layered.py               Layered PSD export
  presets.py               Built-in presets + user presets in ./presets/*.json
  desktop.py               native window (pywebview) around the UI, used by the app bundle
  cli.py, selftest.py
  models/
    registry.py            model catalogue, download with mirrors, loading, device, inference, tiling
    small_archs.py         Lineart generator, HED          (vendored, see THIRD_PARTY_NOTICES.md)
    teed/, pidinet_arch.py TEED / Anyline, PiDiNet networks (vendored)
    depth_anything_v2/     Depth Anything V2 (DINOv2 + DPT), used by "Depth outlines" (vendored)
tests/test_pipeline.py
weights/ output/ presets/  created at runtime, gitignored
```

## How it works

Two stages so sliders feel instant:

1. **Detect** (slow, cached). The photo is resized so its short side = `detail` px, optionally bilateral-smoothed (`texture_smoothing`) and CLAHE'd (`local_contrast`), padded to a multiple of 64 and run through the model. Output: float32 edge-strength map, 0..1, at detail resolution. Cache key = image hash + the params in `detect.DETECT_KEYS`; no other param may change stage 1.
2. **Stylize** (fast). Upsampled 2x (`SUPERSAMPLE`) for anti-aliased strokes, then levels (`faint_cutoff`, `solid_at`), gamma (`darkness`), style (`soft` grey / `hard` threshold / `pen` = skeletonize, then redraw at a fixed radius), `close_gaps`, `despeckle` (area normalised to a 1000 px short side), `line_weight`, `smoothing`. Then resized to the output size (photo size by default) and composed onto white or transparent.

All pixel-valued sliders are in detail-resolution pixels, so preview and export match. The preview is only downscaled for display.

Models in brief:

- `LINEART` family (one generator, three trained styles of Informative Drawings): `lineart` = the "anime" style checkpoint (ControlNet ships it as sk_model.pth "realistic"), `lineart_coarse` = "opensketch", `contour` = "contour". All share inference, tiling (`_lineart_streamed`) and half precision.
- `depth`: Depth Anything V2 Small sees the whole photo at `DEPTH_RES` (770 px short side) regardless of Detail, so it never tiles. `_depth_outlines` turns the relative depth into lines where it jumps: gradient divided by depth (relative jump, so distant edges count), scaled per 1/1000 of the image (same result at any Detail), thinned by non-maximum suppression and widened to a 2-3 px stroke. Silhouettes score ~1-6, flat surfaces ~0.01-0.04. Close-ups and photos of flat things give little, by design.
- `FILTERS` (no network): `canny`, `lineart_standard`, and `xdog` ("Shadow shapes", XDoG by Winnemoeller et al.) whose `shadow_level` slider is a stage 1 parameter (in `DETECT_KEYS`, cache-keyed only for xdog, like Canny's thresholds).
- `_torch_load` reads PyTorch checkpoints and safetensors (told apart by content, so mirrors can mix formats) without executing anything from the file.

Model outputs are normalised so one set of defaults works everywhere: Lineart networks draw 1 = paper, so `_infer` inverts them; TEED, Anyline and HED leave a grey haze that `_lift_haze` removes (5th percentile floor).

### Memory and speed on smaller Macs

The target includes 16 GB machines (MacBook Air class, 8-10 GPU cores). On Apple Silicon the GPU shares RAM with macOS and every other app, so all limits derive from installed RAM (`registry.MEMORY_GB`):

- **Half precision** on MPS (`registry.HALF`): networks run in float16. 2-3x less GPU memory (Lineart at Detail 2048: 8.7 -> 2.9 GB, PiDiNet 10.1 -> 4.8 GB) and 15-40% faster; stylized output matches float32 at IoU >= 0.997 for every model. Whole-network `.half()` beats `torch.autocast` here: autocast keeps InstanceNorm in float32 and made Lineart use *more* memory. PiDiNet's `rd` conv needed a dtype fix. CPU (and the CPU fallback) stays float32.
- **Tiling threshold** `TILE_ABOVE` = what a quarter of RAM holds at ~0.9 GB/MP: 4.4 MP (Detail ~1900) on 16 GB, capped at 9 MP from 32 GB. `MAX_DETAIL` (the slider limit) is 3072 below 24 GB: Lineart at 4096 needs ~9 GB there.
- **Streamed Lineart** keeps activations in half precision, normalises 8 channels at a time in float32, uses 512 px pieces and empties the MPS cache after each layer on < 32 GB: Detail 3072 peaks at 4.9 GB instead of 7.4 GB for ~20% more time.
- `torch.mps.empty_cache()` after every model run hands cached GPU memory back to macOS (otherwise PyTorch keeps several GB).
- The edge cache is bounded by bytes (RAM/32) as well as entries.
- **Preview styling** runs at about display size instead of the export's 2x (`stylize(..., scale)`): 4-7x less work per slider move at high Detail. Pen and Hard strokes change look at low scale (Pen at 1x is 16% lighter), so they keep a minimum scale (Pen 1.5, Hard 1.25); the preview's ink is then within 4% of the export.

Measured on an M1 Max with `LINEWORK_MEMORY_GB=16`: Lineart preview at Detail 1024 in 1 s, 2048 in 5 s, 3072 in 11 s; slider moves afterwards 0.5-1.3 s; server footprint ~4-5 GB at 3072. An M3 Air's GPU has roughly a third of that GPU's cores, so model runs take longer there; styling is CPU-bound and similar. Not yet measured on a real 16 GB machine.

### Large images (tiling)

Above `registry.TILE_ABOVE` (see above) models run in pieces, and the result is identical to the untiled run (half precision: Lineart IoU 0.999, the others exact):

- `_infer_tiled`: overlapping 1536 px tiles. Tile offsets must be multiples of 64 (the networks' downsampling grid; an unaligned tile differs everywhere, not just at its edges). Output within 192 px of a cut edge is wrong for lack of context (PiDiNet worst, HED 128, Lineart 64) and gets zero weight; neighbours crossfade over 64 px.
- Lineart uses InstanceNorm over the whole image, so tiles cannot be normalised independently, and statistics from a downscaled pass made drawings ~25% lighter. `_lineart_streamed` runs it layer by layer instead: each convolution is computed tile by tile on the GPU, full activations live in RAM, every norm sees the whole image.
- `_lift_haze` runs after stitching, never per tile.


### Exports

All formats go through `export.save()`, which picks a free file name (time suffix, then a counter) so nothing is overwritten, in the UI, Batch and the CLI alike.

- **PNG**: stage 2 at photo size.
- **SVG** (`vector.py`, hard/pen only): marching squares at the 0.5 level of the anti-aliased ink map, simplified, Catmull-Rom curves, one even-odd filled path. ~1 s; potrace's pure Python port took 35 s. The viewBox is the ink map grid stretched to the output size; width/height are in inches at the photo's ppi because Photoshop reads SVG `px` as points (1/72 in).
- **PSD** (`layered.py`, psd-tools): Photo, hidden white Paper, transparent Lines; photo's ppi and ICC profile carried over. psd-tools stores layer alpha as a layer *mask* unless the document is RGBA, so the PSD is built as RGBA and the header is set back to 3 channels before saving (otherwise Photoshop shows an "Alpha 1" channel).

Both were checked in Photoshop 2026 (layer structure, resolution, profile; the placed SVG lands on the PSD's Lines layer pixel for pixel). With Photoshop installed, exports can be inspected headlessly via ExtendScript: `osascript -e 'tell application "Adobe Photoshop 2026" to do javascript (read POSIX file "check.jsx")'`. Setting `activeLayer` to a hidden layer from a script makes it visible.

### Watch folder

`watch.Watcher` polls every 2 s (no extra dependency, works on network drives). A photo counts once its size and mtime are unchanged between two polls, so files still being copied are skipped; photos present at start and our own `*_lines_*` outputs are never processed. Settings are a snapshot taken at Start. The watcher thread and the UI share the model and the edge cache, so `detect.detect` runs under a lock.

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

Verified on an Apple Silicon Mac: all six downloadable models load and run on MPS; tiled output is identical to untiled for every model; PNG/SVG/PSD export from the UI, Batch and CLI; the watch folder in the UI, the app (writing to `~/Pictures`) and the CLI; PSD and SVG in Photoshop 2026.

Not verified: `start.command` on a clean machine (venv creation, Gatekeeper prompt); SVG in Illustrator; presets were tuned on small test images and still need tuning on real 20+ MP photos (except "Lots of texture", tuned on 3500-3800 px photos).

## Roadmap

- Tune `presets.py` and `postprocess.DEFAULTS` on real photos, ideally against a hand-traced reference.
- More models only if a different look is needed. Added so far: Contour, Depth outlines, Shadow shapes. Ruled out: OpenSketch (already Lineart coarse), Lineart Anime (already the default Lineart), DexiNed (tested on large photos: less texture than Lineart or Anyline; the "Lots of texture" preset covers that look), anime/manga line models (not wanted).
