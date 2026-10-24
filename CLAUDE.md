# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Linework turns a photo into line art that lines up 1:1 with the photo, for tracing in Photoshop. Local-only Gradio app for a non-technical artist on an Apple Silicon Mac. **Read `DEVELOPMENT.md` first**: architecture, tiling, export details, conventions, verification status and roadmap live there. `README.md` is the end-user guide.

## Commands

```bash
.venv/bin/python app.py                           # UI at http://127.0.0.1:7860 (LINEWORK_NO_BROWSER=1 to skip the browser)
.venv/bin/python -m pytest -q                     # fast tests, no model downloads (pytest is in requirements-dev.txt)
.venv/bin/python -m pytest -q tests/test_pipeline.py::test_lineart_streamed_matches_untiled
.venv/bin/python -m linework.selftest [model ...] # downloads models, writes output/selftest.png
.venv/bin/python -m linework.cli photo.jpg --style pen --svg
```

`./macos/build_app.sh` builds `dist/Linework.app`; its environment lives in `~/Library/Application Support/Linework`, its log in `~/Library/Logs/Linework.log` (see "Mac app" in DEVELOPMENT.md before touching `macos/` or `linework/desktop.py`).

If a Linework instance is already running on 7860, Gradio picks the next free port; use `GRADIO_SERVER_PORT` and `LINEWORK_OUTPUT` (scratch folder) for test servers so the user's instance and output folder are untouched.

## Rules that are easy to break

- Stage 1 (`detect.py`) is cached on `DETECT_KEYS`; a param outside that tuple must never change the model output.
- New parameter: `postprocess.DEFAULTS` + `PARAM_KEYS` and `controls` in `app.py`, same order (assert checks the length).
- Memory limits (tiling threshold, Detail max, cache, streaming piece size) derive from `registry.MEMORY_GB`; test small-Mac behaviour with `LINEWORK_MEMORY_GB=16`. Networks run in half precision on MPS: new models must work with `.half()`.
- Tiled inference (`registry._infer_tiled`, `_lineart_streamed`) must stay pixel-identical to the untiled run: 64-aligned tile offsets, edge margins dropped, whole-image normalisation (`_lift_haze`, InstanceNorm) after stitching or over the full activation.
- UI copy: plain words for an artist, no em dashes. Never overwrite user files. No network access besides the model download, `share=False`, bound to 127.0.0.1.
- New models or vendored code: record the license in `THIRD_PARTY_NOTICES.md`.
