"""Linework: photo -> clean black line drawing, running fully on this computer.

Start with ./start.command (macOS double-click) or `python app.py`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from linework import detect, layered, postprocess, presets, vector
from linework.models import registry

ROOT = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("LINEWORK_OUTPUT", ROOT / "output"))
PREVIEW_MAX = 1800  # long side of the on-screen preview, px

M = registry.MODELS
MODEL_CHOICES = [(m.label, k) for k, m in M.items()]

# Order matters: this is the order of the UI inputs passed to every callback.
PARAM_KEYS = [
    "model", "detail", "texture_smoothing", "local_contrast", "canny_low", "canny_high",
    "faint_cutoff", "solid_at", "darkness", "style", "threshold", "close_gaps", "despeckle",
    "line_weight", "pen_width", "smoothing", "output_size", "background", "line_color",
]


# ------------------------------------------------------------------ helpers --
@lru_cache(maxsize=6)
def _load(path: str, mtime: float):
    rgb = detect.load_image(path)
    return rgb, detect.image_id(rgb)


def load_photo(path: str):
    return _load(path, os.path.getmtime(path))


def params_from(values) -> dict:
    p = dict(zip(PARAM_KEYS, values))
    p["detail"] = int(p["detail"])
    return p


def model_note(key: str) -> str:
    m = M[key]
    note = f"**{m.label}** · {m.blurb}"
    if m.filename and not registry.is_downloaded(key):
        note += f"  \n<small>First use downloads the model (~{m.size_mb:g} MB), then it works offline.</small>"
    return note


def _progress_cb(progress):
    return lambda frac, desc: progress(frac, desc=desc)


def _edges(path, p, progress):
    rgb, img_id = load_photo(path)
    try:
        edges = detect.detect(rgb, p, _progress_cb(progress), img_id=img_id)
    except RuntimeError as e:
        raise gr.Error(f"{e}\n\nIf this is the first use of the model, check the internet connection.") from e
    return rgb, edges


def _checker(h, w, cell=12):
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // cell) + (xx // cell)) % 2).astype(np.float32)
    return (235 + c * 20)[..., None].repeat(3, axis=2)


def _preview_image(rgb, ink, p, view):
    h, w = ink.shape
    k = min(1.0, PREVIEW_MAX / max(h, w))
    if k < 1:
        ink = postprocess.resize_ink(ink, (round(w * k), round(h * k)), crisp=False)
    if view == "Over photo":
        return postprocess.overlay_on_photo(rgb, ink, p)
    if p["background"] == "transparent":
        col = np.array(postprocess.hex_to_rgb(p["line_color"]), np.float32)
        bg = _checker(*ink.shape)
        a = ink[..., None]
        return (bg * (1 - a) + col * a).astype(np.uint8)
    return postprocess.compose(ink, p)


# ---------------------------------------------------------------- callbacks --
def preview(path, view, *values, progress=gr.Progress()):
    if not path:
        return None, ""
    p = params_from(values)
    t0 = time.time()
    rgb, edges = _edges(path, p, progress)
    ink = postprocess.stylize(edges, p)
    img = _preview_image(rgb, ink, p, view)
    eh, ew = edges.shape
    info = (
        f"<small>Photo {rgb.shape[1]}×{rgb.shape[0]} · model sees {ew}×{eh} · "
        f"{registry.DEVICE.type.upper()} · {time.time() - t0:.1f}s</small>"
    )
    return img, info


def export(path, *values, progress=gr.Progress()):
    if not path:
        raise gr.Error("Add a photo first.")
    p = params_from(values)
    rgb, edges = _edges(path, p, progress)
    progress(0.6, desc="Rendering full size")
    img, _ = postprocess.render(rgb, edges, p)
    dest = _new_dest(path, p, ".png")
    Image.fromarray(img).save(dest, compress_level=4)
    h, w = img.shape[:2]
    return gr.update(value=str(dest), visible=True), f"Saved **{dest.name}** ({w}×{h}) in the output folder."


def export_svg(path, *values, progress=gr.Progress()):
    if not path:
        raise gr.Error("Add a photo first.")
    p = params_from(values)
    rgb, edges = _edges(path, p, progress)
    progress(0.6, desc="Tracing lines")
    w, h = postprocess.output_wh(rgb, edges, p["output_size"])
    svg = vector.to_svg(postprocess.stylize(edges, p), (w, h), p, detect.image_info(path)["dpi"])
    dest = _new_dest(path, p, ".svg")
    dest.write_text(svg, encoding="utf-8")
    return gr.update(value=str(dest), visible=True), f"Saved **{dest.name}** ({w}×{h}) in the output folder."


def export_psd(path, *values, progress=gr.Progress()):
    if not path:
        raise gr.Error("Add a photo first.")
    p = params_from(values)
    rgb, edges = _edges(path, p, progress)
    progress(0.6, desc="Building layers")
    psd = layered.to_psd(rgb, edges, p, **detect.image_info(path))
    dest = _new_dest(path, p, ".psd")
    psd.save(dest)
    return gr.update(value=str(dest), visible=True), f"Saved **{dest.name}** (photo + lines as layers) in the output folder."


def _new_dest(path, p, ext) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = (OUT_DIR / detect.output_name(path, p["model"])).with_suffix(ext)
    if dest.exists():  # never overwrite earlier versions
        dest = dest.with_name(f"{dest.stem}_{time.strftime('%H%M%S')}{ext}")
    return dest


def batch(files, *values, progress=gr.Progress()):
    if not files:
        raise gr.Error("Add some photos to the batch first.")
    p = params_from(values)
    folder = OUT_DIR / f"batch_{time.strftime('%Y-%m-%d_%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    done = []
    for i, f in enumerate(files):
        src = f if isinstance(f, str) else f.name
        progress(i / len(files), desc=f"{Path(src).name} ({i + 1}/{len(files)})")
        rgb, img_id = load_photo(src)
        edges = detect.detect(rgb, p, img_id=img_id)
        img, _ = postprocess.render(rgb, edges, p)
        dest = folder / detect.output_name(src, p["model"])
        Image.fromarray(img).save(dest, compress_level=4)
        done.append(dest)
    zpath = folder.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for d in done:
            z.write(d, d.name)
    return gr.update(value=str(zpath), visible=True), f"Done: {len(done)} drawings in **{folder.name}**."


def open_output_folder():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    opener = {"darwin": "open", "win32": "explorer"}.get(sys.platform, "xdg-open")
    try:
        subprocess.Popen([opener, str(OUT_DIR)])
    except Exception:
        pass
    return f"Output folder: `{OUT_DIR}`"


def apply_preset(name):
    p = presets.load(name)
    return [p[k] for k in PARAM_KEYS]


def save_preset(name, *values):
    if not name or not name.strip():
        raise gr.Error("Give the preset a name.")
    saved = presets.save(name, params_from(values))
    return gr.update(choices=presets.names(), value=saved), f"Preset **{saved}** saved."


def visibility(model, style):
    return (
        gr.update(visible=style in ("hard", "pen")),   # threshold
        gr.update(visible=style != "pen"),             # line_weight
        gr.update(visible=style == "pen"),             # pen_width
        gr.update(visible=model == "canny"),           # canny group
        gr.update(visible=style in ("hard", "pen")),   # svg button
        model_note(model),
    )


# --------------------------------------------------------------------- UI ----
CSS = """
#title h1 {font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0}
#title p {opacity: .7; margin-top: 4px}
#preview img {background: #fff}
#result-col {position: sticky; top: 12px; align-self: flex-start}
#result-file {min-height: 0}
.small-note {opacity: .75}
footer {display: none !important}
"""

D = postprocess.DEFAULTS


def make_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        button_primary_background_fill="*neutral_900",
        button_primary_background_fill_hover="*neutral_700",
        button_primary_text_color="white",
    )


def build():
    with gr.Blocks(title="Linework") as demo:
        gr.Markdown(
            "# Linework\nDrop in a photo, get the lines back as black on white. Everything runs on this computer.",
            elem_id="title",
        )
        with gr.Row(equal_height=False):
            # ---------------- controls
            with gr.Column(scale=2, min_width=340):
                photo = gr.Image(type="filepath", label="Photo", sources=["upload", "clipboard"], height=260)
                with gr.Row():
                    preset = gr.Dropdown(presets.names(), value="Default", label="Preset", scale=3)
                model = gr.Dropdown(MODEL_CHOICES, value=D["model"], label="Model")
                note = gr.Markdown(model_note(D["model"]), elem_classes="small-note")
                detail = gr.Slider(256, 4096, D["detail"], step=64, label="Detail",
                                   info="Resolution the model looks at. Higher = finer, more lines. Lower = bolder, simpler. "
                                        "Above about 2500 each change takes several seconds.")

                with gr.Accordion("Lines", open=True):
                    faint_cutoff = gr.Slider(0, 0.9, D["faint_cutoff"], step=0.01, label="Ignore faint lines",
                                             info="Raise to drop weak, noisy edges.")
                    solid_at = gr.Slider(0.1, 1, D["solid_at"], step=0.01, label="Fully black from",
                                         info="Lower = more lines become solid black.")
                    darkness = gr.Slider(0.3, 3, D["darkness"], step=0.05, label="Darken mid-tones",
                                         info="Above 1 makes grey lines darker, below 1 lighter.")
                    style = gr.Radio([("Soft (keeps grey)", "soft"), ("Hard (pure black)", "hard"),
                                      ("Pen (even width)", "pen")], value=D["style"], label="Ink style")
                    threshold = gr.Slider(0.05, 0.95, D["threshold"], step=0.01, label="Ink threshold",
                                          info="Lower = more lines survive.", visible=False)
                    line_weight = gr.Slider(-2, 4, D["line_weight"], step=0.25, label="Line weight",
                                            info="Thin (−) or thicken (+) the strokes.")
                    pen_width = gr.Slider(0.5, 10, D["pen_width"], step=0.25, label="Pen width", visible=False)
                    smoothing = gr.Slider(0, 4, D["smoothing"], step=0.1, label="Smooth stroke edges")

                with gr.Accordion("Clean-up", open=False):
                    despeckle = gr.Slider(0, 2000, D["despeckle"], step=10, label="Remove specks",
                                          info="Deletes isolated dots and short scraps below this size.")
                    close_gaps = gr.Slider(0, 4, D["close_gaps"], step=0.25, label="Join small gaps")
                    texture_smoothing = gr.Slider(0, 10, D["texture_smoothing"], step=0.5, label="Ignore texture",
                                                  info="Smooths skin, grain and fabric before detection. Keeps contours.")
                    local_contrast = gr.Slider(0, 4, D["local_contrast"], step=0.1, label="Find lines in shadows",
                                               info="Boosts local contrast before detection.")
                    with gr.Group(visible=False) as canny_group:
                        canny_low = gr.Slider(0, 255, D["canny_low"], step=1, label="Canny low")
                        canny_high = gr.Slider(0, 255, D["canny_high"], step=1, label="Canny high")

                with gr.Accordion("Output", open=False):
                    output_size = gr.Radio([("Same size as photo", "original"), ("Detail resolution", "detail")],
                                           value=D["output_size"], label="Size")
                    background = gr.Radio([("White", "white"), ("Transparent (lines only)", "transparent")],
                                          value=D["background"], label="Background")
                    line_color = gr.ColorPicker(D["line_color"], label="Line colour")

                with gr.Accordion("Save these settings as a preset", open=False):
                    with gr.Row():
                        preset_name = gr.Textbox(placeholder="e.g. Portraits", show_label=False, scale=3)
                        save_btn = gr.Button("Save", scale=1)

            # ---------------- result
            with gr.Column(scale=3, min_width=420, elem_id="result-col"):
                view = gr.Radio(["Lines", "Over photo"], value="Lines", show_label=False)
                out = gr.Image(label="Result", type="numpy", format="png", height=640,
                               interactive=False, elem_id="preview")
                info = gr.Markdown()
                with gr.Row():
                    export_btn = gr.Button("Save full-size PNG", variant="primary", scale=2)
                    psd_btn = gr.Button("Save PSD", scale=1)
                    svg_btn = gr.Button("Save SVG", scale=1, visible=False)
                    folder_btn = gr.Button("Open output folder", scale=1)
                gr.Markdown("<small>PSD opens in Photoshop with the photo and the lines on separate layers. "
                            "SVG keeps the lines as shapes you can scale and edit in Illustrator "
                            "(Hard and Pen ink styles).</small>", elem_classes="small-note")
                result_file = gr.File(label="Download", interactive=False, visible=False, elem_id="result-file")
                status = gr.Markdown()
                with gr.Accordion("Batch: same settings on many photos", open=False):
                    batch_files = gr.File(file_count="multiple", file_types=["image"], label="Photos")
                    batch_btn = gr.Button("Process all")
                    batch_out = gr.File(label="Zip of all drawings", interactive=False, visible=False)

        controls = [model, detail, texture_smoothing, local_contrast, canny_low, canny_high,
                    faint_cutoff, solid_at, darkness, style, threshold, close_gaps, despeckle,
                    line_weight, pen_width, smoothing, output_size, background, line_color]
        assert len(controls) == len(PARAM_KEYS)

        gr.on(
            triggers=[photo.change, view.change] + [c.change for c in controls],
            fn=preview, inputs=[photo, view] + controls, outputs=[out, info],
            trigger_mode="always_last", show_progress="minimal",
        )
        gr.on([model.change, style.change], visibility, [model, style],
              [threshold, line_weight, pen_width, canny_group, svg_btn, note])
        preset.change(apply_preset, preset, controls)
        save_btn.click(save_preset, [preset_name] + controls, [preset, status])
        export_btn.click(export, [photo] + controls, [result_file, status])
        svg_btn.click(export_svg, [photo] + controls, [result_file, status])
        psd_btn.click(export_psd, [photo] + controls, [result_file, status])
        folder_btn.click(open_output_folder, None, status)
        batch_btn.click(batch, [batch_files] + controls, [batch_out, status])
    return demo


if __name__ == "__main__":
    print(f"Linework · device: {registry.DEVICE} · weights: {registry.WEIGHTS_DIR}")
    build().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        inbrowser=os.environ.get("LINEWORK_NO_BROWSER") != "1",
        allowed_paths=[str(OUT_DIR)],
        theme=make_theme(),
        css=CSS,
    )
