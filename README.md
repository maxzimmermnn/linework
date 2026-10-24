# Linework

Turns a photo into black lines on white (or lines on a transparent layer), ready to draw over in Photoshop. Runs entirely on your own Mac. Models are downloaded once on first use, after that it works offline.

Built and tested on Apple Silicon Macs. It should also run on Linux and Windows (`python app.py` after installing `requirements.txt`; CPU or NVIDIA GPU), but that is untested and the double-click launcher is macOS only. Developer notes are in [DEVELOPMENT.md](DEVELOPMENT.md).

## Start

### As a Mac app (recommended)

1. Download **Linework-…-mac.zip** from the [latest release](https://github.com/maxzimmermnn/linework/releases/latest) and double-click it. (Or build the app yourself: `./macos/build_app.sh` in this folder makes it in `dist`.)
2. Drag **Linework** into **Applications**, and from there into the Dock if you like.
3. Open it. macOS asks you to confirm once because the app isn't registered with Apple: **System Settings › Privacy & Security › Open Anyway** (on older macOS: Control-click the app, **Open**, **Open**). The first start sets everything up (a few minutes, about 1 GB); a message says so and the window opens by itself when it is done. Later starts take a few seconds.
4. Quit with **Cmd+Q** or by closing the window.

Drawings are saved to **Pictures › Linework**.

To remove it completely, delete the app and the folder `~/Library/Application Support/Linework` (Python, models, presets, about 1 GB).

### From the folder, in the browser

Double-click **start.command** (first start: right-click, **Open**, **Open**). It sets up a Python environment in the folder the first time, then opens `http://127.0.0.1:7860` in your browser. Drawings go to the `output` folder. Close the Terminal window to quit.

## Use

1. Drop a photo into **Photo** (JPG, PNG, TIFF, HEIC, WebP).
2. Pick a **Preset** as a starting point, or a **Model**.
3. Adjust sliders. The preview updates live. **Over photo** shows the lines on top of a faded photo so you can judge what got picked up.
4. **Save full-size PNG**. Files go to the output folder (**Open output folder** shows it). Same pixel size as the photo by default, so it drops onto the photo in Photoshop perfectly aligned.

Tip: set **Output → Background → Transparent** and the PNG is just the lines, already a usable layer.

**Save PSD** saves one Photoshop file with the photo, a hidden white *Paper* layer and the lines on top as a transparent layer. Same size, resolution and colour profile as the photo, so you can start tracing right away. Turn *Paper* on to see the lines on white.

**Save SVG** (shown with the Hard and Pen ink styles) saves the lines as vector shapes instead of pixels. Open it in Illustrator to edit or rescale the paths, or place it in Photoshop (File > Place) onto the photo or the PSD: it lands at exactly the photo's size. Line colour and background follow the Output settings.

## What the controls do

| Control | What it does |
|---|---|
| **Model** | The "eye". Lineart = closest to hand inking. Coarse = bolder and simpler. Anyline/TEED = crisp thin contours. PiDiNet = only the main contours. HED = soft and painterly. Lineart standard / Canny = classic filters, no AI. |
| **Detail** | How big the image is when the model looks at it. Higher = finer lines and more of them. Lower = fewer, bolder lines. The single most important slider. Goes up to 4096 for very large photos (3072 on Macs with less than 24 GB of memory); high values take several seconds per change. |
| **Ignore faint lines** | Drops weak edges. Raise it when there is too much noise. |
| **Fully black from** | Edges stronger than this become solid black. |
| **Darken mid-tones** | Pushes grey lines darker (above 1) or lighter (below 1). |
| **Ink style** | Soft keeps the grey tones (pencil feel). Hard makes everything pure black. Pen redraws every line with the same width. |
| **Ink threshold** | Hard/Pen: where grey turns into a line. Lower keeps more. |
| **Line weight / Pen width** | Thinner or thicker strokes. |
| **Smooth stroke edges** | Rounds jagged pixel edges. |
| **Remove specks** | Deletes small isolated dots and scraps. |
| **Join small gaps** | Closes tiny breaks between strokes. |
| **Ignore texture** | Smooths skin, film grain, fabric before the model sees them. Keeps outlines. |
| **Find lines in shadows** | Boosts local contrast first, useful for dark or flat photos. |

Own presets: open **Save these settings as a preset**, give it a name. Saved as small JSON files (in the app: `~/Library/Application Support/Linework/presets`, from the folder: `presets/`).

## Batch

Open **Batch** under the result, drop many photos, choose **Save as** (PNG, PSD or SVG), **Process all**. Uses the current settings. You get a folder plus a zip in `output`. Nothing is ever overwritten: if a name is taken, the new file gets the time added to its name.

## Watch folder

Open **Watch a folder** under the result, check the folder (in the app: **Pictures › Linework › Inbox**), choose **Save as** and press **Start**. From then on, every photo you drop into that folder while Linework is open gets drawn with the current settings, and the drawing appears next to the photo. Photos already in the folder are left alone. Changed the settings? Press **Start** again.

From Terminal, both batch and watch folder work too:

```bash
.venv/bin/python -m linework.cli ~/Shoot/*.jpg -o ~/Lines --preset "Clean outlines" --transparent
.venv/bin/python -m linework.cli photo.jpg --style pen --svg     # vector instead of PNG
.venv/bin/python -m linework.cli photo.jpg --psd                 # layered Photoshop file
.venv/bin/python -m linework.cli --watch ~/Inbox --style hard    # draw every photo dropped into ~/Inbox
```

## Check everything works

```bash
.venv/bin/python -m linework.selftest
```

Downloads all models, runs each on a sample photo and writes `output/selftest.png`.

## License and credits

Linework is MIT licensed (see [LICENSE](LICENSE)). The model code it includes and the model weights it downloads come from other projects and keep their own licenses: Informative Drawings / Lineart (Chan et al.), TEED (Soria et al.), Anyline/MTEED (TheMisto.ai), PiDiNet (Su et al.), HED (Xie & Tu, Apache-2.0 re-implementation by lllyasviel), collected via [huggingface/controlnet_aux](https://github.com/huggingface/controlnet_aux). PiDiNet is for research use and Anyline's weights carry use restrictions; details in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
