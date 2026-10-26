# Third-party notices

Linework's own code is MIT licensed (see `LICENSE`). It includes network definitions from other projects, and it downloads pretrained model weights on first use. Those keep their own licenses. Full license texts are in `licenses/`.

## Vendored code

Most network definitions were taken from [huggingface/controlnet_aux](https://github.com/huggingface/controlnet_aux) (Apache-2.0, `licenses/Apache-2.0-controlnet_aux.txt`), which collected them from the original projects below; Depth Anything V2 comes straight from its own repository. `small_archs.py` was trimmed to the torch modules; tiling and pre/post-processing are Linework's own code in `registry.py`.

| File | Original project | Original license |
|---|---|---|
| `linework/models/small_archs.py` (Lineart generator) | [carolineec/informative-drawings](https://github.com/carolineec/informative-drawings) | MIT (`licenses/MIT-informative-drawings.txt`) |
| `linework/models/small_archs.py` (HED) | [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) Apache-2.0 re-implementation of HED (Xie & Tu) | Apache-2.0 |
| `linework/models/teed/` | [xavysp/TEED](https://github.com/xavysp/TEED) | MIT (`licenses/MIT-TEED.txt`) |
| `linework/models/pidinet_arch.py` | [hellozhuo/pidinet](https://github.com/hellozhuo/pidinet) (modified: half-precision fix) | MIT text with the note "It is just for research purpose, and commercial use should be contacted with authors first." (`licenses/PiDiNet.txt`) |
| `linework/models/depth_anything_v2/` | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) (modified: `dpt.py` without its image helpers), includes DINOv2 code by Meta | Apache-2.0 (`licenses/Apache-2.0-Depth-Anything-V2.txt`) |

## Model weights (downloaded at runtime, not part of this repository)

| Model in the app | Weights | License |
|---|---|---|
| Lineart (realistic), Lineart (coarse) | `sk_model.pth`, `sk_model2.pth` from [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators). Byte-identical to the Informative Drawings authors' "anime" and "opensketch" style checkpoints | Hugging Face repo lists "other"; original project is MIT |
| Contour | The Informative Drawings "contour" style checkpoint, from the [OzzyGT/contour](https://huggingface.co/OzzyGT/contour) safetensors conversion (fallback: a duplicate of the authors' Space) | MIT |
| Depth outlines | `depth_anything_v2_vits.pth` (Small) from [depth-anything/Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) | Apache-2.0 (only Small; the Base/Large/Giant checkpoints are non-commercial and not used) |
| HED (soft) | `ControlNetHED.pth` from lllyasviel/Annotators | Hugging Face repo lists "other"; ControlNet is Apache-2.0 |
| TEED | `5_model.pth` from xavysp/TEED | MIT |
| Anyline | `MTEED.pth` from [TheMistoAI/MistoLine](https://huggingface.co/TheMistoAI/MistoLine) | CreativeML Open RAIL++-M (permissive, with use-based restrictions) |
| PiDiNet | `table5_pidinet.pth` from hellozhuo/pidinet | Research use; contact the authors for commercial use |
| Lineart standard, Canny, Shadow shapes (XDoG) | none (classic filters) | n/a |

This is a summary, not legal advice. For commercial work, check the current license of the model you use; Lineart, Contour, TEED, Depth outlines and the three filters are the least restricted choices.
