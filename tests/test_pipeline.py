"""Fast checks that need no model download: python -m pytest -q"""

import numpy as np
import pytest

from linework import detect, postprocess as pp


def synthetic():
    img = np.full((300, 400, 3), 235, np.uint8)
    img[80:220, 100:300] = (30, 30, 30)
    return img


@pytest.mark.parametrize("model", ["canny", "lineart_standard"])
@pytest.mark.parametrize("style", ["soft", "hard", "pen"])
def test_render_shapes(model, style):
    rgb = synthetic()
    p = {**pp.DEFAULTS, "model": model, "detail": 256, "style": style}
    edges = detect.detect(rgb, p)
    assert edges.dtype == np.float32 and edges.min() >= 0 and edges.max() <= 1
    img, ink = pp.render(rgb, edges, p)
    assert img.shape == (300, 400, 3)
    assert ink.max() > 0.5, "rectangle outline should be drawn"
    assert ink.mean() < 0.2, "most of the page should stay white"


def test_transparent_output_has_alpha():
    rgb = synthetic()
    p = {**pp.DEFAULTS, "model": "canny", "detail": 256, "background": "transparent"}
    img, ink = pp.render(rgb, detect.detect(rgb, p), p)
    assert img.shape[2] == 4
    assert (img[..., 3] == 0).mean() > 0.8


def test_despeckle_removes_dots():
    ink = np.zeros((200, 200), np.float32)
    ink[50, 50] = 1  # single speck
    ink[100, 20:180] = 1  # long line
    out = pp._remove_specks(ink, 10)
    assert out[50, 50] == 0 and out[100, 100] == 1


def test_cache_hits_on_styling_change():
    rgb = synthetic()
    p = {**pp.DEFAULTS, "model": "canny", "detail": 256}
    a = detect.detect(rgb, p)
    b = detect.detect(rgb, {**p, "style": "pen", "faint_cutoff": 0.5})
    assert a is b


def test_hex_colors():
    assert pp.hex_to_rgb("#ff0000") == (255, 0, 0)
    assert pp.hex_to_rgb("#0f0") == (0, 255, 0)
    assert pp.hex_to_rgb("rgba(10, 20, 30, 1)") == (10, 20, 30)


def test_svg_traces_outline_with_hole():
    from linework import vector

    ink = np.zeros((100, 120), np.float32)
    ink[20:80, 30:90] = 1
    ink[30:70, 40:80] = 0  # hollow square -> outer contour + hole
    paths = vector.trace(ink)
    assert len(paths) == 2 and all(d.startswith("M") and d.endswith("Z") for d in paths)
    svg = vector.to_svg(ink, (240, 200), {"background": "transparent", "line_color": "#ff0000"}, dpi=72)
    assert 'width="3.33333in"' in svg and 'viewBox="0 0 120 100"' in svg
    assert 'fill="#ff0000"' in svg and "<rect" not in svg


def test_psd_layers_resolution_and_profile(tmp_path):
    from PIL import ImageCms
    from psd_tools import PSDImage

    from linework import layered

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    rgb = synthetic()
    p = {**pp.DEFAULTS, "model": "canny", "detail": 256, "output_size": "detail", "background": "white"}
    dest = tmp_path / "out.psd"
    layered.to_psd(rgb, detect.detect(rgb, p), p, dpi=300, icc=icc).save(dest)
    psd = PSDImage.open(dest)
    assert psd.size == (400, 300)  # always photo size, even with output_size="detail"
    assert [(l.name, l.visible) for l in psd] == [("Photo", True), ("Paper", False), ("Lines", True)]
    lines = np.asarray(psd[2].topil())
    assert lines.shape == (300, 400, 4) and (lines[..., 3] == 0).mean() > 0.8  # transparent despite white bg
    assert psd.image_resources.get_data(layered.ICC_PROFILE) == icc


def test_tile_blend_weights_sum_to_one():
    from linework.models import registry as R

    n = 4000
    acc = np.zeros(n, np.float32)
    starts = R._starts(n)
    for i, s in enumerate(starts):
        t = min(R.TILE, n - s)
        acc[s : s + t] += R._feather(1, t, False, False, i > 0, s + t < n)[0]
    assert all(s % 64 == 0 for s in starts)
    assert np.allclose(acc, 1)


def test_lineart_streamed_matches_untiled(monkeypatch):
    import torch

    from linework.models import registry as R
    from linework.models.small_archs import LineartGenerator

    torch.manual_seed(0)
    net = LineartGenerator(3, 1, 3).float().eval().to(R.DEVICE)  # random weights, no download
    monkeypatch.setattr(R, "STREAM_TILE", 96)  # many tiles on a small image
    rgb = np.random.default_rng(0).integers(0, 255, (250, 330, 3), dtype=np.uint8)
    with torch.inference_mode():
        ref = R._infer("lineart", net, rgb)
        out = R._lineart_streamed(net, rgb)
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 1e-3


def test_export_formats_never_overwrite(tmp_path):
    from linework import export

    rgb = synthetic()
    p = {**pp.DEFAULTS, "model": "canny", "detail": 256, "style": "hard"}
    edges = detect.detect(rgb, p)
    for fmt in export.FORMATS:
        a, wh = export.save(fmt, "photo.jpg", rgb, edges, p, tmp_path)
        b, _ = export.save(fmt, "photo.jpg", rgb, edges, p, tmp_path)
        c, _ = export.save(fmt, "photo.jpg", rgb, edges, p, tmp_path)
        assert a.name == f"photo_lines_canny.{fmt}" and wh == (400, 300)
        assert len({a, b, c}) == 3 and all(x.exists() for x in (a, b, c))
    with pytest.raises(ValueError):
        export.save("svg", "photo.jpg", rgb, edges, {**p, "style": "soft"}, tmp_path)
