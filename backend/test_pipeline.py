"""End-to-end pipeline test with generated sample artwork."""
import io
import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))

import pyembroidery  # noqa: E402

from digitizer import export, pipeline  # noqa: E402
from digitizer.params import Settings  # noqa: E402


def make_logo_png() -> bytes:
    """Two-color logo on transparency: red circle + blue rounded bar."""
    img = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([40, 40, 200, 200], fill=(220, 30, 40, 255))
    d.rounded_rectangle([220, 90, 380, 150], radius=25, fill=(20, 60, 200, 255))
    d.ellipse([90, 90, 150, 150], fill=(0, 0, 0, 0))  # hole in circle
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def make_lineart_png() -> bytes:
    """Black line drawing on white."""
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.line([(40, 250), (120, 60), (200, 250)], fill=(10, 10, 10), width=6)
    d.line([(70, 180), (170, 180)], fill=(10, 10, 10), width=6)
    d.arc([220, 80, 360, 220], 0, 300, fill=(10, 10, 10), width=6)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def check_dst(plan, label):
    data, mime, filename = export.export_bytes(plan, "dst", "test")
    assert data[:3] == b"LA:", f"{label}: bad DST header start {data[:8]!r}"
    fd, path = tempfile.mkstemp(suffix=".dst")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    rb = pyembroidery.read_dst(path)
    os.unlink(path)
    st = np.array([(s[0], s[1]) for s in rb.stitches])
    deltas = np.abs(np.diff(st, axis=0))
    assert deltas.max() <= 121.5, f"{label}: delta exceeds DST limit"
    ext = rb.extents()
    print(f"  {label}: DST ok, {len(rb.stitches)} entries, "
          f"extents {(ext[2]-ext[0])/10:.1f} x {(ext[3]-ext[1])/10:.1f} mm, "
          f"{len(data)} bytes")


def run(label, png, settings):
    print(f"== {label}")
    a = pipeline.analyze(png, settings)
    print(f"  analyze: {len(a['colors'])} colors: "
          f"{[(c['hex'], c['suggested'], c['regions']) for c in a['colors']]}")
    plan, stats, warnings = pipeline.digitize(png, settings)
    print(f"  stats: {stats}")
    if warnings:
        print(f"  warnings: {warnings}")
    assert stats["stitches"] > 50, f"{label}: too few stitches"
    check_dst(plan, label)
    for fmt in ("pes", "exp", "jef", "svg"):
        blob, _, fn = export.export_bytes(plan, fmt, "test")
        assert len(blob) > 50, f"{label}: {fmt} too small"
    print("  pes/exp/jef/svg exports ok")
    return stats


logo = make_logo_png()
line = make_lineart_png()

s1 = Settings.from_dict({"width_mm": 100, "height_mm": 75, "max_colors": 4})
run("two-color logo (auto)", logo, s1)

s2 = Settings.from_dict({"width_mm": 100, "height_mm": 75,
                         "line_art": True, "line_passes": 2})
run("line art mode", line, s2)

s3 = Settings.from_dict({"width_mm": 60, "height_mm": 45, "max_colors": 4,
                         "underlay": False, "trim_enabled": False,
                         "color_settings": {"0": {"stitch": "running"},
                                            "1": {"stitch": "satin"}}})
run("overrides: running + satin", logo, s3)

s4 = Settings.from_dict({"width_mm": 100, "height_mm": 75, "max_colors": 4,
                         "color_settings": {"1": {"enabled": False}}})
st = run("one color disabled", logo, s4)
assert st["colors"] == 1, "disabled color still present"

print("ALL TESTS PASSED")
