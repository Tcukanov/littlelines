"""Digitizer quality benchmark: run a corpus of images through the
pipeline and score each result on objective quality metrics.

Usage: .venv/bin/python benchmark.py [render_dir] file1 file2 ...
"""
import io
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, ".")

from digitizer import pipeline, segmentation  # noqa: E402
from digitizer.params import Settings  # noqa: E402
from digitizer.plan import CMD_JUMP, CMD_STITCH  # noqa: E402

WIDTH_MM = 100.0


def run_file(path, render_dir=None):
    data = open(path, "rb").read()
    # aspect from the artwork
    rgb, alpha = segmentation.load_image(data)
    fg = segmentation.foreground_mask(rgb, alpha, True)
    rgb, alpha, fg = segmentation.crop_to_foreground(rgb, alpha, fg)
    h, w = fg.shape
    height_mm = WIDTH_MM * h / max(w, 1)
    if height_mm > 240:
        height_mm = 240.0
    s = Settings.from_dict({"width_mm": WIDTH_MM, "height_mm": height_mm,
                            "max_colors": 5})

    t0 = time.time()
    plan, st, warnings = pipeline.digitize(data, s)
    dt = time.time() - t0

    # coverage: rasterize stitches vs artwork mask
    sx = WIDTH_MM / max(w, 1)
    sy = height_mm / max(h, 1)
    mm_per_px = (sx + sy) / 2
    lm, pal = segmentation.quantize(rgb, fg, 5)
    lm, pal = segmentation.collapse_antialias_colors(lm, pal, rgb, mm_per_px)
    occupied = (lm >= 0).astype(np.uint8)
    cover = np.zeros_like(occupied)
    thick = max(2, int(round(0.8 / mm_per_px)))
    xs = [e[1] for e in plan.events]
    ys = [e[2] for e in plan.events]
    if not xs:
        return dict(name=path.split("/")[-1], error="no stitches")
    ox, oy = st.get("cx", -min(xs)), st.get("cy", -min(ys))
    prev = None
    for cmd, x, y in plan.events:
        if cmd == CMD_STITCH:
            if prev is not None:
                cv2.line(cover,
                         (int((prev[0] + ox) / sx), int((prev[1] + oy) / sy)),
                         (int((x + ox) / sx), int((y + oy) / sy)),
                         1, thickness=thick)
            prev = (x, y)
        elif cmd == CMD_JUMP:
            prev = (x, y)
    core = cv2.erode(occupied, np.ones((3, 3), np.uint8))
    if core.sum() < 0.05 * occupied.sum():
        core = occupied  # hairline designs vanish under erosion
    denom = max(int(core.sum()), 1)
    coverage = float((core & cover).sum()) / denom

    covered_mm2 = float(occupied.sum()) * mm_per_px * mm_per_px
    density = st["stitches"] / max(covered_mm2, 1.0)

    issues = []
    if coverage < 0.965:
        issues.append(f"COVERAGE {coverage:.1%}")
    if st["jumps"] > 40:
        issues.append(f"JUMPS {st['jumps']}")
    if st["trims"] > 50:
        issues.append(f"TRIMS {st['trims']}")
    if dt > 10:
        issues.append(f"SLOW {dt:.1f}s")
    if density > 14:
        issues.append(f"DENSE {density:.1f}st/mm2")
    if st["stitches"] > 60000:
        issues.append(f"COUNT {st['stitches']}")

    if render_dir:
        SC = 4
        rox, roy = -min(xs), -min(ys)
        W = int((max(xs) - min(xs) + 8) * SC)
        H = int((max(ys) - min(ys) + 8) * SC)
        out = Image.new("RGB", (W, H), (232, 234, 238))
        dr = ImageDraw.Draw(out)
        ci = 0
        threads = plan.threads or ["#333333"]
        col = threads[0]
        prev = None
        from digitizer.plan import CMD_COLOR_CHANGE
        for cmd, x, y in plan.events:
            if cmd == CMD_STITCH:
                if prev:
                    dr.line([((prev[0] + rox + 4) * SC,
                              (prev[1] + roy + 4) * SC),
                             ((x + rox + 4) * SC, (y + roy + 4) * SC)],
                            fill=col, width=2)
                prev = (x, y)
            elif cmd == CMD_JUMP:
                prev = (x, y)
            elif cmd == CMD_COLOR_CHANGE:
                ci += 1
                col = threads[min(ci, len(threads) - 1)]
        name = path.split("/")[-1].rsplit(".", 1)[0][:40]
        out.save(f"{render_dir}/{name}.png")

    return dict(name=path.split("/")[-1][:38], colors=st["colors"],
                stitches=st["stitches"], jumps=st["jumps"],
                trims=st["trims"], cov=round(coverage, 3),
                dens=round(density, 1), secs=round(dt, 1),
                issues=" ".join(issues))


if __name__ == "__main__":
    render_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not render_dir:
        render_dir = None
    if len(sys.argv) > 2 and sys.argv[2] == "--list":
        files = [ln.strip() for ln in open(sys.argv[3])
                 if ln.strip() and __import__("os").path.getsize(ln.strip()) > 0]
    else:
        files = sys.argv[2:]
    print(f"{'file':<38} {'col':>3} {'stitch':>7} {'jmp':>4} {'trm':>4} "
          f"{'cov':>6} {'dens':>5} {'sec':>5}  issues")
    for f in files:
        try:
            r = run_file(f, render_dir)
            if "error" in r:
                print(f"{r['name']:<38} ERROR: {r['error']}")
                continue
            print(f"{r['name']:<38} {r['colors']:>3} {r['stitches']:>7} "
                  f"{r['jumps']:>4} {r['trims']:>4} {r['cov']:>6.1%} "
                  f"{r['dens']:>5} {r['secs']:>5}  {r['issues']}")
        except Exception as e:  # noqa: BLE001
            print(f"{f.split('/')[-1][:38]:<38} EXCEPTION: {e}")
