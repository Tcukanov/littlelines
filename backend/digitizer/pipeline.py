"""End-to-end digitizing pipeline:
image -> cleanup -> segmentation -> contours -> stitch paths -> plan."""
from __future__ import annotations

import base64
import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from . import fills, lineart, regions, segmentation
from .params import Settings
from .plan import Plan, PlanBuilder


def _hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(c))) for c in rgb)


def analyze(data: bytes, settings: Settings) -> dict:
    """First pass after upload: detect colors/shapes, return a preview."""
    rgb, alpha = segmentation.load_image(data)
    fg = segmentation.foreground_mask(rgb, alpha, settings.remove_background)
    rgb, alpha, fg = segmentation.crop_to_foreground(rgb, alpha, fg)
    h, w = fg.shape
    label_map, palette = segmentation.quantize(rgb, fg, settings.max_colors)

    sx = settings.width_mm / max(w, 1)
    sy = settings.height_mm / max(h, 1)
    mm_per_px = (sx + sy) / 2.0
    min_area_px = settings.min_object_mm2 / max(sx * sy, 1e-9)
    label_map = segmentation.absorb_small_regions(
        label_map, min_area_px, 0.5 / max(mm_per_px, 1e-9))

    colors = []
    render = np.full((h, w, 4), 0, np.uint8)
    for idx, rgb_c in enumerate(palette):
        regs = regions.extract_regions(label_map, idx, min_area_px,
                                       settings.detail, mm_per_px)
        if not regs:
            continue
        # Majority stitch suggestion weighted by area.
        votes = {}
        px_total = 0
        for r in regs:
            s = regions.suggest_stitch(r, mm_per_px, settings.satin_width_mm)
            votes[s] = votes.get(s, 0) + r.area_px
            px_total += r.area_px
            render[r.mask > 0] = (*rgb_c, 255)
        colors.append({
            "index": idx,
            "hex": _hex(rgb_c),
            "pixels": px_total,
            "regions": len(regs),
            "suggested": max(votes, key=votes.get),
        })

    buf = io.BytesIO()
    Image.fromarray(render).save(buf, format="PNG")
    return {
        "width_px": w,
        "height_px": h,
        "aspect": w / max(h, 1),
        "colors": colors,
        "preview_png": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def cleanup(data: bytes, settings: Settings) -> bytes:
    """Vectorize-style cleanup: quantize to flat colors, absorb fragments,
    trace each color to smooth polygons and re-render a clean 2x PNG."""
    rgb, alpha = segmentation.load_image(data)
    fg = segmentation.foreground_mask(rgb, alpha, settings.remove_background)
    rgb, alpha, fg = segmentation.crop_to_foreground(rgb, alpha, fg)
    h, w = fg.shape
    label_map, palette = segmentation.quantize(rgb, fg, settings.max_colors)

    sx = settings.width_mm / max(w, 1)
    sy = settings.height_mm / max(h, 1)
    mm_per_px = (sx + sy) / 2.0
    min_area_px = settings.min_object_mm2 / max(sx * sy, 1e-9)
    label_map = segmentation.absorb_small_regions(
        label_map, min_area_px, 0.5 / max(mm_per_px, 1e-9))

    scale = 2
    out = np.zeros((h * scale, w * scale, 4), np.uint8)
    eps = regions.approx_epsilon(settings.detail)

    for idx, rgb_c in enumerate(palette):
        mask = regions.clean_mask((label_map == idx).astype(np.uint8),
                                  settings.detail)
        if mask.sum() == 0:
            continue
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        layer = np.zeros((h * scale, w * scale), np.uint8)
        smoothed = [_smooth_closed(c.reshape(-1, 2).astype(np.float64), eps)
                    for c in contours]
        for ci, pts in enumerate(smoothed):
            if pts is None or hierarchy[0][ci][3] != -1:
                continue  # holes drawn after outers
            cv2.fillPoly(layer, [(pts * scale).round().astype(np.int32)], 255)
        for ci, pts in enumerate(smoothed):
            if pts is None or hierarchy[0][ci][3] == -1:
                continue
            cv2.fillPoly(layer, [(pts * scale).round().astype(np.int32)], 0)
        # Slight dilation seals hairline seams between adjacent colors;
        # later (smaller, detail) colors simply draw over the overlap.
        layer = cv2.dilate(layer, np.ones((3, 3), np.uint8))
        out[layer > 0] = (*rgb_c, 255)

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue()


def _smooth_closed(pts: np.ndarray, eps: float):
    """Simplify then Chaikin-smooth a closed contour (returns None if tiny)."""
    if len(pts) < 3:
        return None
    ap = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2),
                          eps, True).reshape(-1, 2).astype(np.float64)
    if len(ap) < 3:
        return None
    for _ in range(2):
        nxt = []
        n = len(ap)
        for i in range(n):
            a, b = ap[i], ap[(i + 1) % n]
            nxt.append(a * 0.75 + b * 0.25)
            nxt.append(a * 0.25 + b * 0.75)
        ap = np.array(nxt)
    return ap


def digitize(data: bytes, settings: Settings) -> Tuple[Plan, dict, List[str]]:
    rgb, alpha = segmentation.load_image(data)
    fg = segmentation.foreground_mask(rgb, alpha, settings.remove_background)
    rgb, alpha, fg = segmentation.crop_to_foreground(rgb, alpha, fg)
    h, w = fg.shape
    sx = settings.width_mm / max(w, 1)
    sy = settings.height_mm / max(h, 1)
    mm_per_px = (sx + sy) / 2.0

    builder = PlanBuilder(
        max_stitch_mm=settings.max_stitch_mm,
        max_jump_mm=settings.max_jump_mm,
        trim_enabled=settings.trim_enabled,
        trim_threshold_mm=settings.trim_threshold_mm,
        auto_color_change=settings.auto_color_change,
    )

    if settings.line_art:
        _digitize_line_art(builder, rgb, fg, settings, sx, sy, mm_per_px)
    else:
        _digitize_colors(builder, rgb, fg, settings, sx, sy, mm_per_px)

    return builder.finish()


# ---------------------------------------------------------------- line art

def _digitize_line_art(builder: PlanBuilder, rgb, fg, s: Settings,
                       sx: float, sy: float, mm_per_px: float) -> None:
    mask = segmentation.dark_line_mask(rgb, fg).astype(np.uint8)
    mask = regions.clean_mask(mask, max(s.detail, 30))
    if mask.sum() == 0:
        return
    # Thread color = mean of the dark pixels.
    color = rgb[mask > 0].mean(axis=0)
    builder.start_color(_hex(tuple(color)))

    paths = lineart.centerline_paths(mask, min_len_px=max(3.0, 1.5 / mm_per_px))
    paths = lineart.order_paths(paths)
    paths = lineart.merge_ordered(paths, 0.9 / max(mm_per_px, 1e-9))
    for pts_px, widths_px in paths:
        path_mm = pts_px * np.array([sx, sy])
        widths_mm = widths_px * mm_per_px
        mean_w = float(np.mean(widths_mm)) if len(widths_mm) else 0.0
        if mean_w >= 1.0:
            w_arr = np.clip(widths_mm + 2 * s.pull_comp_mm, 0.8,
                            s.satin_width_mm)
            run = _satin_run(path_mm, w_arr, s)
            if run is not None:
                builder.add_run(run)
        else:
            for run in fills.running_stitch(path_mm, s.stitch_len_mm,
                                            s.line_passes):
                builder.add_run(run)


# ---------------------------------------------------------------- colors

def _digitize_colors(builder: PlanBuilder, rgb, fg, s: Settings,
                     sx: float, sy: float, mm_per_px: float) -> None:
    label_map, palette = segmentation.quantize(rgb, fg, s.max_colors)
    min_area_px = s.min_object_mm2 / max(sx * sy, 1e-9)
    label_map = segmentation.absorb_small_regions(
        label_map, min_area_px, 0.5 / max(mm_per_px, 1e-9))

    # Collect all colors first so we can pick a good stitching order:
    # large fill areas go down first, thin outline-like colors go last so
    # they cover the seams between fills.
    entries = []
    for idx, rgb_c in enumerate(palette):
        cs = s.color_settings.get(idx)
        if cs is not None and not cs.enabled:
            continue
        regs = regions.extract_regions(label_map, idx, min_area_px, s.detail,
                                       mm_per_px)
        if not regs:
            continue
        votes: dict = {}
        total_px = 0
        for r in regs:
            sug = regions.suggest_stitch(r, mm_per_px, s.satin_width_mm)
            votes[sug] = votes.get(sug, 0) + r.area_px
            total_px += r.area_px
        major = max(votes, key=votes.get)
        entries.append((idx, rgb_c, regs, cs, major, total_px))

    entries.sort(key=lambda e: (0 if e[4] == "fill" else 1, -e[5]))

    for idx, rgb_c, regs, cs, major, _ in entries:
        builder.start_color(_hex(rgb_c))
        regs = _order_regions(regs, builder, sx, sy)
        for r in regs:
            stitch = cs.stitch if cs is not None else "auto"
            if stitch == "auto":
                stitch = regions.suggest_stitch(r, mm_per_px, s.satin_width_mm)
            _stitch_region(builder, r, stitch, s, sx, sy, mm_per_px)


def _order_regions(regs, builder: PlanBuilder, sx: float, sy: float):
    remaining = list(regs)
    ordered = []
    pos = builder.pos if builder.pos is not None else np.zeros(2)
    while remaining:
        best, best_d = 0, float("inf")
        for i, r in enumerate(remaining):
            c = np.array([r.centroid[0] * sx, r.centroid[1] * sy])
            d = float(np.linalg.norm(c - pos))
            if d < best_d:
                best, best_d = i, d
        r = remaining.pop(best)
        ordered.append(r)
        pos = np.array([r.centroid[0] * sx, r.centroid[1] * sy])
    return ordered


def _stitch_region(builder: PlanBuilder, r, stitch: str, s: Settings,
                   sx: float, sy: float, mm_per_px: float) -> None:
    polys_mm = fills.contour_paths_mm(r.polys, sx, sy)

    if stitch == "fill":
        # Expand fills slightly under their neighbors so no fabric shows
        # between adjacent colors (later colors stitch over the overlap).
        overlap_px = max(1, int(round(0.35 / max(mm_per_px, 1e-6))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * overlap_px + 1, 2 * overlap_px + 1))
        dil = cv2.dilate(r.mask, kernel)
        contours, _ = cv2.findContours(dil, cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)
        expanded = []
        for c in contours:
            ap = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
            if ap.shape[0] >= 3:
                expanded.append(ap.astype(np.float64))
        if expanded:
            polys_mm = fills.contour_paths_mm(expanded, sx, sy)

    if stitch == "running":
        for poly in polys_mm:
            for run in fills.running_stitch(poly, s.stitch_len_mm, 1):
                builder.add_run(run)
        return

    if stitch == "satin":
        paths = lineart.centerline_paths(r.mask)
        paths = lineart.order_paths(paths)
        paths = lineart.merge_ordered(paths, 0.9 / max(mm_per_px, 1e-9))
        # Satin can only cover up to the width cap; if the shape is locally
        # wider, zigzag would leave bare margins — use fill instead.
        all_w = np.concatenate([w for _, w in paths]) if paths else np.array([])
        if all_w.size and np.percentile(all_w * mm_per_px, 85) > \
                s.satin_width_mm * 1.05:
            paths = []
        emitted = False
        for pts_px, widths_px in paths:
            path_mm = pts_px * np.array([sx, sy])
            if fills.path_length(path_mm) < 1.2:
                continue  # sub-millimeter nub, not worth a satin bead
            w_arr = np.clip(widths_px * mm_per_px + 2 * s.pull_comp_mm,
                            0.8, s.satin_width_mm)
            run = _satin_run(path_mm, w_arr, s)
            if run is not None:
                builder.add_run(run)
                emitted = True
        if emitted:
            return
        stitch = "fill"  # fallback when no usable centerline found

    # Tatami fill. With auto angle, stitch along the shape's long axis —
    # the per-region variation built-in machine patterns have.
    angle = s.fill_angle_deg
    if s.auto_fill_angle:
        angle = _region_angle(r.mask, default=s.fill_angle_deg)
    if s.underlay:
        under = fills.scanline_fill(
            polys_mm, angle + 90.0, spacing=2.5,
            stitch_len=3.5, pull_comp=0.0, inset=0.4)
        for run in under:
            builder.add_run(run)
    top = fills.scanline_fill(
        polys_mm, angle, spacing=s.density_mm,
        stitch_len=s.stitch_len_mm, pull_comp=s.pull_comp_mm)
    for run in top:
        builder.add_run(run)


def _region_angle(mask: np.ndarray, default: float) -> float:
    """Orientation of the region's major axis via image moments.
    Falls back to the default for round-ish shapes with no clear axis."""
    import math
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0:
        return default
    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]
    denom = mu20 + mu02
    if denom <= 0:
        return default
    eccentricity = math.hypot(mu20 - mu02, 2 * mu11) / denom
    if eccentricity < 0.18:
        return default
    return math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02))


def _satin_run(path_mm: np.ndarray, widths, s: Settings):
    zig = fills.satin_along_path(path_mm, widths, spacing=s.density_mm)
    if zig is None:
        return None
    if s.underlay:
        center = fills.resample_path(path_mm, min(s.stitch_len_mm, 2.5))
        if len(center) >= 2:
            return np.concatenate([center, center[::-1], zig])
    return zig
