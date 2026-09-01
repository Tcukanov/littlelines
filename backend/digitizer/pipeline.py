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
from .plan import CMD_JUMP, CMD_STITCH, Plan, PlanBuilder


def _hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(c))) for c in rgb)


def analyze(data: bytes, settings: Settings) -> dict:
    """First pass after upload: detect colors/shapes, return a preview."""
    rgb, alpha = segmentation.load_image(data)
    if settings.photo_mode:
        rgb = segmentation.paper_normalize(rgb)
        fg = segmentation.paper_foreground(rgb)
    else:
        fg = segmentation.foreground_mask(rgb, alpha,
                                          settings.remove_background)
    rgb, alpha, fg = segmentation.crop_to_foreground(rgb, alpha, fg)
    h, w = fg.shape
    label_map, palette = segmentation.quantize(
        rgb, fg, settings.max_colors, settings.palette_hint)

    sx = settings.width_mm / max(w, 1)
    sy = settings.height_mm / max(h, 1)
    mm_per_px = (sx + sy) / 2.0
    min_area_px = settings.min_object_mm2 / max(sx * sy, 1e-9)
    label_map, palette = segmentation.collapse_antialias_colors(
        label_map, palette, rgb, mm_per_px)
    label_map = segmentation.absorb_small_regions(
        label_map, min_area_px, 0.5 / max(mm_per_px, 1e-9),
        palette=palette, keep_px=0.5 / max(mm_per_px * mm_per_px, 1e-9))

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
    if settings.photo_mode:
        rgb = segmentation.paper_normalize(rgb)
        fg = segmentation.paper_foreground(rgb)
    else:
        fg = segmentation.foreground_mask(rgb, alpha,
                                          settings.remove_background)
    rgb, alpha, fg = segmentation.crop_to_foreground(rgb, alpha, fg)
    h, w = fg.shape
    label_map, palette = segmentation.quantize(
        rgb, fg, settings.max_colors, settings.palette_hint)

    sx = settings.width_mm / max(w, 1)
    sy = settings.height_mm / max(h, 1)
    mm_per_px = (sx + sy) / 2.0
    min_area_px = settings.min_object_mm2 / max(sx * sy, 1e-9)
    label_map, palette = segmentation.collapse_antialias_colors(
        label_map, palette, rgb, mm_per_px)
    label_map = segmentation.absorb_small_regions(
        label_map, min_area_px, 0.5 / max(mm_per_px, 1e-9),
        palette=palette, keep_px=0.5 / max(mm_per_px * mm_per_px, 1e-9))

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
    if settings.photo_mode:
        rgb = segmentation.paper_normalize(rgb)
        fg = segmentation.paper_foreground(rgb)
    else:
        fg = segmentation.foreground_mask(rgb, alpha,
                                          settings.remove_background)
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
        walk_mm=settings.walk_connector_mm,
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
    builder.travel_router = _make_travel_router(mask, sx, sy, s.stitch_len_mm)

    # Keep even tiny stub paths: dropping them severs the stroke graph at
    # junctions and turns one continuous walk into dozens of jumps.
    paths = lineart.centerline_paths(mask, min_len_px=2.0)
    walk = lineart.graph_walk(paths)
    for pts_px, widths_px, retrace in walk:
        path_mm = pts_px * np.array([sx, sy])
        widths_mm = widths_px * mm_per_px
        mean_w = float(np.mean(widths_mm)) if len(widths_mm) else 0.0
        if retrace or mean_w < 1.0:
            passes = 1 if retrace else max(1, s.line_passes - 1)
            for run in fills.running_stitch(path_mm, s.stitch_len_mm, passes):
                builder.add_run(run)
        else:
            w_arr = np.clip(widths_mm + 2 * s.pull_comp_mm, 0.8,
                            s.satin_width_mm)
            run = _satin_run(path_mm, w_arr, s)
            if run is not None:
                builder.add_run(run)


# ---------------------------------------------------------------- colors

def _digitize_colors(builder: PlanBuilder, rgb, fg, s: Settings,
                     sx: float, sy: float, mm_per_px: float) -> None:
    label_map, palette = segmentation.quantize(
        rgb, fg, s.max_colors, s.palette_hint)
    min_area_px = s.min_object_mm2 / max(sx * sy, 1e-9)
    label_map, palette = segmentation.collapse_antialias_colors(
        label_map, palette, rgb, mm_per_px)
    label_map = segmentation.absorb_small_regions(
        label_map, min_area_px, 0.5 / max(mm_per_px, 1e-9),
        palette=palette, keep_px=0.5 / max(mm_per_px * mm_per_px, 1e-9))

    # Apply user color merges: pixels of a merged color join their target
    # color, removing a thread block (and enlarging routing territory).
    for m_idx, m_cs in s.color_settings.items():
        tgt = m_cs.merge_into
        hops = 0
        while (0 <= tgt < len(palette) and tgt != m_idx and hops < 12
               and s.color_settings.get(tgt) is not None
               and s.color_settings[tgt].merge_into >= 0):
            tgt = s.color_settings[tgt].merge_into
            hops += 1
        if 0 <= tgt < len(palette) and tgt != m_idx:
            label_map[label_map == m_idx] = tgt

    # Collect all colors first so we can pick a good stitching order:
    # large fill areas go down first, thin outline-like colors go last so
    # they cover the seams between fills. If the small-object threshold
    # would wipe (nearly) everything — dense fine-detail artwork — relax it
    # rather than emitting an empty design.
    occupied_px = max(int((label_map >= 0).sum()), 1)
    entries = []
    for attempt_area in (min_area_px, min_area_px / 5.0, 4.0):
        entries = []
        for idx, rgb_c in enumerate(palette):
            cs = s.color_settings.get(idx)
            if cs is not None and not cs.enabled:
                continue
            regs = regions.extract_regions(label_map, idx, attempt_area,
                                           s.detail, mm_per_px)
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
        kept_px = sum(e[5] for e in entries)
        if kept_px >= 0.55 * occupied_px:
            min_area_px = attempt_area
            break

    entries.sort(key=lambda e: (0 if e[4] == "fill" else 1, -e[5]))

    # The base (largest) color drives merging and burying. Merge it FIRST,
    # then classify: a body divided into cells by sketch lines looks
    # stroke-like until the cells merge into one fill.
    if entries:
        entries.sort(key=lambda e: -e[5])  # largest pixel count first
        idx0, rgb0, regs0, cs0, major0, cnt0 = entries[0]
        mask0 = (label_map == idx0).astype(np.uint8)
        r_px0 = max(1, int(round(1.4 / max(mm_per_px, 1e-9))))
        kern0 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * r_px0 + 1, 2 * r_px0 + 1))
        merged0 = (cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, kern0)
                   & (label_map >= 0).astype(np.uint8))
        m_regs = regions.regions_from_mask(merged0, min_area_px, s.detail,
                                           mm_per_px)
        if m_regs and len(m_regs) <= max(1, len(regs0) // 2):
            votes0: dict = {}
            for r in m_regs:
                sug = regions.suggest_stitch(r, mm_per_px, s.satin_width_mm)
                votes0[sug] = votes0.get(sug, 0) + r.area_px
            major_m = max(votes0, key=votes0.get)
            # Adopt the closed mask only for FILL bodies. Closing distorts
            # stroke networks: it fills junction crotches and the skeleton
            # then shortcuts across them, losing ring segments.
            if major_m == "fill":
                entries[0] = (idx0, rgb0, m_regs, cs0, major_m, cnt0)
        entries[1:] = sorted(
            entries[1:], key=lambda e: (0 if e[4] == "fill" else 1, -e[5]))

    # Fill colors may merge across thin gaps that a LATER color covers
    # (e.g. a body split into cells by sketch lines becomes ONE fill and
    # the lines stitch on top) — never across bare background.
    occupied = (label_map >= 0).astype(np.uint8)
    stitched = np.zeros_like(occupied)

    # Burying pass: detail islands of later colors that sit on the base
    # fill stitch FIRST, connected by free travel runs — the base fill
    # then stitches over those runs and buries them (pro technique).
    # Costs one extra color change, saves an island's trim each.
    buried_ids: set = set()
    islands_excl = None
    prepass_ranges: dict = {}
    if len(entries) >= 2:
        f_idx, f_rgb, f_regs, f_cs, f_major, _ = entries[0]
        f_eff = f_cs.stitch if f_cs is not None and f_cs.stitch != "auto" \
            else f_major
        if f_eff == "fill":
            f_mask_raw = (label_map == f_idx).astype(np.uint8)
            r_px = max(1, int(round(1.4 / max(mm_per_px, 1e-9))))
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
            f_merged = (cv2.morphologyEx(f_mask_raw, cv2.MORPH_CLOSE, kern)
                        & occupied).astype(np.uint8)
            ring_k = np.ones((5, 5), np.uint8)
            island_jobs = []
            for ei in range(1, len(entries)):
                for r in entries[ei][2]:
                    if r.bbox_px * mm_per_px > 40:
                        continue
                    ring = (cv2.dilate(r.mask, ring_k) > 0) & (r.mask == 0)
                    # Eligible if surrounded by ANY artwork that stitches
                    # after this pre-pass (everything does) — the travel
                    # runs get buried regardless of which color covers them.
                    if ring.any() and float(occupied[ring].mean()) > 0.5:
                        island_jobs.append((ei, r))
            if len(island_jobs) >= 2:
                from collections import defaultdict
                by_color = defaultdict(list)
                for ei, r in island_jobs:
                    by_color[ei].append(r)
                    buried_ids.add(id(r))
                islands_mask = np.zeros_like(occupied)
                for ei, regs_i in by_color.items():
                    idx_c, rgb_c, _, cs_c, _, _ = entries[ei]
                    c_mask = (label_map == idx_c).astype(np.uint8)
                    th_hex = _hex(rgb_c)
                    if cs_c is not None and cs_c.thread_hex:
                        th_hex = cs_c.thread_hex
                    builder.start_color(th_hex)
                    _prepass_start = len(builder.plan.events)
                    # Everything stitches after this pre-pass, so travel
                    # may run over the whole artwork footprint.
                    builder.travel_router = _make_travel_router(
                        (occupied | c_mask).astype(np.uint8), sx, sy,
                        s.stitch_len_mm)
                    for r in _order_regions(regs_i, builder, sx, sy):
                        st = cs_c.stitch if cs_c is not None else "auto"
                        if st == "auto":
                            st = regions.suggest_stitch(
                                r, mm_per_px, s.satin_width_mm)
                        _stitch_region(builder, r, st, s, sx, sy, mm_per_px,
                                       allowed=occupied)
                        islands_mask |= r.mask
                    builder.travel_router = None
                    prepass_ranges.setdefault(idx_c, []).append(
                        (_prepass_start, len(builder.plan.events)))
                excl_px = max(1, int(round(0.3 / max(mm_per_px, 1e-9))))
                islands_excl = cv2.dilate(
                    islands_mask, np.ones((2 * excl_px + 1,) * 2, np.uint8))

    # Future-coverage suffix masks: travel for block k may also run under
    # anything a LATER block will stitch over (it gets buried), not only
    # its own color's territory.
    raw_masks = [(label_map == e[0]).astype(np.uint8) for e in entries]
    later_cover = []
    acc = np.zeros_like(occupied)
    for m in reversed(raw_masks):
        later_cover.append(acc.copy())
        acc |= m
    later_cover.reverse()

    for k, (idx, rgb_c, regs, cs, major, _) in enumerate(entries):
        color_mask = raw_masks[k]
        effective = cs.stitch if cs is not None and cs.stitch != "auto" \
            else major
        if effective == "fill":
            r_px = max(1, int(round(1.4 / max(mm_per_px, 1e-9))))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
            closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            allowed = color_mask | (occupied & (1 - stitched))
            merged = (closed & allowed).astype(np.uint8)
            merged_regs = regions.regions_from_mask(
                merged, min_area_px, s.detail, mm_per_px)
            if merged_regs:
                regs = merged_regs
        stitched |= color_mask

        regs = [r for r in regs if id(r) not in buried_ids]
        if not regs:
            continue
        thread_hex = _hex(rgb_c)
        if cs is not None and cs.thread_hex:
            thread_hex = cs.thread_hex
        builder.start_color(thread_hex)
        builder.travel_router = _make_travel_router(
            (color_mask | later_cover[k]).astype(np.uint8), sx, sy,
            s.stitch_len_mm)
        regs = _order_regions(regs, builder, sx, sy)
        excl = islands_excl if idx == entries[0][0] else None
        _main_start = len(builder.plan.events)
        for r in regs:
            stitch = cs.stitch if cs is not None else "auto"
            if stitch == "auto":
                stitch = regions.suggest_stitch(r, mm_per_px, s.satin_width_mm)
            _stitch_region(builder, r, stitch, s, sx, sy, mm_per_px,
                           exclude=excl, allowed=occupied)
        # Verify coverage of this color (incl. buried pre-pass islands) and
        # patch anything the generators missed.
        slices = [builder.plan.events[a:b]
                  for a, b in prepass_ranges.get(idx, [])]
        slices.append(builder.plan.events[_main_start:])
        _patch_uncovered(builder, color_mask, slices, s, sx, sy, mm_per_px)
        builder.travel_router = None


def _patch_uncovered(builder: PlanBuilder, mask: np.ndarray, event_slices,
                     s: Settings, sx: float, sy: float,
                     mm_per_px: float) -> None:
    """Self-healing coverage check: rasterize what was actually stitched
    for this color, diff against its mask, and stitch whatever significant
    area was missed (guards against edge cases in skeleton/walk logic)."""
    cover = np.zeros_like(mask)
    th = max(2, int(round(0.8 / max(mm_per_px, 1e-9))))
    for events in event_slices:
        prev = None
        for cmd, x, y in events:
            if cmd == CMD_STITCH:
                if prev is not None:
                    cv2.line(cover,
                             (int(round(prev[0] / sx)),
                              int(round(prev[1] / sy))),
                             (int(round(x / sx)), int(round(y / sy))),
                             1, thickness=th)
                prev = (x, y)
            elif cmd == CMD_JUMP:
                prev = (x, y)
    core = cv2.erode(mask, np.ones((3, 3), np.uint8))
    uncov = (core & (1 - cover)).astype(np.uint8)
    if not uncov.any():
        return
    min_patch_px = 1.5 / max(mm_per_px * mm_per_px, 1e-9)
    regs = regions.regions_from_mask(uncov, min_patch_px, s.detail, mm_per_px)
    for r in regs:
        stitch = regions.suggest_stitch(r, mm_per_px, s.satin_width_mm)
        _stitch_region(builder, r, stitch, s, sx, sy, mm_per_px)


def _make_travel_router(mask: np.ndarray, sx: float, sy: float,
                        stitch_len_mm: float):
    """BFS pathfinder over a color's footprint: running-stitch travel over
    same-color areas is invisible (covered before or after), so separate
    regions can connect without a trim — the commercial near-zero-trim
    technique."""
    from collections import deque
    # Half-resolution grid keeps BFS fast enough to route across the whole
    # design; dilating first keeps thin strokes connected after sampling.
    # `strict` is the undilated footprint — the truly covered territory.
    grid = (cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0)[::2, ::2]
    strict = (mask > 0)[::2, ::2]
    h, w = grid.shape
    # Component labels let us reject unroutable pairs in O(1) instead of
    # exhausting the BFS budget on every impossible route.
    _, comp_labels = cv2.connectedComponents(grid.astype(np.uint8), 4)

    def to_cell(p):
        return (min(h - 1, max(0, int(round(p[1] / sy / 2)))),
                min(w - 1, max(0, int(round(p[0] / sx / 2)))))

    def snap(c):
        if grid[c]:
            return c
        for r in range(1, 5):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    y, x = c[0] + dy, c[1] + dx
                    if 0 <= y < h and 0 <= x < w and grid[y, x]:
                        return (y, x)
        return None

    def router(a_mm, b_mm):
        a = snap(to_cell(a_mm))
        b = snap(to_cell(b_mm))
        if a is None or b is None:
            router.reason = "SNAP_FAIL"
            return None
        if comp_labels[a] != comp_labels[b]:
            router.reason = "SEPARATE_ISLAND"
            return None
        straight = float(np.hypot(b_mm[0] - a_mm[0], b_mm[1] - a_mm[1]))
        q = deque([a])
        parent = {a: None}
        found = False
        budget = 150000
        while q and budget:
            budget -= 1
            cur = q.popleft()
            if cur == b:
                found = True
                break
            y, x = cur
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] \
                        and (ny, nx) not in parent:
                    parent[(ny, nx)] = cur
                    q.append((ny, nx))
        if not found:
            router.reason = "NO_ROUTE"
            return None
        cells = []
        cur = b
        while cur is not None:
            cells.append(cur)
            cur = parent[cur]
        cells.reverse()
        path = np.array([[c[1] * sx * 2, c[0] * sy * 2] for c in cells])
        # Coverage certainty controls how long a hidden route may be: a
        # route fully on covered territory is allowed at any sane length;
        # one that skirts edges gets the conservative cap.
        coverage = float(np.mean([strict[c] for c in cells])) if cells else 0.0
        if coverage >= 0.99:
            max_len = 400.0
        elif coverage >= 0.95:
            max_len = 180.0
        else:
            max_len = max(80.0, straight * 2.0)
        if fills.path_length(path) > max_len:
            router.reason = "ROUTE_TOO_LONG"
            return None
        return fills.resample_path(path, min(stitch_len_mm, 2.5))

    return router


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
                   sx: float, sy: float, mm_per_px: float,
                   exclude=None, allowed=None) -> None:
    if stitch == "running":
        polys_mm = fills.contour_paths_mm(r.polys, sx, sy)
        for poly in polys_mm:
            for run in fills.running_stitch(poly, s.stitch_len_mm, 1):
                builder.add_run(run)
        return

    # Physical guard, applied even when the user forces satin: a small
    # compact shape (an eye, a dot, a nose) has no meaningful centerline —
    # satin there produces crossing spaghetti. Such shapes must be filled.
    if stitch == "satin":
        compact = (r.area_px / max(r.p85_thickness_px ** 2, 1.0)) < 2.2
        if compact and r.bbox_px * mm_per_px <= 8.0:
            stitch = "fill"

    # Hybrid split: only for STROKE-DOMINATED shapes (an outline network
    # with a few thick blobs). A merged fill body must stay one fill — its
    # narrow leftovers are not strokes to satin, they're part of the fill.
    thin_mask, blob_mask = _split_thick_blobs(r.mask, s, mm_per_px)
    if blob_mask is not None:
        thin_frac = float(thin_mask.sum()) / max(float(r.mask.sum()), 1.0)
        thin_area_mm2 = float(thin_mask.sum()) * mm_per_px * mm_per_px
        if thin_frac >= 0.35 and thin_area_mm2 >= 10.0:
            _fill_mask(builder, blob_mask, s, sx, sy, mm_per_px, exclude,
                       allowed)
            _satin_network(builder, thin_mask, s, sx, sy, mm_per_px)
        else:
            elong0 = r.area_px / max(r.p85_thickness_px ** 2, 1.0)
            if not (elong0 >= 4.0 and r.bbox_px * mm_per_px > 25.0
                    and _stroke_fill(builder, r.mask, s, sx, sy, mm_per_px,
                                     exclude, allowed)):
                _fill_mask(builder, r.mask, s, sx, sy, mm_per_px, exclude,
                           allowed)
        return

    if stitch == "satin":
        if _satin_network(builder, r.mask, s, sx, sy, mm_per_px):
            return
        stitch = "fill"  # fallback when no usable centerline found

    if stitch == "fill":
        elong = r.area_px / max(r.p85_thickness_px ** 2, 1.0)
        if elong >= 4.0 and r.bbox_px * mm_per_px > 25.0:
            if _stroke_fill(builder, r.mask, s, sx, sy, mm_per_px,
                            exclude, allowed):
                return
        _fill_mask(builder, r.mask, s, sx, sy, mm_per_px, exclude, allowed)


def _split_thick_blobs(mask: np.ndarray, s: Settings, mm_per_px: float):
    """Return (thin_mask, blob_mask|None): blobs = areas wider than the
    satin cap, reconstructed from the distance-transform core."""
    thr_px = (s.satin_width_mm / 2.0) / max(mm_per_px, 1e-9)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    core = (dt > thr_px * 1.3).astype(np.uint8)
    if not core.any():
        return mask, None
    # Reconstruct with a radius >= the core threshold so a solid shape
    # comes back whole (no leftover rim ring to satin separately).
    k = 2 * int(round(thr_px * 1.3)) + 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blob = cv2.dilate(core, kernel) & mask
    min_blob_px = 12.0 / max(mm_per_px * mm_per_px, 1e-9)  # >= 12 mm^2
    n, comp, stats, _ = cv2.connectedComponentsWithStats(blob, 8)
    keep = np.zeros_like(blob)
    for ci in range(1, n):
        if stats[ci][4] >= min_blob_px:
            keep[comp == ci] = 1
    if not keep.any():
        return mask, None
    thin = mask.copy()
    thin[keep > 0] = 0
    # Drop crumbs too small to stitch as strokes (blob edges cover them).
    min_crumb_px = 4.0 / max(mm_per_px * mm_per_px, 1e-9)
    n2, comp2, stats2, _ = cv2.connectedComponentsWithStats(thin, 8)
    for ci in range(1, n2):
        if stats2[ci][4] < min_crumb_px:
            thin[comp2 == ci] = 0
    return thin, keep


def _satin_network(builder: PlanBuilder, mask: np.ndarray, s: Settings,
                   sx: float, sy: float, mm_per_px: float) -> bool:
    """Centerline satin over a stroke network via continuous graph walk."""
    if mask.sum() == 0:
        return False
    paths = lineart.centerline_paths(mask, min_len_px=2.0)
    start_near = None
    if builder.pos is not None:
        start_near = (builder.pos[0] / sx, builder.pos[1] / sy)
    emitted = False
    for pts_px, widths_px, retrace in lineart.graph_walk(paths, start_near):
        path_mm = pts_px * np.array([sx, sy])
        if retrace or fills.path_length(path_mm) < 1.2:
            # Travel back along the stitched branch (or a tiny nub):
            # running stitch keeps the walk continuous, no jump.
            for run in fills.running_stitch(path_mm, s.stitch_len_mm, 1):
                builder.add_run(run)
            emitted = emitted or not retrace
            continue
        w_arr = np.clip(widths_px * mm_per_px + 2 * s.pull_comp_mm,
                        0.8, s.satin_width_mm)
        run = _satin_run(path_mm, w_arr, s)
        if run is not None:
            builder.add_run(run)
            emitted = True
    return emitted


def _stroke_fill(builder: PlanBuilder, mask: np.ndarray, s: Settings,
                 sx: float, sy: float, mm_per_px: float,
                 exclude=None, allowed=None) -> bool:
    """Fill a long stroke in short bands whose rows run across it, so the
    stitching follows the stroke's direction instead of cutting it at one
    global angle (which staircases along diagonal and curved edges)."""
    import math

    paths = lineart.centerline_paths(mask, min_len_px=4.0)
    if not paths:
        return False
    seg_px = max(8.0, 12.0 / max(mm_per_px, 1e-9))  # ~12mm bands
    reach = float(max(mask.shape)) * 1.5
    done = np.zeros_like(mask)
    emitted = False
    # One underlay for the whole stroke; bands would each add their own.
    if s.underlay:
        _underlay_only(builder, mask, s, sx, sy, mm_per_px)
    import dataclasses
    s_band = dataclasses.replace(s, underlay=False)

    for pts_px, _w, retrace in lineart.graph_walk(paths):
        if retrace:
            continue
        path = fills.resample_path(pts_px, seg_px)
        if len(path) < 2:
            continue
        for i in range(len(path) - 1):
            p0, p1 = path[i], path[i + 1]
            d = p1 - p0
            n = float(np.hypot(d[0], d[1]))
            if n < 1e-6:
                continue
            ux, uy = d[0] / n, d[1] / n
            px_, py_ = -uy, ux           # perpendicular
            # Band: the slab between the two cut lines, widened to cover
            # the stroke, clipped to the region and to what is left.
            a0 = (p0[0] - ux * 0.5 + px_ * reach, p0[1] - uy * 0.5 + py_ * reach)
            a1 = (p0[0] - ux * 0.5 - px_ * reach, p0[1] - uy * 0.5 - py_ * reach)
            b1 = (p1[0] + ux * 0.5 - px_ * reach, p1[1] + uy * 0.5 - py_ * reach)
            b0 = (p1[0] + ux * 0.5 + px_ * reach, p1[1] + uy * 0.5 + py_ * reach)
            quad = np.array([a0, a1, b1, b0], np.int32)
            band = np.zeros_like(mask)
            cv2.fillPoly(band, [quad], 1)
            band = (band & mask & (1 - done)).astype(np.uint8)
            if band.sum() < 4:
                continue
            done |= band
            # Rows run across the stroke: perpendicular to the tangent.
            angle = math.degrees(math.atan2(uy, ux)) + 90.0
            _fill_mask(builder, band, s_band, sx, sy, mm_per_px, exclude,
                       allowed, angle_deg=angle)
            emitted = True

    leftover = (mask & (1 - done)).astype(np.uint8)
    if emitted and leftover.sum() * mm_per_px * mm_per_px > 2.0:
        _fill_mask(builder, leftover, s_band, sx, sy, mm_per_px, exclude,
                   allowed)
    return emitted


def _underlay_only(builder: PlanBuilder, mask: np.ndarray, s: Settings,
                   sx: float, sy: float, mm_per_px: float) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        ap = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if ap.shape[0] >= 3:
            polys.append(ap.astype(np.float64))
    if not polys:
        return
    polys_mm = fills.contour_paths_mm(polys, sx, sy)
    angle = _region_angle(mask, default=s.fill_angle_deg) + 90.0
    for run in fills.scanline_fill(polys_mm, angle, spacing=2.5,
                                   stitch_len=3.5, pull_comp=0.0, inset=0.4):
        builder.add_run(run)


def _fill_mask(builder: PlanBuilder, mask: np.ndarray, s: Settings,
               sx: float, sy: float, mm_per_px: float,
               exclude=None, allowed=None, angle_deg=None) -> None:
    """Tatami-fill a mask: expand slightly under neighbors so no fabric
    shows between adjacent colors, angle along the shape's long axis,
    underlay first, then top stitching. `exclude` marks already-stitched
    detail islands the fill must stay off of."""
    overlap_px = max(1, int(round(0.35 / max(mm_per_px, 1e-6))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * overlap_px + 1, 2 * overlap_px + 1))
    dil = cv2.dilate(mask, kernel)
    if allowed is not None:
        # Only grow under territory another color will stitch over. Growing
        # into bare fabric would shrink holes (letter counters, highlights).
        dil = (dil & (allowed | mask)).astype(np.uint8)
    if exclude is not None:
        dil = dil.copy()
        dil[exclude > 0] = 0
    contours, _ = cv2.findContours(dil, cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        ap = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if ap.shape[0] >= 3:
            polys.append(ap.astype(np.float64))
    if not polys:
        return
    polys_mm = fills.contour_paths_mm(polys, sx, sy)

    angle = s.fill_angle_deg
    if angle_deg is not None:
        angle = angle_deg
    elif s.auto_fill_angle:
        angle = _region_angle(mask, default=s.fill_angle_deg)
    # Within ONE fill region, short hops (around interior holes) may drag
    # thread instead of trimming: the holes belong to other colors that
    # stitch on top, so the drag ends up covered — commercial files do the
    # same. Long hops still trim.
    saved_threshold = builder.trim_threshold
    builder.trim_threshold = max(saved_threshold, 10.0)
    try:
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
    finally:
        builder.trim_threshold = saved_threshold


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
    # Alternating zigzag: same-side pitch is 2x the sample spacing, so
    # sample at half density to hit the configured density on each edge.
    zig = fills.satin_along_path(path_mm, widths,
                                 spacing=max(0.15, s.density_mm * 0.5))
    if zig is None:
        return None
    if s.underlay:
        center = fills.resample_path(path_mm, min(s.stitch_len_mm, 2.5))
        if len(center) >= 2:
            return np.concatenate([center, center[::-1], zig])
    return zig
