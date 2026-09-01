"""Image loading, background removal and color quantization."""
from __future__ import annotations

import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

MAX_DIM = 900  # working resolution


def load_image(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8 HxWx3 composited on white, alpha uint8 HxW)."""
    Image.MAX_IMAGE_PIXELS = 400_000_000
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGBA")
    w, h = img.size
    if w * h > 4_000_000:
        # Cheap box-filter pre-reduction so huge uploads don't stall LANCZOS.
        factor = max(2, int(((w * h) / 4_000_000) ** 0.5))
        img = img.reduce(factor)
        w, h = img.size
    scale = min(1.0, MAX_DIM / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    alpha = arr[..., 3]
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    rgb = (arr[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a))
    # Median filter kills JPEG noise and anti-aliasing fringe colors that
    # would otherwise become their own (unstitchable) palette entries.
    rgb = cv2.medianBlur(rgb.astype(np.uint8), 3)
    return rgb, alpha


def foreground_mask(rgb: np.ndarray, alpha: np.ndarray,
                    remove_background: bool) -> np.ndarray:
    """Boolean mask of pixels that belong to the artwork."""
    h, w = alpha.shape
    has_alpha = (alpha < 250).sum() > 0.005 * h * w
    if has_alpha:
        fg = alpha >= 128
        # Shave the matte edge: cut-out PNGs carry a 1-3 px light halo that
        # digitizes into ugly slivers around every shape. ADAPTIVE: thin
        # line art would be destroyed by the shave, so back off when the
        # erosion removes too much of the artwork.
        fg = _adaptive_shave(fg, iterations=2)
    else:
        fg = np.ones((h, w), bool)
        if remove_background:
            fg &= ~_border_flood(rgb, fg)
            fg = _adaptive_shave(fg, iterations=1)
    return fg


def _adaptive_shave(fg: np.ndarray, iterations: int) -> np.ndarray:
    total = int(fg.sum())
    if total == 0:
        return fg
    k = np.ones((3, 3), np.uint8)
    for it in range(iterations, 0, -1):
        shaved = cv2.erode(fg.astype(np.uint8), k, iterations=it)
        if int(shaved.sum()) >= 0.72 * total:
            return shaved.astype(bool)
    return fg


def _border_flood(rgb: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Flood fill from the border to find a solid-ish background."""
    h, w = rgb.shape[:2]
    img = rgb.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8) | 8
    tol = (34, 34, 34)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
    for x, y in seeds:
        if fg[y, x] and mask[y + 1, x + 1] == 0:
            cv2.floodFill(img, mask, (x, y), 0, tol, tol, flags)
    return mask[1:-1, 1:-1] > 0


def crop_to_foreground(rgb: np.ndarray, alpha: np.ndarray, fg: np.ndarray,
                       margin: int = 2):
    """Crop all arrays to the artwork bounding box so the requested output
    size applies to the artwork itself, not the canvas. Stray specks are
    ignored for the bbox so junk pixels can't inflate the design size."""
    fg_u8 = fg.astype(np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(fg_u8, 8)
    if n > 2:
        clean = np.zeros_like(fg_u8)
        for ci in range(1, n):
            if stats[ci][4] >= 16:
                clean[comp == ci] = 1
        if clean.any():
            fg = clean.astype(bool)
    ys, xs = np.nonzero(fg)
    if xs.size == 0:
        return rgb, alpha, fg
    h, w = fg.shape
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + 1 + margin)
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + 1 + margin)
    return rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1], fg[y0:y1, x0:x1]


def quantize(rgb: np.ndarray, fg: np.ndarray, max_colors: int,
             palette_hint=None) -> Tuple[np.ndarray,
                                         List[Tuple[int, int, int]]]:
    """Reduce artwork to at most max_colors colors.

    Clustering runs in Lab space (perceptual distance), so dark outlines
    stay separate from dark fills and skin tones don't merge with highlights.
    Returns (label_map HxW int32 with -1 = background, palette [(r,g,b)...])
    ordered by pixel count descending.
    """
    label_map = np.full(fg.shape, -1, np.int32)
    if not fg.any():
        return label_map, []

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pts_lab = lab[fg]
    pts_rgb = rgb[fg].astype(np.float32)

    if palette_hint:
        # Palette from an SVG: use the declared colors, but auto-traced
        # SVGs often declare a separate shade for every anti-alias band.
        # Merge perceptually-close hints into their dominant neighbor and
        # honor the Max colors setting, so the thread list stays minimal.
        hint_rgb = np.array(
            [[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
             for h in palette_hint], np.uint8)
        hint_lab = cv2.cvtColor(hint_rgb.reshape(1, -1, 3),
                                cv2.COLOR_RGB2LAB).reshape(-1, 3) \
            .astype(np.float32)
        d = np.linalg.norm(pts_lab[:, None, :] - hint_lab[None], axis=2)
        assign = d.argmin(axis=1)
        counts = np.bincount(assign, minlength=len(palette_hint)) \
            .astype(np.int64)

        # Agglomerative merge: absorb the smaller of the closest pair while
        # any pair is perceptually close OR we exceed the color budget.
        alive = [i for i in range(len(palette_hint))
                 if counts[i] > 0.001 * max(len(assign), 1)]
        if not alive:
            alive = [int(np.argmax(counts))]
        parent = list(range(len(palette_hint)))
        for i in range(len(palette_hint)):
            if i not in alive:  # negligible hint: fold into nearest kept
                parent[i] = min(alive, key=lambda a: float(
                    np.linalg.norm(hint_lab[i] - hint_lab[a])))
        while len(alive) > 1:
            bi = bj = -1
            bd = float("inf")
            for a in range(len(alive)):
                for b in range(a + 1, len(alive)):
                    dd = float(np.linalg.norm(hint_lab[alive[a]]
                                              - hint_lab[alive[b]]))
                    if dd < bd:
                        bd, bi, bj = dd, alive[a], alive[b]
            if bd >= 14.0 and len(alive) <= max_colors:
                break
            small, big = (bi, bj) if counts[bi] < counts[bj] else (bj, bi)
            parent[small] = big
            counts[big] += counts[small]
            alive.remove(small)

        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i

        if alive:
            final = {c: n for n, c in enumerate(alive)}
            assign2 = np.array([final[root(a)] for a in assign], np.int32)
            counts2 = np.bincount(assign2, minlength=len(alive))
            order = np.argsort(-counts2)
            remap = np.empty(len(alive), np.int32)
            remap[order] = np.arange(len(alive))
            label_map[fg] = remap[assign2]
            return label_map, [tuple(int(v) for v in hint_rgb[alive[i]])
                               for i in order]

    uniq = np.unique(rgb[fg].reshape(-1, 3), axis=0)
    if uniq.shape[0] <= max_colors:
        centers = cv2.cvtColor(uniq.reshape(1, -1, 3),
                               cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
        k = min(max_colors, pts_lab.shape[0])
        _, _, centers = cv2.kmeans(pts_lab, k, None, criteria, 4,
                                   cv2.KMEANS_PP_CENTERS)

    centers = _merge_close(centers, threshold=13.0)

    # Assign every foreground pixel to its nearest center (in Lab).
    d = np.linalg.norm(pts_lab[:, None, :] - centers[None, :, :], axis=2)
    assign = d.argmin(axis=1)

    # Order palette by usage, largest first (stitch big shapes first).
    counts = np.bincount(assign, minlength=centers.shape[0])
    order = np.argsort(-counts)
    remap = np.empty(centers.shape[0], np.int32)
    remap[order] = np.arange(centers.shape[0])
    label_map[fg] = remap[assign]

    # Thread colors: the mean RGB of the pixels actually assigned to each
    # cluster (truer to the artwork than converting Lab centers back).
    palette = []
    for i in order:
        sel = pts_rgb[assign == i]
        c = sel.mean(axis=0) if len(sel) else np.zeros(3)
        palette.append(tuple(min(255, max(0, int(round(v)))) for v in c))
    return label_map, palette


def _merge_close(centers: np.ndarray, threshold: float) -> np.ndarray:
    """Merge cluster centers that are perceptually the same color (Lab)."""
    centers = list(centers)
    merged = True
    while merged and len(centers) > 1:
        merged = False
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if np.linalg.norm(centers[i] - centers[j]) < threshold:
                    centers[i] = (centers[i] + centers[j]) / 2.0
                    centers.pop(j)
                    merged = True
                    break
            if merged:
                break
    return np.array(centers, np.float32)


def collapse_antialias_colors(label_map: np.ndarray, palette,
                              rgb: np.ndarray, mm_per_px: float):
    """Auto-merge anti-aliasing fringe colors into dominant ones.

    A minor color (small share of the artwork) that is fringe-thin or lies
    in Lab space between two dominant colors is an edge-smoothing artifact,
    not a thread color. Its pixels get reassigned pixel-by-pixel to the
    perceptually nearest dominant color. A small but SOLID accent color
    (a red nose) matches neither test and survives."""
    n = len(palette)
    if n <= 2:
        return label_map, palette
    total = int((label_map >= 0).sum())
    if total == 0:
        return label_map, palette
    counts = [int((label_map == i).sum()) for i in range(n)]
    arr = np.clip(np.array(palette, np.float32), 0, 255).astype(np.uint8).reshape(1, -1, 3)
    lab_pal = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3) \
        .astype(np.float32)

    dominants = [i for i in range(n) if counts[i] / total >= 0.12]
    if len(dominants) < 2:
        dominants = sorted(range(n), key=lambda i: -counts[i])[:2]

    aa = []
    for i in range(n):
        if i in dominants or counts[i] == 0:
            continue
        mask = (label_map == i).astype(np.uint8)
        dtm = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        mean_th = 2.0 * float(dtm[mask > 0].mean()) * mm_per_px
        thin = mean_th < 0.8
        between = False
        for a in range(len(dominants)):
            for b in range(a + 1, len(dominants)):
                A = lab_pal[dominants[a]]
                B = lab_pal[dominants[b]]
                P = lab_pal[i]
                AB = B - A
                t = float(np.dot(P - A, AB)) / max(float(np.dot(AB, AB)),
                                                   1e-9)
                if 0.0 <= t <= 1.0 and \
                        float(np.linalg.norm(P - (A + t * AB))) < 20.0:
                    between = True
        if thin or between:
            aa.append(i)
    if not aa:
        return label_map, palette

    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    keep = [i for i in range(n) if i not in aa]
    keep_lab = lab_pal[keep]
    for i in aa:
        sel = label_map == i
        if not sel.any():
            continue
        px = lab_img[sel]
        d = np.linalg.norm(px[:, None, :] - keep_lab[None, :, :], axis=2)
        label_map[sel] = np.array(keep)[d.argmin(axis=1)]

    out = np.full_like(label_map, -1)
    for new_i, old_i in enumerate(keep):
        out[label_map == old_i] = new_i
    return out, [palette[i] for i in keep]


def absorb_small_regions(label_map: np.ndarray, min_area_px: float,
                         min_thick_px: float,
                         palette=None, keep_px: float = 0.0) -> np.ndarray:
    """Reassign unstitchable fragments (too small or too thin) to the
    surrounding color instead of dropping them — dropping leaves bare
    fabric holes inside neighboring fills.

    Exception: a small COMPACT fragment whose color strongly contrasts
    with its surroundings (a white highlight dot inside a black eye) is
    intentional detail — it becomes a hole (background) so the
    surrounding fill leaves it open, instead of being painted over.
    """
    initial_fg = int((label_map >= 0).sum())
    lm = label_map.copy()
    n_colors = int(lm.max()) + 1
    k3 = np.ones((3, 3), np.uint8)
    lab_palette = None
    if palette is not None and len(palette) > 0:
        arr = np.clip(np.array(palette, np.float32), 0, 255).astype(np.uint8).reshape(1, -1, 3)
        lab_palette = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3) \
            .astype(np.float32)
    for _ in range(2):
        changed = False
        for idx in range(n_colors):
            mask = (lm == idx).astype(np.uint8)
            n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for ci in range(1, n):
                x, y, w, h, area = stats[ci]
                x0, y0 = max(0, x - 2), max(0, y - 2)
                x1 = min(lm.shape[1], x + w + 2)
                y1 = min(lm.shape[0], y + h + 2)
                m = (comp[y0:y1, x0:x1] == ci)
                small = area < min_area_px
                if not small:
                    # Thin-sliver cull is for anti-aliasing fringe only; a
                    # LARGE thin structure is real line art that stitches
                    # fine as running stitch — never absorb it.
                    if area > 8 * min_area_px:
                        continue
                    dt = cv2.distanceTransform(m.astype(np.uint8),
                                               cv2.DIST_L2, 3)
                    inner = dt[m]
                    if not (inner.mean() * 2.0 < min_thick_px
                            and dt.max() * 2.0 < min_thick_px * 1.8):
                        continue
                ring = (cv2.dilate(m.astype(np.uint8), k3) > 0) & ~m
                vals = lm[y0:y1, x0:x1][ring]
                vals = vals[(vals >= 0) & (vals != idx)]
                sub = lm[y0:y1, x0:x1]
                if vals.size == 0:
                    sub[m] = -1
                    changed = True
                    continue
                nb = int(np.bincount(vals).argmax())
                if (small and lab_palette is not None and area >= keep_px):
                    dt2 = cv2.distanceTransform(m.astype(np.uint8),
                                                cv2.DIST_L2, 3)
                    compact = float(dt2.max()) * 2.0 >= min_thick_px
                    contrast = float(np.linalg.norm(
                        lab_palette[idx] - lab_palette[nb]))
                    if compact and contrast > 25.0:
                        if lab_palette[idx][0] > lab_palette[nb][0]:
                            # Lighter detail in dark surround (eye sparkle):
                            # keep as a hole, the fabric shows through.
                            sub[m] = -1
                        # Darker detail in light surround (a pupil): keep
                        # its own color so it gets stitched.
                        changed = True
                        continue
                sub[m] = nb
                changed = True
        if not changed:
            break
    removed = initial_fg - int((lm >= 0).sum())
    if initial_fg > 0 and removed > 0.30 * initial_fg:
        # Cleanup would erase much of the artwork (dense fine detail):
        # retry gently; if still destructive, keep the original.
        if min_area_px > 4:
            return absorb_small_regions(label_map, min_area_px / 5.0,
                                        min_thick_px * 0.5, palette, keep_px)
        return label_map
    return lm


def dark_line_mask(rgb: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Line-art mode: keep dark strokes only (Otsu on the foreground)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    vals = gray[fg]
    if vals.size == 0:
        return np.zeros_like(fg)
    thresh, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If artwork is nearly all dark on transparent, Otsu can misfire; clamp.
    thresh = float(min(max(thresh, 60), 200))
    return fg & (gray < thresh)
