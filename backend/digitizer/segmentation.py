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
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGBA")
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
        # digitizes into ugly slivers around every shape.
        fg = cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8),
                       iterations=2).astype(bool)
    else:
        fg = np.ones((h, w), bool)
        if remove_background:
            fg &= ~_border_flood(rgb, fg)
            fg = cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8),
                           iterations=1).astype(bool)
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
    size applies to the artwork itself, not the canvas."""
    ys, xs = np.nonzero(fg)
    if xs.size == 0:
        return rgb, alpha, fg
    h, w = fg.shape
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + 1 + margin)
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + 1 + margin)
    return rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1], fg[y0:y1, x0:x1]


def quantize(rgb: np.ndarray, fg: np.ndarray,
             max_colors: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
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
        palette.append(tuple(int(round(v)) for v in c))
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
    lm = label_map.copy()
    n_colors = int(lm.max()) + 1
    k3 = np.ones((3, 3), np.uint8)
    lab_palette = None
    if palette is not None and len(palette) > 0:
        arr = np.array(palette, np.uint8).reshape(1, -1, 3)
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
                    if area > 64 * min_area_px:
                        continue  # clearly big enough, skip thickness test
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
                        sub[m] = -1  # intentional detail -> keep as a hole
                        changed = True
                        continue
                sub[m] = nb
                changed = True
        if not changed:
            break
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
