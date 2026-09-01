"""Region extraction: connected components, contours and shape features."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Region:
    mask: np.ndarray                 # uint8 component mask (working px)
    area_px: int = 0
    polys: List[np.ndarray] = field(default_factory=list)  # px, outer + holes
    centroid: Tuple[float, float] = (0.0, 0.0)
    max_thickness_px: float = 0.0    # 2 * max distance-transform value
    mean_thickness_px: float = 0.0


def smooth_kernel(detail: int) -> int:
    """detail 0..100 -> odd morphology kernel size (more smoothing when low)."""
    k = 1 + round((100 - detail) / 100.0 * 4)
    return k if k % 2 == 1 else k + 1


def approx_epsilon(detail: int) -> float:
    return 0.8 + (100 - detail) / 100.0 * 2.2


def clean_mask(mask: np.ndarray, detail: int) -> np.ndarray:
    k = smooth_kernel(detail)
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def extract_regions(label_map: np.ndarray, color_index: int,
                    min_area_px: float, detail: int,
                    mm_per_px: float = 0.0) -> List[Region]:
    mask = (label_map == color_index).astype(np.uint8)
    mask = clean_mask(mask, detail)
    n, comp = cv2.connectedComponents(mask, connectivity=8)
    regions: List[Region] = []
    eps = approx_epsilon(detail)
    for ci in range(1, n):
        m = (comp == ci).astype(np.uint8)
        area = int(m.sum())
        if area < max(min_area_px, 4):
            continue
        contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in contours:
            if cv2.contourArea(c) < 2:
                continue
            ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if ap.shape[0] >= 3:
                polys.append(ap.astype(np.float64))
        if not polys:
            continue
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        ys, xs = np.nonzero(m)
        inner = dt[m > 0]
        regions.append(Region(
            mask=m,
            area_px=area,
            polys=polys,
            centroid=(float(xs.mean()), float(ys.mean())),
            max_thickness_px=float(dt.max()) * 2.0,
            mean_thickness_px=float(inner.mean()) * 2.0,
        ))
    return regions


def suggest_stitch(region: Region, mm_per_px: float,
                   satin_max_mm: float) -> str:
    """Pick a sensible stitch type from shape geometry."""
    max_th = region.max_thickness_px * mm_per_px
    area_mm2 = region.area_px * mm_per_px * mm_per_px
    # length/width ratio: ~1 for compact blobs, >3 for stroke-like shapes.
    elongation = region.area_px / max(region.max_thickness_px ** 2, 1.0)
    if max_th <= 1.0 or area_mm2 < 3.0:
        return "running"
    # Satin only for genuinely stroke-like shapes; compact blobs look far
    # better as small tatami fills than as fat zigzag "beads". The shape
    # must also fit inside the satin width cap, or coverage would fall short.
    if max_th <= satin_max_mm and elongation >= 2.2:
        return "satin"
    return "fill"
