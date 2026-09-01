"""Stitch geometry generators. All coordinates here are in mm (y down)."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

Point = np.ndarray  # shape (2,)
Path = np.ndarray   # shape (N, 2)


def path_length(pts: Path) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def resample_path(pts: Path, step: float) -> Path:
    """Resample a polyline at ~step spacing, keeping both endpoints."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total < 1e-9:
        return pts[:1]
    n = max(1, int(math.ceil(total / step)))
    targets = np.linspace(0.0, total, n + 1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    j = 0
    for t in targets:
        while j < len(seg) - 1 and cum[j + 1] < t:
            j += 1
        denom = seg[j] if seg[j] > 1e-12 else 1.0
        alpha = (t - cum[j]) / denom
        alpha = min(max(alpha, 0.0), 1.0)
        out.append(pts[j] * (1 - alpha) + pts[j + 1] * alpha)
    return np.array(out)


def dedupe(pts: Path, min_dist: float = 0.1) -> Path:
    if len(pts) < 2:
        return pts
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) >= min_dist:
            keep.append(p)
    if len(keep) > 1 and not np.array_equal(keep[-1], pts[-1]):
        if np.linalg.norm(pts[-1] - keep[-1]) > 0.02:
            keep.append(pts[-1])
    return np.array(keep)


def running_stitch(path: Path, stitch_len: float, passes: int = 1) -> List[Path]:
    """Running stitch along a path; 2-pass returns over itself, 3-pass again."""
    base = resample_path(path, stitch_len)
    if len(base) < 2:
        return []
    pts = [base]
    for i in range(1, passes):
        pts.append(base[::-1] if i % 2 == 1 else base)
    return [dedupe(np.concatenate(pts))]


def satin_along_path(path: Path, widths, spacing: float,
                     min_width: float = 0.8) -> Optional[Path]:
    """Zigzag satin centered on path. widths: scalar or per-point array (mm)."""
    center = resample_path(path, spacing)
    if len(center) < 2:
        return None
    n = len(center)
    if np.isscalar(widths):
        w = np.full(n, float(widths))
    else:
        # Resample widths to the new sampling by arc position.
        src = np.asarray(widths, np.float64)
        w = np.interp(np.linspace(0, 1, n),
                      np.linspace(0, 1, len(src)), src)
    w = np.maximum(w, min_width)

    # Tangents -> unit normals.
    tang = np.gradient(center, axis=0)
    norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    lens = np.linalg.norm(norm, axis=1, keepdims=True)
    lens[lens < 1e-9] = 1.0
    norm /= lens

    out = []
    side = 1.0
    for i in range(n):
        out.append(center[i] + norm[i] * (w[i] / 2.0) * side)
        out.append(center[i] - norm[i] * (w[i] / 2.0) * side)
        side *= -1.0  # keeps zig direction consistent
    return dedupe(np.array(out), 0.05)


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def scanline_fill(polys: Sequence[Path], angle_deg: float, spacing: float,
                  stitch_len: float, pull_comp: float = 0.0,
                  inset: float = 0.0) -> List[Path]:
    """Serpentine tatami fill of a polygon-with-holes.

    Returns a list of continuously-stitchable runs (mm points). Gaps between
    runs must be bridged by the caller (jump / trim).
    """
    theta = math.radians(angle_deg)
    R = _rot(-theta)
    Rinv = _rot(theta)
    rot_polys = [np.asarray(p, np.float64) @ R.T for p in polys]

    ys = np.concatenate([p[:, 1] for p in rot_polys])
    y0, y1 = ys.min(), ys.max()
    if y1 - y0 < 1e-6:
        return []

    # Collect edges once.
    edges = []
    for p in rot_polys:
        q = np.vstack([p, p[:1]])
        for i in range(len(p)):
            edges.append((q[i], q[i + 1]))

    # Segments per row: (row_index, y, xa, xb)
    segments = []
    y = y0 + spacing * 0.5 + 1e-4
    ri = 0
    while y < y1:
        xs = []
        for a, b in edges:
            ya, yb = a[1], b[1]
            if (ya <= y < yb) or (yb <= y < ya):
                t = (y - ya) / (yb - ya)
                xs.append(a[0] + t * (b[0] - a[0]))
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            xa, xb = xs[i] - pull_comp + inset, xs[i + 1] + pull_comp - inset
            if xb - xa > 0.15:
                segments.append([ri, y, xa, xb])
        y += spacing
        ri += 1

    if not segments:
        return []

    # Chain segments into serpentine columns: from each chain's current
    # segment, continue to the closest reachable segment in the next row.
    # If the next row isn't directly adjacent, allow a short travel run of
    # stitches as long as the straight path stays inside the shape (it will
    # be covered by later rows / neighboring columns).
    by_row: dict = {}
    for i, s in enumerate(segments):
        by_row.setdefault(s[0], []).append(i)
    unused = set(range(len(segments)))
    max_connect = max(2.0, stitch_len)
    travel_max = 14.0
    runs: List[Path] = []

    def inside(pt: np.ndarray) -> bool:
        crossings = 0
        x, y = pt
        for a, b in edges:
            ya, yb = a[1], b[1]
            if (ya <= y < yb) or (yb <= y < ya):
                t = (y - ya) / (yb - ya)
                if a[0] + t * (b[0] - a[0]) > x:
                    crossings += 1
        return crossings % 2 == 1

    def can_travel(p0: np.ndarray, p1: np.ndarray) -> bool:
        d = float(np.linalg.norm(p1 - p0))
        if d < 0.3:
            return True
        n = max(2, int(d / 0.6))
        for k in range(1, n):
            if not inside(p0 + (p1 - p0) * (k / n)):
                return False
        return True

    while unused:
        start = min(unused, key=lambda i: (segments[i][0], segments[i][2]))
        unused.remove(start)
        ri, y, xa, xb = segments[start]
        pts = _sample_row(xa, xb, y, stitch_len, phase=(ri % 4) / 4.0)
        while True:
            cur = pts[-1]
            # 1) Direct serpentine continuation in the adjacent row.
            best, best_d, best_rev = -1, float("inf"), False
            for i in by_row.get(ri + 1, []):
                if i not in unused:
                    continue
                _, ny, ca, cb = segments[i]
                for rev, xe in ((False, ca), (True, cb)):
                    d = abs(xe - cur[0])
                    if d < best_d:
                        best, best_d, best_rev = i, d, rev
            travel_pts: List[np.ndarray] = []
            if best < 0 or best_d > max_connect:
                # 2) Travel inside the shape to a nearby unused segment.
                best, best_d, best_rev = -1, float("inf"), False
                for dr in range(1, 8):
                    for i in by_row.get(ri + dr, []):
                        if i not in unused:
                            continue
                        _, ny, ca, cb = segments[i]
                        for rev, xe in ((False, ca), (True, cb)):
                            d = float(np.hypot(xe - cur[0], ny - cur[1]))
                            if d < best_d:
                                best, best_d, best_rev = i, d, rev
                    if best >= 0 and best_d <= max_connect:
                        break
                if best < 0 or best_d > travel_max:
                    break
                _, ny, ca, cb = segments[best]
                entry_pt = np.array([cb if best_rev else ca, ny])
                if not can_travel(cur, entry_pt):
                    break
                step = min(stitch_len, 2.5)
                n = max(1, int(np.ceil(best_d / step)))
                travel_pts = [cur + (entry_pt - cur) * (k / n)
                              for k in range(1, n)]
            unused.remove(best)
            ri, y, ca, cb = segments[best]
            entry, exit_ = (cb, ca) if best_rev else (ca, cb)
            pts.extend(travel_pts)
            pts.extend(_sample_row(entry, exit_, y, stitch_len,
                                   phase=(ri % 4) / 4.0))
        runs.append(np.array(pts))

    # Rotate back to design space.
    return [dedupe(r @ Rinv.T, 0.08) for r in runs if len(r) >= 2]


def _sample_row(x_entry: float, x_exit: float, y: float, stitch_len: float,
                phase: float) -> List[np.ndarray]:
    """Needle points along one row, phase-shifted to avoid banding."""
    length = abs(x_exit - x_entry)
    direction = 1.0 if x_exit >= x_entry else -1.0
    ts = [0.0]
    t = phase * stitch_len if phase > 0 else stitch_len
    while t < length - 0.3:
        ts.append(t)
        t += stitch_len
    ts.append(length)
    return [np.array([x_entry + direction * t, y]) for t in ts]


def contour_paths_mm(polys_px: Sequence[np.ndarray], sx: float,
                     sy: float) -> List[Path]:
    """Convert closed px contours to closed mm paths."""
    out = []
    for p in polys_px:
        q = np.asarray(p, np.float64).copy()
        q[:, 0] *= sx
        q[:, 1] *= sy
        out.append(np.vstack([q, q[:1]]))
    return out
