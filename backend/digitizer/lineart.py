"""Centerline extraction for line art and satin regions.

Skeletonize a mask, walk the skeleton into polylines and measure local
stroke width via the distance transform.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

Pixel = Tuple[int, int]  # (x, y)


def skeletonize(img: np.ndarray) -> np.ndarray:
    """Guo-Hall thinning, vectorized in NumPy (replaces scikit-image).
    Chosen over Zhang-Suen, which leaves 2-px staircases on diagonals.
    Runs on the content bounding box only — masks are often tiny regions
    inside a large frame."""
    full = (img > 0)
    ys, xs = np.nonzero(full)
    if ys.size == 0:
        return np.zeros_like(full)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    out = np.zeros_like(full)
    out[y0:y1, x0:x1] = _guo_hall(full[y0:y1, x0:x1])
    return out


def _guo_hall(img: np.ndarray) -> np.ndarray:
    skel = img.astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(skel, 1)
            p2 = P[:-2, 1:-1]
            p3 = P[:-2, 2:]
            p4 = P[1:-1, 2:]
            p5 = P[2:, 2:]
            p6 = P[2:, 1:-1]
            p7 = P[2:, :-2]
            p8 = P[1:-1, :-2]
            p9 = P[:-2, :-2]
            C = (((1 - p2) & (p3 | p4)) + ((1 - p4) & (p5 | p6))
                 + ((1 - p6) & (p7 | p8)) + ((1 - p8) & (p9 | p2)))
            N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8)
            N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9)
            N = np.minimum(N1, N2)
            if step == 0:
                m = (p6 | p7 | (1 - p9)) & p8
            else:
                m = (p2 | p3 | (1 - p5)) & p4
            cond = (skel == 1) & (C == 1) & (N >= 2) & (N <= 3) & (m == 0)
            if cond.any():
                skel[cond] = 0
                changed = True
    return skel.astype(bool)

_NBRS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def centerline_paths(mask: np.ndarray, min_len_px: float = 4.0
                     ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return [(path_px Nx2 float, widths_px N float)] for a binary mask."""
    skel = skeletonize(mask > 0)
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    ys, xs = np.nonzero(skel)
    pixels: Set[Pixel] = set(zip(xs.tolist(), ys.tolist()))
    if not pixels:
        return []

    def neighbors(p: Pixel) -> List[Pixel]:
        x, y = p
        return [(x + dx, y + dy) for dx, dy in _NBRS if (x + dx, y + dy) in pixels]

    degree: Dict[Pixel, int] = {p: len(neighbors(p)) for p in pixels}
    terminals = {p for p, d in degree.items() if d != 2}

    visited: Set[Pixel] = set()
    raw_paths: List[List[Pixel]] = []

    def walk(start: Pixel, nxt: Pixel) -> List[Pixel]:
        path = [start, nxt]
        if nxt not in terminals:
            visited.add(nxt)
        prev, cur = start, nxt
        while cur not in terminals:
            options = [n for n in neighbors(cur) if n != prev and
                       (n in terminals or n not in visited)]
            if not options:
                break
            nxt2 = options[0]
            path.append(nxt2)
            if nxt2 not in terminals:
                visited.add(nxt2)
            prev, cur = cur, nxt2
            if cur == start:  # closed loop back to start
                break
        return path

    seen_edges: Set[frozenset] = set()
    for t in terminals:
        for n in neighbors(t):
            if n in terminals:
                key = frozenset((t, n))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                raw_paths.append([t, n])
            elif n not in visited:
                raw_paths.append(walk(t, n))

    # Pure cycles (no terminals touched).
    for p in pixels:
        if p not in visited and p not in terminals and degree.get(p) == 2:
            n = neighbors(p)[0]
            visited.add(p)
            cyc = walk(p, n)
            if len(cyc) > 3:
                cyc.append(p)
                raw_paths.append(cyc)

    out = []
    for rp in raw_paths:
        if len(rp) < 2:
            continue
        pts = np.array(rp, np.float64)
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if length < min_len_px:
            continue
        pts = _smooth(pts)
        w = np.array([2.0 * dt[int(round(y)), int(round(x))]
                      for x, y in pts], np.float64)
        out.append((pts, w))
    return out


def _smooth(pts: np.ndarray, epsilon: float = 1.2) -> np.ndarray:
    """Douglas-Peucker then one round of corner-cutting."""
    if len(pts) < 3:
        return pts
    ap = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2),
                          epsilon, False).reshape(-1, 2).astype(np.float64)
    if len(ap) < 3:
        return ap
    # Chaikin corner cutting (keep endpoints).
    out = [ap[0]]
    for i in range(len(ap) - 1):
        a, b = ap[i], ap[i + 1]
        out.append(a * 0.75 + b * 0.25)
        out.append(a * 0.25 + b * 0.75)
    out.append(ap[-1])
    return np.array(out)


def order_paths(paths: List[Tuple[np.ndarray, np.ndarray]]
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Greedy nearest-neighbor ordering, flipping paths when it helps."""
    if not paths:
        return []
    remaining = list(paths)
    ordered = [remaining.pop(0)]
    while remaining:
        cur_end = ordered[-1][0][-1]
        best_i, best_d, best_flip = 0, float("inf"), False
        for i, (p, _) in enumerate(remaining):
            d0 = np.linalg.norm(p[0] - cur_end)
            d1 = np.linalg.norm(p[-1] - cur_end)
            if d0 < best_d:
                best_i, best_d, best_flip = i, d0, False
            if d1 < best_d:
                best_i, best_d, best_flip = i, d1, True
        p, w = remaining.pop(best_i)
        if best_flip:
            p, w = p[::-1], w[::-1]
        ordered.append((p, w))
    return ordered


def graph_walk(paths: List[Tuple[np.ndarray, np.ndarray]],
               start_near=None
               ) -> List[Tuple[np.ndarray, np.ndarray, bool]]:
    """Traverse the stroke network like a commercial digitizer: DFS over the
    graph of strokes, stitching each branch on first visit and traveling
    back along it (retrace=True) when backtracking. The result is one
    continuous polyline per connected component — jumps only remain
    between separate components.

    Returns [(pts, widths, retrace)] in stitching order.
    """
    if not paths:
        return []

    # Snap nearby endpoints to one node: skeleton junctions span 2-3 px,
    # and paths attaching to different pixels of the same junction must
    # still count as connected.
    snap = 3.0
    node_xy: List[Tuple[float, float]] = []
    end_node: List[int] = []
    for pts, _ in paths:
        for p in (pts[0], pts[-1]):
            found = -1
            for i, q in enumerate(node_xy):
                if abs(p[0] - q[0]) <= snap and abs(p[1] - q[1]) <= snap:
                    found = i
                    break
            if found < 0:
                node_xy.append((float(p[0]), float(p[1])))
                found = len(node_xy) - 1
            end_node.append(found)

    adj: Dict[int, list] = {}
    for eid in range(len(paths)):
        a, b = end_node[2 * eid], end_node[2 * eid + 1]
        adj.setdefault(a, []).append((eid, b, True))
        adj.setdefault(b, []).append((eid, a, False))

    # Group nodes into connected components first (node-level BFS), so we
    # can pick each component's DFS root as the node nearest the needle —
    # the walk then STARTS right where the previous piece ended, letting
    # short connector stitches replace jumps between adjacent pieces.
    node_comp = {}
    comp_nodes: List[List[int]] = []
    for start in adj:
        if start in node_comp:
            continue
        cid = len(comp_nodes)
        queue = [start]
        node_comp[start] = cid
        members = [start]
        while queue:
            nd = queue.pop()
            for _, other, _ in adj[nd]:
                if other not in node_comp:
                    node_comp[other] = cid
                    members.append(other)
                    queue.append(other)
        comp_nodes.append(members)

    used = [False] * len(paths)

    def walk_component(root) -> List[Tuple[np.ndarray, np.ndarray, bool]]:
        segs: List[Tuple[np.ndarray, np.ndarray, bool]] = []
        stack = [[root, 0, None, None]]
        while stack:
            frame = stack[-1]
            node, idx = frame[0], frame[1]
            edges = adj[node]
            advanced = False
            while idx < len(edges):
                eid, other, fwd = edges[idx]
                idx += 1
                if used[eid]:
                    continue
                used[eid] = True
                frame[1] = idx
                pts, w = paths[eid]
                if fwd:
                    segs.append((pts, w, False))
                else:
                    segs.append((pts[::-1], w[::-1], False))
                stack.append([other, 0, eid, fwd])
                advanced = True
                break
            if not advanced:
                frame[1] = idx
                stack.pop()
                if frame[2] is not None:
                    eid, fwd = frame[2], frame[3]
                    pts, w = paths[eid]
                    if fwd:
                        segs.append((pts[::-1], w[::-1], True))
                    else:
                        segs.append((pts, w, True))
        return segs

    out: List[Tuple[np.ndarray, np.ndarray, bool]] = []
    remaining = list(range(len(comp_nodes)))
    pos = None if start_near is None else np.asarray(start_near, float)
    while remaining:
        # Nearest component AND its nearest node to the current position.
        best_ci, best_node, best_d = remaining[0], comp_nodes[remaining[0]][0], \
            float("inf")
        for ci in remaining:
            for nd in comp_nodes[ci]:
                if pos is None:
                    d = 0.0 if ci == remaining[0] else 1.0
                else:
                    d = float(np.hypot(node_xy[nd][0] - pos[0],
                                       node_xy[nd][1] - pos[1]))
                if d < best_d:
                    best_ci, best_node, best_d = ci, nd, d
            if pos is None:
                break
        remaining.remove(best_ci)
        segs = walk_component(best_node)
        if segs:
            out.extend(segs)
            pos = segs[-1][0][-1]
        elif pos is None:
            pos = np.array(node_xy[best_node])
    return out


def merge_ordered(paths: List[Tuple[np.ndarray, np.ndarray]],
                  max_gap_px: float
                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Concatenate consecutive ordered paths whose ends meet, so satin runs
    continuously through junctions instead of restarting in short beads."""
    merged: List[Tuple[np.ndarray, np.ndarray]] = []
    for p, w in paths:
        if merged:
            lp, lw = merged[-1]
            if np.linalg.norm(p[0] - lp[-1]) <= max_gap_px:
                merged[-1] = (np.vstack([lp, p]), np.concatenate([lw, w]))
                continue
        merged.append((p, w))
    return merged
