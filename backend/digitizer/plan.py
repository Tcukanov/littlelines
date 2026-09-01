"""Stitch plan: ordered machine events with jump/trim/color-change logic,
long-stitch splitting, validation and statistics."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

CMD_STITCH = 0
CMD_JUMP = 1
CMD_COLOR_CHANGE = 2
CMD_TRIM = 3


@dataclass
class Plan:
    events: List[Tuple[int, float, float]] = field(default_factory=list)
    threads: List[str] = field(default_factory=list)  # hex colors in order
    stops: List[dict] = field(default_factory=list)   # jump/trim diagnostics


class PlanBuilder:
    def __init__(self, max_stitch_mm: float, max_jump_mm: float,
                 trim_enabled: bool, trim_threshold_mm: float,
                 auto_color_change: bool, walk_mm: float = 2.5):
        self.plan = Plan()
        self.max_stitch = max_stitch_mm
        self.max_jump = max_jump_mm
        self.trim_enabled = trim_enabled
        self.trim_threshold = trim_threshold_mm
        self.auto_color_change = auto_color_change
        self.walk_mm = walk_mm
        self.pos: Optional[np.ndarray] = None
        self._color_started = False
        self._needs_tie_in = True  # thread not anchored yet
        # Optional per-color router: returns an mm polyline through the
        # color's own footprint (invisible travel) or None.
        self.travel_router = None

    def _lock(self, p: np.ndarray) -> None:
        """Tie stitches: 3 tiny stitches so the thread can't pull loose."""
        for dx in (0.4, -0.4, 0.0):
            self.plan.events.append(
                (CMD_STITCH, float(p[0] + dx), float(p[1])))

    def start_color(self, hex_color: str) -> None:
        if self._color_started:
            if self.auto_color_change:
                if self.pos is not None:
                    self._lock(self.pos)  # tie-off before the cut
                    if self.trim_enabled:
                        self.plan.events.append(
                            (CMD_TRIM, float(self.pos[0]), float(self.pos[1])))
                p = self.pos if self.pos is not None else np.zeros(2)
                self.plan.events.append(
                    (CMD_COLOR_CHANGE, float(p[0]), float(p[1])))
                self.plan.stops.append(dict(
                    kind="color_change", x=float(p[0]), y=float(p[1]),
                    gap=0.0, reason="COLOR_CHANGE"))
                self.plan.threads.append(hex_color)
                self._needs_tie_in = True
            # else: same thread keeps running, no event
        else:
            self.plan.threads.append(hex_color)
            self._color_started = True

    def add_run(self, pts: np.ndarray) -> None:
        """One continuously stitched polyline; bridges the gap from the
        previous position with jumps (and a trim when far)."""
        pts = np.asarray(pts, np.float64)
        if len(pts) < 1:
            return
        if self.pos is not None:
            gap = float(np.linalg.norm(pts[0] - self.pos))
            if gap > 0.8:
                if gap <= self.walk_mm:
                    # Walk stitches instead of a jump: a short connector is
                    # invisible but saves a full machine stop/start cycle.
                    self._emit_move(CMD_STITCH, pts[0], self.max_stitch)
                else:
                    travel = None
                    if self.travel_router is not None:
                        travel = self.travel_router(self.pos, pts[0])
                    if travel is not None:
                        # Hidden travel: running stitches routed along the
                        # color's own footprint — invisible, no stop.
                        for p in travel:
                            self._emit_move(CMD_STITCH, np.asarray(p),
                                            self.max_stitch)
                    else:
                        reason = "NO_ROUTER"
                        if self.travel_router is not None:
                            reason = getattr(self.travel_router, "reason",
                                             "NO_ROUTE")
                        trimmed = (self.trim_enabled
                                   and gap > self.trim_threshold)
                        self.plan.stops.append(dict(
                            kind="trim" if trimmed else "jump",
                            x=float(pts[0][0]), y=float(pts[0][1]),
                            fx=float(self.pos[0]), fy=float(self.pos[1]),
                            gap=round(gap, 1), reason=reason))
                        if trimmed:
                            self._lock(self.pos)  # tie-off before the cut
                            self.plan.events.append(
                                (CMD_TRIM, float(self.pos[0]),
                                 float(self.pos[1])))
                            self._needs_tie_in = True
                        self._emit_move(CMD_JUMP, pts[0], self.max_jump)
        if self._needs_tie_in:
            self._lock(pts[0])  # tie-in: anchor the new thread start
            self.pos = pts[0]
            self._needs_tie_in = False
        if len(pts) == 1:
            self.plan.events.append((CMD_STITCH, float(pts[0][0]),
                                     float(pts[0][1])))
            self.pos = pts[0]
            return
        for p in pts:
            if self.pos is None:
                self.plan.events.append((CMD_STITCH, float(p[0]), float(p[1])))
                self.pos = p
                continue
            d = float(np.linalg.norm(p - self.pos))
            if d < 0.05:
                continue
            self._emit_move(CMD_STITCH, p, self.max_stitch)

    def _emit_move(self, cmd: int, target: np.ndarray, max_len: float) -> None:
        start = self.pos if self.pos is not None else target
        d = float(np.linalg.norm(target - start))
        n = max(1, int(math.ceil(d / max_len)))
        for i in range(1, n + 1):
            p = start + (target - start) * (i / n)
            self.plan.events.append((cmd, float(p[0]), float(p[1])))
        self.pos = np.asarray(target, np.float64)

    def finish(self) -> Tuple[Plan, dict, List[str]]:
        if self.pos is not None:
            self._lock(self.pos)  # final tie-off before END
        plan = self.plan
        xs = [e[1] for e in plan.events if e[0] in (CMD_STITCH, CMD_JUMP)]
        ys = [e[2] for e in plan.events if e[0] in (CMD_STITCH, CMD_JUMP)]
        cx = cy = 0.0
        if xs:
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            plan.events = [(c, x - cx, y - cy) for c, x, y in plan.events]
            for st in plan.stops:
                st["x"] = round(st["x"] - cx, 2)
                st["y"] = round(st["y"] - cy, 2)
                if "fx" in st:
                    st["fx"] = round(st["fx"] - cx, 2)
                    st["fy"] = round(st["fy"] - cy, 2)
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
        else:
            width = height = 0.0

        stitches = sum(1 for e in plan.events if e[0] == CMD_STITCH)
        # Count jump MOVES (a long move split into hops is one machine stop).
        jumps = sum(1 for i, e in enumerate(plan.events)
                    if e[0] == CMD_JUMP
                    and (i == 0 or plan.events[i - 1][0] != CMD_JUMP))
        trims = sum(1 for e in plan.events if e[0] == CMD_TRIM)
        changes = sum(1 for e in plan.events if e[0] == CMD_COLOR_CHANGE)
        # Realistic machine time: jumps and trims dominate stroke-heavy
        # designs (frame creep + stop/cut/restart cycles), not stitches.
        minutes = (stitches / 700.0 + jumps * 1.2 / 60.0
                   + trims * 5 / 60.0 + changes * 20 / 60.0)

        stats = {
            "stitches": stitches,
            "jumps": jumps,
            "trims": trims,
            "color_changes": changes,
            "colors": len(plan.threads),
            "width_mm": round(width, 1),
            "height_mm": round(height, 1),
            "est_minutes": round(minutes, 1),
            # Center offset in artwork coordinates (for coverage checks).
            "cx": round(cx, 2),
            "cy": round(cy, 2),
        }

        warnings = self._validate(plan, width, height, stitches)
        return plan, stats, warnings

    def _validate(self, plan: Plan, width: float, height: float,
                  stitches: int) -> List[str]:
        warnings = []
        if stitches == 0:
            warnings.append("No stitches were generated — try lowering the "
                            "'remove small objects' value or raising detail.")
        if stitches > 120000:
            warnings.append(f"Very high stitch count ({stitches:,}); consider "
                            "a smaller size or lower density.")
        jumps = sum(1 for i, e in enumerate(plan.events)
                    if e[0] == CMD_JUMP
                    and (i == 0 or plan.events[i - 1][0] != CMD_JUMP))
        trims = sum(1 for e in plan.events if e[0] == CMD_TRIM)
        if not self.trim_enabled and jumps > 30:
            warnings.append(
                f"Trims are OFF but the design has {jumps} jumps — the "
                "machine will drag loose threads across the design between "
                "shapes. Turn 'Trim between objects' ON.")
        if width > 360 or height > 360:
            warnings.append("Design exceeds 360 mm and will not fit most hoops.")
        elif width > 180 or height > 180:
            warnings.append("Design is larger than a typical 180 mm hoop — "
                            "double-check your hoop size.")
        # Hard guarantee for the encoder: no delta may exceed 12.1 mm.
        prev = None
        for c, x, y in plan.events:
            if c in (CMD_STITCH, CMD_JUMP):
                if prev is not None:
                    if abs(x - prev[0]) > 12.15 or abs(y - prev[1]) > 12.15:
                        warnings.append("Internal: over-long stitch detected.")
                        break
                prev = (x, y)
        if abs(width) > 999 or abs(height) > 999:
            warnings.append("Design exceeds DST coordinate limits.")
        return warnings
