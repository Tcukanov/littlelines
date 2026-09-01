"""Settings parsing with beginner-safe defaults. All physical units are mm."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


def _f(d: dict, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(d.get(key, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _i(d: dict, key: str, default: int, lo: int, hi: int) -> int:
    return int(round(_f(d, key, default, lo, hi)))


def _b(d: dict, key: str, default: bool) -> bool:
    v = d.get(key, default)
    return bool(v)


@dataclass
class ColorSetting:
    enabled: bool = True
    stitch: str = "auto"  # auto | fill | satin | running


@dataclass
class Settings:
    # Size
    width_mm: float = 100.0
    height_mm: float = 100.0
    # Image preparation
    max_colors: int = 4
    detail: int = 50            # 0 = very simplified, 100 = keep detail
    min_object_mm2: float = 2.0
    remove_background: bool = True
    # Stitch settings
    density_mm: float = 0.40    # row/zigzag spacing
    stitch_len_mm: float = 3.0
    satin_width_mm: float = 2.5  # max satin width
    fill_angle_deg: float = 45.0
    auto_fill_angle: bool = True  # per-shape angle along its long axis
    underlay: bool = True
    pull_comp_mm: float = 0.15
    trim_enabled: bool = True
    auto_color_change: bool = True
    # Line art mode
    line_art: bool = False
    line_passes: int = 2        # 1 | 2 | 3
    # Per-color overrides, keyed by palette index
    color_settings: Dict[int, ColorSetting] = field(default_factory=dict)

    # Hard machine limits
    max_stitch_mm: float = 12.0
    max_jump_mm: float = 12.0
    trim_threshold_mm: float = 4.0

    @staticmethod
    def from_dict(d: Optional[dict]) -> "Settings":
        d = d or {}
        s = Settings()
        s.width_mm = _f(d, "width_mm", 100.0, 5.0, 400.0)
        s.height_mm = _f(d, "height_mm", 100.0, 5.0, 400.0)
        s.max_colors = _i(d, "max_colors", 4, 1, 12)
        s.detail = _i(d, "detail", 50, 0, 100)
        s.min_object_mm2 = _f(d, "min_object_mm2", 2.0, 0.0, 100.0)
        s.remove_background = _b(d, "remove_background", True)
        s.density_mm = _f(d, "density_mm", 0.40, 0.25, 1.0)
        s.stitch_len_mm = _f(d, "stitch_len_mm", 3.0, 1.0, 7.0)
        s.satin_width_mm = _f(d, "satin_width_mm", 2.5, 1.0, 7.0)
        s.fill_angle_deg = _f(d, "fill_angle_deg", 45.0, 0.0, 180.0)
        s.auto_fill_angle = _b(d, "auto_fill_angle", True)
        s.underlay = _b(d, "underlay", True)
        s.pull_comp_mm = _f(d, "pull_comp_mm", 0.15, 0.0, 1.0)
        s.trim_enabled = _b(d, "trim_enabled", True)
        s.auto_color_change = _b(d, "auto_color_change", True)
        s.line_art = _b(d, "line_art", False)
        s.line_passes = _i(d, "line_passes", 2, 1, 3)
        cs = d.get("color_settings") or {}
        for k, v in cs.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            stitch = str(v.get("stitch", "auto"))
            if stitch not in ("auto", "fill", "satin", "running"):
                stitch = "auto"
            s.color_settings[idx] = ColorSetting(
                enabled=bool(v.get("enabled", True)), stitch=stitch
            )
        return s
