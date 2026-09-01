"""Encode a stitch plan into embroidery file formats via pyembroidery."""
from __future__ import annotations

import os
import tempfile
from typing import Tuple

import pyembroidery

from .plan import CMD_COLOR_CHANGE, CMD_JUMP, CMD_STITCH, CMD_TRIM, Plan

FORMATS = {
    "dst": ("application/octet-stream", pyembroidery.write_dst),
    "pes": ("application/octet-stream", pyembroidery.write_pes),
    "exp": ("application/octet-stream", pyembroidery.write_exp),
    "jef": ("application/octet-stream", pyembroidery.write_jef),
}


def build_pattern(plan: Plan) -> pyembroidery.EmbPattern:
    pat = pyembroidery.EmbPattern()
    for hex_color in plan.threads:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        thread = pyembroidery.EmbThread()
        thread.set_color(r, g, b)
        thread.description = hex_color
        pat.add_thread(thread)
    for cmd, x, y in plan.events:
        # pyembroidery native units are 0.1 mm.
        ux, uy = x * 10.0, y * 10.0
        if cmd == CMD_STITCH:
            pat.add_stitch_absolute(pyembroidery.STITCH, ux, uy)
        elif cmd == CMD_JUMP:
            pat.add_stitch_absolute(pyembroidery.JUMP, ux, uy)
        elif cmd == CMD_TRIM:
            pat.add_stitch_absolute(pyembroidery.TRIM, ux, uy)
        elif cmd == CMD_COLOR_CHANGE:
            pat.add_stitch_absolute(pyembroidery.COLOR_CHANGE, ux, uy)
    pat.end()
    return pat


def export_bytes(plan: Plan, fmt: str, name: str = "design"
                 ) -> Tuple[bytes, str, str]:
    """Return (data, mime, filename). Uses a temp file that is always removed."""
    fmt = fmt.lower()
    if fmt == "svg":
        return export_svg(plan).encode("utf-8"), "image/svg+xml", f"{name}.svg"
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")
    mime, writer = FORMATS[fmt]
    pat = build_pattern(plan)
    fd, path = tempfile.mkstemp(suffix=f".{fmt}")
    try:
        os.close(fd)
        writer(pat, path)
        with open(path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if fmt == "dst":
        _verify_dst(data, plan)
    return data, mime, f"{name}.{fmt}"


def _verify_dst(data: bytes, plan: Plan) -> None:
    """Read the file back and sanity-check it parses as a real DST."""
    fd, path = tempfile.mkstemp(suffix=".dst")
    try:
        os.close(fd)
        with open(path, "wb") as f:
            f.write(data)
        readback = pyembroidery.read_dst(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if readback is None or len(readback.stitches) == 0:
        raise RuntimeError("Generated DST failed validation on read-back.")


def export_svg(plan: Plan) -> str:
    """Simple stitch-line SVG preview (mm units)."""
    xs = [x for c, x, _ in plan.events if c in (CMD_STITCH, CMD_JUMP)]
    ys = [y for c, _, y in plan.events if c in (CMD_STITCH, CMD_JUMP)]
    if not xs:
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'
    pad = 2.0
    x0, y0 = min(xs) - pad, min(ys) - pad
    w, h = (max(xs) - min(xs)) + 2 * pad, (max(ys) - min(ys)) + 2 * pad

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {y0:.2f} '
        f'{w:.2f} {h:.2f}" width="{w:.1f}mm" height="{h:.1f}mm">'
    ]
    color_i = 0
    color = plan.threads[0] if plan.threads else "#000000"
    d: list = []
    prev = None

    def flush():
        if d:
            parts.append(f'<path d="{" ".join(d)}" fill="none" '
                         f'stroke="{color}" stroke-width="0.25" '
                         'stroke-linecap="round"/>')
            d.clear()

    for cmd, x, y in plan.events:
        if cmd == CMD_STITCH:
            if prev is None:
                d.append(f"M{x:.2f} {y:.2f}")
            else:
                d.append(f"L{x:.2f} {y:.2f}")
            prev = (x, y)
        elif cmd == CMD_JUMP:
            prev = (x, y)
            if d:
                d.append(f"M{x:.2f} {y:.2f}")
        elif cmd == CMD_COLOR_CHANGE:
            flush()
            color_i += 1
            if color_i < len(plan.threads):
                color = plan.threads[color_i]
            prev = (x, y)
    flush()
    parts.append("</svg>")
    return "".join(parts)
