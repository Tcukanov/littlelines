"""PNG to DST — digitizing backend.

Stateless by design: uploads are processed in memory, temp files used during
encoding are deleted immediately, nothing is ever stored.
"""
from __future__ import annotations

import json
import os
import re

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from digitizer import export, pipeline
from digitizer.params import Settings

app = FastAPI(title="PNG to DST")

# Extra origins (e.g. a deployed frontend) via env:
#   ALLOWED_ORIGINS=https://myapp.vercel.app,https://other.example
_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
_origins += [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
             if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD = 20 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg"}


async def _read_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Please upload a PNG or JPG image.")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is too large (max 20 MB).")
    if len(data) == 0:
        raise HTTPException(400, "Empty file.")
    return data


def _settings(raw: str) -> Settings:
    try:
        return Settings.from_dict(json.loads(raw) if raw else {})
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid settings JSON.")


@app.middleware("http")
async def _restore_vercel_path(request: Request, call_next):
    # On Vercel every request is rewritten to /api/index with the original
    # path tucked into the __path query param — restore it before routing.
    orig = request.query_params.get("__path")
    if orig is not None and request.scope["path"].endswith("/api/index"):
        request.scope["path"] = orig if orig.startswith("/") else "/" + orig
    return await call_next(request)


@app.get("/")
async def root():
    return {"service": "png-to-dst", "ok": True}


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), settings: str = Form("{}")):
    data = await _read_upload(file)
    s = _settings(settings)
    try:
        return pipeline.analyze(data, s)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Could not analyze image: {e}")


@app.post("/api/cleanup")
async def cleanup(file: UploadFile = File(...), settings: str = Form("{}")):
    data = await _read_upload(file)
    s = _settings(settings)
    try:
        png = pipeline.cleanup(data, s)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Cleanup failed: {e}")
    import base64
    return {"png": base64.b64encode(png).decode("ascii")}


@app.post("/api/digitize")
async def digitize(file: UploadFile = File(...), settings: str = Form("{}")):
    data = await _read_upload(file)
    s = _settings(settings)
    try:
        plan, stats, warnings = pipeline.digitize(data, s)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Digitizing failed: {e}")
    return {
        "threads": plan.threads,
        "stitches": [[c, round(x, 2), round(y, 2)]
                     for c, x, y in plan.events],
        "stats": stats,
        "warnings": warnings,
        "stops": plan.stops,
    }


@app.post("/api/export")
async def export_file(file: UploadFile = File(...), settings: str = Form("{}"),
                      format: str = Form("dst")):
    data = await _read_upload(file)
    s = _settings(settings)
    fmt = format.lower()
    if fmt not in ("dst", "pes", "exp", "jef", "svg"):
        raise HTTPException(400, f"Unsupported format: {format}")
    try:
        plan, stats, warnings = pipeline.digitize(data, s)
        if stats["stitches"] == 0:
            raise HTTPException(422, "No stitches generated — nothing to export.")
        name = re.sub(r"[^A-Za-z0-9_-]+", "_",
                      (file.filename or "design").rsplit(".", 1)[0]) or "design"
        blob, mime, filename = export.export_bytes(plan, fmt, name)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Export failed: {e}")
    return Response(
        content=blob,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)):
    """Parse an existing embroidery file (DST/PES/JEF/EXP) into the same
    preview format the digitizer produces, for side-by-side inspection."""
    import tempfile

    import pyembroidery

    data = await file.read()
    if len(data) == 0 or len(data) > 5 * 1024 * 1024:
        raise HTTPException(400, "Empty or oversized file.")
    suffix = "." + (file.filename or "x.dst").rsplit(".", 1)[-1].lower()
    if suffix not in (".dst", ".pes", ".jef", ".exp"):
        raise HTTPException(415, "Supported: DST, PES, JEF, EXP.")
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.close(fd)
        with open(path, "wb") as f:
            f.write(data)
        pat = pyembroidery.read(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if pat is None or not pat.stitches:
        raise HTTPException(422, "Could not read that embroidery file.")

    mask = pyembroidery.COMMAND_MASK
    palette = ["#111827", "#2563eb", "#dc2626", "#059669", "#d97706",
               "#7c3aed", "#0891b2", "#be185d", "#4d7c0f", "#9333ea"]
    threads = []
    for i, t in enumerate(pat.threadlist):
        try:
            threads.append("#%06x" % (t.color & 0xFFFFFF))
        except Exception:  # noqa: BLE001
            threads.append(palette[i % len(palette)])
    if not threads:
        threads = [palette[0]]

    events = []
    stops = []
    color_i = 0
    prev_jump = False
    for x, y, raw in pat.stitches:
        c = raw & mask
        mx, my = x / 10.0, y / 10.0
        if c == pyembroidery.STITCH:
            events.append([0, mx, my])
            prev_jump = False
        elif c == pyembroidery.JUMP:
            events.append([1, mx, my])
            if not prev_jump:
                stops.append(dict(kind="jump", x=mx, y=my, gap=0.0,
                                  reason="IMPORTED"))
            prev_jump = True
        elif c == pyembroidery.TRIM:
            events.append([3, mx, my])
            stops.append(dict(kind="trim", x=mx, y=my, gap=0.0,
                              reason="IMPORTED"))
            prev_jump = False
        elif c == pyembroidery.COLOR_CHANGE:
            events.append([2, mx, my])
            stops.append(dict(kind="color_change", x=mx, y=my, gap=0.0,
                              reason="COLOR_CHANGE"))
            color_i += 1
            if color_i >= len(threads):
                threads.append(palette[color_i % len(palette)])
            prev_jump = False

    xs = [e[1] for e in events if e[0] in (0, 1)]
    ys = [e[2] for e in events if e[0] in (0, 1)]
    if not xs:
        raise HTTPException(422, "No stitches in that file.")
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    events = [[c, round(x - cx, 2), round(y - cy, 2)] for c, x, y in events]
    for st in stops:
        st["x"] = round(st["x"] - cx, 2)
        st["y"] = round(st["y"] - cy, 2)

    n_st = sum(1 for e in events if e[0] == 0)
    n_jmoves = sum(1 for s in stops if s["kind"] == "jump")
    n_trims = sum(1 for s in stops if s["kind"] == "trim")
    n_changes = sum(1 for s in stops if s["kind"] == "color_change")
    minutes = (n_st / 700.0 + n_jmoves * 1.2 / 60.0 + n_trims * 5 / 60.0
               + n_changes * 20 / 60.0)
    stats = {
        "stitches": n_st, "jumps": n_jmoves, "trims": n_trims,
        "color_changes": n_changes, "colors": max(1, color_i + 1),
        "width_mm": round(max(xs) - min(xs), 1),
        "height_mm": round(max(ys) - min(ys), 1),
        "est_minutes": round(minutes, 1), "cx": 0, "cy": 0,
    }
    return {"threads": threads, "stitches": events, "stats": stats,
            "warnings": [], "stops": stops}
