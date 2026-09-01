"""PNG to DST — digitizing backend.

Stateless by design: uploads are processed in memory, temp files used during
encoding are deleted immediately, nothing is ever stored.
"""
from __future__ import annotations

import json
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from digitizer import export, pipeline
from digitizer.params import Settings

app = FastAPI(title="PNG to DST")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
