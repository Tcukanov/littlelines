# PNG → DST

Personal web app that turns simple PNG/JPG artwork (logos, kids' drawings,
line art) into real, machine-ready Tajima **.DST** embroidery files.

The image is **actually digitized** — cleanup → color segmentation →
contour detection → stitch path generation → DST encoding — not renamed.

## Run it

```bash
./start.sh
```

Then open **http://localhost:3000**. Stop with Ctrl-C.

(That script starts the Python digitizing service on port 8000 and the web
app on port 3000.)

## Workflow

**UPLOAD → SIZE → AUTO DIGITIZE → PREVIEW → DOWNLOAD DST**

1. **Upload** — drag & drop a PNG (transparent background works best) or JPG.
2. **Prepare** — colors and shapes are detected automatically; untick colors
   you don't want, set the finished size in mm or inches, adjust detail /
   small-object removal.
3. **AUTO DIGITIZE** — picks fill / satin / running stitch per shape
   automatically. Override per color if you like. Changes after the first
   digitize re-render the preview automatically.
4. **Preview** — real stitches with thread colors, jumps (dashed), start
   (green) / end (red) markers. Scroll to zoom, drag to pan. Shows stitch
   count, colors, trims, size and estimated time.
5. **Export** — DOWNLOAD DST (validated by reading the file back), plus
   PES / EXP / JEF / SVG.

**Line Art Mode** (in step 2) is for one-color line drawings: it extracts
the centerline of dark strokes and uses satin for wide lines and 1/2/3-pass
running stitch for thin ones.

## Notes

- Uploads are processed in memory; nothing is stored on disk.
- Stitch limits are enforced (max 12.1 mm stitch/jump, DST coordinate range,
  hoop-size warnings).
- Stack: Next.js + TypeScript + Tailwind frontend; FastAPI + OpenCV +
  NumPy + scikit-image + pyembroidery backend (`backend/digitizer/`).
- Backend tests: `cd backend && .venv/bin/python test_pipeline.py`
