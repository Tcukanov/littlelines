"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  CMD_COLOR_CHANGE,
  CMD_JUMP,
  CMD_STITCH,
  DigitizeResult,
  StopInfo,
} from "@/lib/types";

const REASONS: Record<string, string> = {
  SEPARATE_ISLAND:
    "this piece is separate — bare fabric on all sides, no hidden route exists (a professional file would also stop here)",
  NO_ROUTE: "no connected path found through the stitchable area",
  ROUTE_TOO_LONG: "a hidden route exists but is too long to be safe",
  SNAP_FAIL: "the connection point sits off the artwork",
  NO_ROUTER: "no routing context for this move",
  COLOR_CHANGE: "thread color change",
};

interface Props {
  result: DigitizeResult;
}

interface View {
  scale: number; // px per mm
  ox: number; // canvas px offset
  oy: number;
}

export default function StitchPreview({ result }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<View>({ scale: 4, ox: 0, oy: 0 });
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const fitScaleRef = useRef(4);
  const movedRef = useRef(false);
  const [selectedStop, setSelectedStop] = useState<StopInfo | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const dpr = window.devicePixelRatio || 1;
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { scale, ox, oy } = viewRef.current;

    ctx.fillStyle = "#dce0e5";
    ctx.fillRect(0, 0, w, h);

    // 10 mm grid
    ctx.strokeStyle = "#cfd3d9";
    ctx.lineWidth = 1;
    const gridMm = 10;
    const step = gridMm * scale;
    if (step > 8) {
      const startX = ((ox % step) + step) % step;
      const startY = ((oy % step) + step) % step;
      ctx.beginPath();
      for (let x = startX; x < w; x += step) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
      }
      for (let y = startY; y < h; y += step) {
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
      }
      ctx.stroke();
    }
    // Axes through origin
    ctx.strokeStyle = "#bec3ca";
    ctx.beginPath();
    ctx.moveTo(ox, 0);
    ctx.lineTo(ox, h);
    ctx.moveTo(0, oy);
    ctx.lineTo(w, oy);
    ctx.stroke();

    const X = (mm: number) => ox + mm * scale;
    const Y = (mm: number) => oy + mm * scale;

    // Stitches — batched per color block, drawn as thread: a soft dark
    // shadow pass under a colored pass so even white thread is visible.
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    let colorIdx = 0;
    let color = result.threads[0] ?? "#333";
    let prev: [number, number] | null = null;
    let thread = new Path2D();
    let jumpsPath = new Path2D();
    let hasThread = false;
    let hasJumps = false;
    const markers: [number, number][] = [];

    const flush = () => {
      if (hasJumps) {
        ctx.strokeStyle = "#a8a8b4";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke(jumpsPath);
        ctx.setLineDash([]);
        jumpsPath = new Path2D();
        hasJumps = false;
      }
      if (hasThread) {
        ctx.strokeStyle = "rgba(25, 25, 35, 0.35)";
        ctx.lineWidth = Math.max(1.8, scale * 0.55);
        ctx.stroke(thread);
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(1.2, scale * 0.42);
        ctx.stroke(thread);
        // Sheen: a lighter core line makes thread look round and glossy.
        ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
        ctx.lineWidth = Math.max(0.6, scale * 0.14);
        ctx.stroke(thread);
        thread = new Path2D();
        hasThread = false;
      }
    };

    for (const [cmd, x, y] of result.stitches) {
      if (cmd === CMD_STITCH) {
        if (prev) {
          thread.moveTo(X(prev[0]), Y(prev[1]));
          thread.lineTo(X(x), Y(y));
          hasThread = true;
        }
        prev = [x, y];
      } else if (cmd === CMD_JUMP) {
        if (prev) {
          jumpsPath.moveTo(X(prev[0]), Y(prev[1]));
          jumpsPath.lineTo(X(x), Y(y));
          hasJumps = true;
        }
        prev = [x, y];
      } else if (cmd === CMD_COLOR_CHANGE) {
        flush();
        colorIdx += 1;
        color = result.threads[colorIdx] ?? color;
        if (prev) markers.push([prev[0], prev[1]]);
      }
    }
    flush();
    ctx.fillStyle = "#8888ff";
    for (const [mx, my] of markers) {
      ctx.beginPath();
      ctx.arc(X(mx), Y(my), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // At loupe zoom, show actual needle penetration points to judge gaps.
    if (scale >= 8) {
      const dots = new Path2D();
      let n = 0;
      for (const [cmd, x, y] of result.stitches) {
        if (cmd !== CMD_STITCH) continue;
        const px = X(x);
        const py = Y(y);
        if (px < -5 || px > w + 5 || py < -5 || py > h + 5) continue;
        dots.moveTo(px + 1.6, py);
        dots.arc(px, py, 1.6, 0, Math.PI * 2);
        if (++n > 20000) break;
      }
      ctx.fillStyle = "rgba(15, 15, 25, 0.55)";
      ctx.fill(dots);
    }

    // Stop markers (clickable): trims red, jumps amber, changes blue.
    for (const s of result.stops ?? []) {
      const px = X(s.x);
      const py = Y(s.y);
      if (px < -10 || px > w + 10 || py < -10 || py > h + 10) continue;
      ctx.beginPath();
      ctx.arc(px, py, selectedStop === s ? 7 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle =
        s.kind === "trim"
          ? "rgba(220, 60, 60, 0.9)"
          : s.kind === "jump"
            ? "rgba(235, 160, 40, 0.9)"
            : "rgba(90, 110, 235, 0.9)";
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#fff";
      ctx.stroke();
    }

    // Start / end markers
    const first = result.stitches.find(
      (s) => s[0] === CMD_STITCH || s[0] === CMD_JUMP,
    );
    let last: [number, number, number] | undefined;
    for (const s of result.stitches) {
      if (s[0] === CMD_STITCH) last = s;
    }
    if (first) {
      ctx.fillStyle = "#16a34a";
      ctx.beginPath();
      ctx.arc(X(first[1]), Y(first[2]), 5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (last) {
      ctx.fillStyle = "#dc2626";
      ctx.beginPath();
      ctx.arc(X(last[1]), Y(last[2]), 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Scale bar (10 mm)
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(16, h - 16);
    ctx.lineTo(16 + 10 * scale, h - 16);
    ctx.stroke();
    ctx.fillStyle = "#333";
    ctx.font = "11px sans-serif";
    ctx.fillText("10 mm", 18, h - 22);
  }, [result, selectedStop]);

  const fit = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const pts = result.stitches.filter(
      (s) => s[0] === CMD_STITCH || s[0] === CMD_JUMP,
    );
    if (pts.length === 0) return;
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const [, x, y] of pts) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    const scale = Math.min(
      (w - 60) / Math.max(maxX - minX, 1),
      (h - 60) / Math.max(maxY - minY, 1),
    );
    fitScaleRef.current = scale;
    viewRef.current = {
      scale,
      ox: w / 2 - ((minX + maxX) / 2) * scale,
      oy: h / 2 - ((minY + maxY) / 2) * scale,
    };
    draw();
  }, [result, draw]);

  const zoomTo = useCallback(
    (mult: number) => {
      const wrap = wrapRef.current;
      if (!wrap) return;
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      const v = viewRef.current;
      const ns = Math.min(80, fitScaleRef.current * mult);
      // Keep whatever is at the canvas center in place while zooming.
      const cx = (w / 2 - v.ox) / v.scale;
      const cy = (h / 2 - v.oy) / v.scale;
      viewRef.current = { scale: ns, ox: w / 2 - cx * ns, oy: h / 2 - cy * ns };
      draw();
    },
    [draw],
  );

  useEffect(() => {
    fit();
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [fit, draw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const v = viewRef.current;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const ns = Math.min(60, Math.max(0.5, v.scale * factor));
      v.ox = mx - ((mx - v.ox) / v.scale) * ns;
      v.oy = my - ((my - v.oy) / v.scale) * ns;
      v.scale = ns;
      draw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [draw]);

  return (
    <div
      ref={wrapRef}
      className="relative h-[480px] w-full overflow-hidden rounded-xl border border-gray-200 bg-gray-50"
    >
      <canvas
        ref={canvasRef}
        className="h-full w-full cursor-grab active:cursor-grabbing"
        onPointerDown={(e) => {
          dragRef.current = { x: e.clientX, y: e.clientY };
          movedRef.current = false;
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!dragRef.current) {
            // Hover feedback: pointer cursor near a clickable stop marker.
            const rect = (e.target as HTMLElement).getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const { scale, ox, oy } = viewRef.current;
            let near = false;
            for (const s of result.stops ?? []) {
              if (
                Math.hypot(ox + s.x * scale - mx, oy + s.y * scale - my) < 22
              ) {
                near = true;
                break;
              }
            }
            (e.target as HTMLElement).style.cursor = near ? "pointer" : "";
            return;
          }
          const dx = e.clientX - dragRef.current.x;
          const dy = e.clientY - dragRef.current.y;
          if (Math.abs(dx) + Math.abs(dy) > 8) movedRef.current = true;
          viewRef.current.ox += dx;
          viewRef.current.oy += dy;
          dragRef.current = { x: e.clientX, y: e.clientY };
          draw();
        }}
        onPointerUp={(e) => {
          dragRef.current = null;
          if (movedRef.current) return;
          // Click: select the nearest stop marker.
          const rect = (e.target as HTMLElement).getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const my = e.clientY - rect.top;
          const { scale, ox, oy } = viewRef.current;
          let best: StopInfo | null = null;
          let bestD = 22;
          for (const s of result.stops ?? []) {
            const d = Math.hypot(ox + s.x * scale - mx, oy + s.y * scale - my);
            if (d < bestD) {
              bestD = d;
              best = s;
            }
          }
          setSelectedStop(best);
        }}
      />
      <div className="absolute right-3 top-3 flex overflow-hidden rounded-lg border border-gray-300 bg-white text-xs text-gray-600 shadow-sm">
        <button onClick={fit} className="px-3 py-1 hover:bg-gray-100">
          Fit
        </button>
        <button
          onClick={() => zoomTo(5)}
          className="border-l border-gray-200 px-3 py-1 hover:bg-gray-100"
        >
          ×5
        </button>
        <button
          onClick={() => zoomTo(10)}
          className="border-l border-gray-200 px-3 py-1 hover:bg-gray-100"
        >
          ×10
        </button>
        <button
          onClick={() => zoomTo(20)}
          className="border-l border-gray-200 px-3 py-1 hover:bg-gray-100"
        >
          ×20
        </button>
      </div>
      <div className="pointer-events-none absolute bottom-3 right-3 rounded bg-white/80 px-2 py-1 text-[10px] text-gray-500">
        scroll = zoom · drag = pan · click a{" "}
        <span className="text-red-600">●</span>
        <span className="text-amber-500">●</span> stop marker for details
      </div>
      {selectedStop && (
        <div className="absolute bottom-3 left-3 max-w-md rounded-lg border border-gray-300 bg-white/95 px-3 py-2 text-xs text-gray-700 shadow">
          <b>
            {selectedStop.kind === "trim"
              ? "✂ Trim"
              : selectedStop.kind === "jump"
                ? "→ Jump"
                : "● Color change"}
          </b>
          {selectedStop.gap > 0 && <> · {selectedStop.gap} mm move</>}
          <div className="mt-0.5 text-gray-500">
            {REASONS[selectedStop.reason] ?? selectedStop.reason}
          </div>
        </div>
      )}
    </div>
  );
}
