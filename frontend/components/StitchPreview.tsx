"use client";

import { useCallback, useEffect, useRef } from "react";
import {
  CMD_COLOR_CHANGE,
  CMD_JUMP,
  CMD_STITCH,
  DigitizeResult,
} from "@/lib/types";

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
        ctx.lineWidth = Math.max(1.6, scale * 0.5);
        ctx.stroke(thread);
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(1, scale * 0.36);
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
  }, [result]);

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
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!dragRef.current) return;
          viewRef.current.ox += e.clientX - dragRef.current.x;
          viewRef.current.oy += e.clientY - dragRef.current.y;
          dragRef.current = { x: e.clientX, y: e.clientY };
          draw();
        }}
        onPointerUp={() => (dragRef.current = null)}
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
        scroll = zoom · drag = pan · <span className="text-green-600">●</span>{" "}
        start · <span className="text-red-600">●</span> end · ▪▪ jump
      </div>
    </div>
  );
}
