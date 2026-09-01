"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface TextMeta {
  recommendedWidthMm?: number;
  singleColor: boolean;
  satinWidthMm?: number;
  connectorMm?: number;
}

interface Props {
  onFile: (file: File, meta: TextMeta) => void;
}

// Connected script fonts stitch as ONE continuous shape per word — the
// low-jump lettering style professional embroidery designs use.
const FONTS = [
  "Pacifico",
  "Yellowtail",
  "Dancing Script",
  "Great Vibes",
  "Graduate",
  "Alfa Slab One",
  "Luckiest Guy",
  "Lobster",
  "Helvetica",
  "Arial Black",
  "Futura",
  "Georgia",
  "Baskerville",
  "Times New Roman",
  "Verdana",
  "Trebuchet MS",
  "Impact",
  "Courier New",
  "Comic Sans MS",
  "Marker Felt",
  "Chalkboard SE",
  "Brush Script MT",
];

const SIZE = 260;

function drawChar(
  g: CanvasRenderingContext2D,
  ch: string,
  x: number,
  y: number,
  angle: number,
  color: string,
  outline: boolean,
) {
  g.save();
  g.translate(x, y);
  g.rotate(angle);
  if (outline) {
    // Full collegiate letter, inside out: core, inline, edge band, gap,
    // outer ring. Carved bands are transparent so the fabric shows.
    // Band widths must stay thinner than half a letter limb, or the carve
    // cuts straight through and the restore refills it solid.
    g.lineJoin = "round";
    g.miterLimit = 2;
    g.strokeStyle = color;
    g.fillStyle = color;
    g.lineWidth = SIZE * 0.21; // material out to the outer ring
    g.strokeText(ch, 0, 0);
    g.fillText(ch, 0, 0); // letter body
    g.globalCompositeOperation = "destination-out";
    g.lineWidth = SIZE * 0.115; // carve gap outside + inline channel inside
    g.strokeText(ch, 0, 0);
    g.globalCompositeOperation = "source-over";
    g.lineWidth = SIZE * 0.054; // restore the letter's edge band
    g.strokeText(ch, 0, 0);
  } else {
    g.fillStyle = color;
    g.fillText(ch, 0, 0);
  }
  g.restore();
}

async function renderText(
  text: string,
  font: string,
  bold: boolean,
  italic: boolean,
  color: string,
  outline: boolean,
  archPct: number,
): Promise<HTMLCanvasElement | null> {
  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) return null;
  const style = `${italic ? "italic " : ""}${bold ? "bold " : ""}${SIZE}px "${font}"`;
  try {
    await document.fonts.load(style, text);
  } catch {
    /* system fonts don't need loading */
  }

  const extra = outline ? SIZE * 0.08 : 0;
  const archRad = (archPct / 100) * 2.0; // up to ~115°

  const meas = document.createElement("canvas").getContext("2d")!;
  meas.font = style;
  const layout = lines.map((line) => {
    const chars = Array.from(line);
    const widths = chars.map((c) => meas.measureText(c).width);
    const W = widths.reduce((a, b) => a + b, 0) + extra * (chars.length - 1);
    return { chars, widths, W };
  });

  const maxW = Math.max(...layout.map((l) => l.W));
  const rise =
    archRad > 0.02 ? (maxW / archRad) * (1 - Math.cos(archRad / 2)) : 0;
  const margin = SIZE * 0.5;
  const lineStep = SIZE * 1.25 + rise;
  // Generous canvas — the digitizer crops to the artwork bounding box.
  const c = document.createElement("canvas");
  c.width = Math.ceil(maxW + margin * 2 + rise);
  c.height = Math.ceil(lines.length * lineStep + margin * 2);
  const g = c.getContext("2d")!;
  g.clearRect(0, 0, c.width, c.height);
  g.font = style;
  g.textAlign = "center";
  g.textBaseline = "alphabetic";

  const cx = c.width / 2;
  layout.forEach(({ chars, widths, W }, li) => {
    const baseY = margin + SIZE + li * lineStep;
    if (archRad < 0.02) {
      let x = cx - W / 2;
      chars.forEach((ch, i) => {
        drawChar(g, ch, x + widths[i] / 2, baseY, 0, color, outline);
        x += widths[i] + extra;
      });
    } else {
      const R = W / archRad;
      let s = 0;
      chars.forEach((ch, i) => {
        const mid = s + widths[i] / 2;
        const theta = -archRad / 2 + (mid / W) * archRad;
        const x = cx + R * Math.sin(theta);
        const y = baseY + R * (1 - Math.cos(theta));
        drawChar(g, ch, x, y, theta, color, outline);
        s += widths[i] + extra;
      });
    }
  });
  return c;
}

export default function TextMaker({ onFile }: Props) {
  const [text, setText] = useState("");
  const [font, setFont] = useState(FONTS[0]);
  const [bold, setBold] = useState(true);
  const [italic, setItalic] = useState(false);
  const [color, setColor] = useState("#1a2a52");
  const [outline, setOutline] = useState(false);
  const [connect, setConnect] = useState(true);
  const [arch, setArch] = useState(0);
  const [preview, setPreview] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let stale = false;
    const t = setTimeout(async () => {
      const c = await renderText(text, font, bold, italic, color, outline, arch);
      if (stale) return;
      canvasRef.current = c;
      setPreview(c ? c.toDataURL("image/png") : null);
    }, 250);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [text, font, bold, italic, color, outline, arch]);

  // Smallest output width (mm) where the finest feature of this design —
  // the varsity inline channel (0.03 × SIZE px) — is still ≥ 0.75 mm.
  const recommendedWidth = useCallback(() => {
    const c = canvasRef.current;
    if (!c) return undefined;
    if (outline) {
      const mmPerPx = 0.75 / (SIZE * 0.03);
      return Math.min(280, Math.ceil((c.width * mmPerPx) / 5) * 5);
    }
    // Plain lettering: professional designs run letters 25 mm+ tall so
    // strokes become plump wide satin. Recommend a width that gets there.
    const w = (25 * c.width) / Math.max(c.height, 1);
    return Math.max(60, Math.min(260, Math.ceil(w / 5) * 5));
  }, [outline]);

  const use = useCallback(() => {
    const c = canvasRef.current;
    if (!c) return;
    c.toBlob((blob) => {
      if (!blob) return;
      const safe =
        text
          .replace(/[^A-Za-z0-9]+/g, "_")
          .slice(0, 20)
          .replace(/^_+|_+$/g, "") || "text";
      onFile(new File([blob], `${safe}.png`, { type: "image/png" }), {
        recommendedWidthMm: recommendedWidth(),
        singleColor: true,
        satinWidthMm: 7, // lettering strokes stay satin, not fill
        connectorMm: connect ? 5 : undefined,
      });
    }, "image/png");
  }, [onFile, text, recommendedWidth, connect]);

  return (
    <div className="flex flex-col gap-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={"Type your text here…\n(multiple lines are fine)"}
        rows={2}
        className="w-full rounded-xl border border-gray-300 px-4 py-3 text-lg text-gray-800 focus:border-indigo-500 focus:outline-none"
      />
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={font}
          onChange={(e) => setFont(e.target.value)}
          className="rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-700"
          style={{ fontFamily: font }}
        >
          {FONTS.map((f) => (
            <option key={f} value={f} style={{ fontFamily: f }}>
              {f}
            </option>
          ))}
        </select>
        <label className="flex items-center gap-1.5 text-sm text-gray-700">
          <input
            type="checkbox"
            checked={bold}
            onChange={(e) => setBold(e.target.checked)}
            className="h-4 w-4 accent-indigo-600"
          />
          <b>Bold</b>
        </label>
        <label className="flex items-center gap-1.5 text-sm text-gray-700">
          <input
            type="checkbox"
            checked={italic}
            onChange={(e) => setItalic(e.target.checked)}
            className="h-4 w-4 accent-indigo-600"
          />
          <i>Italic</i>
        </label>
        <label className="flex items-center gap-1.5 text-sm text-gray-700">
          <input
            type="checkbox"
            checked={outline}
            onChange={(e) => setOutline(e.target.checked)}
            className="h-4 w-4 accent-indigo-600"
          />
          🏈 Varsity outline
        </label>
        <label className="flex items-center gap-1.5 text-sm text-gray-700">
          <input
            type="checkbox"
            checked={connect}
            onChange={(e) => setConnect(e.target.checked)}
            className="h-4 w-4 accent-indigo-600"
          />
          Connect letters
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-700">
          Thread color
          <input
            type="color"
            value={color}
            onChange={(e) => setColor(e.target.value)}
            className="h-8 w-12 cursor-pointer rounded border border-gray-300"
          />
        </label>
      </div>
      <label className="flex flex-col gap-1 text-sm text-gray-600">
        <span>
          Arch: <b>{arch === 0 ? "off" : `${arch}%`}</b>
        </span>
        <input
          type="range"
          min={0}
          max={100}
          value={arch}
          onChange={(e) => setArch(Number(e.target.value))}
          className="accent-indigo-600"
        />
      </label>
      {preview ? (
        <div className="rounded-xl border border-gray-200 bg-[repeating-conic-gradient(#f3f4f6_0_25%,white_0_50%)] bg-[length:16px_16px] p-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preview}
            alt="Text preview"
            className="mx-auto max-h-44 object-contain"
          />
        </div>
      ) : (
        <p className="py-6 text-center text-sm text-gray-400">
          The preview appears as you type
        </p>
      )}
      <button
        onClick={use}
        disabled={!preview}
        className="w-full rounded-xl bg-indigo-600 py-3 text-base font-bold tracking-wide text-white shadow transition hover:bg-indigo-700 disabled:opacity-40"
      >
        ➜ USE THIS TEXT
      </button>
      <p className="text-center text-xs text-gray-400">
        {outline
          ? "Varsity outline has fine details — the width in step 2 is set automatically to the minimum stitchable size."
          : "Tip: script fonts (Pacifico, Yellowtail…) connect the letters, so a word stitches as ONE piece with almost no jumps — the professional embroidery lettering style. Block fonts need one stop per letter."}
      </p>
    </div>
  );
}
