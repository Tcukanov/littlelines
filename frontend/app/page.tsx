"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import StitchPreview from "@/components/StitchPreview";
import TextMaker, { TextMeta } from "@/components/TextMaker";
import UploadZone from "@/components/UploadZone";
import * as api from "@/lib/api";
import {
  AnalyzeResult,
  AppSettings,
  DEFAULT_SETTINGS,
  DigitizeResult,
  StitchType,
} from "@/lib/types";

const MM_PER_INCH = 25.4;

function svgPalette(svgText: string): string[] {
  // The SVG's declared fills/strokes are the authoritative thread colors.
  const ctx = document.createElement("canvas").getContext("2d")!;
  const found = new Set<string>();
  const re = /(?:fill|stroke)\s*[:=]\s*["']?([^"';>\s}]+)/g;
  let m;
  while ((m = re.exec(svgText))) {
    const v = m[1].trim();
    if (!v || v === "none" || v === "transparent" || v.startsWith("url"))
      continue;
    ctx.fillStyle = "#000000";
    ctx.fillStyle = v;
    const norm = ctx.fillStyle;
    if (/^#[0-9a-f]{6}$/.test(norm)) found.add(norm);
  }
  return [...found].slice(0, 12);
}

async function svgToPng(
  file: File,
): Promise<{ file: File; palette: string[] }> {
  let text = await file.text();
  const load = (src: string) =>
    new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  let url = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
  let img = await load(url);
  if (!img.naturalWidth || !img.naturalHeight) {
    // SVG with only a viewBox: inject explicit dimensions.
    const m = text.match(/viewBox\s*=\s*"([-\d.\s]+)"/);
    let w = 1200,
      h = 1200;
    if (m) {
      const parts = m[1].trim().split(/\s+/).map(Number);
      if (parts.length === 4 && parts[2] > 0 && parts[3] > 0) {
        const s = 1200 / Math.max(parts[2], parts[3]);
        w = Math.round(parts[2] * s);
        h = Math.round(parts[3] * s);
      }
    }
    URL.revokeObjectURL(url);
    text = text.replace(/<svg/, `<svg width="${w}" height="${h}"`);
    url = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
    img = await load(url);
  }
  const scale = 1200 / Math.max(img.naturalWidth, img.naturalHeight, 1);
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d")!.drawImage(img, 0, 0, w, h);
  URL.revokeObjectURL(url);
  const blob = await new Promise<Blob | null>((r) => c.toBlob(r, "image/png"));
  if (!blob) throw new Error("Could not rasterize the SVG.");
  const base = file.name.replace(/\.svg$/i, "");
  return {
    file: new File([blob], `${base}.png`, { type: "image/png" }),
    palette: svgPalette(text),
  };
}

function Section(props: {
  step: number;
  title: string;
  children: React.ReactNode;
  dimmed?: boolean;
}) {
  return (
    <section
      className={`rounded-2xl border border-gray-200 bg-white p-6 shadow-sm transition ${props.dimmed ? "pointer-events-none opacity-40" : ""}`}
    >
      <h2 className="mb-4 flex items-center gap-3 text-lg font-semibold text-gray-800">
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-indigo-600 text-sm font-bold text-white">
          {props.step}
        </span>
        {props.title}
      </h2>
      {props.children}
    </section>
  );
}

function Num(props: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  suffix?: string;
}) {
  return (
    <label className="flex flex-col gap-1 text-sm text-gray-600">
      <span>{props.label}</span>
      <span className="flex items-center gap-1">
        <input
          type="number"
          className="w-24 rounded-lg border border-gray-300 px-2 py-1.5 text-gray-800 focus:border-indigo-500 focus:outline-none"
          value={props.value}
          min={props.min}
          max={props.max}
          step={props.step}
          onChange={(e) => props.onChange(Number(e.target.value))}
        />
        {props.suffix && (
          <span className="text-xs text-gray-400">{props.suffix}</span>
        )}
      </span>
    </label>
  );
}

function Toggle(props: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-sm text-gray-700">
      <input
        type="checkbox"
        className="h-4 w-4 accent-indigo-600"
        checked={props.checked}
        onChange={(e) => props.onChange(e.target.checked)}
      />
      {props.label}
    </label>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [unit, setUnit] = useState<"mm" | "inch">("mm");
  const [lockAspect, setLockAspect] = useState(true);
  const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null);
  const [result, setResult] = useState<DigitizeResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [digitizing, setDigitizing] = useState(false);
  const [exporting, setExporting] = useState<string | null>(null);
  const [originalFile, setOriginalFile] = useState<File | null>(null);
  const [cleaning, setCleaning] = useState(false);
  const [sourceTab, setSourceTab] = useState<"image" | "text">("image");
  const [error, setError] = useState<string | null>(null);
  const [backendUp, setBackendUp] = useState(true);
  const digitizedOnce = useRef(false);

  useEffect(() => {
    api.health().then(setBackendUp);
  }, []);

  const set = useCallback(<K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setSettings((s) => ({ ...s, [key]: value }));
  }, []);

  const swapFile = useCallback((f: File) => {
    setFile(f);
    setImageUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(f);
    });
    setResult(null);
    digitizedOnce.current = false;
    setSettings((s) => ({ ...s, color_settings: {} }));
  }, []);

  const onFile = useCallback(
    (f: File) => {
      setOriginalFile(null);
      if (f.type === "image/svg+xml" || /\.svg$/i.test(f.name)) {
        setError(null);
        svgToPng(f)
          .then(({ file: png, palette }) => {
            swapFile(png);
            setSettings((s) => ({
              ...s,
              palette_hint: palette,
              max_colors: Math.max(s.max_colors,
                Math.min(12, palette.length || s.max_colors)),
            }));
          })
          .catch(() => setError("Could not read that SVG file."));
        return;
      }
      swapFile(f);
      setSettings((s) => ({ ...s, palette_hint: [] }));
    },
    [swapFile],
  );

  const onTextFile = useCallback(
    (f: File, meta: TextMeta) => {
      setOriginalFile(null);
      swapFile(f);
      setSettings((s) => ({
        ...s,
        ...(meta.singleColor ? { max_colors: 1 } : {}),
        ...(meta.recommendedWidthMm
          ? { width_mm: Math.max(s.width_mm, meta.recommendedWidthMm) }
          : {}),
        ...(meta.satinWidthMm
          ? { satin_width_mm: Math.max(s.satin_width_mm, meta.satinWidthMm) }
          : {}),
        walk_connector_mm: meta.connectorMm ?? 2.5,
      }));
    },
    [swapFile],
  );

  const runCleanup = useCallback(async () => {
    // Always clean from the ORIGINAL image so changing settings after a
    // cleanup can't compound losses on an already-flattened file.
    const source = originalFile ?? file;
    if (!source) return;
    setCleaning(true);
    setError(null);
    try {
      const cleaned = await api.cleanup(source, settings);
      setOriginalFile(source);
      swapFile(cleaned);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCleaning(false);
    }
  }, [file, originalFile, settings, swapFile]);

  const revertCleanup = useCallback(() => {
    if (!originalFile) return;
    swapFile(originalFile);
    setOriginalFile(null);
  }, [originalFile, swapFile]);

  // Analyze (debounced) whenever the file or prep settings change.
  const prepKey = JSON.stringify([
    settings.max_colors,
    settings.detail,
    settings.min_object_mm2,
    settings.remove_background,
    settings.width_mm,
    settings.height_mm,
    settings.satin_width_mm,
  ]);
  useEffect(() => {
    if (!file) return;
    const t = setTimeout(async () => {
      setAnalyzing(true);
      setError(null);
      try {
        const a = await api.analyze(file, settings);
        setAnalysis(a);
        setBackendUp(true);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setAnalyzing(false);
      }
    }, 400);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, prepKey]);

  // If a cleaned file is active and prep settings change, regenerate the
  // cleanup from the original so the new settings actually apply.
  useEffect(() => {
    if (!originalFile) return;
    const t = setTimeout(runCleanup, 600);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prepKey]);

  // Keep aspect ratio locked to the artwork.
  useEffect(() => {
    if (!analysis || !lockAspect) return;
    const h = settings.width_mm / analysis.aspect;
    if (Math.abs(h - settings.height_mm) > 0.05) {
      set("height_mm", Math.round(h * 10) / 10);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis, lockAspect, settings.width_mm]);

  const runDigitize = useCallback(async () => {
    if (!file) return;
    setDigitizing(true);
    setError(null);
    try {
      const r = await api.digitize(file, settings);
      setResult(r);
      digitizedOnce.current = true;
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setDigitizing(false);
    }
  }, [file, settings]);

  // Re-digitize automatically after the first run when settings change.
  const settingsKey = JSON.stringify(settings);
  useEffect(() => {
    if (!file || !digitizedOnce.current) return;
    const t = setTimeout(runDigitize, 700);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settingsKey]);

  const autoDigitize = useCallback(() => {
    setSettings((s) => ({
      ...DEFAULT_SETTINGS,
      width_mm: s.width_mm,
      height_mm: s.height_mm,
      max_colors: s.max_colors,
      detail: s.detail,
      min_object_mm2: s.min_object_mm2,
      remove_background: s.remove_background,
      line_art: s.line_art,
      line_passes: s.line_passes,
      color_settings: Object.fromEntries(
        Object.entries(s.color_settings).map(([k, v]) => [
          k,
          { ...v, stitch: "auto" as StitchType },
        ]),
      ),
    }));
    digitizedOnce.current = true; // the settings change triggers a run
    setTimeout(runDigitize, 0);
  }, [runDigitize]);

  const download = useCallback(
    async (format: string) => {
      if (!file) return;
      setExporting(format);
      setError(null);
      try {
        await api.exportFile(file, settings, format);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setExporting(null);
      }
    },
    [file, settings],
  );

  const dim = useCallback(
    (mm: number) =>
      unit === "mm" ? Math.round(mm * 10) / 10 : Math.round((mm / MM_PER_INCH) * 100) / 100,
    [unit],
  );
  const toMm = useCallback(
    (v: number) => (unit === "mm" ? v : v * MM_PER_INCH),
    [unit],
  );

  const stats = result?.stats;
  const sizeLabel = useMemo(() => {
    const w = dim(settings.width_mm);
    const h = dim(settings.height_mm);
    return `${w} × ${h} ${unit === "mm" ? "mm" : "in"}`;
  }, [settings.width_mm, settings.height_mm, dim, unit]);

  return (
    <main className="mx-auto max-w-4xl px-4 py-10">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-gray-900">
          PNG <span className="text-indigo-600">→</span> DST
        </h1>
        <p className="mt-2 text-sm text-gray-500">
          Upload → Size → Auto digitize → Preview → Download DST
        </p>
      </header>

      {!backendUp && (
        <div className="mb-6 rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-800">
          The digitizing service isn&apos;t running. Start it with{" "}
          <code className="rounded bg-amber-100 px-1">
            cd backend && .venv/bin/uvicorn main:app --port 8000
          </code>{" "}
          and reload this page.
        </div>
      )}
      {error && (
        <div className="mb-6 rounded-xl border border-red-300 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="flex flex-col gap-6">
        <Section step={1} title="Upload image or create text">
          <div className="mb-4 flex overflow-hidden rounded-xl border border-gray-200 text-sm font-semibold">
            {(
              [
                ["image", "🖼 Image"],
                ["text", "🔤 Text"],
              ] as const
            ).map(([tab, label]) => (
              <button
                key={tab}
                onClick={() => setSourceTab(tab)}
                className={`flex-1 py-2.5 transition ${
                  sourceTab === tab
                    ? "bg-indigo-600 text-white"
                    : "bg-white text-gray-500 hover:bg-gray-50"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {sourceTab === "image" ? (
            <UploadZone
              onFile={onFile}
              fileName={file?.name ?? null}
              imageUrl={imageUrl}
            />
          ) : (
            <>
              <TextMaker onFile={onTextFile} />
              {file && imageUrl && (
                <p className="mt-2 text-center text-xs text-gray-500">
                  Current artwork: <b>{file.name}</b>
                </p>
              )}
            </>
          )}
          <div
            className={`mt-4 flex-col items-center gap-2 ${sourceTab === "image" ? "flex" : "hidden"}`}
          >
            <button
              onClick={originalFile ? revertCleanup : runCleanup}
              disabled={!file || cleaning}
                className={`w-full rounded-xl py-3 text-base font-bold tracking-wide shadow transition disabled:opacity-40 ${
                originalFile
                  ? "border border-gray-300 bg-white text-gray-600 hover:bg-gray-50"
                  : "bg-violet-600 text-white hover:bg-violet-700"
              }`}
            >
              {cleaning
                ? "Cleaning…"
                : originalFile
                  ? "↩ USE ORIGINAL IMAGE"
                  : "🪄 CLEAN UP ARTWORK"}
            </button>
            <span
              className={`text-xs ${originalFile ? "font-medium text-violet-600" : "text-gray-400"}`}
            >
              {originalFile
                ? "Using the cleaned artwork — click above to go back to the original"
                : "Optional: traces the image into smooth, flat colors before digitizing (uses the Max colors setting below)"}
            </span>
          </div>
        </Section>

        <Section step={2} title="Image preparation & size" dimmed={!file}>
          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">
                  Detected colors {analyzing && "…"}
                </span>
                <span className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">
                  {sizeLabel}
                </span>
              </div>
              {analysis?.preview_png && (
                <div className="mb-3 rounded-lg border border-gray-200 bg-[repeating-conic-gradient(#f3f4f6_0_25%,white_0_50%)] bg-[length:16px_16px] p-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/png;base64,${analysis.preview_png}`}
                    alt="Detected shapes"
                    className="mx-auto max-h-44 object-contain"
                  />
                </div>
              )}
              <div className="flex flex-col gap-2">
                {analysis?.colors.map((c) => {
                  const cs = settings.color_settings[c.index] ?? {
                    enabled: true,
                    stitch: "auto" as StitchType,
                  };
                  return (
                    <div
                      key={c.index}
                      className={`flex items-center gap-3 rounded-lg border px-3 py-2 ${cs.enabled ? "border-gray-200" : "border-gray-100 opacity-50"}`}
                    >
                      <input
                        type="checkbox"
                        className="h-4 w-4 accent-indigo-600"
                        checked={cs.enabled}
                        onChange={(e) =>
                          set("color_settings", {
                            ...settings.color_settings,
                            [c.index]: { ...cs, enabled: e.target.checked },
                          })
                        }
                      />
                      <span
                        className="h-6 w-6 rounded-full border border-gray-300"
                        style={{ background: c.hex }}
                      />
                      <span className="flex-1 text-xs text-gray-500">
                        {c.hex} · {c.regions} shape{c.regions !== 1 && "s"}
                      </span>
                      <select
                        className="rounded-lg border border-gray-300 px-2 py-1 text-xs text-gray-700"
                        value={cs.stitch}
                        onChange={(e) =>
                          set("color_settings", {
                            ...settings.color_settings,
                            [c.index]: {
                              ...cs,
                              stitch: e.target.value as StitchType,
                            },
                          })
                        }
                      >
                        <option value="auto">Auto ({c.suggested})</option>
                        <option value="fill">Fill stitch</option>
                        <option value="satin">Satin stitch</option>
                        <option value="running">Running stitch</option>
                      </select>
                    </div>
                  );
                })}
                {file && !analysis && !analyzing && (
                  <p className="text-sm text-gray-400">Analyzing…</p>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-4">
              <div className="flex items-end gap-3">
                <Num
                  label={`Width (${unit})`}
                  value={dim(settings.width_mm)}
                  min={unit === "mm" ? 10 : 0.4}
                  max={unit === "mm" ? 400 : 15}
                  step={unit === "mm" ? 1 : 0.1}
                  onChange={(v) => set("width_mm", toMm(v))}
                />
                <Num
                  label={`Height (${unit})`}
                  value={dim(settings.height_mm)}
                  min={unit === "mm" ? 10 : 0.4}
                  max={unit === "mm" ? 400 : 15}
                  step={unit === "mm" ? 1 : 0.1}
                  onChange={(v) => {
                    setLockAspect(false);
                    set("height_mm", toMm(v));
                  }}
                />
                <div className="flex overflow-hidden rounded-lg border border-gray-300 text-xs">
                  {(["mm", "inch"] as const).map((u) => (
                    <button
                      key={u}
                      onClick={() => setUnit(u)}
                      className={`px-3 py-2 ${unit === u ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}
                    >
                      {u}
                    </button>
                  ))}
                </div>
              </div>
              <Toggle
                label="Lock aspect ratio"
                checked={lockAspect}
                onChange={setLockAspect}
              />
              <label className="flex flex-col gap-1 text-sm text-gray-600">
                <span>
                  Max colors: <b>{settings.max_colors}</b>
                </span>
                <input
                  type="range"
                  min={1}
                  max={12}
                  value={settings.max_colors}
                  onChange={(e) => set("max_colors", Number(e.target.value))}
                  className="accent-indigo-600"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm text-gray-600">
                <span>
                  Detail level: <b>{settings.detail}</b>{" "}
                  <span className="text-xs text-gray-400">
                    (lower = simpler, easier to stitch)
                  </span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={settings.detail}
                  onChange={(e) => set("detail", Number(e.target.value))}
                  className="accent-indigo-600"
                />
              </label>
              <Num
                label="Remove objects smaller than"
                value={settings.min_object_mm2}
                min={0}
                max={50}
                step={0.5}
                suffix="mm²"
                onChange={(v) => set("min_object_mm2", v)}
              />
              <Toggle
                label="Remove background automatically"
                checked={settings.remove_background}
                onChange={(v) => set("remove_background", v)}
              />

              <div
                className={`rounded-xl border p-3 ${settings.line_art ? "border-indigo-400 bg-indigo-50" : "border-gray-200"}`}
              >
                <Toggle
                  label="✏️ Line Art Mode — for one-color line drawings"
                  checked={settings.line_art}
                  onChange={(v) => set("line_art", v)}
                />
                {settings.line_art && (
                  <div className="mt-2 flex items-center gap-2 text-sm text-gray-600">
                    Passes:
                    {[1, 2, 3].map((p) => (
                      <button
                        key={p}
                        onClick={() => set("line_passes", p)}
                        className={`rounded-lg border px-3 py-1 text-xs ${settings.line_passes === p ? "border-indigo-600 bg-indigo-600 text-white" : "border-gray-300 bg-white text-gray-600"}`}
                      >
                        {p}-pass
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </Section>

        <Section step={3} title="Embroidery settings" dimmed={!file}>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Num
              label="Stitch density (row gap)"
              value={settings.density_mm}
              min={0.25}
              max={1}
              step={0.05}
              suffix="mm"
              onChange={(v) => set("density_mm", v)}
            />
            <Num
              label="Stitch length"
              value={settings.stitch_len_mm}
              min={1}
              max={7}
              step={0.5}
              suffix="mm"
              onChange={(v) => set("stitch_len_mm", v)}
            />
            <Num
              label="Satin width (max)"
              value={settings.satin_width_mm}
              min={1}
              max={7}
              step={0.5}
              suffix="mm"
              onChange={(v) => set("satin_width_mm", v)}
            />
            <div className="flex flex-col gap-1 text-sm text-gray-600">
              <span>Fill angle</span>
              <span className="flex items-center gap-2">
                <label className="flex items-center gap-1 text-xs">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-indigo-600"
                    checked={settings.auto_fill_angle}
                    onChange={(e) => set("auto_fill_angle", e.target.checked)}
                  />
                  Auto
                </label>
                <input
                  type="number"
                  className="w-16 rounded-lg border border-gray-300 px-2 py-1.5 text-gray-800 focus:border-indigo-500 focus:outline-none disabled:opacity-40"
                  value={settings.fill_angle_deg}
                  min={0}
                  max={180}
                  step={15}
                  disabled={settings.auto_fill_angle}
                  onChange={(e) =>
                    set("fill_angle_deg", Number(e.target.value))
                  }
                />
                <span className="text-xs text-gray-400">°</span>
              </span>
            </div>
            <Num
              label="Pull compensation"
              value={settings.pull_comp_mm}
              min={0}
              max={1}
              step={0.05}
              suffix="mm"
              onChange={(v) => set("pull_comp_mm", v)}
            />
            <div className="col-span-2 flex flex-col justify-center gap-2 md:col-span-3">
              <div className="flex flex-wrap gap-x-6 gap-y-2">
                <Toggle
                  label="Underlay"
                  checked={settings.underlay}
                  onChange={(v) => set("underlay", v)}
                />
                <Toggle
                  label="Trim between objects"
                  checked={settings.trim_enabled}
                  onChange={(v) => set("trim_enabled", v)}
                />
                <Toggle
                  label="Automatic color changes"
                  checked={settings.auto_color_change}
                  onChange={(v) => set("auto_color_change", v)}
                />
              </div>
            </div>
          </div>
          <button
            onClick={autoDigitize}
            disabled={!file || digitizing}
            className="mt-6 w-full rounded-xl bg-indigo-600 py-4 text-lg font-bold tracking-wide text-white shadow transition hover:bg-indigo-700 disabled:opacity-40"
          >
            {digitizing ? "Digitizing…" : "✨ AUTO DIGITIZE"}
          </button>
        </Section>

        <Section step={4} title="Stitch preview" dimmed={!result}>
          {result ? (
            <>
              <StitchPreview result={result} />
              {stats && (
                <div className="mt-4 grid grid-cols-3 gap-3 text-center md:grid-cols-6">
                  {[
                    ["Stitches", stats.stitches.toLocaleString()],
                    ["Colors", stats.colors],
                    ["Trims", stats.trims],
                    ["Jumps", stats.jumps],
                    ["Size", `${stats.width_mm}×${stats.height_mm}mm`],
                    ["Est. time", `${stats.est_minutes} min`],
                  ].map(([label, value]) => (
                    <div
                      key={label as string}
                      className="rounded-xl bg-gray-50 p-3"
                    >
                      <div className="text-lg font-bold text-gray-800">
                        {value}
                      </div>
                      <div className="text-xs text-gray-500">{label}</div>
                    </div>
                  ))}
                </div>
              )}
              {result.warnings.length > 0 && (
                <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                  {result.warnings.map((w, i) => (
                    <p key={i}>⚠ {w}</p>
                  ))}
                </div>
              )}
              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-gray-500">
                Thread order:
                {result.threads.map((t, i) => (
                  <span key={i} className="flex items-center gap-1">
                    <span
                      className="h-4 w-4 rounded-full border border-gray-300"
                      style={{ background: t }}
                    />
                    {i < result.threads.length - 1 && "→"}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <p className="py-10 text-center text-sm text-gray-400">
              Press AUTO DIGITIZE to see the stitch preview
            </p>
          )}
        </Section>

        <Section step={5} title="Export" dimmed={!result}>
          <button
            onClick={() => download("dst")}
            disabled={!result || exporting !== null}
            className="w-full rounded-xl bg-emerald-600 py-4 text-lg font-bold tracking-wide text-white shadow transition hover:bg-emerald-700 disabled:opacity-40"
          >
            {exporting === "dst" ? "Generating…" : "⬇ DOWNLOAD DST"}
          </button>
          <div className="mt-3 flex justify-center gap-2">
            {["pes", "exp", "jef", "svg"].map((f) => (
              <button
                key={f}
                onClick={() => download(f)}
                disabled={!result || exporting !== null}
                className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 disabled:opacity-40"
              >
                {exporting === f ? "…" : f.toUpperCase()}
              </button>
            ))}
          </div>
          <p className="mt-4 text-center text-xs text-gray-400">
            Files are processed in memory and never stored.
          </p>
        </Section>
      </div>
    </main>
  );
}
