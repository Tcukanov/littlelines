import { AnalyzeResult, AppSettings, DigitizeResult } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function form(file: File, settings: AppSettings): FormData {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("settings", JSON.stringify(settings));
  return fd;
}

async function post<T>(path: string, fd: FormData): Promise<T> {
  const res = await fetch(`${API}${path}`, { method: "POST", body: fd });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      if (j.detail) msg = String(j.detail);
    } catch {
      /* keep default */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export function analyze(file: File, settings: AppSettings) {
  return post<AnalyzeResult>("/api/analyze", form(file, settings));
}

export async function cleanup(file: File, settings: AppSettings): Promise<File> {
  const res = await post<{ png: string }>("/api/cleanup", form(file, settings));
  const bytes = Uint8Array.from(atob(res.png), (c) => c.charCodeAt(0));
  const base = file.name.replace(/\.[^.]+$/, "");
  return new File([bytes], `${base}-clean.png`, { type: "image/png" });
}

export function digitize(file: File, settings: AppSettings) {
  return post<DigitizeResult>("/api/digitize", form(file, settings));
}

export async function exportFile(
  file: File,
  settings: AppSettings,
  format: string,
): Promise<void> {
  const fd = form(file, settings);
  fd.append("format", format);
  const res = await fetch(`${API}/api/export`, { method: "POST", body: fd });
  if (!res.ok) {
    let msg = `Export failed (${res.status})`;
    try {
      const j = await res.json();
      if (j.detail) msg = String(j.detail);
    } catch {
      /* keep default */
    }
    throw new Error(msg);
  }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") ?? "";
  const m = /filename="([^"]+)"/.exec(cd);
  const name = m?.[1] ?? `design.${format}`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function health(): Promise<boolean> {
  try {
    const res = await fetch(`${API}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}
