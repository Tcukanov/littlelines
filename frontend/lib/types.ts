export type StitchType = "auto" | "fill" | "satin" | "running";

export interface ColorInfo {
  index: number;
  hex: string;
  pixels: number;
  regions: number;
  suggested: StitchType;
}

export interface AnalyzeResult {
  width_px: number;
  height_px: number;
  aspect: number;
  colors: ColorInfo[];
  preview_png: string; // base64
}

export interface ColorSetting {
  enabled: boolean;
  stitch: StitchType;
}

export interface AppSettings {
  width_mm: number;
  height_mm: number;
  max_colors: number;
  detail: number;
  min_object_mm2: number;
  remove_background: boolean;
  density_mm: number;
  stitch_len_mm: number;
  satin_width_mm: number;
  fill_angle_deg: number;
  underlay: boolean;
  pull_comp_mm: number;
  trim_enabled: boolean;
  auto_color_change: boolean;
  line_art: boolean;
  line_passes: number;
  color_settings: Record<number, ColorSetting>;
}

export const DEFAULT_SETTINGS: AppSettings = {
  width_mm: 100,
  height_mm: 100,
  max_colors: 4,
  detail: 50,
  min_object_mm2: 2,
  remove_background: true,
  density_mm: 0.4,
  stitch_len_mm: 3,
  satin_width_mm: 2.5,
  fill_angle_deg: 45,
  underlay: true,
  pull_comp_mm: 0.15,
  trim_enabled: true,
  auto_color_change: true,
  line_art: false,
  line_passes: 2,
  color_settings: {},
};

// Stitch event commands (must match backend plan.py)
export const CMD_STITCH = 0;
export const CMD_JUMP = 1;
export const CMD_COLOR_CHANGE = 2;
export const CMD_TRIM = 3;

export interface Stats {
  stitches: number;
  jumps: number;
  trims: number;
  color_changes: number;
  colors: number;
  width_mm: number;
  height_mm: number;
  est_minutes: number;
}

export interface DigitizeResult {
  threads: string[];
  stitches: [number, number, number][]; // [cmd, x_mm, y_mm]
  stats: Stats;
  warnings: string[];
}
