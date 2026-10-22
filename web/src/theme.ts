// The one source of truth for colour. CSS reads these as custom properties
// (applyTheme writes them onto :root before the first render), and the
// three.js materials and SVG charts import the hex values directly, so a
// palette change here reaches the page, the 3D viewer and the charts at once.
//
// Instrument: chrome is grey; colour means data or status. Dark only.
// Contrast is checked against these exact values (see web/DESIGN.md), so
// re-check it when changing one.

export const palette = {
  // Surfaces, darkest to lightest.
  bg: "#0E1013", // page and 3D viewport (also the three.js clear colour)
  surface: "#15181D", // topbar, inspector, timeline
  raised: "#1C2026", // controls, selected rows, popovers, modals
  border: "#2B3039",
  borderStrong: "#3B414C",

  // Text.
  text: "#ECEEF2",
  textMuted: "#A1A8B5",
  textDisabled: "#737B88", // disabled controls only, never essential information

  // The primary (filled) button.
  accent: "#ECEEF2",
  onAccent: "#0E1013",

  // Data: what each arm or trace is.
  real: "#E6E8EC", // the logged real arm
  stock: "#7C8494", // the sim on the stock model
  sim: "#4D8DFF", // the sim on calibrated physics (a lock or the live fit)
  comp: "#C86BC4", // the sim driven by lag-compensated commands

  // Status.
  good: "#43C38A", // an improvement
  warn: "#E8B341", // an anomaly, a partial run
  critical: "#FF6A55", // contact, the sim disagrees, an error
} as const;

export type PaletteKey = keyof typeof palette;

// camelCase key -> CSS custom property name (borderStrong -> --border-strong).
export const cssVar = (k: PaletteKey) => `--${k.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`;

export function applyTheme(root: HTMLElement = document.documentElement) {
  for (const k of Object.keys(palette) as PaletteKey[]) root.style.setProperty(cssVar(k), palette[k]);
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

export const rgba = (hex: string, alpha: number) => `rgba(${hexToRgb(hex).join(", ")}, ${alpha})`;

// Which physics a sim arm runs on decides its colour (stock grey or calibrated blue).
export type SimTone = "stock" | "sim";

// The user's OS setting: no ghost tween, no glow animation, no spinners spinning.
export const prefersReducedMotion = () =>
  typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
