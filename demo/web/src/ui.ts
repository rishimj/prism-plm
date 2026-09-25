import * as d3 from "d3";

export const $ = <T extends HTMLElement = HTMLElement>(sel: string, root: ParentNode = document) =>
  root.querySelector(sel) as T;

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  html = "",
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (html) node.innerHTML = html;
  return node;
}

export function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

export const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Tooltip ---------------------------------------------------------------------
const tip = () => document.getElementById("tooltip")!;
export function showTip(html: string, ev: { clientX: number; clientY: number }) {
  const t = tip();
  t.innerHTML = html;
  t.classList.add("show");
  const pad = 14;
  const { width, height } = t.getBoundingClientRect();
  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  if (x + width > window.innerWidth - 8) x = ev.clientX - width - pad;
  if (y + height > window.innerHeight - 8) y = ev.clientY - height - pad;
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${Math.max(8, y)}px`;
}
export function hideTip() {
  tip().classList.remove("show");
}
export const ttRow = (k: string, v: string) => `<div class="tt-row"><span>${k}</span><b>${v}</b></div>`;

// Color scales ------------------------------------------------------------------
/** Diverging blue-gray-red scale centred on `mid` (used for activations). */
export function divergingScale(values: number[], mid?: number) {
  const vals = values.filter((v) => Number.isFinite(v));
  const center = mid ?? d3.median(vals) ?? 0;
  const lo = d3.quantile(vals, 0.02) ?? d3.min(vals) ?? 0;
  const hi = d3.quantile(vals, 0.98) ?? d3.max(vals) ?? 1;
  const span = Math.max(center - lo, hi - center, 1e-6);
  const interp = d3.piecewise(d3.interpolateLab, [
    cssVar("--div-neg-2"),
    cssVar("--div-neg-1"),
    cssVar("--div-mid"),
    cssVar("--div-pos-1"),
    cssVar("--div-pos-2"),
  ]);
  return (v: number) => interp(Math.max(0, Math.min(1, 0.5 + (v - center) / (2 * span))));
}

/** Single-hue sequential scale from the surface toward a colour. */
export function sequentialScale(color: string, domain: [number, number] = [0, 1]) {
  const interp = d3.interpolateLab(cssVar("--div-mid"), color);
  return (v: number) => interp(Math.max(0, Math.min(1, (v - domain[0]) / (domain[1] - domain[0] || 1))));
}

export const FEATURE_META: Record<string, { label: string; color: string }> = {
  helix: { label: "Alpha helix", color: "--s2" },
  strand: { label: "Beta strand", color: "--s1" },
  transmembrane: { label: "Transmembrane", color: "--s3" },
  binding: { label: "Binding site", color: "--ink" },
  active: { label: "Active site", color: "--ink" },
  motif: { label: "Motif", color: "--ink" },
  zinc_finger: { label: "Zinc finger", color: "--ink" },
  disulfide: { label: "Disulfide", color: "--ink" },
  signal: { label: "Signal peptide", color: "--ink" },
  dna_binding: { label: "DNA binding", color: "--ink" },
  propeptide: { label: "Propeptide", color: "--ink" },
  domain: { label: "Domain", color: "--ink" },
  region: { label: "Region", color: "--ink" },
  turn: { label: "Turn", color: "--ink" },
  site: { label: "Site", color: "--ink" },
};

export const fmt = {
  int: d3.format(","),
  pct: d3.format(".0%"),
  pct1: d3.format(".1%"),
  f2: d3.format(".2f"),
  f3: d3.format(".3f"),
  compact: (n: number) => d3.format(".3~s")(n).replace("G", "B"),
  q: (q: number) => (q < 1e-3 ? d3.format(".1e")(q) : d3.format(".3f")(q)),
};

export const neuronId = (l: number, u: number) => `L${l}·N${u}`;

/** Tiny sparkline of a per-residue track. */
export function sparkline(values: number[], w = 92, h = 24): string {
  const x = d3.scaleLinear().domain([0, values.length - 1]).range([1, w - 1]);
  const ext = d3.extent(values) as [number, number];
  const y = d3.scaleLinear().domain(ext[0] === ext[1] ? [ext[0] - 1, ext[1] + 1] : ext).range([h - 2, 2]);
  const line = d3.line<number>().x((_, i) => x(i)).y((v) => y(v)).curve(d3.curveMonotoneX);
  return `<svg viewBox="0 0 ${w} ${h}" aria-hidden="true"><path d="${line(values)}" fill="none" stroke="var(--s1)" stroke-width="1.5"/></svg>`;
}

/** Reveal-on-scroll for elements with .reveal */
export function initReveal() {
  const io = new IntersectionObserver(
    (entries) =>
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      }),
    { threshold: 0.12 },
  );
  document.querySelectorAll(".reveal").forEach((n) => io.observe(n));
}

/** Re-render callbacks when the theme changes (charts read CSS variables). */
const themeListeners: (() => void)[] = [];
export const onTheme = (fn: () => void) => themeListeners.push(fn);
export function initTheme() {
  const btn = document.getElementById("theme-toggle")!;
  let saved: string | null = null;
  try {
    saved = localStorage.getItem("prism-theme");
  } catch {
    /* storage unavailable */
  }
  if (saved) document.documentElement.dataset.theme = saved;
  btn.addEventListener("click", () => {
    const dark =
      document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("prism-theme", next);
    } catch {
      /* ignore */
    }
    themeListeners.forEach((fn) => fn());
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => themeListeners.forEach((fn) => fn()));
}
