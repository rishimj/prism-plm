// Client for the optional live backend (FastAPI on Azure). The site is fully
// usable without it; live features switch on when /api/health answers.

import type { FingerprintItem, Track } from "./data";

export type Health = {
  status: string;
  model: string;
  layers: number;
  hidden: number;
  max_length: number;
  concepts: { key: string; title: string; layer: number }[];
};

export type AnalyzeResult = {
  sequence: string;
  length: number;
  layer: number;
  probes: Record<string, number[]>;
  tracks: Track[];
  fingerprint: FingerprintItem[];
  p_native: number[] | null;
  entropy: number[] | null;
};

export type SteerResult = {
  concept: string;
  layer: number;
  r: number;
  mult: number;
  seq: string;
  identity: number;
  p_native: number;
  concept_score: number;
  per_pos: number[];
  drift: number[];
  motifs?: [number, number][];
};

function resolveBase(): string {
  const qs = new URLSearchParams(location.search).get("api");
  if (qs) return qs.replace(/\/$/, "");
  return (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
}

export const API_BASE = resolveBase();

let health: Health | null = null;
const listeners: ((h: Health | null) => void)[] = [];

export function onHealth(fn: (h: Health | null) => void) {
  listeners.push(fn);
  fn(health);
}

export function isLive() {
  return health !== null;
}

export async function checkHealth(): Promise<Health | null> {
  if (!API_BASE) {
    health = null;
  } else {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 6000);
      const r = await fetch(`${API_BASE}/api/health`, { signal: ctrl.signal });
      clearTimeout(t);
      health = r.ok ? await r.json() : null;
    } catch {
      health = null;
    }
  }
  listeners.forEach((fn) => fn(health));
  return health;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = `Request failed (${r.status})`;
    try {
      const j = await r.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* keep default */
    }
    throw new Error(msg);
  }
  return r.json();
}

export const analyze = (sequence: string, layer = 6) => post<AnalyzeResult>("/api/analyze", { sequence, layer });
export const steer = (sequence: string, concept: string, strength: number) =>
  post<SteerResult>("/api/steer", { sequence, concept, strength });
