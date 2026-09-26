// Static data produced by demo/pipeline (real ESM-2 runs), served next to the site.

export type Track = {
  l: number;
  u: number;
  kind: "feature" | "probe" | "top" | "requested";
  note: string;
  auc?: number | null;
  feature?: string | null;
  v: number[];
  z: number;
  label?: string;
};

export type Feature = { type: string; start: number; end: number; label: string };

export type FingerprintItem = { l: number; u: number; z: number; label: string; q: number };

export type Showcase = {
  acc: string;
  short: string;
  title: string;
  name: string;
  organism: string;
  blurb: string;
  focus: string;
  sequence: string;
  features: Feature[];
  structure: boolean;
  p_native: (number | null)[] | null;
  entropy: (number | null)[] | null;
  probes: Record<string, number[]>;
  tracks: Track[];
  fingerprint: FingerprintItem[];
};

export type ShowcaseIndexItem = { acc: string; short: string; title: string; organism: string; length: number; blurb: string };

// Atlas: per-layer files keep keys short to stay small
export type AtlasCluster = [size: number, meanAct: number, term: number, q: number, examples: number[]];
export type AtlasNeuron = {
  u: number;
  p: number;
  m: number;
  sd: number;
  h: number[];
  hr: [number, number];
  t: [term: number, k: number, q: number, fold: number][];
  c: AtlasCluster[];
  tp: number[];
  ta: number[];
};
export type AtlasLayer = { layer: number; pca_var: number; neurons: AtlasNeuron[] };
export type AtlasIndex = {
  model: string;
  n_layers: number;
  hidden: number;
  n_proteins: number;
  top_k: number;
  n_clusters: number;
  pca_dims: number;
  layers: { layer: number; neurons: [poly: number, term: number, neglogq: number][]; median_poly: number; sig_frac: number }[];
};
export type Term = [id: string, name: string, ns: "BP" | "MF" | "CC" | "FAM"] | null;
export type AtlasProtein = [id: string, name: string, organism: string, family: string, location: string, length: number];

export type ProbeCell = { auc: number; acc: number; base_rate: number; neuron: { u: number; sign: number; auc: number } };
export type Probing = {
  model: string;
  n_struct_proteins: number;
  n_tm_proteins: number;
  n_residues: number;
  layers: ({ layer: number } & Record<"helix" | "strand" | "transmembrane", ProbeCell>)[];
  best_layer: Record<string, number>;
};

export type Scaling = {
  n_proteins: number;
  top_k: number;
  units_per_layer: number;
  models: {
    key: string;
    name: string;
    params: number;
    n_layers: number;
    hidden: number;
    median: number;
    mean: number;
    sig_frac: number;
    layers: { layer: number; depth: number; median: number; q25: number; q75: number; sig_frac: number; poly: number[] }[];
  }[];
};

export type SteerRun = {
  r: number;
  mult: number;
  seq: string;
  identity: number;
  p_native: number;
  concept_score: number;
  per_pos: number[];
  drift: number[];
  motifs?: [number, number][];
  concept_prob?: number;
  concept_shift?: number;
};
export type Steering = {
  model: string;
  layer: number;
  relative_strengths: number[];
  targets: { acc: string; short: string; title: string; sequence: string }[];
  concepts: {
    key: string;
    title: string;
    kind: "motif" | "set";
    blurb: string;
    pattern: string | null;
    n_pos: number;
    n_neg: number;
    vector_norm: number;
    top_residues: string[];
    logodds: number[];
    runs: { acc: string; scale: number; runs: SteerRun[] }[];
  }[];
};

export type Summary = {
  proteins: number;
  neurons_profiled: number;
  models: number;
  go_terms: number;
  residues_probed: number;
  organisms: number;
};

const cache = new Map<string, Promise<unknown>>();

export function dataUrl(path: string): string {
  return `${import.meta.env.BASE_URL}data/${path}`;
}

export function load<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(dataUrl(path)).then((r) => {
        if (!r.ok) throw new Error(`Failed to load ${path}: ${r.status}`);
        return path.endsWith(".pdb") ? r.text() : r.json();
      }),
    );
  }
  return cache.get(path) as Promise<T>;
}

export const loadAtlasLayer = (l: number) => load<AtlasLayer>(`atlas/layer-${String(l).padStart(2, "0")}.json`);
