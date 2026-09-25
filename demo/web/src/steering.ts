import { lineChart } from "./charts";
import { isLive, onHealth, steer, type SteerResult } from "./api";
import { load, type Steering, type SteerRun } from "./data";
import { $, cssVar, divergingScale, el, esc, fmt, onTheme } from "./ui";

let data: Steering;
let conceptKey = "zinc_finger";
let targetAcc = "";
let idx = 0;
let playing: number | null = null;

export async function initSteering() {
  data = await load<Steering>("steering.json");
  conceptKey = data.concepts[0].key;
  targetAcc = data.targets[0].acc;
  idx = Math.max(0, data.relative_strengths.indexOf(2));

  const grid = $("#concept-grid");
  for (const c of data.concepts) {
    const b = el(
      "button",
      { class: "concept", "data-k": c.key },
      `<div class="t">${esc(c.title)}</div><div class="d">${esc(c.blurb)}</div><div class="d" style="margin-top:4px">${
        c.pattern ? `motif <span class="mono">${esc(c.pattern)}</span> · ` : ""
      }${c.n_pos} vs ${c.n_neg} proteins</div>`,
    );
    b.addEventListener("click", () => {
      conceptKey = c.key;
      render();
      renderLive();
    });
    grid.append(b);
  }
  const seg = $("#target-select");
  for (const t of data.targets) {
    const b = el("button", { "data-acc": t.acc }, esc(t.short));
    b.title = t.title;
    b.addEventListener("click", () => {
      targetAcc = t.acc;
      render();
    });
    seg.append(b);
  }
  const slider = $("#steer-slider") as HTMLInputElement;
  slider.max = String(data.relative_strengths.length - 1);
  slider.value = String(idx);
  slider.addEventListener("input", () => {
    idx = +slider.value;
    render();
  });
  $("#steer-play").addEventListener("click", togglePlay);
  onTheme(() => render());
  onHealth(() => renderLive());
  let t: number | undefined;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = window.setTimeout(() => render(), 150);
  });
  render();
}

function togglePlay() {
  const btn = $("#steer-play");
  if (playing !== null) {
    clearInterval(playing);
    playing = null;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
    return;
  }
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>`;
  playing = window.setInterval(() => {
    idx = (idx + 1) % data.relative_strengths.length;
    ($("#steer-slider") as HTMLInputElement).value = String(idx);
    render();
  }, 900);
}

const concept = () => data.concepts.find((c) => c.key === conceptKey)!;
const target = () => data.targets.find((t) => t.acc === targetAcc)!;
const runsFor = () => concept().runs.find((r) => r.acc === targetAcc)!;

function strengthLabel(r: number) {
  return `${r > 0 ? "+" : ""}${r.toFixed(2)}×`;
}

function render() {
  document.querySelectorAll("#concept-grid .concept").forEach((b) => b.classList.toggle("active", (b as HTMLElement).dataset.k === conceptKey));
  document.querySelectorAll("#target-select button").forEach((b) => b.classList.toggle("active", (b as HTMLElement).dataset.acc === targetAcc));
  const c = concept();
  const tr = runsFor();
  const run = tr.runs[idx];
  const base = tr.runs.find((r) => r.r === 0)!;
  $("#strength-read").textContent = strengthLabel(run.r);
  $("#steer-scale-note").innerHTML = `Strength is the size of the added vector relative to the average activation norm at layer ${data.layer} (α = ${fmt.f2(
    run.mult,
  )} × concept vector). Negative values push away from the concept.`;
  renderTiles($("#steer-tiles"), run, base, c.kind === "motif");
  renderSeq($("#steer-seq"), target().sequence, run, c.pattern !== null);
  renderCharts(tr.runs, run);
}

function renderTiles(host: HTMLElement, run: SteerRun | SteerResult, base: SteerRun | null, motif: boolean) {
  const dScore = base ? run.concept_score - base.concept_score : null;
  const tiles = [
    { l: "Identity kept", v: fmt.pct(run.identity), d: "positions still predicted as the native residue" },
    {
      l: "Concept signature",
      v: `${run.concept_score >= 0 ? "+" : ""}${fmt.f3(run.concept_score)}`,
      d: dScore !== null ? `${dScore >= 0 ? "+" : ""}${fmt.f3(dScore)} vs unsteered` : "mean log-odds per residue",
    },
    motif
      ? { l: "Motif matches", v: String(run.motifs?.length ?? 0), d: "in the model's predicted sequence" }
      : { l: "P(native residue)", v: fmt.pct(run.p_native), d: "average model confidence in the real protein" },
    "concept_shift" in run && run.concept_shift !== undefined
      ? { l: "Concept probability shift", v: `${run.concept_shift >= 0 ? "+" : ""}${fmt.f3(run.concept_shift * 100)} pp`, d: "repo metric: P(motif | steered) − P(motif | baseline)" }
      : { l: "Steering multiplier α", v: fmt.f2(run.mult), d: "raw scale applied to the concept vector" },
  ];
  host.innerHTML = tiles.map((t) => `<div class="card stat-tile"><div class="l">${t.l}</div><div class="v">${t.v}</div><div class="d">${t.d}</div></div>`).join("");
}

function renderSeq(host: HTMLElement, native: string, run: { seq: string; per_pos: number[]; motifs?: [number, number][] }, showMotif: boolean) {
  const heat = divergingScale(run.per_pos, 0);
  const inMotif = new Set<number>();
  for (const [s, e] of run.motifs ?? []) for (let i = s; i < e; i++) inMotif.add(i);
  const nat = native.split("").map((a) => `<span class="res">${a}</span>`).join("");
  const st = run.seq
    .split("")
    .map((a, i) => `<span class="res${a !== native[i] ? " changed" : ""}${showMotif && inMotif.has(i) ? " motif" : ""}" title="${native[i]}${i + 1}${a}">${a}</span>`)
    .join("");
  const hm = run.per_pos.map((v) => `<span class="heat" style="background:${heat(v)}"></span>`).join("");
  const row = (label: string, body: string) => `<div class="row"><span class="row-label">${label}</span><span class="track">${body}</span></div>`;
  host.innerHTML = row("Native", nat) + row("Steered", st) + row("Concept pull", hm);
}

function renderCharts(runs: SteerRun[], run: SteerRun) {
  const color = cssVar("--s1");
  lineChart($("#steer-concept-chart"), {
    height: 210,
    xLabel: "Strength",
    yLabel: "Log-odds",
    xFormat: (v) => `${v}×`,
    yFormat: fmt.f3,
    marker: run.r,
    refX: { x: 0, label: "unsteered" },
    series: [{ name: "Concept signature", color, points: runs.map((r) => ({ x: r.r, y: r.concept_score })) }],
  });
  lineChart($("#steer-identity-chart"), {
    height: 210,
    xLabel: "Strength",
    yLabel: "Identity",
    xFormat: (v) => `${v}×`,
    yFormat: fmt.pct,
    yDomain: [0, 1],
    marker: run.r,
    refX: { x: 0, label: "unsteered" },
    series: [{ name: "Identity kept", color, points: runs.map((r) => ({ x: r.r, y: r.identity })) }],
  });
  lineChart($("#steer-drift-chart"), {
    height: 210,
    xLabel: "Layer",
    yLabel: "Cosine similarity",
    xFormat: (v) => (v === 0 ? "emb" : String(v)),
    yFormat: fmt.f3,
    refX: { x: data.layer + 1, label: "injected" },
    series: [{ name: `Strength ${strengthLabel(run.r)}`, color, points: run.drift.map((y, x) => ({ x, y })) }],
  });
}

// Live steering -------------------------------------------------------------------------
function renderLive() {
  const host = $("#steer-live");
  const c = concept();
  const live = isLive();
  host.innerHTML = `
    <h3>Steer your own sequence ${live ? `<span class="tag accent" style="margin-left:6px">Live</span>` : ""}</h3>
    <div class="sub">${
      live
        ? `Runs on the live ESM-2 35M API with the <b>${esc(c.title)}</b> concept at any strength.`
        : "Available when the live model is online. The precomputed runs above cover four proteins and twelve strengths."
    }</div>
    <div style="display:grid;gap:10px;margin-top:12px">
      <textarea class="seq-input" id="steer-custom" style="min-height:70px" placeholder="Paste a protein sequence (up to 400 residues)" ${live ? "" : "disabled"}></textarea>
      <div class="slider-row" style="margin-top:0">
        <input type="range" id="steer-live-strength" min="-2" max="3" step="0.25" value="1.5" ${live ? "" : "disabled"} aria-label="Live steering strength"/>
        <span class="strength-read" id="steer-live-read">+1.50×</span>
        <button class="btn primary" id="steer-live-run" ${live ? "" : "disabled"}>Steer</button>
      </div>
      <span id="steer-live-status" class="muted"></span>
      <div class="stat-tiles" id="steer-live-tiles" style="margin:0"></div>
      <div class="seq-compare" id="steer-live-seq"></div>
    </div>`;
  if (!live) return;
  const slider = $("#steer-live-strength") as HTMLInputElement;
  slider.addEventListener("input", () => ($("#steer-live-read").textContent = strengthLabel(+slider.value)));
  $("#steer-live-run").addEventListener("click", async () => {
    const seq = ($("#steer-custom") as HTMLTextAreaElement).value;
    const status = $("#steer-live-status");
    status.className = "muted";
    status.innerHTML = `<span class="spinner"></span> Steering…`;
    try {
      const r = await steer(seq, conceptKey, +slider.value);
      status.textContent = "";
      const clean = seq.split("\n").filter((l) => !l.startsWith(">")).join("").replace(/[\s\d*]/g, "").toUpperCase();
      renderTiles($("#steer-live-tiles"), r, null, !!r.motifs);
      renderSeq($("#steer-live-seq"), clean, r, !!r.motifs);
    } catch (e) {
      status.className = "error";
      status.textContent = (e as Error).message;
    }
  });
}
