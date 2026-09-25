import * as d3 from "d3";
import { analyze, isLive, onHealth } from "./api";
import { load, type Showcase, type ShowcaseIndexItem, type Track } from "./data";
import {
  $,
  cssVar,
  divergingScale,
  el,
  esc,
  FEATURE_META,
  fmt,
  hideTip,
  neuronId,
  onTheme,
  sequentialScale,
  showTip,
  sparkline,
  ttRow,
} from "./ui";

type ColorMode = "neuron" | "plddt" | "surprise";

const CELL = 14;
const EXAMPLES: Record<string, string> = {
  "Ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
  "Melittin (bee venom)": "MKFLVNVALVFMVVYISYIYAAPEPEPAPEPEAEADAEADPEAGIGAVLKVLTTGLPALISWIKRKRQQG",
  "Insulin B-chain": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
};

let current: Showcase | null = null;
let activeTrack: Track | null = null;
let colorMode: ColorMode = "neuron";
let pdbText: string | null = null;
let plddt: number[] = [];
let viewer: any = null;
let $3Dmol: any = null;
let highlighted: number | null = null;

export async function initExplorer() {
  const index = await load<ShowcaseIndexItem[]>("showcase/index.json");
  const chips = $("#protein-chips");
  for (const p of index) {
    const b = el("button", { class: "chip", role: "tab", "data-acc": p.acc }, `<div class="t">${esc(p.short)}</div><div class="s">${esc(p.title)} · ${p.length} aa</div>`);
    b.addEventListener("click", () => selectProtein(p.acc));
    chips.append(b);
  }
  const live = el("button", { class: "chip live-chip", "data-acc": "custom" }, `<div class="t">+ Your sequence</div><div class="s">Live ESM-2 analysis</div>`);
  live.addEventListener("click", () => showCustom());
  chips.append(live);
  onHealth((h) => {
    live.querySelector(".s")!.textContent = h ? "Live ESM-2 analysis" : "Needs the live model";
  });

  $("#color-mode").addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest("button");
    if (!btn || btn.hasAttribute("disabled")) return;
    colorMode = btn.dataset.mode as ColorMode;
    document.querySelectorAll("#color-mode button").forEach((b) => b.classList.toggle("active", b === btn));
    paintStructure();
    renderLegend();
  });
  onTheme(() => {
    if (viewer) viewer.setBackgroundColor(cssVar("--card"));
    paintStructure();
    renderSequence();
    renderLegend();
  });
  window.addEventListener("resize", () => viewer?.resize());
  await selectProtein(index[0].acc);
}

function setActiveChip(acc: string) {
  document.querySelectorAll("#protein-chips .chip").forEach((c) => c.classList.toggle("active", (c as HTMLElement).dataset.acc === acc));
}

async function selectProtein(acc: string) {
  setActiveChip(acc);
  const p = await load<Showcase>(`showcase/${acc}.json`);
  current = p;
  activeTrack = pickDefaultTrack(p);
  pdbText = p.structure ? await load<string>(`structures/${acc}.pdb`) : null;
  plddt = pdbText ? parsePlddt(pdbText) : [];
  renderPanel();
  await renderStructure();
  renderSequence();
  renderLegend();
  renderFingerprint();
}

function trackNote(t: Track): string {
  if (t.kind === "feature" && t.feature) {
    const name = (FEATURE_META[t.feature]?.label ?? t.feature).toLowerCase();
    return `Fires on ${name} residues · AUROC ${fmt.f2(t.auc ?? 0)}`;
  }
  return t.note;
}

function pickDefaultTrack(p: Showcase): Track {
  const byFocus = p.tracks.filter((t) => t.kind === "feature" && t.feature === p.focus);
  if (byFocus.length) return byFocus[0];
  return p.tracks.find((t) => t.kind === "feature") ?? p.tracks[0];
}

function parsePlddt(pdb: string): number[] {
  const out: number[] = [];
  for (const line of pdb.split("\n")) {
    if (line.startsWith("ATOM") && line.slice(12, 16).trim() === "CA") out.push(+line.slice(60, 66));
  }
  return out;
}

// Panel ---------------------------------------------------------------------------
function renderPanel() {
  const p = current!;
  const panel = $("#protein-panel");
  const groups: [string, Track[]][] = [
    [
      "Feature detectors (best match within this protein)",
      p.tracks
        // tiny terminal propeptides are trivially separable and not informative
        .filter((t) => t.kind === "feature" && t.feature !== "propeptide")
        .sort((a, b) => Number(b.feature === p.focus) - Number(a.feature === p.focus) || (b.auc ?? 0) - (a.auc ?? 0)),
    ],
    [
      "Structure neurons (selected on held-out proteins)",
      p.tracks
        .filter((t) => t.kind === "probe")
        .sort((a, b) => (b.auc ?? 0) - (a.auc ?? 0))
        .slice(0, 6),
    ],
    ["Strongest firing on this protein", p.tracks.filter((t) => t.kind === "top" || t.kind === "requested").sort((a, b) => b.z - a.z).slice(0, 8)],
  ];
  const isCustom = p.acc === "custom";
  panel.innerHTML = `
    <div class="protein-head">
      <div class="title">${esc(p.title)}</div>
      <div class="meta">
        ${p.organism ? `<span class="tag"><i>${esc(p.organism)}</i></span>` : ""}
        <span class="tag">${p.sequence.length} amino acids</span>
        ${isCustom ? `<span class="tag accent">Live model</span>` : `<a class="tag accent" href="https://www.uniprot.org/uniprotkb/${p.acc}" target="_blank" rel="noopener">UniProt ${p.acc} ↗</a>`}
      </div>
      <p class="blurb">${esc(p.blurb)}</p>
    </div>
    <div class="neuron-list" id="neuron-list"></div>`;
  const list = $("#neuron-list", panel);
  for (const [title, tracks] of groups) {
    if (!tracks.length) continue;
    list.append(el("div", { class: "neuron-group-title" }, esc(title)));
    for (const t of tracks) {
      const row = el(
        "button",
        { class: "neuron-row" + (t === activeTrack ? " active" : "") },
        `<span class="id">${neuronId(t.l, t.u)}</span><span class="note">${esc(trackNote(t))}${t.label ? `<br><span class="muted">Atlas: ${esc(t.label)}</span>` : ""}</span>${sparkline(t.v)}`,
      );
      row.addEventListener("click", () => {
        activeTrack = t;
        list.querySelectorAll(".neuron-row").forEach((r) => r.classList.remove("active"));
        row.classList.add("active");
        if (colorMode !== "neuron") {
          colorMode = "neuron";
          document.querySelectorAll("#color-mode button").forEach((b) => b.classList.toggle("active", (b as HTMLElement).dataset.mode === "neuron"));
        }
        paintStructure();
        renderSequence();
        renderLegend();
      });
      list.append(row);
    }
  }
}

// Structure -------------------------------------------------------------------------
async function renderStructure() {
  const host = $("#viewer");
  const hasStructure = !!pdbText;
  document.querySelectorAll<HTMLButtonElement>("#color-mode button").forEach((b) => {
    if (b.dataset.mode === "plddt") b.disabled = !hasStructure;
  });
  if (!hasStructure) {
    if (viewer) {
      viewer.clear();
      viewer.render();
    }
    host.querySelector(".viewer-empty")?.remove();
    host.append(
      el(
        "div",
        { class: "viewer-empty" },
        `<div><b>No 3D structure for custom sequences</b><br/>Residue-level results are shown below. Structures are included for the showcase proteins (from AlphaFold DB).</div>`,
      ),
    );
    return;
  }
  host.querySelector(".viewer-empty")?.remove();
  if (!$3Dmol) {
    host.append(el("div", { class: "viewer-empty", id: "viewer-loading" }, `<span class="spinner"></span>`));
    $3Dmol = await import("3dmol");
    host.querySelector("#viewer-loading")?.remove();
  }
  if (!viewer) {
    viewer = $3Dmol.createViewer(host, { backgroundColor: cssVar("--card"), antialias: true });
  }
  viewer.clear();
  viewer.addModel(pdbText, "pdb");
  viewer.setHoverable(
    {},
    true,
    (atom: any, _v: any, ev: MouseEvent) => {
      if (!atom) return;
      const i = atom.resi - 1;
      showTip(residueTip(i), ev ?? { clientX: 0, clientY: 0 });
    },
    () => hideTip(),
  );
  paintStructure();
  // frame the confidently predicted core rather than floppy termini
  viewer.zoomTo({ predicate: (a: any) => a.b > 70 });
  viewer.zoom(1.15);
  viewer.render();
}

function residueColors(): (i: number) => string {
  const p = current!;
  if (colorMode === "plddt" && plddt.length) {
    return (i) => {
      const v = plddt[i] ?? 0;
      return v > 90 ? "#0053d6" : v > 70 ? "#65cbf3" : v > 50 ? "#ffdb13" : "#ff7d45";
    };
  }
  if (colorMode === "surprise" && p.p_native) {
    const s = sequentialScale(cssVar("--div-pos-2"), [0, 1]);
    return (i) => s(1 - (p.p_native![i] ?? 1));
  }
  const vals = activeTrack!.v;
  const s = divergingScale(vals);
  return (i) => s(vals[i] ?? 0);
}

function paintStructure() {
  if (!viewer || !pdbText) return;
  const c = residueColors();
  viewer.setStyle({}, { cartoon: { colorfunc: (atom: any) => c(atom.resi - 1), thickness: 0.4 } });
  if (highlighted !== null) {
    viewer.addStyle({ resi: highlighted + 1 }, { stick: { radius: 0.25, colorscheme: "default" } });
    viewer.addStyle({ resi: highlighted + 1, atom: "CA" }, { sphere: { radius: 1.1, color: cssVar("--ink") } });
  }
  viewer.render();
}

function highlight(i: number | null) {
  if (highlighted === i) return;
  highlighted = i;
  paintStructure();
}

function renderLegend() {
  const host = $("#viewer-legend");
  const p = current!;
  if (colorMode === "plddt") {
    host.innerHTML = [
      ["#0053d6", "Very high (>90)"],
      ["#65cbf3", "Confident"],
      ["#ffdb13", "Low"],
      ["#ff7d45", "Very low (<50)"],
    ]
      .map(([c, l]) => `<span style="display:inline-flex;align-items:center;gap:5px"><i class="swatch" style="background:${c}"></i>${l}</span>`)
      .join(" ");
    return;
  }
  if (colorMode === "surprise") {
    host.innerHTML = `<span>expected</span><span class="bar" style="background:linear-gradient(90deg,${cssVar("--div-mid")},${cssVar("--div-pos-2")})"></span><span>surprising residue</span>`;
    return;
  }
  const t = activeTrack!;
  host.innerHTML = `<b class="mono" style="font-size:12px">${neuronId(t.l, t.u)}</b><span>low</span><span class="bar" style="background:linear-gradient(90deg,${cssVar("--div-neg-2")},${cssVar(
    "--div-mid",
  )},${cssVar("--div-pos-2")})"></span><span>high activation</span>`;
  void p;
}

// Sequence lanes -----------------------------------------------------------------------
type Lane = { key: string; label: string; h: number; draw: (g: d3.Selection<SVGGElement, unknown, null, undefined>) => void };

function residueTip(i: number): string {
  const p = current!;
  const aa = p.sequence[i];
  const feats = p.features.filter((f) => f.start - 1 <= i && i <= f.end - 1 && FEATURE_META[f.type]);
  let html = `<div class="tt-title">${aa}${i + 1}</div>`;
  if (activeTrack) html += ttRow(`${neuronId(activeTrack.l, activeTrack.u)} activation`, fmt.f2(activeTrack.v[i]));
  for (const k of ["helix", "strand", "transmembrane"]) {
    if (p.probes[k]) html += ttRow(`Predicted ${FEATURE_META[k].label.toLowerCase()}`, fmt.pct(p.probes[k][i]));
  }
  if (p.p_native) html += ttRow("P(native residue)", fmt.pct(p.p_native[i] ?? 0));
  if (plddt.length) html += ttRow("AlphaFold pLDDT", fmt.int(plddt[i] ?? 0));
  if (feats.length) {
    html += `<div style="margin-top:4px;color:var(--ink-2)">${feats
      .slice(0, 4)
      .map((f) => esc(FEATURE_META[f.type].label + (f.label ? `: ${f.label}` : "")))
      .join("<br>")}</div>`;
  }
  return html;
}

function renderSequence() {
  const p = current!;
  const host = $("#seq-view");
  host.innerHTML = "";
  const L = p.sequence.length;
  const width = L * CELL;
  const labelW = 150;
  const t = activeTrack!;
  const act = divergingScale(t.v);
  const lanes: Lane[] = [];

  lanes.push({
    key: "ruler",
    label: "",
    h: 14,
    draw: (g) => {
      for (let i = 9; i < L; i += 10) {
        g.append("text").attr("x", i * CELL + CELL / 2).attr("y", 10).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", "var(--muted)").text(i + 1);
      }
    },
  });
  const [vmin, vmax] = d3.extent(t.v) as [number, number];
  lanes.push({
    key: "bars",
    label: `${neuronId(t.l, t.u)} activity`,
    h: 40,
    draw: (g) => {
      const y = d3.scaleLinear().domain([vmin, vmax]).range([38, 2]);
      t.v.forEach((v, i) => {
        const top = y(v);
        g.append("rect").attr("x", i * CELL + 2).attr("width", CELL - 4).attr("y", top).attr("height", Math.max(1, 38 - top)).attr("rx", 2).attr("fill", act(v));
      });
    },
  });
  lanes.push({
    key: "seq",
    label: "Sequence",
    h: 20,
    draw: (g) => {
      p.sequence.split("").forEach((aa, i) => {
        g.append("rect").attr("x", i * CELL).attr("y", 1).attr("width", CELL - 1).attr("height", 18).attr("rx", 3).attr("fill", act(t.v[i]));
        g.append("text").attr("x", i * CELL + (CELL - 1) / 2).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600).attr("fill", "var(--ink)").text(aa);
      });
    },
  });
  const featTypes = Array.from(new Set(p.features.map((f) => f.type))).filter((k) => FEATURE_META[k]);
  const order = ["helix", "strand", "transmembrane", "signal", "binding", "active", "motif", "zinc_finger", "dna_binding", "disulfide", "propeptide", "domain", "region"];
  featTypes.sort((a, b) => order.indexOf(a) - order.indexOf(b));
  for (const type of featTypes.filter((k) => order.includes(k))) {
    const meta = FEATURE_META[type];
    lanes.push({
      key: `f-${type}`,
      label: meta.label,
      h: 14,
      draw: (g) => {
        g.append("line").attr("x1", 0).attr("x2", width).attr("y1", 7).attr("y2", 7).attr("stroke", "var(--grid)");
        for (const f of p.features.filter((f) => f.type === type)) {
          g.append("rect")
            .attr("x", (f.start - 1) * CELL + 1)
            .attr("width", Math.max(CELL - 2, (f.end - f.start + 1) * CELL - 2))
            .attr("y", 2)
            .attr("height", 10)
            .attr("rx", 3)
            .attr("fill", `var(${meta.color})`)
            .attr("opacity", meta.color === "--ink" ? 0.75 : 1);
        }
      },
    });
  }
  for (const k of ["helix", "strand", "transmembrane"]) {
    if (!p.probes[k]) continue;
    const s = sequentialScale(cssVar(FEATURE_META[k].color), [0, 1]);
    lanes.push({
      key: `p-${k}`,
      label: `Model: ${FEATURE_META[k].label.toLowerCase()}`,
      h: 12,
      draw: (g) => p.probes[k].forEach((v, i) => g.append("rect").attr("x", i * CELL).attr("width", CELL).attr("y", 1).attr("height", 10).attr("fill", s(v))),
    });
  }
  if (p.p_native) {
    const s = sequentialScale(cssVar("--div-pos-2"), [0, 1]);
    lanes.push({
      key: "surprise",
      label: "Model surprise",
      h: 12,
      draw: (g) => p.p_native!.forEach((v, i) => g.append("rect").attr("x", i * CELL).attr("width", CELL).attr("y", 1).attr("height", 10).attr("fill", s(1 - (v ?? 1)))),
    });
  }

  const gap = 6;
  const totalH = lanes.reduce((s, l) => s + l.h + gap, 0);
  const wrap = el("div", { style: `display:flex;min-width:${labelW + width}px` });
  const labels = d3
    .select(wrap)
    .append("svg")
    .attr("width", labelW)
    .attr("height", totalH)
    .style("position", "sticky")
    .style("left", "0")
    .style("background", "var(--card)")
    .style("z-index", "2")
    .style("flex", "none");
  const svg = d3.select(wrap).append("svg").attr("class", "seq-svg").attr("width", width).attr("height", totalH);
  let y = 0;
  for (const lane of lanes) {
    labels.append("text").attr("class", "lane-label").attr("x", 0).attr("y", y + lane.h / 2 + 4).text(lane.label);
    const g = svg.append("g").attr("transform", `translate(0,${y})`);
    lane.draw(g);
    y += lane.h + gap;
  }
  const hl = svg.append("rect").attr("y", 0).attr("height", totalH).attr("width", CELL).attr("fill", "var(--ink)").attr("opacity", 0).attr("pointer-events", "none");
  svg
    .append("rect")
    .attr("width", width)
    .attr("height", totalH)
    .attr("fill", "transparent")
    .on("mousemove", (ev: MouseEvent) => {
      const i = Math.max(0, Math.min(L - 1, Math.floor(d3.pointer(ev)[0] / CELL)));
      hl.attr("x", i * CELL).attr("opacity", 0.08);
      showTip(residueTip(i), ev);
      highlight(i);
    })
    .on("mouseleave", () => {
      hl.attr("opacity", 0);
      hideTip();
      highlight(null);
    });
  host.append(wrap);

  const legend = $("#seq-legend");
  legend.innerHTML = ["helix", "strand", "transmembrane"]
    .map((k) => `<span><i class="swatch" style="background:var(${FEATURE_META[k].color})"></i>${FEATURE_META[k].label}</span>`)
    .join("") + `<span><i class="swatch" style="background:var(--ink);opacity:.75"></i>Other annotation</span>`;
}

// Fingerprint ----------------------------------------------------------------------------
function renderFingerprint() {
  const host = $("#fingerprint");
  host.innerHTML = "";
  for (const f of current!.fingerprint) {
    const b = el(
      "button",
      { class: "fp-item" },
      `<span class="z">z ${fmt.f2(f.z)}</span><span><span class="lab">${esc(f.label)}</span><br><span class="id">${neuronId(f.l, f.u)} · FDR ${fmt.q(f.q)}</span></span>`,
    );
    b.addEventListener("click", () => window.dispatchEvent(new CustomEvent("open-neuron", { detail: { l: f.l, u: f.u } })));
    host.append(b);
  }
  if (!current!.fingerprint.length) host.innerHTML = `<p class="muted">No strongly enriched neurons for this sequence.</p>`;
}

// Custom sequence (live) ------------------------------------------------------------------
function showCustom() {
  setActiveChip("custom");
  const panel = $("#protein-panel");
  const live = isLive();
  panel.innerHTML = `
    <div class="protein-head">
      <div class="title">Analyze your own sequence</div>
      <p class="blurb" style="margin-top:8px">${
        live
          ? "Paste a protein sequence (raw or FASTA, up to 400 residues). The live ESM-2 35M model on Azure computes neuron activity, structure-probe predictions and per-residue surprise in a few seconds."
          : "The live model is offline right now, so custom sequences are unavailable. Every showcase protein still works from precomputed results."
      }</p>
    </div>
    <div class="live-panel" style="margin-top:14px">
      <textarea class="seq-input" id="custom-seq" placeholder=">my_protein&#10;MKTAYIAKQRQISFVKSHFSRQ..." ${live ? "" : "disabled"}></textarea>
      <div style="display:flex;gap:8px;flex-wrap:wrap" id="examples"></div>
      <div style="display:flex;gap:10px;align-items:center">
        <button class="btn primary" id="run-custom" ${live ? "" : "disabled"}>Analyze with ESM-2</button>
        <span id="custom-status" class="muted"></span>
      </div>
    </div>`;
  const ta = $("#custom-seq") as HTMLTextAreaElement;
  for (const [name, seq] of Object.entries(EXAMPLES)) {
    const b = el("button", { class: "btn small" }, esc(name));
    if (!live) b.setAttribute("disabled", "");
    b.addEventListener("click", () => (ta.value = seq));
    $("#examples").append(b);
  }
  $("#run-custom").addEventListener("click", async () => {
    const status = $("#custom-status");
    status.className = "muted";
    status.innerHTML = `<span class="spinner"></span> Running ESM-2…`;
    try {
      const r = await analyze(ta.value);
      current = {
        acc: "custom",
        short: "Custom",
        title: "Your sequence",
        name: "Your sequence",
        organism: "",
        blurb: "Analyzed live by ESM-2 35M. Tracks show structure neurons and the neurons that fire hardest on this sequence.",
        focus: "",
        sequence: r.sequence,
        features: [],
        structure: false,
        p_native: r.p_native,
        entropy: r.entropy,
        probes: r.probes,
        tracks: r.tracks,
        fingerprint: r.fingerprint,
      };
      activeTrack = r.tracks.find((t) => t.kind === "top") ?? r.tracks[0];
      pdbText = null;
      plddt = [];
      renderPanel();
      await renderStructure();
      renderSequence();
      renderLegend();
      renderFingerprint();
    } catch (e) {
      status.className = "error";
      status.textContent = (e as Error).message;
    }
  });
}
