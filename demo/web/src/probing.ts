import { lineChart } from "./charts";
import { load, type Probing } from "./data";
import { $, cssVar, el, FEATURE_META, fmt, neuronId, onTheme } from "./ui";

const FEATURES = ["helix", "strand", "transmembrane"] as const;

export async function initProbing() {
  const data = await load<Probing>("probing.json");
  const render = () => draw(data);
  render();
  onTheme(render);
  let t: number | undefined;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = window.setTimeout(render, 150);
  });
}

function draw(data: Probing) {
  const probeColor = cssVar("--s1");
  const neuronColor = cssVar("--s2");
  $("#probing-legend").innerHTML = `<span><i class="key-line" style="background:${probeColor}"></i>Linear probe (all 480 neurons)</span><span><i class="key-line" style="background:${neuronColor}"></i>Best single neuron</span>`;
  const host = $("#probing-charts");
  host.innerHTML = "";
  for (const f of FEATURES) {
    const card = el("div", { class: "card card-pad" });
    const best = data.layers.reduce((a, b) => (b[f].auc > a[f].auc ? b : a));
    card.innerHTML = `<h3>${FEATURE_META[f].label}</h3><div class="sub">Held-out AUROC by layer. Peak ${fmt.f3(best[f].auc)} at layer ${best.layer}.</div>`;
    const chart = el("div", { class: "chart" });
    card.append(chart);
    host.append(card);
    lineChart(chart, {
      height: 220,
      xLabel: "Layer",
      yLabel: "AUROC",
      yDomain: [0.5, 1],
      xFormat: (v) => (v === 0 ? "emb" : String(v)),
      yFormat: fmt.f2,
      refY: { y: 0.5, label: "chance" },
      series: [
        { name: "Linear probe", color: probeColor, points: data.layers.map((r) => ({ x: r.layer, y: r[f].auc })) },
        { name: "Best single neuron", color: neuronColor, points: data.layers.map((r) => ({ x: r.layer, y: r[f].neuron.auc })) },
      ],
      table: true,
    });
  }

  const ins = $("#probing-insights");
  const emb = data.layers[0];
  const bestOf = (f: (typeof FEATURES)[number]) => data.layers.reduce((a, b) => (b[f].auc > a[f].auc ? b : a));
  const h = bestOf("helix");
  const s = bestOf("strand");
  const tm = bestOf("transmembrane");
  const tmNeuron = data.layers.reduce((a, b) => (b.transmembrane.neuron.auc > a.transmembrane.neuron.auc ? b : a));
  ins.innerHTML = `
    <div class="card insight"><div class="v">${fmt.f2(emb.helix.auc)} → ${fmt.f2(h.helix.auc)}</div><div class="l">Helix AUROC from the raw amino-acid embedding to layer ${h.layer}: context turns letters into structure.</div></div>
    <div class="card insight"><div class="v">${fmt.f2(s.strand.auc)}</div><div class="l">Beta-strand AUROC at layer ${s.layer}, on proteins the probe never saw.</div></div>
    <div class="card insight"><div class="v">${fmt.f2(tmNeuron.transmembrane.neuron.auc)}</div><div class="l">A single neuron, ${neuronId(tmNeuron.layer, tmNeuron.transmembrane.neuron.u)}, separates membrane-spanning residues with this AUROC (probe: ${fmt.f2(tm.transmembrane.auc)}).</div></div>`;
  $("#probing-note").textContent = `Probes: L2-regularised logistic regression on standardised activations, 80/20 split by protein. Data: ${fmt.int(
    data.n_struct_proteins,
  )} SwissProt proteins with PDB-derived secondary structure and ${fmt.int(data.n_tm_proteins)} with annotated transmembrane segments (${fmt.int(
    data.n_residues,
  )} residues). Model: ${data.model}. The single-neuron line picks the neuron on training proteins and scores it on test proteins.`;
}
