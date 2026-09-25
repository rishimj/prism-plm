import * as d3 from "d3";
import { load, loadAtlasLayer, type AtlasIndex, type AtlasNeuron, type AtlasProtein, type Term } from "./data";
import { $, cssVar, el, esc, fmt, hideTip, neuronId, onTheme, showTip, ttRow } from "./ui";

const NS: Record<string, string> = { BP: "Biological process", MF: "Molecular function", CC: "Cellular component", FAM: "Protein family" };
const SIG = -Math.log10(0.05);

let index: AtlasIndex;
let terms: Term[];
let proteins: AtlasProtein[];
let layer = 6;
let selected = { l: 6, u: -1 };

export async function initAtlas() {
  [index, terms, proteins] = await Promise.all([
    load<AtlasIndex>("atlas/index.json"),
    load<Term[]>("atlas/terms.json"),
    load<AtlasProtein[]>("atlas/proteins.json"),
  ]);
  const total = index.n_layers * index.hidden;
  const sigAll = d3.sum(index.layers, (l) => l.neurons.filter((n) => n[2] > SIG).length);
  $("#atlas-intro").textContent = `${fmt.int(total)} neurons (${index.n_layers} layers × ${index.hidden}) profiled on ${fmt.int(
    index.n_proteins,
  )} proteins; ${fmt.pct(sigAll / total)} earn a significant biological label. Each dot is a neuron: further right means its top proteins split into more distinct groups (polysemantic); higher means those proteins share a sharper biological label. Search for a concept or click any dot.`;

  const seg = $("#layer-select");
  for (let l = 1; l <= index.n_layers; l++) {
    const b = el("button", { "data-l": String(l) }, `L${l}`);
    b.addEventListener("click", () => selectLayer(l));
    seg.append(b);
  }
  initSearch();
  window.addEventListener("open-neuron", (e) => {
    const { l, u } = (e as CustomEvent).detail;
    document.getElementById("atlas")!.scrollIntoView({ behavior: "smooth" });
    selectLayer(l, u);
  });
  onTheme(() => drawScatter());
  let t: number | undefined;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = window.setTimeout(drawScatter, 150);
  });
  // open on the most specific neuron in the middle layer
  const mid = index.layers[layer - 1].neurons;
  const best = d3.greatestIndex(mid, (n) => n[2]);
  await selectLayer(layer, best ?? 0);
}

async function selectLayer(l: number, u?: number) {
  layer = l;
  document.querySelectorAll("#layer-select button").forEach((b) => b.classList.toggle("active", (b as HTMLElement).dataset.l === String(l)));
  const info = index.layers[l - 1];
  if (u === undefined) u = d3.greatestIndex(info.neurons, (n) => n[2]) ?? 0;
  selected = { l, u };
  $("#scatter-title").textContent = `Layer ${l} of ${index.n_layers}`;
  $("#scatter-sub").textContent = `${fmt.pct(info.sig_frac)} of neurons carry a significant label · median polysemanticity ${fmt.f3(info.median_poly)}`;
  drawScatter();
  renderTopList();
  await showNeuron(l, u);
}

function renderTopList() {
  const host = $("#layer-top");
  host.innerHTML = "";
  const ranked = index.layers[layer - 1].neurons
    .map((n, u) => ({ u, t: n[1], s: n[2] }))
    .sort((a, b) => b.s - a.s)
    .slice(0, 8);
  for (const r of ranked) {
    const b = el(
      "button",
      { class: r.u === selected.u ? "active" : "" },
      `<span class="mono">${neuronId(layer, r.u)}</span><span>${esc(termName(r.t))}</span><span class="muted">FDR ${fmt.q(10 ** -r.s)}</span>`,
    );
    b.addEventListener("click", () => {
      selected = { l: layer, u: r.u };
      drawScatter();
      renderTopList();
      showNeuron(layer, r.u);
    });
    host.append(b);
  }
}

function termName(t: number) {
  const term = terms[t];
  return term ? term[1] : "Unlabelled";
}

function drawScatter() {
  const host = $("#atlas-scatter");
  host.innerHTML = "";
  const data = index.layers[layer - 1].neurons.map((n, u) => ({ u, p: n[0], t: n[1], s: n[2] }));
  const width = Math.max(280, host.clientWidth);
  const height = 420;
  const m = { top: 14, right: 16, bottom: 42, left: 50 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  // domains shared across layers so switching layer is comparable
  const allP = index.layers.flatMap((l) => l.neurons.map((n) => n[0]));
  const allS = index.layers.flatMap((l) => l.neurons.map((n) => Math.min(n[2], 60)));
  const x = d3
    .scaleLinear()
    .domain(d3.extent(allP) as [number, number])
    .nice()
    .range([0, iw]);
  const y = d3.scaleSymlog().constant(3).domain([0, d3.max(allS)!]).range([ih, 0]);
  const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("height", height);
  const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
  const yt = [0, 1.3, 3, 5, 10, 20, 40, 60].filter((v) => v <= y.domain()[1]);
  g.append("g").attr("class", "grid").call(d3.axisLeft(y).tickValues(yt).tickSize(-iw).tickFormat(() => ""));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
  g.append("g")
    .attr("class", "axis")
    .call(d3.axisLeft(y).tickValues(yt).tickSize(0).tickPadding(8).tickFormat((v) => d3.format("~g")(+v)))
    .call((s) => s.select(".domain").remove());
  g.append("text").attr("class", "axis-title").attr("x", iw / 2).attr("y", ih + 34).attr("text-anchor", "middle").text("Polysemanticity score (1 − mean centroid similarity)");
  g.append("text").attr("class", "axis-title").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -38).attr("text-anchor", "middle").text("Label strength, −log10 FDR");
  g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(SIG)).attr("y2", y(SIG)).attr("stroke", "var(--axis)");
  g.append("text").attr("class", "direct-label").attr("x", iw).attr("y", y(SIG) - 5).attr("text-anchor", "end").style("font-weight", 500).text("FDR 0.05");

  const blue = cssVar("--s1");
  const muted = cssVar("--muted");
  g.append("g")
    .selectAll("circle")
    .data(data)
    .join("circle")
    .attr("cx", (d) => x(d.p))
    .attr("cy", (d) => y(Math.min(d.s, 60)))
    .attr("r", 4)
    .attr("fill", (d) => (d.s > SIG ? blue : muted))
    .attr("fill-opacity", (d) => (d.s > SIG ? 0.75 : 0.4))
    .attr("stroke", "var(--card)")
    .attr("stroke-width", 1)
    .style("cursor", "pointer")
    .on("mousemove", (ev: MouseEvent, d) =>
      showTip(
        `<div class="tt-title">${neuronId(layer, d.u)}</div><div style="margin-bottom:4px">${esc(d.s > SIG ? termName(d.t) : "No significant label")}</div>${ttRow(
          "Polysemanticity",
          fmt.f3(d.p),
        )}${ttRow("−log10 FDR", fmt.f2(d.s))}`,
        ev,
      ),
    )
    .on("mouseleave", hideTip)
    .on("click", (_, d) => {
      selected = { l: layer, u: d.u };
      drawScatter();
      renderTopList();
      showNeuron(layer, d.u);
    });
  const sel = data[selected.u];
  if (sel && selected.l === layer) {
    g.append("circle").attr("cx", x(sel.p)).attr("cy", y(Math.min(sel.s, 60))).attr("r", 8).attr("fill", "none").attr("stroke", "var(--ink)").attr("stroke-width", 2);
  }
}

async function showNeuron(l: number, u: number) {
  const card = $("#neuron-card");
  const layerData = await loadAtlasLayer(l);
  const n = layerData.neurons[u];
  const top = n.t[0];
  const sig = top && top[2] < 0.05;
  const headline = sig ? termName(top[0]) : "No significant biological label";
  const ns = sig ? terms[top[0]]?.[2] ?? "" : "";
  card.innerHTML = `
    <span class="eyebrow">Layer ${l} · Neuron ${u}</span>
    <div class="headline">${esc(headline)}</div>
    <div style="display:flex;gap:6px;flex-wrap:wrap">${ns ? `<span class="tag">${NS[ns]}</span>` : ""}${
      sig ? `<span class="tag accent">${top[1]} of top ${index.top_k} proteins · ${fmt.f2(top[3])}× enriched · FDR ${fmt.q(top[2])}</span>` : `<span class="tag">Top proteins share no over-represented term</span>`
    }</div>
    <div class="kv">
      <div><div class="k">Polysemanticity</div><div class="v">${fmt.f3(n.p)}</div></div>
      <div><div class="k">Clusters (k-means)</div><div class="v">${n.c.length}</div></div>
      <div><div class="k">Mean activation</div><div class="v">${fmt.f2(n.m)}</div></div>
      <div><div class="k">Std. deviation</div><div class="v">${fmt.f2(n.sd)}</div></div>
    </div>
    <div class="sub-h">Activation across ${fmt.int(index.n_proteins)} proteins</div>
    <div class="chart" id="neuron-hist"></div>
    ${
      n.t.length
        ? `<div class="sub-h">Most enriched annotations</div><table class="data"><thead><tr><th>Term</th><th>Type</th><th class="num">In top ${index.top_k}</th><th class="num">Fold</th><th class="num">FDR</th></tr></thead><tbody>${n.t
            .map(
              ([t, k, q, f]) =>
                `<tr><td>${esc(termName(t))}${terms[t]?.[0] ? ` <span class="muted mono" style="font-size:10.5px">${terms[t]![0]}</span>` : ""}</td><td>${terms[t]?.[2] ?? ""}</td><td class="num">${k}</td><td class="num">${fmt.f2(
                  f,
                )}</td><td class="num">${fmt.q(q)}</td></tr>`,
            )
            .join("")}</tbody></table>`
        : ""
    }
    <div class="sub-h">Sub-populations among its top ${index.top_k} proteins</div>
    <div id="clusters"></div>
    <div class="sub-h">Strongest activating proteins</div>
    <div class="protein-list">${n.tp
      .map((i, k) => {
        const p = proteins[i];
        return `<div><a href="https://www.uniprot.org/uniprotkb/${p[0]}" target="_blank" rel="noopener">${esc(p[1])}</a> <span class="muted">${fmt.f2(n.ta[k])}</span><br><span class="muted" style="font-size:11.5px">${esc(
          p[3] || p[2],
        )}</span></div>`;
      })
      .join("")}</div>`;
  drawHistogram($("#neuron-hist", card), n);
  const cl = $("#clusters", card);
  for (const [size, mean, t, q, ex] of n.c) {
    const label = t >= 0 && q < 0.05 ? termName(t) : "Mixed / no significant label";
    cl.append(
      el(
        "div",
        { class: "cluster-row" },
        `<div class="size">${size}</div><div><div>${esc(label)}${t >= 0 && q < 0.05 ? ` <span class="muted">FDR ${fmt.q(q)}</span>` : ""}</div><div class="bar"><i style="width:${size}%"></i></div><div class="ex">mean activation ${fmt.f2(
          mean,
        )} · e.g. ${ex
          .slice(0, 3)
          .map((i) => esc(proteins[i][1]))
          .join(", ")}</div></div>`,
      ),
    );
  }
}

function drawHistogram(host: HTMLElement, n: AtlasNeuron) {
  const width = Math.max(260, host.clientWidth);
  const height = 90;
  const bins = n.h.length;
  const [lo, hi] = n.hr;
  const step = (hi - lo) / bins;
  const x = d3.scaleLinear().domain([lo, hi]).range([0, width]);
  const y = d3.scaleSymlog().constant(5).domain([0, d3.max(n.h)!]).range([height - 18, 0]);
  const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("height", height);
  const thresh = Math.min(...n.ta);
  const bw = Math.max(1, width / bins - 2);
  n.h.forEach((c, i) => {
    const x0 = x(lo + i * step) + 1;
    const top = y(c);
    const inTop = lo + (i + 1) * step > thresh;
    svg
      .append("rect")
      .attr("x", x0)
      .attr("width", bw)
      .attr("y", top)
      .attr("height", Math.max(0, height - 18 - top))
      .attr("rx", Math.min(2, bw / 2))
      .attr("fill", inTop ? cssVar("--s2") : cssVar("--s1"))
      .attr("fill-opacity", inTop ? 1 : 0.55)
      .on("mousemove", (ev: MouseEvent) => showTip(`${ttRow("Activation", `${fmt.f2(lo + i * step)} to ${fmt.f2(lo + (i + 1) * step)}`)}${ttRow("Proteins", fmt.int(c))}`, ev))
      .on("mouseleave", hideTip);
  });
  svg.append("line").attr("x1", 0).attr("x2", width).attr("y1", height - 18).attr("y2", height - 18).attr("stroke", "var(--axis)");
  svg.append("text").attr("x", 0).attr("y", height - 4).attr("font-size", 10.5).attr("fill", "var(--muted)").text(fmt.f2(lo));
  svg.append("text").attr("x", width).attr("y", height - 4).attr("text-anchor", "end").attr("font-size", 10.5).attr("fill", "var(--muted)").text(fmt.f2(hi));
  svg.append("text").attr("x", width / 2).attr("y", height - 4).attr("text-anchor", "middle").attr("font-size", 10.5).attr("fill", "var(--ink-2)").text("orange: bins reaching the strongest-activating proteins");
}

function initSearch() {
  const input = $("#atlas-search") as HTMLInputElement;
  const box = $("#search-results");
  const run = () => {
    const q = input.value.trim().toLowerCase();
    box.innerHTML = "";
    if (q.length < 2) {
      box.classList.remove("show");
      return;
    }
    const hits: { l: number; u: number; name: string; s: number }[] = [];
    for (const L of index.layers) {
      L.neurons.forEach(([, t, s], u) => {
        if (s > SIG && t >= 0 && terms[t] && terms[t]![1].toLowerCase().includes(q)) hits.push({ l: L.layer, u, name: terms[t]![1], s });
      });
    }
    hits.sort((a, b) => b.s - a.s);
    if (!hits.length) box.innerHTML = `<div class="muted" style="padding:10px">No neuron is labelled with “${esc(q)}”. Try heme, kinase, membrane, ribosome, zinc, DNA.</div>`;
    for (const h of hits.slice(0, 40)) {
      const b = el("button", {}, `<span class="mono">${neuronId(h.l, h.u)}</span><span>${esc(h.name)}</span><span class="muted">FDR ${fmt.q(10 ** -h.s)}</span>`);
      b.addEventListener("click", () => {
        box.classList.remove("show");
        selectLayer(h.l, h.u);
      });
      box.append(b);
    }
    box.classList.add("show");
  };
  input.addEventListener("input", run);
  input.addEventListener("focus", run);
  document.addEventListener("click", (e) => {
    if (!(e.target as HTMLElement).closest(".search")) box.classList.remove("show");
  });
}
