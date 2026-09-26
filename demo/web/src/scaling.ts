import { barChart, lineChart } from "./charts";
import { load, type Scaling } from "./data";
import { $, cssVar, fmt, onTheme } from "./ui";

// Model size is ordinal, so it gets one hue stepped light -> dark (not categorical colours).
const RAMP = ["--seq-250", "--seq-400", "--seq-550", "--seq-700"];

export async function initScaling() {
  let data: Scaling;
  try {
    data = await load<Scaling>("scaling.json");
  } catch {
    $("#scaling-finding").textContent = "Scaling results are not available in this build.";
    return;
  }
  const render = () => draw(data);
  render();
  onTheme(render);
  let t: number | undefined;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = window.setTimeout(render, 150);
  });
}

const label = (m: Scaling["models"][number]) => `ESM-2 ${m.key} (${m.n_layers} layers)`;

function draw(data: Scaling) {
  const models = data.models;
  const colors = models.map((_, i) => cssVar(RAMP[Math.min(i + (4 - models.length), 3)]));
  $("#scaling-legend").innerHTML = models
    .map((m, i) => `<span><i class="key-line" style="background:${colors[i]}"></i>${label(m)}</span>`)
    .join("");
  lineChart($("#scaling-depth"), {
    height: 280,
    xLabel: "Relative depth (layer / total layers)",
    yLabel: "Median polysemanticity",
    xDomain: [0, 1],
    xFormat: fmt.pct,
    yFormat: fmt.f3,
    series: models.map((m, i) => ({
      name: m.key,
      color: colors[i],
      points: m.layers.map((l) => ({ x: l.depth, y: l.median })),
    })),
    table: true,
  });
  barChart($("#scaling-sig"), {
    height: 280,
    xLabel: "Model",
    yLabel: "Share of neurons labelled",
    yDomain: [0, 1],
    yFormat: fmt.pct,
    bars: models.map((m, i) => ({
      key: m.key,
      value: m.sig_frac,
      color: colors[i],
      note: `${fmt.int(m.layers.length * data.units_per_layer)} neurons sampled from ${m.n_layers} layers × ${m.hidden} units`,
    })),
  });

  const first = models[0];
  const last = models[models.length - 1];
  const polyDir = last.median < first.median ? "lower" : "higher";
  const labDiff = last.sig_frac - first.sig_frac;
  const labText =
    Math.abs(labDiff) < 0.05
      ? `the share of neurons with a significant biological label barely changes (${fmt.pct(first.sig_frac)} → ${fmt.pct(last.sig_frac)})`
      : `${labDiff > 0 ? "more" : "fewer"} neurons earn a significant biological label (${fmt.pct(first.sig_frac)} → ${fmt.pct(last.sig_frac)})`;
  $("#scaling-finding").innerHTML = `<b>Finding.</b> Going from ${first.key} to ${last.key} parameters, the median neuron polysemanticity is ${polyDir} (${fmt.f3(
    first.median,
)} → ${fmt.f3(last.median)}) and ${labText}. Scores are computed in a shared 64-dimensional PCA space so models of different width are comparable; each model sees the same ${fmt.int(
    data.n_proteins,
  )} proteins and ${data.units_per_layer} sampled neurons per layer.${
    polyDir === "higher"
      ? " Bigger models do not give cleaner individual neurons. That is consistent with superposition, where larger models pack more features into shared units, and it motivates sparse dictionary methods as the next step."
      : ""
  } The research pipeline extends this to ESM-2 3B on GPU.`;
}
