// Minimal chart kit: thin marks, recessive axes, hover layer, table fallback.
import * as d3 from "d3";
import { el, esc, hideTip, showTip, ttRow } from "./ui";

export type Series = { name: string; color: string; points: { x: number; y: number }[]; label?: boolean };

export type LineOpts = {
  series: Series[];
  height?: number;
  xLabel: string;
  yLabel: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  xTicks?: number;
  refX?: { x: number; label: string };
  refY?: { y: number; label: string };
  marker?: number; // x value to highlight on every series
  table?: boolean;
};

const M = { top: 14, right: 18, bottom: 38, left: 58 };

function frame(host: HTMLElement, height: number) {
  host.innerHTML = "";
  const width = Math.max(260, host.clientWidth || 360);
  const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("height", height);
  return { svg, width, iw: width - M.left - M.right, ih: height - M.top - M.bottom };
}

function axes(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  x: d3.ScaleLinear<number, number> | d3.ScaleBand<string>,
  y: d3.ScaleLinear<number, number>,
  iw: number,
  ih: number,
  o: { xLabel: string; yLabel: string; xFormat?: (v: number) => string; yFormat?: (v: number) => string; xTicks?: number },
) {
  g.append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(y).ticks(5).tickSize(-iw).tickFormat(() => ""));
  const xa =
    "bandwidth" in x
      ? d3.axisBottom(x as d3.ScaleBand<string>).tickSizeOuter(0)
      : d3
          .axisBottom(x as d3.ScaleLinear<number, number>)
          .ticks(o.xTicks ?? 6)
          .tickSizeOuter(0)
          .tickFormat((v) => (o.xFormat ? o.xFormat(+v) : String(v)));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(xa as any);
  g.append("g")
    .attr("class", "axis")
    .call(
      d3
        .axisLeft(y)
        .ticks(5)
        .tickSize(0)
        .tickPadding(8)
        .tickFormat((v) => (o.yFormat ? o.yFormat(+v) : String(v))),
    )
    .call((s) => s.select(".domain").remove());
  g.append("text").attr("class", "axis-title").attr("x", iw / 2).attr("y", ih + 32).attr("text-anchor", "middle").text(o.xLabel);
  g.append("text")
    .attr("class", "axis-title")
    .attr("transform", `rotate(-90)`)
    .attr("x", -ih / 2)
    .attr("y", -46)
    .attr("text-anchor", "middle")
    .text(o.yLabel);
}

export function lineChart(host: HTMLElement, o: LineOpts) {
  const height = o.height ?? 240;
  const { svg, iw, ih } = frame(host, height);
  const all = o.series.flatMap((s) => s.points);
  const x = d3
    .scaleLinear()
    .domain(o.xDomain ?? (d3.extent(all, (p) => p.x) as [number, number]))
    .range([0, iw]);
  const yd = o.yDomain ?? (d3.extent(all, (p) => p.y) as [number, number]);
  const y = d3.scaleLinear().domain(yd).nice().range([ih, 0]);
  const g = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);
  axes(g, x, y, iw, ih, o);
  const xf = o.xFormat ?? ((v: number) => String(v));
  const yf = o.yFormat ?? d3.format(".3f");

  if (o.refY) {
    g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(o.refY.y)).attr("y2", y(o.refY.y)).attr("stroke", "var(--axis)");
    g.append("text").attr("class", "direct-label").attr("x", iw).attr("y", y(o.refY.y) - 5).attr("text-anchor", "end").style("font-weight", 500).text(o.refY.label);
  }
  if (o.refX) {
    g.append("line").attr("x1", x(o.refX.x)).attr("x2", x(o.refX.x)).attr("y1", 0).attr("y2", ih).attr("stroke", "var(--axis)");
    g.append("text").attr("class", "direct-label").attr("x", x(o.refX.x) + 5).attr("y", 10).style("font-weight", 500).text(o.refX.label);
  }

  const line = d3
    .line<{ x: number; y: number }>()
    .x((p) => x(p.x))
    .y((p) => y(p.y))
    .curve(d3.curveMonotoneX);
  for (const s of o.series) {
    g.append("path").attr("d", line(s.points)).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2).attr("stroke-linejoin", "round").attr("stroke-linecap", "round");
    const last = s.points[s.points.length - 1];
    g.append("circle").attr("cx", x(last.x)).attr("cy", y(last.y)).attr("r", 4).attr("fill", s.color).attr("stroke", "var(--card)").attr("stroke-width", 2);
    if (s.label) {
      g.append("text").attr("class", "direct-label").attr("x", x(last.x) + 8).attr("y", y(last.y) + 4).text(s.name);
    }
  }
  if (o.marker !== undefined) {
    const mx = x(o.marker);
    g.append("line").attr("x1", mx).attr("x2", mx).attr("y1", 0).attr("y2", ih).attr("stroke", "var(--ink-2)").attr("stroke-width", 1);
    for (const s of o.series) {
      const p = s.points.reduce((a, b) => (Math.abs(b.x - o.marker!) < Math.abs(a.x - o.marker!) ? b : a));
      g.append("circle").attr("cx", x(p.x)).attr("cy", y(p.y)).attr("r", 5).attr("fill", s.color).attr("stroke", "var(--card)").attr("stroke-width", 2);
    }
  }

  // hover crosshair
  const xs = Array.from(new Set(all.map((p) => p.x))).sort((a, b) => a - b);
  const cross = g.append("line").attr("y1", 0).attr("y2", ih).attr("stroke", "var(--axis)").attr("opacity", 0);
  const dots = g.append("g");
  g.append("rect")
    .attr("width", iw)
    .attr("height", ih)
    .attr("fill", "transparent")
    .on("mousemove", (ev: MouseEvent) => {
      const mx = x.invert(d3.pointer(ev)[0]);
      const xv = xs.reduce((a, b) => (Math.abs(b - mx) < Math.abs(a - mx) ? b : a));
      cross.attr("x1", x(xv)).attr("x2", x(xv)).attr("opacity", 1);
      dots.selectAll("*").remove();
      let html = `<div class="tt-title">${esc(o.xLabel)}: ${xf(xv)}</div>`;
      for (const s of o.series) {
        const p = s.points.find((q) => q.x === xv);
        if (!p) continue;
        dots.append("circle").attr("cx", x(p.x)).attr("cy", y(p.y)).attr("r", 4).attr("fill", s.color).attr("stroke", "var(--card)").attr("stroke-width", 2);
        html += ttRow(`<i class="swatch" style="background:${s.color};margin-right:6px"></i>${esc(s.name)}`, yf(p.y));
      }
      showTip(html, ev);
    })
    .on("mouseleave", () => {
      cross.attr("opacity", 0);
      dots.selectAll("*").remove();
      hideTip();
    });

  if (o.table) addTable(host, o.xLabel, o.series, xf, yf);
}

export function barChart(
  host: HTMLElement,
  o: {
    bars: { key: string; value: number; color: string; note?: string }[];
    height?: number;
    xLabel: string;
    yLabel: string;
    yDomain?: [number, number];
    yFormat?: (v: number) => string;
  },
) {
  const height = o.height ?? 240;
  const { svg, iw, ih } = frame(host, height);
  const x = d3
    .scaleBand()
    .domain(o.bars.map((b) => b.key))
    .range([0, iw])
    .padding(0.35);
  const y = d3
    .scaleLinear()
    .domain(o.yDomain ?? [0, d3.max(o.bars, (b) => b.value)!])
    .nice()
    .range([ih, 0]);
  const g = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);
  axes(g, x, y, iw, ih, { ...o, xFormat: undefined });
  const yf = o.yFormat ?? d3.format(".2f");
  const bw = Math.min(40, x.bandwidth());
  for (const b of o.bars) {
    const bx = x(b.key)! + (x.bandwidth() - bw) / 2;
    const top = y(b.value);
    const h = ih - top;
    const r = Math.min(4, h);
    g.append("path")
      .attr("d", `M${bx},${ih}V${top + r}Q${bx},${top} ${bx + r},${top}H${bx + bw - r}Q${bx + bw},${top} ${bx + bw},${top + r}V${ih}Z`)
      .attr("fill", b.color)
      .on("mousemove", (ev: MouseEvent) => showTip(`<div class="tt-title">${esc(b.key)}</div>${ttRow(esc(o.yLabel), yf(b.value))}${b.note ? `<div class="muted">${esc(b.note)}</div>` : ""}`, ev))
      .on("mouseleave", hideTip);
    g.append("text").attr("class", "direct-label").attr("x", bx + bw / 2).attr("y", top - 6).attr("text-anchor", "middle").text(yf(b.value));
  }
}

function addTable(host: HTMLElement, xLabel: string, series: Series[], xf: (v: number) => string, yf: (v: number) => string) {
  const btn = el("button", { class: "table-toggle" }, "Show data table");
  const box = el("div", { hidden: "" });
  const xs = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.x)))).sort((a, b) => a - b);
  box.innerHTML = `<table class="data"><thead><tr><th>${esc(xLabel)}</th>${series.map((s) => `<th class="num">${esc(s.name)}</th>`).join("")}</tr></thead><tbody>${xs
    .map(
      (xv) =>
        `<tr><td>${xf(xv)}</td>${series
          .map((s) => {
            const p = s.points.find((q) => q.x === xv);
            return `<td class="num">${p ? yf(p.y) : ""}</td>`;
          })
          .join("")}</tr>`,
    )
    .join("")}</tbody></table>`;
  btn.addEventListener("click", () => {
    box.hidden = !box.hidden;
    btn.textContent = box.hidden ? "Show data table" : "Hide data table";
  });
  host.append(btn, box);
}
