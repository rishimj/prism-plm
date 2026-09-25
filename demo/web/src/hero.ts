// Hero visual: the real AlphaFold C-alpha trace of a showcase protein, slowly
// rotating, coloured by a real neuron's per-residue activation.
import * as d3 from "d3";
import { load, type Showcase } from "./data";
import { esc, neuronId } from "./ui";

type P3 = [number, number, number];

function parseCA(pdb: string): P3[] {
  const pts: P3[] = [];
  for (const line of pdb.split("\n")) {
    if (line.startsWith("ATOM") && line.slice(12, 16).trim() === "CA") {
      pts.push([+line.slice(30, 38), +line.slice(38, 46), +line.slice(46, 54)]);
    }
  }
  return pts;
}

function catmull(points: P3[], values: number[], steps = 5) {
  const out: { p: P3; v: number }[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    for (let s = 0; s < steps; s++) {
      const t = s / steps;
      const t2 = t * t;
      const t3 = t2 * t;
      const p = [0, 1, 2].map(
        (k) =>
          0.5 *
          (2 * p1[k] + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3),
      ) as P3;
      out.push({ p, v: values[i] + (values[i + 1] - values[i]) * t });
    }
  }
  return out;
}

export async function initHero(acc = "P01116") {
  const canvas = document.getElementById("hero-canvas") as HTMLCanvasElement;
  const caption = document.getElementById("hero-caption")!;
  let protein: Showcase;
  let pdb: string;
  try {
    [protein, pdb] = await Promise.all([load<Showcase>(`showcase/${acc}.json`), load<string>(`structures/${acc}.pdb`)]);
  } catch {
    return;
  }
  const ca = parseCA(pdb);
  const track =
    protein.tracks.find((t) => t.kind === "feature" && t.feature === protein.focus) ??
    protein.tracks.find((t) => t.kind === "feature") ??
    protein.tracks[0];
  const n = Math.min(ca.length, track.v.length);
  const lo = d3.quantile(track.v, 0.05)!;
  const hi = d3.quantile(track.v, 0.995)!;
  const norm = track.v.slice(0, n).map((v) => Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1))));
  const centre = [0, 1, 2].map((k) => d3.median(ca, (p) => p[k])!) as P3;
  const pts = ca.slice(0, n).map((p) => [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]] as P3);
  // frame the folded core; disordered termini may run off the edge
  const radius = d3.quantile(pts.map((p) => Math.hypot(...p)).sort(d3.ascending), 0.9)!;
  const curve = catmull(pts, norm);
  const color = d3.scaleLinear<string>().domain([0, 0.6, 1]).range(["#2c4d78", "#9ad7ff", "#ffe38a"]).interpolate(d3.interpolateLab);

  caption.innerHTML = `<b>${esc(protein.short)}</b> · AlphaFold structure, coloured by neuron <b>${neuronId(track.l, track.u)}</b><br/>${esc(
    track.note,
  )}${track.auc ? ` (AUROC ${track.auc.toFixed(2)})` : ""}<div class="hero-legend"><span>quiet</span><span class="bar"></span><span>firing</span></div>`;

  const ctx = canvas.getContext("2d")!;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let angle = 0.6;
  let visible = true;
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(canvas);

  function frame() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const scale = (Math.min(w, h) / (2 * radius)) * 0.8;
    const ca_ = Math.cos(angle);
    const sa = Math.sin(angle);
    const tilt = 0.35;
    const ct = Math.cos(tilt);
    const st = Math.sin(tilt);
    const proj = curve.map(({ p, v }) => {
      const x = p[0] * ca_ + p[2] * sa;
      const z0 = -p[0] * sa + p[2] * ca_;
      const y = p[1] * ct - z0 * st;
      const z = p[1] * st + z0 * ct;
      const persp = 1 / (1 - z / (radius * 5));
      return { x: w * 0.47 + x * scale * persp, y: h * 0.42 - y * scale * persp, z, v };
    });
    const segs = d3.range(proj.length - 1).sort((a, b) => proj[a].z - proj[b].z);
    ctx.lineCap = "round";
    for (const i of segs) {
      const a = proj[i];
      const b = proj[i + 1];
      const depth = (a.z / radius + 1) / 2; // 0 back .. 1 front
      const v = (a.v + b.v) / 2;
      // depth cue: blend toward the background instead of using alpha (avoids dotted joints)
      ctx.strokeStyle = d3.interpolateLab("#06110d", color(v))(0.3 + 0.7 * depth);
      ctx.lineWidth = 2.2 + 4.2 * depth + 3 * v;
      if (v > 0.55) {
        ctx.shadowColor = color(v);
        ctx.shadowBlur = 14 * v;
      } else {
        ctx.shadowBlur = 0;
      }
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
    if (!reduce && visible) angle += 0.0035;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
