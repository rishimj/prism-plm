import "./styles.css";
import { API_BASE, checkHealth, onHealth } from "./api";
import { initAtlas } from "./atlas";
import { load, type Summary } from "./data";
import { initExplorer } from "./explorer";
import { initHero } from "./hero";
import { initProbing } from "./probing";
import { initScaling } from "./scaling";
import { initSteering } from "./steering";
import { $, fmt, initReveal, initTheme } from "./ui";

async function heroStats() {
  try {
    const s = await load<Summary>("summary.json");
    const items = [
      [fmt.int(s.proteins), "real proteins analyzed"],
      [fmt.int(s.neurons_profiled), "neurons profiled"],
      [String(s.models), "ESM-2 model sizes compared"],
      [fmt.int(s.go_terms), "GO terms & families tested"],
    ];
    $("#hero-stats").innerHTML = items.map(([v, l]) => `<div class="hero-stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  } catch {
    /* stats are decorative */
  }
}

function statusPill() {
  const pill = $("#api-status");
  onHealth((h) => {
    pill.classList.toggle("live", !!h);
    pill.querySelector(".txt")!.textContent = h ? "Live model online" : "Precomputed mode";
    pill.title = h
      ? `Live ESM-2 (${h.model}) running on Azure. You can analyze and steer your own sequences.`
      : API_BASE
        ? "The live API is not reachable right now; everything shown uses precomputed results from real model runs."
        : "Showing precomputed results from real model runs.";
  });
  checkHealth();
  setInterval(checkHealth, 60_000);
}

function navSpy() {
  const links = Array.from(document.querySelectorAll<HTMLAnchorElement>(".nav-links a"));
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
      }
    },
    { rootMargin: "-45% 0px -50% 0px" },
  );
  links.forEach((a) => {
    const s = document.querySelector(a.getAttribute("href")!);
    if (s) io.observe(s);
  });
}

const ICONS: Record<string, string> = {
  gpu: '<path d="M4 7h16v10H4zM8 11h8M2 10h2M2 14h2M20 10h2M20 14h2"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5zM3 13l9 5 9-5"/>',
  shield: '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6l-8-3z"/>',
  flask: '<path d="M9 3h6M10 3v6L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2l-6-10V3"/>',
  cog: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.9 1.2V21a2 2 0 1 1-4 0v-.1A1.7 1.7 0 0 0 7 19.4a1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 1 15H.9a2 2 0 1 1 0-4H1a1.7 1.7 0 0 0 1.6-2.2 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H7a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
};
const icon = (k: string) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[k]}</svg>`;

function engineering() {
  $("#arch-diagram").innerHTML = `
  <h3>System architecture</h3>
  <div class="sub" style="margin-bottom:14px">Offline batch analysis feeds a static site; a small live service handles custom sequences.</div>
  <svg viewBox="0 0 640 400" role="img" aria-label="Architecture: research pipeline produces JSON artifacts served by GitHub Pages; the browser also calls a FastAPI service on an Azure B2s VM through Caddy.">
    <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--ink-2)"/></marker></defs>
    <rect class="zone" x="8" y="18" width="292" height="200" rx="12"/>
    <text class="zone-label" x="22" y="38">Offline · research pipeline</text>
    <rect class="box" x="24" y="52" width="260" height="56" rx="10"/>
    <text x="40" y="76" font-weight="650">ESM-2 8M → 3B (PyTorch)</text>
    <text class="small" x="40" y="94">SLURM jobs on GPU cluster (PACE) or CPU</text>
    <rect class="box" x="24" y="128" width="260" height="72" rx="10"/>
    <text x="40" y="152" font-weight="650">PRISM-Bio analyses</text>
    <text class="small" x="40" y="170">neuron atlas · GO enrichment · probing</text>
    <text class="small" x="40" y="186">scaling study · activation steering</text>
    <path class="flow" d="M154 108v18"/>
    <rect class="zone" x="340" y="18" width="292" height="200" rx="12"/>
    <text class="zone-label" x="354" y="38">GitHub Pages · static</text>
    <rect class="box" x="356" y="52" width="260" height="56" rx="10"/>
    <text x="372" y="76" font-weight="650" textLength="236" lengthAdjust="spacingAndGlyphs">JSON results + AlphaFold models</text>
    <text class="small" x="372" y="94">versioned with the code, served via CDN</text>
    <rect class="box" x="356" y="128" width="260" height="72" rx="10"/>
    <text x="372" y="152" font-weight="650">This site</text>
    <text class="small" x="372" y="170">Vite · TypeScript · D3 · 3Dmol.js</text>
    <text class="small" x="372" y="186">deployed by GitHub Actions on push</text>
    <path class="flow" d="M286 164H352"/>
    <path class="flow" d="M486 108v18"/>
    <rect class="box accent" x="210" y="258" width="220" height="44" rx="10"/>
    <text x="320" y="285" text-anchor="middle" font-weight="650">Your browser</text>
    <path class="flow" d="M486 202V230H340V254"/>
    <rect class="zone" x="8" y="322" width="624" height="70" rx="12"/>
    <text class="zone-label" x="22" y="342">Azure B2s VM · live API</text>
    <rect class="box" x="24" y="350" width="170" height="34" rx="8"/>
    <text x="109" y="372" text-anchor="middle">Caddy · auto HTTPS</text>
    <rect class="box" x="232" y="350" width="176" height="34" rx="8"/>
    <text x="320" y="372" text-anchor="middle">FastAPI · rate limit</text>
    <rect class="box" x="446" y="350" width="170" height="34" rx="8"/>
    <text x="531" y="372" text-anchor="middle">ESM-2 35M · CPU</text>
    <path class="flow" d="M196 367H228"/>
    <path class="flow" d="M410 367H442"/>
    <path class="flow live" d="M290 304V318"/>
    <path class="flow live" d="M350 318V306" />
  </svg>`;
  const items: [string, string, string][] = [
    ["gpu", "Research code, not a mock-up", "Every figure comes from running this repository's own modules on real SwissProt proteins: SteeringVector, SteeringHook, compute_polysemanticity and the concept-probability-shift metric."],
    ["cog", "Configurable pipeline", "Pydantic-validated configs with a CLI > ENV > YAML > defaults hierarchy, a registry of clustering and dimensionality-reduction backends, and SLURM scripts for cluster runs."],
    ["flask", "Statistics done carefully", "Hypergeometric enrichment with Benjamini-Hochberg FDR, probes evaluated on held-out proteins, and PCA-normalised scores so models of different widths are comparable."],
    ["layers", "Reproducible data build", "demo/pipeline regenerates every JSON file from scratch with fixed seeds; the scaling study runs four model sizes on identical inputs."],
    ["shield", "Production-minded API", "Validated inputs, per-IP rate limiting, response caching, a single-flight inference lock sized for a 4 GB VM, CORS locked to this site, and a non-root container."],
    ["globe", "Works even when the VM is off", "The site degrades gracefully to precomputed results, so it is always demo-ready; the status pill shows whether the live model is reachable."],
  ];
  $("#tech-list").innerHTML =
    items.map(([ic, h, p]) => `<div class="tech"><div class="ic">${icon(ic)}</div><div><h4>${h}</h4><p>${p}</p></div></div>`).join("") +
    `<div class="badges">${["Python", "PyTorch", "HuggingFace Transformers", "scikit-learn", "SciPy", "FastAPI", "Docker", "Caddy", "Azure", "TypeScript", "D3.js", "3Dmol.js", "GitHub Actions"]
      .map((b) => `<span class="tag">${b}</span>`)
      .join("")}</div>`;
}

function guard(name: string, p: Promise<unknown>) {
  p.catch((e) => console.error(`[${name}]`, e));
}

initTheme();
statusPill();
navSpy();
engineering();
guard("stats", heroStats());
guard("hero", initHero());
guard("explorer", initExplorer());
guard("probing", initProbing());
guard("atlas", initAtlas());
guard("scaling", initScaling());
guard("steering", initSteering());
initReveal();
