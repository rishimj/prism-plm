"""Shared helpers for the PRISM-Bio demo data pipeline.

Everything here wraps the same primitives the research scripts use
(HuggingFace ESM-2, mean-pooled hidden states, top-k clustering, GO
enrichment) so the numbers on the demo site come from real model runs.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO_ROOT / "demo"
CACHE_DIR = Path(os.environ.get("PRISM_DEMO_CACHE", DEMO_ROOT / ".cache"))
WEB_DATA_DIR = DEMO_ROOT / "web" / "public" / "data"
BACKEND_ASSETS_DIR = DEMO_ROOT / "backend" / "assets"

sys.path.insert(0, str(REPO_ROOT))

MODELS = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}
ATLAS_MODEL = "35M"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SEED = 42

UNIPROT = "https://rest.uniprot.org/uniprotkb"
GO_PATTERN = re.compile(r"(.+?) \[(GO:\d{7})\]")
GO_COLUMNS = {
    "BP": "Gene Ontology (biological process)",
    "MF": "Gene Ontology (molecular function)",
    "CC": "Gene Ontology (cellular component)",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dirs() -> None:
    for d in (CACHE_DIR, WEB_DATA_DIR, BACKEND_ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj, compact: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if compact:
            json.dump(obj, f, separators=(",", ":"))
        else:
            json.dump(obj, f, indent=2)
    log(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def rnd(x, nd: int = 3):
    """Round floats (recursively) so the JSON stays small."""
    if isinstance(x, (list, tuple)):
        return [rnd(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return [rnd(v, nd) for v in x.tolist()]
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if not np.isfinite(v):
            return None
        return round(v, nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def parse_go(row: Dict) -> List[tuple]:
    """Return [(go_id, name, namespace)] from SwissProt GO columns."""
    out = []
    for ns, col in GO_COLUMNS.items():
        value = row.get(col) or ""
        if value == "None":
            continue
        for part in value.split("; "):
            m = GO_PATTERN.match(part.strip())
            if m:
                out.append((m.group(2), m.group(1), ns))
    return out


def parse_families(row: Dict) -> List[str]:
    value = row.get("Protein families") or ""
    if not value or value == "None":
        return []
    fams = []
    for part in re.split(r"[;,] ", value):
        part = part.strip().rstrip(".")
        if part and "family" in part.lower():
            fams.append(part[0].upper() + part[1:])
    return fams


def short_location(row: Dict) -> str:
    value = row.get("Subcellular location [CC]") or ""
    if not value or value == "None":
        return ""
    value = value.replace("SUBCELLULAR LOCATION: ", "")
    value = re.sub(r"\{[^}]*\}", "", value)
    value = re.sub(r"\[[^\]]*\]:", "", value)
    first = value.split(".")[0].strip()
    return first[:60]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(key: str, masked_lm: bool = False):
    import torch
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    name = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    cls = AutoModelForMaskedLM if masked_lm else AutoModel
    model = cls.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    return tok, model


def mean_pooled_hidden_states(
    tok, model, sequences: Sequence[str], max_residues: int = 512, batch_tokens: int = 8192
) -> np.ndarray:
    """Mean-pooled hidden states for every layer: [n_layers+1, n_seq, hidden].

    Pooling mirrors scripts/run_single_neuron.py (attention-mask mean).
    Sequences are length-bucketed so CPU batches carry little padding.
    """
    import torch

    order = np.argsort([len(s) for s in sequences])
    n_layers = model.config.num_hidden_layers + 1
    out = np.zeros((n_layers, len(sequences), model.config.hidden_size), dtype=np.float32)
    lengths = np.array([min(len(s), max_residues) + 2 for s in sequences])[order]
    i = 0
    n_batches = 0
    t0 = time.time()
    while i < len(order):
        bs = 1
        # ascending lengths: the last sequence in the batch is the longest
        while i + bs < len(order) and lengths[i + bs] * (bs + 1) <= batch_tokens:
            bs += 1
        idx = order[i : i + bs]
        batch = [sequences[j][:max_residues] for j in idx]
        enc = tok(batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).float()
        denom = mask.sum(1).clamp(min=1e-9)
        for l, h in enumerate(hs):
            out[l, idx] = ((h * mask).sum(1) / denom).numpy()
        i += bs
        n_batches += 1
        if n_batches % 25 == 0:
            log(f"  pooled {i}/{len(order)} ({time.time() - t0:.0f}s)")
    return np.nan_to_num(out)


def residue_hidden_states(tok, model, sequence: str, max_residues: int = 512) -> np.ndarray:
    """Per-residue hidden states for all layers: [n_layers+1, L, hidden] (special tokens removed)."""
    import torch

    enc = tok([sequence[:max_residues]], return_tensors="pt")
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    return np.stack([h[0, 1:-1].numpy() for h in hs])


def masked_marginals(tok, model_mlm, sequence: str, batch: int = 32) -> Dict[str, np.ndarray]:
    """Mask each residue in turn; return P(native aa) and entropy per position."""
    import torch

    enc = tok([sequence], return_tensors="pt")
    ids = enc["input_ids"][0]
    L = len(sequence)
    aa_ids = [tok.convert_tokens_to_ids(a) for a in AMINO_ACIDS]
    p_native = np.zeros(L)
    entropy = np.zeros(L)
    for start in range(0, L, batch):
        pos = list(range(start, min(L, start + batch)))
        x = ids.unsqueeze(0).repeat(len(pos), 1)
        for r, p in enumerate(pos):
            x[r, p + 1] = tok.mask_token_id
        with torch.no_grad():
            logits = model_mlm(input_ids=x).logits
        for r, p in enumerate(pos):
            probs = torch.softmax(logits[r, p + 1, aa_ids], -1).numpy()
            native = AMINO_ACIDS.find(sequence[p])
            p_native[p] = probs[native] if native >= 0 else np.nan
            entropy[p] = float(-(probs * np.log(probs + 1e-12)).sum())
    return {"p_native": p_native, "entropy": entropy}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


class Enricher:
    """Hypergeometric term enrichment of a study set against a fixed population.

    Mirrors src/analysis/go_enrichment.perform_go_enrichment (one-sided
    over-representation, BH-FDR) but vectorised so thousands of neurons
    can be profiled in minutes on a laptop CPU.
    """

    def __init__(self, protein_terms: List[List[int]], n_terms: int, min_pop: int = 3):
        from scipy.sparse import csr_matrix

        rows, cols = [], []
        for i, terms in enumerate(protein_terms):
            for t in set(terms):
                rows.append(i)
                cols.append(t)
        M = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(protein_terms), n_terms))
        self.pop_counts = np.asarray(M.sum(0)).ravel()
        self.keep = np.where(self.pop_counts >= min_pop)[0]
        self.M = M[:, self.keep].tocsc().tocsr()
        self.K = self.pop_counts[self.keep]
        self.N = len(protein_terms)

    def run(self, study: np.ndarray, min_count: int = 3, max_terms: int = 5):
        from scipy.stats import hypergeom

        n = len(study)
        k = np.asarray(self.M[study].sum(0)).ravel()
        tested = np.where(k >= 1)[0]
        if len(tested) == 0:
            return []
        p = hypergeom.sf(k[tested] - 1, self.N, self.K[tested], n)
        # BH across every term present in the population (conservative)
        q_all = np.ones(len(self.K))
        q_all[tested] = p
        q_all = bh_fdr(q_all)
        res = []
        for j in tested[np.argsort(p)]:
            if k[j] < min_count:
                continue
            fold = (k[j] / n) / (self.K[j] / self.N)
            res.append((int(self.keep[j]), int(k[j]), float(q_all[j]), float(fold)))
            if len(res) >= max_terms:
                break
        return res


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def fast_auroc_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC of every column of X for binary y via rank statistics."""
    from scipy.stats import rankdata

    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.full(X.shape[1], np.nan)
    ranks = rankdata(X, axis=0)
    return (ranks[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------------------
# UniProt / AlphaFold
# ---------------------------------------------------------------------------

def http_get(url: str, params: Optional[dict] = None, retries: int = 4, **kw):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60, **kw)
            if r.status_code == 200:
                return r
            log(f"  HTTP {r.status_code} for {url}")
        except requests.RequestException as e:
            log(f"  request error {e}")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {url}")


def uniprot_entry(acc: str) -> dict:
    cache = CACHE_DIR / "uniprot" / f"{acc}.json"
    if cache.exists():
        return read_json(cache)
    data = http_get(f"{UNIPROT}/{acc}.json").json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def uniprot_search(query: str, fields: str, max_results: int) -> List[dict]:
    """Paginated UniProt search returning JSON results."""
    key = re.sub(r"[^A-Za-z0-9]+", "_", f"{query}_{fields}_{max_results}")[:150]
    cache = CACHE_DIR / "uniprot" / f"search_{key}.json"
    if cache.exists():
        return read_json(cache)
    url = f"{UNIPROT}/search"
    params = {"query": query, "fields": fields, "size": 500, "format": "json"}
    results: List[dict] = []
    while url and len(results) < max_results:
        r = http_get(url, params=params)
        results.extend(r.json()["results"])
        url = r.links.get("next", {}).get("url")
        params = None
    results = results[:max_results]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(results))
    return results


FEATURE_TYPES = {
    "Helix": "helix",
    "Beta strand": "strand",
    "Turn": "turn",
    "Transmembrane": "transmembrane",
    "Signal": "signal",
    "Binding site": "binding",
    "Active site": "active",
    "Motif": "motif",
    "Zinc finger": "zinc_finger",
    "Disulfide bond": "disulfide",
    "Domain": "domain",
    "Region": "region",
    "Modified residue": "modified",
    "Site": "site",
    "DNA binding": "dna_binding",
    "Chain": "chain",
    "Propeptide": "propeptide",
}


def residue_masks(entry: dict, L: int) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for f in entry.get("features", []):
        kind = FEATURE_TYPES.get(f["type"])
        if not kind:
            continue
        loc = f["location"]
        try:
            s = int(loc["start"]["value"]) - 1
            e = int(loc["end"]["value"])
        except (TypeError, ValueError, KeyError):
            continue
        m = masks.setdefault(kind, np.zeros(L, dtype=bool))
        if kind == "disulfide":
            # disulfide features span the bonded pair; mark just the two cysteines
            for p in (s, e - 1):
                if 0 <= p < L:
                    m[p] = True
        else:
            m[max(0, s) : min(L, e)] = True
    return masks


def feature_list(entry: dict) -> List[dict]:
    feats = []
    for f in entry.get("features", []):
        kind = FEATURE_TYPES.get(f["type"])
        if not kind or kind in ("chain", "modified"):
            continue
        loc = f["location"]
        try:
            s = int(loc["start"]["value"])
            e = int(loc["end"]["value"])
        except (TypeError, ValueError, KeyError):
            continue
        desc = f.get("description") or ""
        lig = (f.get("ligand") or {}).get("name")
        if lig:
            desc = f"{lig}" + (f" ({desc})" if desc else "")
        feats.append({"type": kind, "start": s, "end": e, "label": desc})
    return feats


def alphafold_pdb(acc: str) -> Optional[str]:
    cache = CACHE_DIR / "alphafold" / f"{acc}.pdb"
    if cache.exists():
        return cache.read_text()
    try:
        meta = http_get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}").json()[0]
        pdb = http_get(meta["pdbUrl"]).text
    except Exception as e:  # noqa: BLE001
        log(f"  no AlphaFold model for {acc}: {e}")
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(pdb)
    return pdb


def slim_pdb(pdb: str) -> str:
    """Keep ATOM records only (drops headers/HETATM) to shrink the payload."""
    keep = [ln for ln in pdb.splitlines() if ln.startswith(("ATOM", "TER"))]
    return "\n".join(keep) + "\nEND\n"


def chunks(xs: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]
