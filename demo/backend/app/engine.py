"""Inference engine behind the PRISM-Bio live demo API.

Loads ESM-2 35M once and serves three analyses on CPU:
  * analyze: per-residue neuron activity, structure-probe predictions,
    masked-marginal residue plausibility and an atlas "fingerprint"
  * steer: inject a concept vector (src/steering) and read out what the
    model now predicts at every position
All reference data (concept vectors, probes, atlas statistics) is produced
by demo/pipeline and shipped in demo/backend/assets.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ASSETS = Path(os.environ.get("PRISM_ASSETS", Path(__file__).resolve().parents[1] / "assets"))
REPO_ROOT = Path(os.environ.get("PRISM_REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO_ROOT))

from src.steering.analysis import compare_hidden_states  # noqa: E402
from src.steering.hooks import SteeringHook  # noqa: E402

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/esm2_t12_35M_UR50D")
MAX_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", "400"))


def clean_sequence(raw: str) -> str:
    """Accept raw or FASTA input; return an upper-case amino-acid string."""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip() and not ln.startswith(">")]
    seq = re.sub(r"[\s\d*]", "", "".join(lines)).upper()
    if not seq:
        raise ValueError("Sequence is empty.")
    bad = sorted(set(seq) - set(AMINO_ACIDS + "XBZUO"))
    if bad:
        raise ValueError(f"Unsupported characters in sequence: {''.join(bad)}")
    if len(seq) < 10:
        raise ValueError("Sequence must be at least 10 residues.")
    if len(seq) > MAX_LENGTH:
        raise ValueError(f"Sequence is {len(seq)} residues; the live demo accepts up to {MAX_LENGTH}.")
    return seq


def _r(x, nd=3):
    return [round(float(v), nd) for v in np.asarray(x).ravel()]


class Engine:
    def __init__(self):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden = self.model.config.hidden_size
        self.aa_ids = [self.tok.convert_tokens_to_ids(a) for a in AMINO_ACIDS]
        self.lock = threading.Lock()  # one forward pass at a time keeps memory flat on a small VM

        self.concepts = {c["key"]: c for c in json.loads((ASSETS / "concepts.json").read_text())}
        for c in self.concepts.values():
            c["tensor"] = torch.tensor(c["vector"], dtype=torch.float32)
            c["logodds_np"] = np.array(c["logodds"])
        self.probes = json.loads((ASSETS / "probes.json").read_text())
        for p in self.probes.values():
            for k in ("mean", "scale", "coef"):
                p[k] = np.array(p[k])
        self.probe_neurons = json.loads((ASSETS / "probe_neurons.json").read_text())
        atlas = np.load(ASSETS / "atlas_stats.npz")
        self.atlas_mean, self.atlas_std = atlas["mean"], atlas["std"] + 1e-6
        res = np.load(ASSETS / "residue_stats.npz")
        self.res_mean, self.res_std = res["mean"], res["std"]
        self.labels = json.loads((ASSETS / "atlas_labels.json").read_text())

    # -- helpers -----------------------------------------------------------
    def _forward(self, input_ids: torch.Tensor):
        with torch.no_grad():
            return self.model(input_ids=input_ids, output_hidden_states=True)

    def _marginals(self, ids: torch.Tensor, seq: str, batch: int = 48):
        L = len(seq)
        p_native = np.zeros(L)
        entropy = np.zeros(L)
        for start in range(0, L, batch):
            pos = list(range(start, min(L, start + batch)))
            x = ids.repeat(len(pos), 1)
            for r, p in enumerate(pos):
                x[r, p + 1] = self.tok.mask_token_id
            with torch.no_grad():
                logits = self.model(input_ids=x).logits
            probs = torch.softmax(logits[torch.arange(len(pos)), torch.tensor(pos) + 1][:, self.aa_ids], -1).numpy()
            for r, p in enumerate(pos):
                i = AMINO_ACIDS.find(seq[p])
                p_native[p] = probs[r, i] if i >= 0 else np.nan
                entropy[p] = -(probs[r] * np.log(probs[r] + 1e-12)).sum()
        return p_native, entropy

    # -- public API --------------------------------------------------------
    def info(self) -> Dict:
        return {"model": MODEL_NAME, "layers": self.n_layers, "hidden": self.hidden, "max_length": MAX_LENGTH,
                "concepts": [{"key": c["key"], "title": c["title"], "layer": c["layer"]} for c in self.concepts.values()]}

    def analyze(self, seq: str, layer: int = 6, units: Optional[List[int]] = None, marginals: bool = True) -> Dict:
        layer = int(np.clip(layer, 1, self.n_layers))
        enc = self.tok([seq], return_tensors="pt")
        with self.lock:
            out = self._forward(enc["input_ids"])
            p_native, entropy = self._marginals(enc["input_ids"], seq) if marginals else (None, None)
        hs = np.stack([h[0].numpy() for h in out.hidden_states])  # [n_layers+1, L+2, H]
        res = hs[:, 1:-1]
        pooled = hs[1:].mean(1)

        probes = {}
        for k, p in self.probes.items():
            z = (res[p["layer"]] - p["mean"]) / p["scale"]
            probes[k] = _r(1 / (1 + np.exp(-(z @ p["coef"] + p["intercept"]))), 2)

        tracks = []
        seen = set()

        def add(l, u, kind, note):
            if (l, u) in seen:
                return
            seen.add((l, u))
            v = res[l, :, u]
            label = self.labels[(l - 1) * self.hidden + u]
            tracks.append({"l": l, "u": u, "kind": kind, "note": note, "v": _r(v, 2),
                           "z": round(float(((v - self.res_mean[l, u]) / self.res_std[l, u]).max()), 2),
                           "label": label[2]})

        for k, rows in self.probe_neurons.items():
            best = max(rows, key=lambda r: r["auc"])
            if best["sign"] > 0:
                add(best["layer"], best["u"], "probe", f"{k.capitalize()} neuron (held-out AUROC {best['auc']:.2f})")
        z = ((res[layer] - self.res_mean[layer]) / self.res_std[layer]).max(0)
        for u in np.argsort(z)[-6:][::-1]:
            add(layer, int(u), "top", f"Fires strongly on this protein (peak z = {z[u]:.1f})")
        for u in units or []:
            if 0 <= u < self.hidden:
                add(layer, int(u), "requested", "Requested neuron")

        zp = (pooled - self.atlas_mean) / self.atlas_std
        fingerprint = []
        for idx in np.argsort(zp.ravel())[::-1]:
            l, u = divmod(int(idx), self.hidden)
            label = self.labels[l * self.hidden + u]
            if label[2]:
                fingerprint.append({"l": l + 1, "u": u, "z": round(float(zp[l, u]), 2), "label": label[2], "q": label[3]})
            if len(fingerprint) >= 8:
                break

        return {"sequence": seq, "length": len(seq), "layer": layer, "probes": probes, "tracks": tracks,
                "fingerprint": fingerprint,
                "p_native": _r(p_native, 3) if p_native is not None else None,
                "entropy": _r(entropy, 2) if entropy is not None else None}

    def steer(self, seq: str, concept_key: str, strength: float) -> Dict:
        c = self.concepts.get(concept_key)
        if c is None:
            raise ValueError(f"Unknown concept '{concept_key}'. Options: {', '.join(self.concepts)}")
        strength = float(np.clip(strength, -3.0, 4.0))
        layer = c["layer"]
        enc = self.tok([seq], return_tensors="pt")
        with self.lock:
            base = self._forward(enc["input_ids"])
            h_norm = base.hidden_states[layer + 1][0].norm(dim=-1).mean().item()
            mult = strength * h_norm / c["tensor"].norm().item()
            with SteeringHook(self.model, layer, c["tensor"], multiplier=mult):
                steered = self._forward(enc["input_ids"])
        probs = torch.softmax(steered.logits[0, 1:-1][:, self.aa_ids], -1).numpy()
        argmax = "".join(AMINO_ACIDS[i] for i in probs.argmax(1))
        native = np.array([AMINO_ACIDS.find(a) for a in seq])
        valid = native >= 0
        p_native = probs[np.arange(len(seq))[valid], native[valid]].mean() if valid.any() else float("nan")
        cscore = probs @ c["logodds_np"]
        drift = compare_hidden_states(list(base.hidden_states), list(steered.hidden_states))
        result = {
            "concept": concept_key, "layer": layer, "r": strength, "mult": round(mult, 3), "seq": argmax,
            "identity": round(float(np.mean([a == b for a, b in zip(argmax, seq)])), 3),
            "p_native": round(float(p_native), 3), "concept_score": round(float(cscore.mean()), 3),
            "per_pos": _r(cscore, 2),
            "drift": [round(drift["per_layer"][f"layer_{i}"]["cosine_similarity"], 4) for i in range(drift["n_layers"])],
        }
        if c.get("pattern"):
            result["motifs"] = [[m.start(), m.end()] for m in re.finditer(c["pattern"], argmax)]
        return result
