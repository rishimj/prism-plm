"""Scaling study: does polysemanticity change with ESM-2 model size?

Replicates scripts/run_neuron_batch.py (top-k=100, k-means k=5, PCA-normalised
embeddings, centroid-cosine polysemanticity) across four ESM-2 sizes on the
same 1,000 SwissProt proteins, sampling 128 residual-stream units per layer.
"""
import gc

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from common import CACHE_DIR, MODELS, SEED, WEB_DATA_DIR, Enricher, load_model, log, mean_pooled_hidden_states, read_json, rnd, write_json
from scripts.run_single_neuron import compute_polysemanticity

N_PROTEINS = 1000
MAX_RES = 256
UNITS_PER_LAYER = 128
TOP_K = 100
N_CLUSTERS = 5
PCA_DIMS = 64


def build_terms(proteins):
    vocab = {}
    per_protein = []
    for p in proteins:
        ids = []
        for go_id, name, ns in p["go"]:
            ids.append(vocab.setdefault(go_id, len(vocab)))
        for fam in p["families"]:
            ids.append(vocab.setdefault("FAM:" + fam, len(vocab)))
        per_protein.append(ids)
    return per_protein, len(vocab)


def neuron_scores(pooled_layer, pca, unit, enricher):
    acts = pooled_layer[:, unit]
    top = np.argsort(acts)[-TOP_K:][::-1]
    emb = pca.transform(pooled_layer[top])
    labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, max_iter=300, random_state=SEED).fit_predict(emb)
    poly = compute_polysemanticity(emb, labels)["polysemanticity_score"]
    terms = enricher.run(top, min_count=3, max_terms=1)
    best_q = terms[0][2] if terms else 1.0
    return poly, best_q


def main():
    proteins = read_json(CACHE_DIR / "sample.json")[:N_PROTEINS]
    seqs = [p["sequence"] for p in proteins]
    per_protein, n_terms = build_terms(proteins)
    enricher = Enricher(per_protein, n_terms)
    out = {"n_proteins": N_PROTEINS, "top_k": TOP_K, "n_clusters": N_CLUSTERS, "pca_dims": PCA_DIMS,
           "units_per_layer": UNITS_PER_LAYER, "max_residues": MAX_RES, "models": []}
    for key in ["8M", "35M", "150M", "650M"]:
        cache = CACHE_DIR / f"scaling_pooled_{key}.npy"
        if cache.exists():
            pooled = np.load(cache).astype(np.float32)
        else:
            log(f"[{key}] loading {MODELS[key]}")
            tok, model = load_model(key)
            pooled = mean_pooled_hidden_states(tok, model, seqs, max_residues=MAX_RES, batch_tokens=4096)
            np.save(cache, pooled.astype(np.float16))
            del model
            gc.collect()
        n_layers = pooled.shape[0] - 1
        hidden = pooled.shape[2]
        rng = np.random.default_rng(SEED)
        layers = []
        for l in range(1, n_layers + 1):
            X = pooled[l]
            pca = PCA(n_components=PCA_DIMS, random_state=SEED).fit(X)
            units = rng.choice(hidden, size=min(UNITS_PER_LAYER, hidden), replace=False)
            polys, qs = [], []
            for u in units:
                p, q = neuron_scores(X, pca, int(u), enricher)
                polys.append(p)
                qs.append(q)
            polys, qs = np.array(polys), np.array(qs)
            layers.append({
                "layer": l,
                "depth": rnd(l / n_layers),
                "poly": rnd(polys),
                "median": rnd(np.median(polys)),
                "q25": rnd(np.percentile(polys, 25)),
                "q75": rnd(np.percentile(polys, 75)),
                "sig_frac": rnd(float((qs < 0.05).mean())),
                "pca_var": rnd(float(pca.explained_variance_ratio_.sum())),
            })
            log(f"[{key}] layer {l}/{n_layers} median poly {np.median(polys):.3f} sig {np.mean(qs < 0.05):.2f}")
        all_poly = np.concatenate([np.array(x["poly"], dtype=float) for x in layers])
        out["models"].append({
            "key": key,
            "name": MODELS[key],
            "params": {"8M": 8e6, "35M": 35e6, "150M": 150e6, "650M": 650e6}[key],
            "n_layers": n_layers,
            "hidden": hidden,
            "layers": layers,
            "median": rnd(np.median(all_poly)),
            "mean": rnd(np.mean(all_poly)),
            "sig_frac": rnd(float(np.mean([x["sig_frac"] for x in layers]))),
        })
        write_json(WEB_DATA_DIR / "scaling.json", out)


if __name__ == "__main__":
    main()
