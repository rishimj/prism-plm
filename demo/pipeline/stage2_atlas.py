"""Stage 2: neuron atlas for every residual-stream unit of ESM-2 35M (12 layers x 480 units).

For each neuron this runs the PRISM-Bio single-neuron recipe
(scripts/run_single_neuron.py): take the top-100 activating proteins,
cluster their embeddings with k-means (k=5) in a PCA-64 space, score
polysemanticity as 1 - mean cosine similarity between cluster centroids,
then describe the neuron (and each cluster) with GO / protein-family
enrichment against the 3,000-protein background.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from common import (ATLAS_MODEL, BACKEND_ASSETS_DIR, CACHE_DIR, SEED, WEB_DATA_DIR, Enricher, load_model, log,
                    mean_pooled_hidden_states, read_json, rnd, write_json)
from scripts.run_single_neuron import compute_polysemanticity

TOP_K = 100
N_CLUSTERS = 5
PCA_DIMS = 64
N_BINS = 24


def build_terms(proteins):
    vocab, table, per_protein = {}, [], []
    for p in proteins:
        ids = []
        for go_id, name, ns in p["go"]:
            if go_id not in vocab:
                vocab[go_id] = len(table)
                table.append([go_id, name, ns])
            ids.append(vocab[go_id])
        for fam in p["families"]:
            key = "FAM:" + fam
            if key not in vocab:
                vocab[key] = len(table)
                table.append(["", fam, "FAM"])
            ids.append(vocab[key])
        per_protein.append(ids)
    return table, per_protein


def main():
    proteins = read_json(CACHE_DIR / "sample.json")
    seqs = [p["sequence"] for p in proteins]
    cache = CACHE_DIR / "atlas_pooled_35M.npy"
    if cache.exists():
        pooled = np.load(cache)
    else:
        tok, model = load_model(ATLAS_MODEL)
        pooled = mean_pooled_hidden_states(tok, model, seqs, max_residues=512)
        np.save(cache, pooled)
    n_layers = pooled.shape[0] - 1
    hidden = pooled.shape[2]

    table, per_protein = build_terms(proteins)
    enricher = Enricher(per_protein, len(table))
    used_terms = set()

    write_json(WEB_DATA_DIR / "atlas" / "proteins.json", [
        [p["id"], p["name"], p["organism"], (p["families"][0] if p["families"] else ""), p["location"], len(p["sequence"])]
        for p in proteins
    ])

    index = {"model": "facebook/esm2_t12_35M_UR50D", "n_layers": n_layers, "hidden": hidden, "n_proteins": len(proteins),
             "top_k": TOP_K, "n_clusters": N_CLUSTERS, "pca_dims": PCA_DIMS, "layers": []}
    labels_for_backend = []
    for l in range(1, n_layers + 1):
        X = pooled[l]
        pca = PCA(n_components=PCA_DIMS, random_state=SEED).fit(X)
        neurons, compact = [], []
        for u in range(hidden):
            acts = X[:, u]
            top = np.argsort(acts)[-TOP_K:][::-1]
            emb = pca.transform(X[top])
            labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, max_iter=300, random_state=SEED).fit_predict(emb)
            poly = compute_polysemanticity(emb, labels)
            terms = enricher.run(top, min_count=3, max_terms=5)
            clusters = []
            for c in range(N_CLUSTERS):
                members = top[labels == c]
                if len(members) == 0:
                    continue
                ct = enricher.run(members, min_count=2, max_terms=1)
                clusters.append([
                    int(len(members)),
                    rnd(acts[members].mean()),
                    ct[0][0] if ct else -1,
                    float(f"{ct[0][2]:.3g}") if ct else 1.0,
                    [int(i) for i in members[:4]],
                ])
                if ct:
                    used_terms.add(ct[0][0])
            clusters.sort(key=lambda c: -c[0])
            lo, hi = float(acts.min()), float(acts.max())
            hist = np.histogram(acts, bins=N_BINS, range=(lo, hi))[0]
            for t in terms:
                used_terms.add(t[0])
            neurons.append({
                "u": u,
                "p": rnd(poly["polysemanticity_score"]),
                "m": rnd(acts.mean()),
                "sd": rnd(acts.std()),
                "h": hist.tolist(),
                "hr": [rnd(lo), rnd(hi)],
                "t": [[t[0], t[1], float(f"{t[2]:.3g}"), rnd(t[3], 2)] for t in terms],
                "c": clusters,
                "tp": [int(i) for i in top[:12]],
                "ta": rnd(acts[top[:12]]),
            })
            best = terms[0] if terms else None
            compact.append([rnd(poly["polysemanticity_score"]), best[0] if best else -1,
                            rnd(-np.log10(max(best[2], 1e-300)), 2) if best else 0.0])
            labels_for_backend.append([l, u, table[best[0]][1] if best and best[2] < 0.05 else "",
                                       float(f"{best[2]:.3g}") if best else 1.0])
        write_json(WEB_DATA_DIR / "atlas" / f"layer-{l:02d}.json", {"layer": l, "pca_var": rnd(pca.explained_variance_ratio_.sum()), "neurons": neurons})
        polys = np.array([c[0] for c in compact])
        sig = np.array([c[2] for c in compact]) > -np.log10(0.05)
        index["layers"].append({"layer": l, "neurons": compact, "median_poly": rnd(np.median(polys)), "sig_frac": rnd(sig.mean())})
        log(f"layer {l}: median poly {np.median(polys):.3f}, {sig.mean():.0%} neurons with significant label")

    # term table (only terms referenced anywhere, but keep original indices)
    terms_out = [t if i in used_terms else None for i, t in enumerate(table)]
    write_json(WEB_DATA_DIR / "atlas" / "terms.json", terms_out)
    write_json(WEB_DATA_DIR / "atlas" / "index.json", index)

    # stats the live backend uses to z-score a new protein against the atlas
    np.savez_compressed(BACKEND_ASSETS_DIR / "atlas_stats.npz",
                        mean=pooled[1:].mean(1).astype(np.float32), std=pooled[1:].std(1).astype(np.float32))
    write_json(BACKEND_ASSETS_DIR / "atlas_labels.json", labels_for_backend)


if __name__ == "__main__":
    main()
