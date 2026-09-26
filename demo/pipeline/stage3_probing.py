"""Stage 3: layer-wise probing for protein structure inside ESM-2 35M.

ESM-2 is trained only on sequences, yet secondary structure and membrane
topology become linearly decodable from its residual stream. For every
layer we fit (a) a linear probe and (b) find the single best neuron for
alpha-helix, beta-strand and transmembrane residues, using experimentally
derived UniProt annotations (PDB secondary structure, TM segments).
Proteins are split 80/20 by protein, so test residues come from unseen proteins.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import (ATLAS_MODEL, BACKEND_ASSETS_DIR, CACHE_DIR, SEED, WEB_DATA_DIR, fast_auroc_matrix, load_model, log,
                    residue_hidden_states, residue_masks, rnd, uniprot_search, write_json)
from showcase import SHOWCASE

N_STRUCT = 450
N_TM = 220
RES_PER_PROTEIN = 60
FIELDS = "accession,sequence,ft_helix,ft_strand,ft_transmem"


def collect(query, n, rng, min_cover=0.0):
    results = uniprot_search(query, FIELDS, 2500)
    rng.shuffle(results)
    exclude = {p["acc"] for p in SHOWCASE}
    chosen = []
    for r in results:
        acc = r["primaryAccession"]
        seq = r["sequence"]["value"]
        if acc in exclude or set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
            continue
        masks = residue_masks(r, len(seq))
        cover = (masks.get("helix", np.zeros(len(seq), bool)) | masks.get("strand", np.zeros(len(seq), bool))).mean()
        if cover < min_cover:
            continue
        chosen.append((acc, seq, masks))
        if len(chosen) >= n:
            break
    return chosen


def main():
    rng = np.random.default_rng(SEED)
    struct = collect("reviewed:true AND structure_3d:true AND length:[80 TO 400]", N_STRUCT, rng, min_cover=0.45)
    tm = collect("reviewed:true AND ft_transmem:* AND length:[80 TO 400]", N_TM, rng)
    log(f"structure set {len(struct)}, TM set {len(tm)}")
    tok, model = load_model(ATLAS_MODEL)

    def featurise(items):
        X, labels, groups = [], {"helix": [], "strand": [], "transmembrane": []}, []
        for gi, (acc, seq, masks) in enumerate(items):
            hs = residue_hidden_states(tok, model, seq)  # [13, L, H]
            L = hs.shape[1]
            pick = rng.choice(L, size=min(RES_PER_PROTEIN, L), replace=False)
            X.append(hs[:, pick].astype(np.float16))
            for k in labels:
                labels[k].append(masks.get(k, np.zeros(L, bool))[pick])
            groups.append(np.full(len(pick), gi))
            if gi % 100 == 0:
                log(f"  featurised {gi}/{len(items)}")
        return np.concatenate(X, 1), {k: np.concatenate(v) for k, v in labels.items()}, np.concatenate(groups)

    Xs, ys, gs = featurise(struct)
    Xt, yt, gt = featurise(tm)

    def split(groups):
        ids = np.unique(groups)
        r = np.random.default_rng(SEED)
        test_ids = set(r.choice(ids, size=len(ids) // 5, replace=False).tolist())
        test = np.array([g in test_ids for g in groups])
        return ~test, test

    tr_s, te_s = split(gs)
    tr_t, te_t = split(gt)
    tasks = {
        "helix": (Xs, ys["helix"], tr_s, te_s),
        "strand": (Xs, ys["strand"], tr_s, te_s),
        "transmembrane": (Xt, yt["transmembrane"], tr_t, te_t),
    }
    n_layers = Xs.shape[0]
    layers = []
    best = {k: (-1, None) for k in tasks}
    for l in range(n_layers):
        row = {"layer": l}
        for name, (X, y, tr, te) in tasks.items():
            Xl = X[l].astype(np.float32)
            scaler = StandardScaler().fit(Xl[tr])
            Ztr, Zte = scaler.transform(Xl[tr]), scaler.transform(Xl[te])
            clf = LogisticRegression(C=0.05, max_iter=3000).fit(Ztr, y[tr])
            auc = roc_auc_score(y[te], clf.decision_function(Zte))
            acc = float((clf.predict(Zte) == y[te]).mean())
            neuron_auc = fast_auroc_matrix(Xl[tr], y[tr])
            strength = np.abs(neuron_auc - 0.5)
            u = int(np.nanargmax(strength))
            sign = 1 if neuron_auc[u] >= 0.5 else -1
            test_auc = roc_auc_score(y[te], sign * Xl[te, u])
            row[name] = {"auc": rnd(auc), "acc": rnd(acc), "neuron": {"u": u, "sign": sign, "auc": rnd(test_auc)},
                         "base_rate": rnd(y[te].mean())}
            if auc > best[name][0]:
                best[name] = (auc, {"layer": l, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                                    "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "auc": float(auc)})
        layers.append(row)
        log(f"layer {l}: " + ", ".join(f"{k} AUROC {row[k]['auc']:.3f} (neuron {row[k]['neuron']['u']} {row[k]['neuron']['auc']:.2f})" for k in tasks))

    out = {
        "model": "facebook/esm2_t12_35M_UR50D",
        "n_struct_proteins": len(struct), "n_tm_proteins": len(tm),
        "n_residues": int(Xs.shape[1] + Xt.shape[1]),
        "layers": layers,
        "best_layer": {k: v[1]["layer"] for k, v in best.items()},
    }
    write_json(WEB_DATA_DIR / "probing.json", out)
    write_json(BACKEND_ASSETS_DIR / "probes.json", {k: v[1] for k, v in best.items()})
    write_json(BACKEND_ASSETS_DIR / "probe_neurons.json",
               {k: [dict(layer=row["layer"], **row[k]["neuron"]) for row in layers[1:]] for k in tasks})


if __name__ == "__main__":
    main()
