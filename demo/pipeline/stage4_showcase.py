"""Stage 4: per-residue neuron activity for the showcase proteins.

For each protein we store:
  * UniProt features and the AlphaFold structure (for the 3D viewer)
  * masked-marginal P(native residue) from ESM-2 (the model's "surprise")
  * linear-probe predictions for helix / strand / transmembrane
  * activation tracks for selected neurons:
      - probe neurons (selected on held-out proteins in stage 3)
      - feature detectors (best in-protein AUROC for each annotated feature)
      - the most strongly firing neurons at each layer
  * a pooled "fingerprint": neurons this protein drives furthest above
    the atlas background, with their atlas descriptions
"""
import numpy as np

from common import (ATLAS_MODEL, BACKEND_ASSETS_DIR, WEB_DATA_DIR, alphafold_pdb, fast_auroc_matrix, feature_list,
                    load_model, log, masked_marginals, mean_pooled_hidden_states, read_json, residue_hidden_states, residue_masks, rnd, slim_pdb,
                    uniprot_entry, write_json)
from showcase import SHOWCASE

DETECTOR_FEATURES = ["binding", "active", "motif", "zinc_finger", "transmembrane", "helix", "strand", "disulfide",
                     "signal", "dna_binding", "propeptide"]


def protein_name(entry):
    try:
        return entry["proteinDescription"]["recommendedName"]["fullName"]["value"]
    except KeyError:
        return entry.get("uniProtkbId", "")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    tok, model = load_model(ATLAS_MODEL, masked_lm=True)
    probing = read_json(WEB_DATA_DIR / "probing.json")
    probes = read_json(BACKEND_ASSETS_DIR / "probes.json")
    stats = np.load(BACKEND_ASSETS_DIR / "atlas_stats.npz")
    labels = read_json(BACKEND_ASSETS_DIR / "atlas_labels.json")
    hidden = stats["mean"].shape[1]

    items = []
    for sp in SHOWCASE:
        entry = uniprot_entry(sp["acc"])
        seq = entry["sequence"]["value"][:512]
        hs = residue_hidden_states(tok, model, seq)  # [13, L, H]
        items.append((sp, entry, seq, hs))
    # residue-level background per unit, pooled over all showcase residues
    allres = np.concatenate([it[3] for it in items], axis=1)  # [13, R, H]
    res_mean, res_std = allres.mean(1), allres.std(1) + 1e-6
    np.savez_compressed(BACKEND_ASSETS_DIR / "residue_stats.npz", mean=res_mean.astype(np.float32),
                        std=res_std.astype(np.float32))

    index = []
    for sp, entry, seq, hs in items:
        L = len(seq)
        log(f"{sp['short']} (L={L})")
        masks = residue_masks(entry, L)
        mm = masked_marginals(tok, model, seq)
        probe_out = {}
        for k, p in probes.items():
            z = (hs[p["layer"]] - np.array(p["mean"])) / np.array(p["scale"])
            probe_out[k] = rnd(sigmoid(z @ np.array(p["coef"]) + p["intercept"]), 2)

        tracks = {}

        def add(l, u, kind, note, auc=None, feature=None):
            key = f"{l}-{u}"
            if key in tracks:
                if kind == "feature" and tracks[key]["kind"] != "feature":
                    tracks[key].update(kind=kind, note=note, auc=auc, feature=feature)
                return
            v = hs[l, :, u]
            tracks[key] = {"l": int(l), "u": int(u), "kind": kind, "note": note, "auc": auc, "feature": feature,
                           "v": rnd(v, 2), "z": rnd(((v - res_mean[l, u]) / res_std[l, u]).max(), 2)}

        # feature detectors: best neuron within this protein for each annotated feature
        for feat in DETECTOR_FEATURES:
            m = masks.get(feat)
            if m is None or m.sum() < 2 or m.mean() > 0.75:
                continue
            best = []
            for l in range(1, hs.shape[0]):
                a = fast_auroc_matrix(hs[l], m)
                u = int(np.nanargmax(a))
                best.append((a[u], l, u))
            best.sort(reverse=True)
            for auc, l, u in best[:2]:
                add(l, u, "feature", f"Best in-protein match for {feat.replace('_', ' ')}", rnd(auc), feat)

        # probe neurons selected on independent proteins (stage 3)
        for row in probing["layers"][1:]:
            for k in ("helix", "strand", "transmembrane"):
                n = row[k]["neuron"]
                if n["sign"] > 0:
                    add(row["layer"], n["u"], "probe", f"{k.capitalize()} neuron (held-out AUROC {n['auc']:.2f})",
                        n["auc"], k)

        # strongest firing neurons per layer (residue z-score)
        for l in range(1, hs.shape[0]):
            z = ((hs[l] - res_mean[l]) / res_std[l]).max(0)
            for u in np.argsort(z)[-2:][::-1]:
                add(l, int(u), "top", f"Fires strongly on this protein (peak z = {z[u]:.1f})")

        # pooled fingerprint vs atlas background
        pooled = mean_pooled_hidden_states(tok, model, [seq])[1:, 0]  # same pooling as the atlas
        zp = (pooled - stats["mean"]) / (stats["std"] + 1e-6)
        flat = np.argsort(zp.ravel())[::-1]
        fingerprint = []
        for idx in flat:
            l, u = divmod(int(idx), hidden)
            label = labels[l * hidden + u]
            if label[2]:
                fingerprint.append({"l": l + 1, "u": u, "z": rnd(zp[l, u], 2), "label": label[2], "q": label[3]})
            if len(fingerprint) >= 8:
                break

        pdb = alphafold_pdb(sp["acc"])
        has_structure = pdb is not None
        if has_structure:
            (WEB_DATA_DIR / "structures").mkdir(parents=True, exist_ok=True)
            (WEB_DATA_DIR / "structures" / f"{sp['acc']}.pdb").write_text(slim_pdb(pdb))

        organism = entry.get("organism", {}).get("scientificName", "")
        out = {
            "acc": sp["acc"], "short": sp["short"], "title": sp["title"], "name": protein_name(entry),
            "organism": organism, "blurb": sp["blurb"], "focus": sp["focus"], "sequence": seq,
            "features": feature_list(entry), "structure": has_structure,
            "p_native": rnd(mm["p_native"], 3), "entropy": rnd(mm["entropy"], 2),
            "probes": probe_out,
            "tracks": sorted(tracks.values(), key=lambda t: ({"feature": 0, "probe": 1, "top": 2}[t["kind"]], t["l"], t["u"])),
            "fingerprint": fingerprint,
        }
        write_json(WEB_DATA_DIR / "showcase" / f"{sp['acc']}.json", out)
        index.append({"acc": sp["acc"], "short": sp["short"], "title": sp["title"], "organism": organism,
                      "length": L, "blurb": sp["blurb"]})
    write_json(WEB_DATA_DIR / "showcase" / "index.json", index)


if __name__ == "__main__":
    main()
