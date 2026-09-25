"""Stage 5: activation steering with the repo's own src/steering module.

Concept vectors are derived exactly as scripts/run_activation_steering.py does
(SteeringVector: mean activation of a positive set minus a negative set) and
injected with SteeringHook. For each concept x target protein x strength we
record what ESM-2 now predicts at every position, how far each layer's
representation drifts (compare_hidden_states), and, for motif concepts, the
repo's concept-probability-shift metric.
"""
import re

import numpy as np
import torch

from common import (ATLAS_MODEL, AMINO_ACIDS, BACKEND_ASSETS_DIR, CACHE_DIR, SEED, WEB_DATA_DIR, load_model, log,
                    read_json, rnd, uniprot_entry, write_json)
from showcase import SHOWCASE, STEERING_TARGETS
from src.steering import SteeringHook, SteeringVector
from src.steering.analysis import compare_hidden_states, compute_concept_probability_shift, create_motif_evaluator

LAYER = 6
N_SET = 150
RELATIVE_STRENGTHS = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

CONCEPTS = [
    {"key": "zinc_finger", "title": "C2H2 zinc finger", "kind": "motif",
     "pattern": "C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H",
     "blurb": "DNA-reading fingers held together by a zinc ion coordinated by two Cys and two His."},
    {"key": "p_loop", "title": "P-loop (Walker A)", "kind": "motif", "pattern": "G.{4}GK[ST]",
     "blurb": "The phosphate-binding loop shared by ATP- and GTP-hydrolysing enzymes."},
    {"key": "membrane", "title": "Multi-pass membrane", "kind": "set",
     "blurb": "Proteins that thread back and forth through the lipid bilayer."},
    {"key": "secreted", "title": "Secreted", "kind": "set",
     "blurb": "Proteins exported out of the cell, like hormones, toxins and digestive enzymes."},
]


def concept_sets(pool, concept, rng):
    if concept["kind"] == "motif":
        return None, None  # SteeringVector.from_motif does the split
    sub = [p.get("subcellular", "") for p in pool]
    if concept["key"] == "membrane":
        pos = [p for p, s in zip(pool, sub) if "Multi-pass membrane protein" in s]
        neg = [p for p, s in zip(pool, sub) if s and "membrane" not in s.lower()]
    else:
        pos = [p for p, s in zip(pool, sub) if "Secreted" in s and "membrane" not in s.lower()]
        neg = [p for p, s in zip(pool, sub) if ("Cytoplasm" in s or "Nucleus" in s) and "Secreted" not in s
               and "membrane" not in s.lower()]
    pos = [pos[i] for i in rng.choice(len(pos), min(N_SET, len(pos)), replace=False)]
    neg = [neg[i] for i in rng.choice(len(neg), min(N_SET, len(neg)), replace=False)]
    return pos, neg


def composition(seqs):
    counts = np.ones(20)
    for s in seqs:
        for a in s:
            i = AMINO_ACIDS.find(a)
            if i >= 0:
                counts[i] += 1
    return counts / counts.sum()


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    pool = read_json(CACHE_DIR / "pool.json")
    tok, model = load_model(ATLAS_MODEL, masked_lm=True)
    aa_ids = [tok.convert_tokens_to_ids(a) for a in AMINO_ACIDS]

    targets = []
    for acc in STEERING_TARGETS:
        entry = uniprot_entry(acc)
        sp = next(s for s in SHOWCASE if s["acc"] == acc)
        targets.append({"acc": acc, "short": sp["short"], "title": sp["title"], "sequence": entry["sequence"]["value"]})

    out = {"model": "facebook/esm2_t12_35M_UR50D", "layer": LAYER, "relative_strengths": RELATIVE_STRENGTHS,
           "targets": [{k: t[k] for k in ("acc", "short", "title", "sequence")} for t in targets], "concepts": []}
    backend_concepts = []

    for concept in CONCEPTS:
        sv = SteeringVector(model=model, tokenizer=tok, layer_id=LAYER, device=torch.device("cpu"), max_length=512)
        pos, neg = concept_sets(pool, concept, rng)
        if concept["kind"] == "motif":
            pat = re.compile(concept["pattern"])
            pos_all = [p for p in pool if pat.search(p["sequence"])]
            neg_all = [p for p in pool if not pat.search(p["sequence"])]
            vec = sv.from_motif(pool, concept["pattern"], n_positive=N_SET, n_negative=N_SET, random_seed=SEED)
            n_pos, n_neg = min(N_SET, len(pos_all)), min(N_SET, len(neg_all))
            # composition from a same-sized random draw of each side
            fpos = composition([p["sequence"] for p in pos_all[:N_SET]])
            fneg = composition([p["sequence"] for p in neg_all[:N_SET]])
        else:
            vec = sv.compute(pos, neg)
            n_pos, n_neg = len(pos), len(neg)
            fpos = composition([p["sequence"] for p in pos])
            fneg = composition([p["sequence"] for p in neg])
        logodds = np.log(fpos / fneg)
        vec = vec.detach().float()
        log(f"{concept['key']}: {n_pos} positives / {n_neg} negatives, |v| = {vec.norm():.3f}")

        evaluator = create_motif_evaluator(concept["pattern"]) if concept["kind"] == "motif" else None
        concept_out = {k: concept[k] for k in ("key", "title", "kind", "blurb")}
        concept_out.update({"pattern": concept.get("pattern"), "n_pos": n_pos, "n_neg": n_neg,
                            "vector_norm": rnd(vec.norm().item()),
                            "top_residues": [AMINO_ACIDS[i] for i in np.argsort(logodds)[::-1][:5]],
                            "logodds": rnd(logodds, 3), "runs": []})

        for t in targets:
            enc = tok([t["sequence"]], return_tensors="pt")
            with torch.no_grad():
                base = model(**enc, output_hidden_states=True)
            h_norm = base.hidden_states[LAYER + 1][0].norm(dim=-1).mean().item()
            scale = h_norm / vec.norm().item()
            runs = []
            for r in RELATIVE_STRENGTHS:
                mult = r * scale
                with SteeringHook(model, LAYER, vec, multiplier=mult):
                    with torch.no_grad():
                        steered = model(**enc, output_hidden_states=True)
                probs = torch.softmax(steered.logits[0, 1:-1][:, aa_ids], -1).numpy()
                argmax = "".join(AMINO_ACIDS[i] for i in probs.argmax(1))
                native_idx = np.array([AMINO_ACIDS.find(a) for a in t["sequence"]])
                p_native = probs[np.arange(len(native_idx)), native_idx]
                cscore = probs @ logodds
                drift = compare_hidden_states(list(base.hidden_states), list(steered.hidden_states))
                run = {
                    "r": r, "mult": rnd(mult, 3), "seq": argmax,
                    "identity": rnd(float(np.mean([a == b for a, b in zip(argmax, t["sequence"])]))),
                    "p_native": rnd(float(p_native.mean())),
                    "concept_score": rnd(float(cscore.mean())),
                    "per_pos": rnd(cscore, 2),
                    "drift": [rnd(drift["per_layer"][f"layer_{i}"]["cosine_similarity"], 4)
                              for i in range(drift["n_layers"])],
                }
                if evaluator is not None:
                    pat = re.compile(concept["pattern"])
                    run["motifs"] = [[m.start(), m.end()] for m in pat.finditer(argmax)]
                    shift = compute_concept_probability_shift(base.logits, steered.logits, tok, evaluator,
                                                              input_ids=enc["input_ids"], method="masked_token")
                    run["concept_prob"] = rnd(shift["steered_concept_prob"], 4)
                    run["concept_shift"] = rnd(shift["probability_shift"], 4)
                runs.append(run)
            concept_out["runs"].append({"acc": t["acc"], "scale": rnd(scale, 3), "runs": runs})
            last = runs[-1]
            log(f"  {t['short']}: identity at r=3 {last['identity']:.2f}, concept score "
                f"{runs[2]['concept_score']:.3f} -> {last['concept_score']:.3f}")
        out["concepts"].append(concept_out)
        backend_concepts.append({"key": concept["key"], "title": concept["title"], "kind": concept["kind"],
                                 "pattern": concept.get("pattern"), "layer": LAYER, "vector": rnd(vec.numpy(), 5),
                                 "logodds": rnd(logodds, 4)})

    write_json(WEB_DATA_DIR / "steering.json", out)
    write_json(BACKEND_ASSETS_DIR / "concepts.json", backend_concepts)


if __name__ == "__main__":
    main()
