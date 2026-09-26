"""Stage 6: headline numbers for the site hero (all derived from the other stages' outputs)."""
from collections import Counter

from common import CACHE_DIR, WEB_DATA_DIR, read_json, write_json


def main():
    sample = read_json(CACHE_DIR / "sample.json")
    probing = read_json(WEB_DATA_DIR / "probing.json")
    index = read_json(WEB_DATA_DIR / "atlas" / "index.json")
    scaling = read_json(WEB_DATA_DIR / "scaling.json")
    showcase = read_json(WEB_DATA_DIR / "showcase" / "index.json")

    term_counts = Counter()
    for p in sample:
        term_counts.update({g[0] for g in p["go"]} | {"FAM:" + f for f in p["families"]})
    tested = sum(1 for c in term_counts.values() if c >= 3)

    scaling_neurons = sum(len(m["layers"]) * scaling["units_per_layer"] for m in scaling["models"])
    write_json(WEB_DATA_DIR / "summary.json", {
        "proteins": len(sample) + probing["n_struct_proteins"] + probing["n_tm_proteins"] + len(showcase),
        "neurons_profiled": index["n_layers"] * index["hidden"] + scaling_neurons,
        "models": len(scaling["models"]),
        "go_terms": tested,
        "residues_probed": probing["n_residues"],
        "organisms": len({p["organism"] for p in sample}),
    }, compact=False)


if __name__ == "__main__":
    main()
