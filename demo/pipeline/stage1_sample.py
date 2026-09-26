"""Stage 1: sample reviewed proteins from SwissProt (lhallee/SwissProt, as in configs/default.yaml).

Outputs (cache):
  pool.json    ~40k proteins used to derive steering vectors (motif / GO sets)
  sample.json  3,000-protein analysis set used for the neuron atlas
"""
import random

from common import CACHE_DIR, SEED, ensure_dirs, log, parse_families, parse_go, short_location, write_json

POOL_SIZE = 40_000
SAMPLE_SIZE = 3_000


def main():
    from datasets import load_dataset

    ensure_dirs()
    ds = load_dataset("lhallee/SwissProt", split="train", streaming=True).shuffle(seed=SEED, buffer_size=50_000)
    pool = []
    for row in ds:
        seq = row["Sequence"]
        if not seq or not (50 <= len(seq) <= 1000) or set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
            continue
        pool.append(
            {
                "id": row["Entry"],
                "name": row["Entry Name"],
                "sequence": seq,
                "organism": (row.get("Organism") or "").split(" (")[0],
                "families": parse_families(row),
                "go": parse_go(row),
                "location": short_location(row),
                "subcellular": row.get("Subcellular location [CC]") or "",
            }
        )
        if len(pool) % 5000 == 0:
            log(f"pool {len(pool)}")
        if len(pool) >= POOL_SIZE:
            break
    rng = random.Random(SEED)
    sample = rng.sample(pool, SAMPLE_SIZE)
    write_json(CACHE_DIR / "pool.json", pool)
    write_json(CACHE_DIR / "sample.json", sample)
    log(f"organisms in sample: {len(set(p['organism'] for p in sample))}")


if __name__ == "__main__":
    main()
