#!/usr/bin/env bash
# Rebuild every data file used by the demo site and the live API.
# CPU is fine (about 1.5 hours on 4 cores, dominated by the 650M scaling run).
#   pip install -r demo/pipeline/requirements.txt
#   bash demo/pipeline/run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

python stage1_sample.py        # SwissProt sample + steering pool
python stage2_atlas.py         # neuron atlas for ESM-2 35M
python stage3_probing.py       # layer-wise structure probes
python stage4_showcase.py      # per-residue data + AlphaFold structures
python stage5_steering.py      # concept vectors and steering sweeps
python stage_scaling.py        # 8M / 35M / 150M / 650M polysemanticity
python stage6_summary.py       # hero numbers
