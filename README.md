# PRISM-Bio

**Opening up protein language models: which neurons encode real biology, and how to steer them.**

### ▶ [Live demo: rishimj.github.io/prism-plm](https://rishimj.github.io/prism-plm/)

PRISM-Bio is a mechanistic interpretability toolkit for Meta's ESM-2 protein language models. It profiles
every neuron, labels it with the biology it tracks, measures how many concepts each neuron mixes
(polysemanticity), and steers the model toward biological motifs by injecting concept vectors into its
hidden states. It builds on the [PRISM](https://github.com/rishimj/prism) framework (NeurIPS 2025).

The live demo runs a real ESM-2 model behind a public API: paste any protein sequence and see neuron
activity painted onto its AlphaFold structure, or steer it in real time.

## Highlights

- **16,128 neurons profiled** across four ESM-2 models (8M → 650M parameters)
- **3,679 real proteins** from 554 organisms, labeled against **2,905 GO terms and protein families**
- **Complete neuron atlas** of ESM-2 35M: all 5,760 neurons with polysemanticity scores and biological labels
- **Activation steering** for C2H2 zinc fingers, P-loops, membrane segments and secretion signals
- **Full-stack deployment**: research pipeline → FastAPI inference service → interactive TypeScript/D3/3Dmol.js site

## Results

**The model learns protein structure without ever seeing it.** Linear probes on ESM-2 35M hidden states,
evaluated on held-out proteins with experimental annotations (40,200 residues):

| Structure | Best probe AUROC | Accuracy | Best single-neuron AUROC |
|---|---|---|---|
| Transmembrane segment | **0.967** (layer 9) | 92.1% | 0.892 (L8 neuron 460) |
| α-helix | **0.914** (layer 11) | 85.4% | 0.675 |
| β-strand | **0.911** (layer 11) | 88.0% | 0.669 |

Structure signal climbs steadily from the embedding layer (AUROC ≈ 0.6) to the upper layers (>0.9), and a
single neuron alone detects membrane-spanning segments with 0.89 AUROC.

**Scaling study: biology is everywhere, and it's mixed.** 1,000 proteins, 128 neurons sampled per layer,
every layer of four model sizes:

| Model | Params | Layers | Neurons with a significant biological label (q < 0.05) | Median polysemanticity |
|---|---|---|---|---|
| ESM-2 8M | 8M | 6 | 80.3% | 0.951 |
| ESM-2 35M | 35M | 12 | 83.4% | 0.971 |
| ESM-2 150M | 150M | 30 | 82.0% | 0.989 |
| ESM-2 650M | 650M | 33 | 78.4% | 1.017 |

Roughly four in five neurons carry a statistically significant biological signal at every scale, while
polysemanticity rises with model size: larger models pack more concepts into each neuron.

## Using the project

### 1. Try it in the browser

Open **[rishimj.github.io/prism-plm](https://rishimj.github.io/prism-plm/)**:

- **Protein explorer**: neuron activity on AlphaFold structures of KRAS, hemoglobin, GFP, p53 and more
- **Emergent structure**: watch helices, strands and membrane segments appear layer by layer
- **Neuron atlas**: search all 5,760 neurons by label or polysemanticity
- **Scaling**: compare the four model sizes
- **Steering**: push a protein toward a motif and see how its predicted sequence changes
- **Your own sequence**: paste any protein to analyze or steer it with the live model

### 2. Call the API

```bash
curl -X POST https://prism-bio.20.25.227.252.sslip.io/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"sequence": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEY", "layer": 6}'
```

| Endpoint | Returns |
|---|---|
| `GET /api/health` | model info and available concepts |
| `POST /api/analyze` | per-residue neuron tracks, helix/strand/membrane predictions, atlas fingerprint |
| `POST /api/steer` | steered sequence, concept score, per-layer drift, motif matches |

### 3. Run it locally

```bash
git clone https://github.com/rishimj/prism-plm.git && cd prism-plm
uv sync                                     # or: pip install -r requirements.txt

# Describe neurons in any ESM-2 model
python scripts/run_feature_description.py --config configs/experiments/quick_test.yaml

# Steer the model toward a concept
python scripts/run_activation_steering.py --config configs/steering/default.yaml

# Run the demo API and site
pip install -r demo/backend/requirements.txt
uvicorn demo.backend.app.main:app --port 8000
cd demo/web && npm install && npm run dev   # http://localhost:5173
```

Everything is configurable through YAML, environment variables or CLI flags, and SLURM scripts are
included for GPU clusters (`slurm/`).

## Validation

- **Held-out evaluation**: structure probes are trained and scored on separate proteins with experimental
  UniProt annotations, so reported AUROCs measure generalization rather than memorization.
- **Statistical labeling**: neuron labels come from hypergeometric enrichment against GO terms and protein
  families with false-discovery-rate correction (q < 0.05).
- **Reproducible pipeline**: `demo/pipeline/run_all.sh` regenerates every number on the site from raw
  model runs.
- **198 automated tests** covering configuration, data loading, analysis, steering, visualization and
  the API (`pytest tests/ demo/backend/tests`).
- **Production checks**: the deployed API is verified end to end over HTTPS, including CORS for the site
  and real inference calls.

## Tech stack

**Research**: PyTorch, Hugging Face Transformers, ESM-2, scikit-learn, UMAP, HDBSCAN
**Serving**: FastAPI, Uvicorn, systemd, Caddy (automatic HTTPS), Azure VM
**Frontend**: TypeScript, Vite, D3, 3Dmol.js, GitHub Actions → GitHub Pages

## License

MIT
