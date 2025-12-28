# PRISM-Bio

**Protein Mechanistic Interpretability with PRISM**

PRISM-Bio adapts the PRISM framework for mechanistic interpretability of protein language models (ESM-2, ProtTrans, etc.).

## Features

- **Configurable**: Everything is configurable via YAML files and environment variables
- **Modular**: Switch datasets, models, clustering algorithms via config
- **Comprehensive Logging**: PRISM-style logging with config dumps
- **Visualization**: UMAP/t-SNE/PCA embedding visualizations with clustering
- **SLURM Support**: Ready-to-use SBATCH scripts for PACE cluster

## Quick Start

```bash
# 1. Create conda environment
conda create -n prism-bio python=3.11
conda activate prism-bio

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run quick test locally
python scripts/run_feature_description.py --config configs/experiments/quick_test.yaml

# 4. Or submit to SLURM
sbatch slurm/submit_quick_test.sh
```

## Configuration

Configuration follows a priority hierarchy: **CLI > ENV > YAML > Defaults**

### Using YAML Config

```bash
python scripts/run_feature_description.py --config configs/default.yaml
```

### Using Environment Variables

```bash
export PRISM_BIO_MODEL__MODEL_NAME="facebook/esm2_t6_8M_UR50D"
export PRISM_BIO_CLUSTERING__N_CLUSTERS=10
python scripts/run_feature_description.py
```

### Using CLI Overrides

```bash
python scripts/run_feature_description.py \
    --model-name facebook/esm2_t33_650M_UR50D \
    --layer-ids 16 17 \
    --n-clusters 10 \
    --max-samples 1000
```

## Available Models

Configure via `model.model_name`:

| Model | Params | Layers | Config |
|-------|--------|--------|--------|
| `facebook/esm2_t6_8M_UR50D` | 8M | 6 | `configs/models/esm2_8m.yaml` |
| `facebook/esm2_t12_35M_UR50D` | 35M | 12 | - |
| `facebook/esm2_t30_150M_UR50D` | 150M | 30 | - |
| `facebook/esm2_t33_650M_UR50D` | 650M | 33 | `configs/models/esm2_650m.yaml` |
| `facebook/esm2_t36_3B_UR50D` | 3B | 36 | `configs/models/esm2_3b.yaml` |

## Clustering Algorithms

Configure via `clustering.algorithm`:

- `kmeans` (default) - K-Means clustering
- `hdbscan` - Density-based clustering
- `spectral` - Spectral clustering
- `agglomerative` - Hierarchical clustering
- `dbscan` - DBSCAN clustering

## Visualization Methods

Configure via `visualization.reduction_method`:

- `umap` (default) - UMAP dimensionality reduction
- `tsne` - t-SNE dimensionality reduction
- `pca` - PCA dimensionality reduction

## Project Structure

```
prism-bio/
├── configs/                  # Configuration files
│   ├── default.yaml         # Default configuration
│   ├── datasets/            # Dataset-specific configs
│   ├── models/              # Model-specific configs
│   └── experiments/         # Experiment configs
├── src/                     # Source code
│   ├── config/              # Configuration system
│   ├── data/                # Data loading
│   ├── analysis/            # Analysis tools
│   └── visualization/       # Visualization
├── scripts/                 # Entry point scripts
├── slurm/                   # SLURM job scripts
├── tests/                   # Test suite
├── descriptions/            # Output: feature descriptions
├── visualizations/          # Output: plots
├── results/                 # Output: evaluation results
└── logs/                    # Output: log files
```

## Running on PACE

```bash
# Quick test (1 hour, V100)
sbatch slurm/submit_quick_test.sh

# Full analysis (24 hours, A100)
sbatch slurm/submit_feature_description.sh configs/experiments/full_analysis.yaml

# With custom config
sbatch slurm/submit_feature_description.sh configs/my_experiment.yaml
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/ -v -m unit
```

## Output Files

Outputs match the PRISM format:

- `descriptions/{model}/{target}/{model}_layer-{L}_unit-{U}_{timestamp}.csv`
- `visualizations/embedding_space_{method}.png`
- `visualizations/cluster_grid.png`
- `logs/{model}_layer-{L}_{timestamp}.log`
- `results/cosy-evaluation_{method}_{target}_{timestamp}.csv`

## License

MIT License

