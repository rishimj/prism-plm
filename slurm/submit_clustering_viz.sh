#!/bin/bash
#SBATCH --job-name=prism-bio-cluster-viz
#SBATCH --account=gts-crozell3-paid
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:V100:1
#SBATCH --time=4:00:00
#SBATCH --output=slurm-results/slurm-cluster-viz-%j.out
#SBATCH --error=slurm-results/slurm-cluster-viz-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@gatech.edu

# ============================================================================
# PRISM-Bio Clustering Visualization SLURM Job
# ============================================================================
#
# Usage:
#   sbatch slurm/submit_clustering_viz.sh
#   sbatch slurm/submit_clustering_viz.sh activations/embeddings.pt
#
# ============================================================================

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname $0)/..}"
PROJECT_DIR=$(pwd)

echo "=============================================="
echo "PRISM-Bio Clustering Visualization"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Directory: ${PROJECT_DIR}"
echo "Start time: $(date)"
echo "=============================================="

mkdir -p slurm-results
mkdir -p logs
mkdir -p visualizations

source slurm/setup_env.sh

# Get embeddings file from argument
EMBEDDINGS_FILE="${1:-}"

# Build arguments
ARGS=""
if [[ -n "${EMBEDDINGS_FILE}" ]]; then
    ARGS="--embeddings ${EMBEDDINGS_FILE}"
fi

# Add clustering and visualization options
ARGS="${ARGS} --reduction-method ${REDUCTION_METHOD:-umap}"
ARGS="${ARGS} --clustering-method ${CLUSTERING_METHOD:-kmeans}"
ARGS="${ARGS} --n-clusters ${N_CLUSTERS:-5}"
ARGS="${ARGS} --verbose"

echo ""
echo "=============================================="
echo "Running clustering visualization..."
echo "=============================================="
echo "Command: python scripts/run_clustering_viz.py ${ARGS}"
echo ""

python scripts/run_clustering_viz.py ${ARGS}

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job Complete"
echo "=============================================="
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=============================================="

exit ${EXIT_CODE}

