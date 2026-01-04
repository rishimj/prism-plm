#!/bin/bash
#SBATCH --job-name=prism-neuron-batch
#SBATCH -t 4:00:00
#SBATCH --mem-per-gpu=24G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --output=slurm-results/slurm-neuron-batch-%j.out
#SBATCH --error=slurm-results/slurm-neuron-batch-%j.err

# ============================================================================
# PRISM-Bio Batch Neuron Analysis SLURM Job
# ============================================================================
#
# Usage:
#   # Analyze top 50 most variable neurons in layer 18
#   sbatch slurm/submit_neuron_batch.sh --layer-id 18 --auto-select 50
#
#   # Analyze specific neurons
#   sbatch slurm/submit_neuron_batch.sh --layer-id 18 --unit-ids 10 50 100 200 300
#
#   # Analyze neurons from a CSV file
#   sbatch slurm/submit_neuron_batch.sh --neuron-file neurons.csv
#
# ============================================================================

set -e

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname $0)/..}"
PROJECT_DIR=$(pwd)

# Create output directories
mkdir -p slurm-results
mkdir -p outputs/neuron_batch

# Source environment setup
source slurm/setup_env.sh

# Set PyTorch memory allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "PRISM-Bio Batch Neuron Analysis"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Directory: ${PROJECT_DIR}"
echo "Arguments: $@"
echo "=============================================="

# Default config
CONFIG="${PRISM_CONFIG:-configs/models/esm2_8m.yaml}"

# Run the analysis
python3 scripts/run_neuron_batch.py --config "$CONFIG" "$@" --verbose --save-individual

echo "=============================================="
echo "Analysis complete!"
echo "=============================================="

