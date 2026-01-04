#!/bin/bash
#SBATCH --job-name=prism-single-neuron
#SBATCH -t 2:00:00
#SBATCH --mem-per-gpu=16G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --output=slurm-results/slurm-single-neuron-%j.out
#SBATCH --error=slurm-results/slurm-single-neuron-%j.err

# ============================================================================
# PRISM-Bio Single Neuron Analysis SLURM Job
# ============================================================================
#
# Usage:
#   sbatch slurm/submit_single_neuron.sh --layer-id 18 --unit-id 100
#   sbatch slurm/submit_single_neuron.sh --layer-id 18 --unit-id 100 --config configs/models/esm2_3b.yaml
#
# ============================================================================

set -e

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname $0)/..}"
PROJECT_DIR=$(pwd)

# Create output directories
mkdir -p slurm-results
mkdir -p outputs/single_neuron

# Source environment setup
source slurm/setup_env.sh

# Set PyTorch memory allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "PRISM-Bio Single Neuron Analysis"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Directory: ${PROJECT_DIR}"
echo "Arguments: $@"
echo "=============================================="

# Run the analysis
python3 scripts/run_single_neuron.py "$@" --verbose

echo "=============================================="
echo "Analysis complete!"
echo "=============================================="

