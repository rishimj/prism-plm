#!/bin/bash
#SBATCH --job-name=prism-steering
#SBATCH -t 2:00:00
#SBATCH --mem-per-gpu=16G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --output=slurm-results/slurm-steering-%j.out
#SBATCH --error=slurm-results/slurm-steering-%j.err

# ============================================================================
# PRISM-Bio Activation Steering SLURM Job
# ============================================================================
#
# Usage:
#   # Test with 8M model using motif mode
#   sbatch slurm/submit_steering.sh --model facebook/esm2_t6_8M_UR50D \
#       --layer-id 3 --mode motif --motif-pattern "C.{2,4}C"
#
#   # Steering with 650M model
#   sbatch slurm/submit_steering.sh --model facebook/esm2_t33_650M_UR50D \
#       --layer-id 16 --mode motif --motif-pattern "C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H"
#
# ============================================================================

set -e

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname $0)/..}"
PROJECT_DIR=$(pwd)

# Create output directories
mkdir -p slurm-results
mkdir -p outputs/steering

# Source environment setup
source slurm/setup_env.sh

# Set PyTorch memory allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "PRISM-Bio Activation Steering"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Directory: ${PROJECT_DIR}"
echo "Arguments: $@"
echo "=============================================="

# Run the steering experiment
python3 scripts/run_activation_steering.py "$@" --verbose

echo "=============================================="
echo "Steering experiment complete!"
echo "=============================================="

