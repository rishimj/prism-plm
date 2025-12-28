#!/bin/bash
#SBATCH --job-name=prism-bio-feat-desc
#SBATCH --account=gts-crozell3-paid
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-results/slurm-feat-desc-%j.out
#SBATCH --error=slurm-results/slurm-feat-desc-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@gatech.edu

# ============================================================================
# PRISM-Bio Feature Description SLURM Job
# ============================================================================
#
# Usage:
#   sbatch slurm/submit_feature_description.sh
#   sbatch slurm/submit_feature_description.sh --config configs/experiments/quick_test.yaml
#
# Environment variables:
#   PRISM_BIO_CONFIG - Path to config file (default: configs/default.yaml)
#   PRISM_BIO_EXPERIMENT_NAME - Override experiment name
#   PRISM_BIO_MODEL__MODEL_NAME - Override model name
#   PRISM_BIO_DATASET__MAX_SAMPLES - Override max samples
#
# ============================================================================

set -e  # Exit on error

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname $0)/..}"
PROJECT_DIR=$(pwd)

echo "=============================================="
echo "PRISM-Bio Feature Description"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Directory: ${PROJECT_DIR}"
echo "Start time: $(date)"
echo "=============================================="

# Create output directories
mkdir -p slurm-results
mkdir -p logs
mkdir -p descriptions
mkdir -p visualizations

# Source environment setup
source slurm/setup_env.sh

# Get config file from argument or environment
CONFIG_FILE="${1:-${PRISM_BIO_CONFIG:-configs/default.yaml}}"
echo "Using config: ${CONFIG_FILE}"

# Build arguments
ARGS="--config ${CONFIG_FILE}"

# Add verbose flag for debugging
ARGS="${ARGS} --verbose"

# Run the script
echo ""
echo "=============================================="
echo "Running feature description..."
echo "=============================================="
echo "Command: python scripts/run_feature_description.py ${ARGS}"
echo ""

python scripts/run_feature_description.py ${ARGS}

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job Complete"
echo "=============================================="
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=============================================="

exit ${EXIT_CODE}

