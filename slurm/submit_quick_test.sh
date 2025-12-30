#!/bin/bash
#SBATCH --job-name=prism-bio-test
#SBATCH -t 1:00:00
#SBATCH --mem-per-gpu=16G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --output=slurm-results/slurm-test-%j.out
#SBATCH --error=slurm-results/slurm-test-%j.err

# ============================================================================
# PRISM-Bio Quick Test SLURM Job
# ============================================================================
#
# This runs a quick test with minimal resources to verify everything works.
#
# Usage:
#   sbatch slurm/submit_quick_test.sh
#
# ============================================================================

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname $0)/..}"
PROJECT_DIR=$(pwd)

echo "=============================================="
echo "PRISM-Bio Quick Test"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Directory: ${PROJECT_DIR}"
echo "Start time: $(date)"
echo "=============================================="

mkdir -p slurm-results
mkdir -p logs

source slurm/setup_env.sh

# Run with quick_test config
CONFIG_FILE="configs/experiments/quick_test.yaml"

echo ""
echo "=============================================="
echo "Running quick test..."
echo "=============================================="
echo "Config: ${CONFIG_FILE}"
echo ""

python scripts/run_feature_description.py \
    --config ${CONFIG_FILE} \
    --experiment-name "quick_test_${SLURM_JOB_ID}" \
    --verbose

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Quick Test Complete"
echo "=============================================="
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=============================================="

exit ${EXIT_CODE}




