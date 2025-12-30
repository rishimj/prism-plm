#!/bin/bash
#SBATCH --job-name=prism-bio-eval
#SBATCH -t 8:00:00
#SBATCH --mem-per-gpu=16G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --output=slurm-results/slurm-eval-%j.out
#SBATCH --error=slurm-results/slurm-eval-%j.err

# ============================================================================
# PRISM-Bio Evaluation SLURM Job
# ============================================================================
#
# Usage:
#   sbatch slurm/submit_evaluation.sh descriptions/model/target/descriptions.csv
#
# ============================================================================

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname $0)/..}"
PROJECT_DIR=$(pwd)

echo "=============================================="
echo "PRISM-Bio Evaluation"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Directory: ${PROJECT_DIR}"
echo "Start time: $(date)"
echo "=============================================="

mkdir -p slurm-results
mkdir -p logs
mkdir -p results

source slurm/setup_env.sh

# Get descriptions file from argument
DESCRIPTIONS_FILE="${1:-}"

if [[ -z "${DESCRIPTIONS_FILE}" ]]; then
    echo "ERROR: Must provide descriptions file as argument"
    echo "Usage: sbatch slurm/submit_evaluation.sh path/to/descriptions.csv"
    exit 1
fi

ARGS="--descriptions ${DESCRIPTIONS_FILE}"
ARGS="${ARGS} --verbose"

echo ""
echo "=============================================="
echo "Running evaluation..."
echo "=============================================="
echo "Command: python scripts/run_evaluation.py ${ARGS}"
echo ""

python scripts/run_evaluation.py ${ARGS}

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job Complete"
echo "=============================================="
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=============================================="

exit ${EXIT_CODE}




