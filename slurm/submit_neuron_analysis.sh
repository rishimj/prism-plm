#!/bin/bash
#SBATCH --job-name=prism-bio-neuron
#SBATCH --output=slurm-results/slurm-neuron-%j.out
#SBATCH --error=slurm-results/slurm-neuron-%j.err
#SBATCH --time=04:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu=16G

# PRISM-Bio Per-Neuron Analysis SLURM Script
# Usage: sbatch slurm/submit_neuron_analysis.sh [config_file] [neuron_args...]
# Examples:
#   sbatch slurm/submit_neuron_analysis.sh configs/models/esm2_8m.yaml --top-variable 10
#   sbatch slurm/submit_neuron_analysis.sh configs/models/esm2_3b.yaml --neuron-ids 0 1 2 3 4
#   sbatch slurm/submit_neuron_analysis.sh configs/models/esm2_650m.yaml --random-neurons 20

set -e  # Exit on error

# Change to project directory (use SLURM_SUBMIT_DIR if available)
cd "${SLURM_SUBMIT_DIR:-$(dirname $0)/..}"
PROJECT_DIR=$(pwd)

# Create output directories
mkdir -p slurm-results
mkdir -p outputs/neuron_analysis
mkdir -p outputs/logs

# Source environment setup
source slurm/setup_env.sh

# Set up environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print environment info
echo "=========================================="
echo "PRISM-Bio Per-Neuron Analysis"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Directory: ${PROJECT_DIR}"
echo "Start time: $(date)"
echo "=========================================="

# Get config file (default to esm2_8m if not provided)
CONFIG_FILE="${1:-configs/models/esm2_8m.yaml}"
shift || true  # Remove first argument (config file)

# Default to top 10 most variable neurons if no neuron args provided
if [ $# -eq 0 ]; then
    NEURON_ARGS="--top-variable 10"
else
    NEURON_ARGS="$@"
fi

echo "Config: $CONFIG_FILE"
echo "Neuron args: $NEURON_ARGS"
echo "=========================================="

# Run the per-neuron analysis
python3 scripts/run_neuron_description.py \
    --config "$CONFIG_FILE" \
    --top-k-sequences 50 \
    $NEURON_ARGS

echo "=========================================="
echo "Analysis complete!"
echo "=========================================="

