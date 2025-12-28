#!/bin/bash
# Environment setup for PRISM-Bio on PACE
# Source this file before running any PRISM-Bio scripts

# Load required modules
module purge
module load anaconda3
module load cuda/12.1

# Activate conda environment
# Change this to your environment name
CONDA_ENV="${PRISM_BIO_CONDA_ENV:-prism-bio}"

if conda info --envs | grep -q "^${CONDA_ENV}"; then
    conda activate "${CONDA_ENV}"
    echo "Activated conda environment: ${CONDA_ENV}"
else
    echo "Warning: Conda environment '${CONDA_ENV}' not found"
    echo "Create it with: conda create -n ${CONDA_ENV} python=3.11"
    echo "Then install requirements: pip install -r requirements.txt"
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Set HuggingFace cache directory (optional, but recommended)
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Optional: Set HuggingFace token from file
HF_TOKEN_FILE="${HOME}/.huggingface/token"
if [[ -f "${HF_TOKEN_FILE}" ]]; then
    export HUGGING_FACE_HUB_TOKEN=$(cat "${HF_TOKEN_FILE}")
    echo "Loaded HuggingFace token from ${HF_TOKEN_FILE}"
fi

# Print configuration
echo "======================================"
echo "PRISM-Bio Environment Configuration"
echo "======================================"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "HF_HOME: ${HF_HOME}"
echo "======================================"

