#!/bin/bash
# Environment setup for PRISM-Bio on PACE
# Source this file before running any PRISM-Bio scripts

# Load required modules
module purge
module load cuda/12.1

# Setup uv (if not in PATH, adjust path as needed)
# uv should be installed in ~/.cargo/bin or ~/.local/bin
if ! command -v uv &> /dev/null; then
    # Try common installation locations
    if [ -f "${HOME}/.cargo/bin/uv" ]; then
        export PATH="${HOME}/.cargo/bin:${PATH}"
    elif [ -f "${HOME}/.local/bin/uv" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    else
        echo "Warning: uv not found in PATH"
        echo "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
fi

# Activate uv virtual environment
# Determine project directory (assume setup_env.sh is in slurm/ subdirectory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${PRISM_BIO_VENV:-${PROJECT_DIR}/.venv}"
if [ -d "${VENV_DIR}" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated uv virtual environment: ${VENV_DIR}"
else
    echo "Warning: Virtual environment '${VENV_DIR}' not found"
    echo "Create it with: cd ${PROJECT_DIR} && uv venv"
    echo "Then install requirements: cd ${PROJECT_DIR} && uv pip install -r requirements.txt"
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Set HuggingFace cache directory (use storage directory to avoid quota issues)
HF_CACHE_DIR="${PROJECT_DIR}/.hf_cache"
mkdir -p "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"

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




