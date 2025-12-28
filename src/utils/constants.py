"""Path constants for PRISM-Bio outputs - matches PRISM structure."""
from pathlib import Path
from typing import Optional

# Base paths (can be overridden by config)
BASE_OUTPUT_DIR = Path(".")

# Output directories
DESCRIPTIONS_PATH = "descriptions"  # Feature description CSVs
ASSETS_PATH = "assets"  # Assets folder
EXPLANATIONS_PATH = "assets/explanations"  # Feature explanations
RESULTS_PATH = "results"  # Evaluation results
LOGS_PATH = "logs"  # Log files
ACTIVATIONS_PATH = "activations"  # Cached activations
VISUALIZATIONS_PATH = "visualizations"  # Generated plots
SLURM_RESULTS_PATH = "slurm-results"  # SLURM outputs

# File naming templates
DESCRIPTION_FILENAME = "{model}_layer-{layer}_unit-{unit}_{timestamp}.csv"
EXPLANATION_FILENAME = "{model}_layer{layers}_{n_samples}-samples.csv"
LOG_FILENAME = "{model}_layer-{layer}_unit-{unit}_{timestamp}.log"
EVAL_LOG_FILENAME = "evaluation_{model}_{method}_{timestamp}.log"
META_EVAL_LOG_FILENAME = "meta-evaluation_{model}_{method}_{timestamp}.log"
ACTIVATION_FILENAME = (
    "{activation_type}_{model}_layer{layer}_{aggregation}_{dataset}_{n_samples}.pt"
)

# Visualization file naming
VIZ_EMBEDDING_SPACE = "embedding_space_{method}.png"
VIZ_CLUSTER_GRID = "cluster_grid_{method}.png"
VIZ_ACTIVATION_HEATMAP = "activation_heatmap_{layer}_{unit}.png"
CLUSTER_STATS_FILENAME = "cluster_statistics.json"
REPRESENTATIVE_SEQS_FILENAME = "representative_sequences.txt"


def get_base_output_dir() -> Path:
    """Get the base output directory."""
    return BASE_OUTPUT_DIR


def get_descriptions_dir(
    model_name: str, target_model: str, base_dir: Optional[Path] = None
) -> Path:
    """Get descriptions output directory for a model.

    Args:
        model_name: Name of the description generator model
        target_model: Name of the target model being analyzed
        base_dir: Optional base directory override

    Returns:
        Path to descriptions directory
    """
    base = base_dir or BASE_OUTPUT_DIR
    return base / DESCRIPTIONS_PATH / model_name / target_model


def get_explanations_dir(method_name: str, base_dir: Optional[Path] = None) -> Path:
    """Get explanations output directory for a method.

    Args:
        method_name: Name of the explanation method
        base_dir: Optional base directory override

    Returns:
        Path to explanations directory
    """
    base = base_dir or BASE_OUTPUT_DIR
    return base / EXPLANATIONS_PATH / method_name


def get_results_dir(base_dir: Optional[Path] = None) -> Path:
    """Get results output directory.

    Args:
        base_dir: Optional base directory override

    Returns:
        Path to results directory
    """
    base = base_dir or BASE_OUTPUT_DIR
    return base / RESULTS_PATH


def get_logs_dir(base_dir: Optional[Path] = None) -> Path:
    """Get logs output directory.

    Args:
        base_dir: Optional base directory override

    Returns:
        Path to logs directory
    """
    base = base_dir or BASE_OUTPUT_DIR
    return base / LOGS_PATH


def get_activations_dir(model_name: str, base_dir: Optional[Path] = None) -> Path:
    """Get activations output directory for a model.

    Args:
        model_name: Name of the model
        base_dir: Optional base directory override

    Returns:
        Path to activations directory
    """
    base = base_dir or BASE_OUTPUT_DIR
    return base / ACTIVATIONS_PATH / model_name


def get_visualizations_dir(base_dir: Optional[Path] = None) -> Path:
    """Get visualizations output directory.

    Args:
        base_dir: Optional base directory override

    Returns:
        Path to visualizations directory
    """
    base = base_dir or BASE_OUTPUT_DIR
    return base / VISUALIZATIONS_PATH


def get_slurm_results_dir(base_dir: Optional[Path] = None) -> Path:
    """Get SLURM results output directory.

    Args:
        base_dir: Optional base directory override

    Returns:
        Path to SLURM results directory
    """
    base = base_dir or BASE_OUTPUT_DIR
    return base / SLURM_RESULTS_PATH


def ensure_output_dirs(base_dir: Optional[Path] = None) -> None:
    """Create all output directories if they don't exist.

    Args:
        base_dir: Optional base directory override
    """
    base = base_dir or BASE_OUTPUT_DIR
    dirs = [
        base / DESCRIPTIONS_PATH,
        base / EXPLANATIONS_PATH,
        base / RESULTS_PATH,
        base / LOGS_PATH,
        base / ACTIVATIONS_PATH,
        base / VISUALIZATIONS_PATH,
        base / SLURM_RESULTS_PATH,
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def set_base_output_dir(path: Path) -> None:
    """Set the base output directory.

    Args:
        path: New base output directory path
    """
    global BASE_OUTPUT_DIR
    BASE_OUTPUT_DIR = Path(path)

